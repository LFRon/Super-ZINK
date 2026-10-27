/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * NIR to SPIR-V translation.
 *
 * Scope, deliberately: this covers the NIR that st/mesa hands a driver for a
 * GLSL 1.10/1.20 style program -- derefs of shader inputs and outputs, constants,
 * straight-line and branchy ALU, and the single-block bodies that make up the
 * overwhelming majority of shaders in the wild. Everything it does not cover
 * fails with the NIR opcode named, rather than emitting something plausible and
 * wrong: a driver that draws garbage is much more expensive to debug than one
 * that refuses to link and says why.
 *
 * The structure is a two-pass walk. The first pass declares IO variables and
 * allocates a SPIR-V id per NIR block, because OpEntryPoint has to list the
 * interface in full and a branch may target a block that has not been emitted
 * yet. The second pass emits blocks in list order with the entry block first.
 *
 * Types live in a per-def table rather than being recomputed, because NIR has
 * instructions whose result type is not recoverable from the instruction itself
 * (a `load_const` carries only a bit pattern; whether it is a float or an int
 * depends on who reads it).
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/glsl_types.h"
#include "pipe/p_state.h"
#include "compiler/nir/nir.h"
#include "compiler/spirv/GLSL.std.450.h"
#include "util/log.h"
#include "util/macros.h"

#include "sz_compiler.h"
#include "sz_debug.h"
#include "sz_spirv.h"

/* Locations 0..15 hold the legacy varying slots (COL0, TEX0..TEX7, ...), so
 * generic varyings start above them. Both stages use the same offset, which is
 * all the interface matching Vulkan requires. */
#define SZ_LOC_VAR_BASE 16

#define SZ_MAX_VARIABLES 96

/* Gallium's floor for constant buffer slots per stage: the same number the
 * descriptor set layout declares. */
#define SZ_MAX_UBOS 16

enum sz_base {
   SZ_BASE_FLOAT,
   SZ_BASE_INT,
   SZ_BASE_UINT,
   SZ_BASE_BOOL,
};

struct sz_dtype {
   enum sz_base base;
   unsigned bits;
   unsigned ncomp;
};

struct sz_translate {
   struct sz_spv_module *m;
   struct nir_shader *nir;
   char **error;
   bool failed;

   /* Vertex transform feedback. Mesa hands the driver a list of captured
    * outputs whose `register_index` is the *rank* of the varying among
    * outputs_written, not its location, so the mapping has to be reconstructed
    * from the shader's own output mask. */
   const struct pipe_stream_output_info *so;

   /* Indexed by nir_def::index. */
   struct sz_dtype *def_types;
   uint32_t *defs_id;
   /* Constants are untyped in NIR and typed in SPIR-V, so a def that reaches a
    * phi as a different base type needs its own typed constant; remembering the
    * instruction here is what makes that possible. */
   nir_load_const_instr **consts;

   /* Set once OpCapability DerivativeControl has been emitted. */
   bool derivative_control;

   struct {
      const nir_variable *var;
      uint32_t id;
      uint32_t pointee;
   } vars[SZ_MAX_VARIABLES];
   unsigned num_vars;

   /* Uniform blocks are declared on first use rather than in the variable walk,
    * because the load is what carries the binding: after
    * nir_lower_uniforms_to_ubo() there is no nir_var_uniform deref left, only
    * load_ubo with an index that names a gallium constant buffer slot. */
   struct {
      unsigned index;
      /* Pointer to the block's runtime array of 32-bit words, one access chain
       * away from a typed scalar. */
      uint32_t array_ptr;
   } ubos[SZ_MAX_UBOS];
   unsigned num_ubos;

   /* Samplers are declared on first use for the same reason uniform blocks are:    * nir_lower_samplers() has already replaced every sampler deref with a plain
    * sampler_index, so the NIR no longer carries a type to declare from -- but
    * that index is exactly the texture unit whose binding the layout holds. */
   struct {
      unsigned index;
      SpvDim dim;
      bool arrayed;
      bool depth;
      /* OpTypeImage, kept so OpImageFetch and the query ops can name it. */
      uint32_t image_type;
      /* The value OpLoad produced from the variable, which is what the sampling
       * ops take as their image operand. */
      uint32_t value;
   } samplers[SZ_MAX_SAMPLERS];
   unsigned num_samplers;

   uint32_t *interfaces;
   unsigned num_interfaces;

   /* Control flow. NIR is already structured, so the emitter walks its if/loop
    * tree directly; these are the pieces of state that walking needs. Block
    * labels are assigned up front because a phi and a branch both name blocks
    * that have not been emitted yet -- the back edge of a loop always does. */
   uint32_t *labels;
   bool block_open;
   uint32_t loop_continue;
   uint32_t loop_break;
   /* Only needed to recognise the back edge: a continue issued by the continue
    * target block itself means "back to the top". */
   uint32_t loop_header;
   /* A loop without a continue construct still gets a continue target block of
    * its own, so the phis in the loop header are told the back edge arrives from
    * that block rather than from the NIR predecessor it was derived from. */
   nir_block *back_edge_pred;
   uint32_t back_edge_label;
   /* Emitted after the last block of the function, for a construct at the end of
    * the body that has no following block to merge into. */
   uint32_t end_label;

   /* Phi operands that name a value defined later, resolved once the whole
    * function has been translated. */
   struct {
      size_t offset;
      unsigned def;
   } *phi_fixups;
   unsigned num_phi_fixups;
   unsigned phi_fixups_capacity;
};

static bool sz_emit_instr(struct sz_translate *t, nir_instr *instr);
static bool sz_emit_phi(struct sz_translate *t, nir_phi_instr *phi);
static void sz_emit_cf_list_from(struct sz_translate *t, struct exec_list *list,
                                 nir_cf_node *after, uint32_t fallthrough,
                                 nir_loop *loop);

/* ------------------------------------------------------------------------- */
/* Diagnostics                                                               */
/* ------------------------------------------------------------------------- */

static void
sz_fail(struct sz_translate *t, const char *fmt, ...)
{
   if (t->failed)
      return;

   t->failed = true;

   char *msg = NULL;
   va_list ap;
   va_start(ap, fmt);
   if (vasprintf(&msg, fmt, ap) < 0)
      msg = NULL;
   va_end(ap);

   if (t->error)
      *t->error = msg;
   else
      free(msg);
}

/* ------------------------------------------------------------------------- */
/* Types                                                                     */
/* ------------------------------------------------------------------------- */

static bool
sz_base_from_glsl(enum glsl_base_type base, enum sz_base *out)
{
   switch (base) {
   case GLSL_TYPE_FLOAT:
   case GLSL_TYPE_FLOAT16:
      *out = SZ_BASE_FLOAT;
      return true;
   case GLSL_TYPE_INT:
   case GLSL_TYPE_INT16:
   case GLSL_TYPE_INT8:
      *out = SZ_BASE_INT;
      return true;
   case GLSL_TYPE_UINT:
   case GLSL_TYPE_UINT16:
   case GLSL_TYPE_UINT8:
      *out = SZ_BASE_UINT;
      return true;
   case GLSL_TYPE_BOOL:
      *out = SZ_BASE_BOOL;
      return true;
   default:
      return false;
   }
}

static bool
sz_base_from_alu_type(nir_alu_type type, enum sz_base *out)
{
   switch (nir_alu_type_get_base_type(type)) {
   case nir_type_float:
      *out = SZ_BASE_FLOAT;
      return true;
   case nir_type_int:
      *out = SZ_BASE_INT;
      return true;
   case nir_type_uint:
      *out = SZ_BASE_UINT;
      return true;
   case nir_type_bool:
      *out = SZ_BASE_BOOL;
      return true;
   default:
      return false;
   }
}

static uint32_t
sz_scalar_type(struct sz_translate *t, enum sz_base base, unsigned bits)
{
   /* A one-bit integer and a bool are the same thing in SPIR-V, and only one of
    * them exists: OpTypeInt requires 8, 16, 32 or 64 bits, so `OpTypeInt 1 0`
    * is rejected -- by the validation layer if it is installed, and by a crash
    * inside RADV's parser if it is not. NIR records a bool as a one-bit value
    * whose base type the opcode table may still call uint (iand, ior, inot on
    * booleans), so the bit width decides. */
   if (bits == 1)
      return sz_spv_type_bool(t->m);

   switch (base) {
   case SZ_BASE_FLOAT:
      return sz_spv_type_float(t->m, bits);
   case SZ_BASE_INT:
      return sz_spv_type_int(t->m, bits, true);
   case SZ_BASE_UINT:
      return sz_spv_type_int(t->m, bits, false);
   case SZ_BASE_BOOL:
      return sz_spv_type_bool(t->m);
   }
   UNREACHABLE("bad base type");
}

static uint32_t
sz_dtype_type(struct sz_translate *t, const struct sz_dtype *type)
{
   const uint32_t scalar = sz_scalar_type(t, type->base, type->bits);
   if (type->ncomp == 1)
      return scalar;
   return sz_spv_type_vec(t->m, scalar, type->ncomp);
}

static bool
sz_dtype_from_glsl(struct sz_translate *t, const struct glsl_type *glsl,
                   struct sz_dtype *out)
{
   if (glsl_type_is_array(glsl) || glsl_type_is_struct_or_ifc(glsl) ||
       glsl_type_is_matrix(glsl)) {
      sz_fail(t, "unsupported shader variable type `%s`",
              glsl_get_type_name(glsl));
      return false;
   }

   if (!sz_base_from_glsl(glsl_get_base_type(glsl), &out->base)) {
      sz_fail(t, "unsupported shader variable base type in `%s`",
              glsl_get_type_name(glsl));
      return false;
   }

   out->bits = glsl_base_type_get_bit_size(glsl_get_base_type(glsl));
   out->ncomp = glsl_type_is_vector(glsl) ? glsl_get_vector_elements(glsl) : 1;
   return true;
}

/* vecN, mov and bcsel are polymorphic: their tables declare nir_type_uint as a
 * placeholder, so reading it would type every load they touch as unsigned. The
 * type has to be looked for one step further, at the instruction that consumes
 * their result. */
static bool
sz_alu_is_passthrough(nir_op op)
{
   switch (op) {
   case nir_op_mov:
   case nir_op_vec2:
   case nir_op_vec3:
   case nir_op_vec4:
   case nir_op_vec8:
   case nir_op_vec16:
   case nir_op_bcsel:
      return true;
   default:
      return false;
   }
}

/* Walks the uses of `def` until it finds an instruction that states a base type.
 * The depth limit keeps a use cycle (a loop feeding its own induction value)
 * from turning into infinite recursion; nothing this driver compiles needs more
 * than a couple of hops. */
static bool
sz_infer_def_base(nir_def *def, unsigned depth, enum sz_base *out)
{
   if (depth > 4)
      return false;

   nir_foreach_use (src, def) {
      nir_instr *parent = nir_src_use_instr(src);

      if (parent->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(parent);
         const nir_op_info *info = &nir_op_infos[alu->op];

         if (sz_alu_is_passthrough(alu->op)) {
            if (sz_infer_def_base(&alu->def, depth + 1, out))
               return true;
            continue;
         }

         for (unsigned i = 0; i < info->num_inputs; i++) {
            if (&alu->src[i].src != src)
               continue;
            enum sz_base base;
            if (sz_base_from_alu_type(info->input_types[i], &base)) {
               *out = base;
               return true;
            }
         }
      } else if (parent->type == nir_instr_type_intrinsic) {
         nir_intrinsic_instr *intrin = nir_instr_as_intrinsic(parent);

         if (intrin->intrinsic != nir_intrinsic_store_deref ||
             src != &intrin->src[1])
            continue;

         const nir_variable *var = nir_intrinsic_get_var(intrin, 0);
         if (!var)
            continue;

         enum sz_base base;
         if (sz_base_from_glsl(glsl_get_base_type(var->type), &base)) {
            *out = base;
            return true;
         }
      } else if (parent->type == nir_instr_type_phi) {
         /* The initial value of a loop induction variable is a bare integer
          * constant, which carries no type of its own; the phi it feeds has one,
          * through the instruction that updates the variable. Following the phi
          * is what keeps an integer 0 from being emitted as a float 0.0. */
         if (sz_infer_def_base(&nir_instr_as_phi(parent)->def, depth + 1, out))
            return true;
      }
   }

   return false;
}

/* NIR does not record a base type on defs, so instructions that do not carry one
 * -- load_const, undef, and both untyped uniform-buffer loads -- are typed by
 * looking at their uses. GLSL's default for an otherwise ambiguous constant is
 * float, and that is also what the frontend produces for the cases this driver
 * can reach, so it is the fallback. */
static void
sz_guess_def_type(struct sz_translate *t, const nir_def *def,
                  struct sz_dtype *out)
{
   out->base = SZ_BASE_FLOAT;
   out->bits = def->bit_size;
   out->ncomp = def->num_components;

   if (def->bit_size == 1) {
      out->base = SZ_BASE_BOOL;
      return;
   }

   enum sz_base base;
   if (sz_infer_def_base((nir_def *)def, 0, &base))
      out->base = base;
}

/* ------------------------------------------------------------------------- */
/* Variables                                                                 */
/* ------------------------------------------------------------------------- */

static bool
sz_varying_location(struct sz_translate *t, const nir_variable *var,
                    SpvBuiltIn *builtin, bool *has_builtin, unsigned *location)
{
   const int slot = var->data.location;
   const bool frag = t->nir->info.stage == MESA_SHADER_FRAGMENT;
   *has_builtin = false;

   if (var->data.mode == nir_var_shader_in) {
      if (!frag) {
         /* Vertex elements are packed in slot order, so the element index of an
          * attribute is its rank among the attributes the shader reads, not the
          * slot itself: st_atom_array.cpp places elements with
          * util_bitcount(inputs_read & BITFIELD_MASK(slot)). Fixed-function
          * attributes (VERT_ATTRIB_POS and friends) sit below
          * VERT_ATTRIB_GENERIC0 and still take part in that ranking. */
         unsigned rank = 0;
         nir_foreach_shader_in_variable (other, t->nir) {
            if (other == var || other->data.location >= slot)
               continue;
            rank++;
         }
         *location = rank;
         return true;
      }

      if (slot == VARYING_SLOT_POS) {
         *builtin = SpvBuiltInFragCoord;
         *has_builtin = true;
         return true;
      }
   } else {
      if (!frag) {
         if (slot == VARYING_SLOT_POS) {
            /* SpvBuiltInPosition is zero, so the builtin value cannot double as
             * its own "was set" flag. */
            *builtin = SpvBuiltInPosition;
            *has_builtin = true;
            return true;
         }
      } else {
         if (slot == FRAG_RESULT_DEPTH) {
            *builtin = SpvBuiltInFragDepth;
            *has_builtin = true;
            return true;
         }
         if (slot == FRAG_RESULT_COLOR) {
            *location = 0;
            return true;
         }
         if (slot >= FRAG_RESULT_DATA0) {
            *location = slot - FRAG_RESULT_DATA0;
            return true;
         }
         sz_fail(t, "unsupported fragment output slot %d (`%s`)", slot,
                 var->name);
         return false;
      }
   }

   if (slot >= VARYING_SLOT_VAR0) {
      *location = slot - VARYING_SLOT_VAR0 + SZ_LOC_VAR_BASE;
      return true;
   }

   if (slot >= 0 && slot < SZ_LOC_VAR_BASE) {
      *location = slot;
      return true;
   }

   sz_fail(t, "unsupported varying slot %d for `%s`", slot, var->name);
   return false;
}

/* Locates the shader output variable declared at `slot`. */
static nir_variable *
sz_find_output(struct sz_translate *t, unsigned slot)
{
   nir_foreach_variable_with_modes (var, t->nir, nir_var_shader_out) {
      if ((unsigned)var->data.location == slot)
         return var;
   }
   return NULL;
}

/* Rank of `slot` among the shader's written outputs, in ascending slot order.
 *
 * st_translate_stream_output_info() builds the same ranking on the Mesa side and
 * stores it in pipe_stream_output::register_index. One subtlety is carried over:
 * st appends gl_PointSize to outputs_written when a shader writes it implicitly,
 * and excludes it from the transform-feedback list. A PointSize the application
 * did not declare explicitly is therefore not counted here either, or every
 * output above it would be captured from the wrong variable. */
static unsigned
sz_output_rank(struct sz_translate *t, unsigned slot)
{
   unsigned rank = 0;

   for (unsigned s = 0; s < slot; s++) {
      if (!(t->nir->info.outputs_written & BITFIELD64_BIT(s)))
         continue;
      if (s == VARYING_SLOT_PSIZ) {
         nir_variable *var = sz_find_output(t, s);
         if (!var || !var->data.explicit_location)
            continue;
      }
      rank++;
   }

   return rank;
}

/* Decorates one output variable with the transform-feedback placement of every
 * capture that names it. */
static void
sz_decorate_stream_output(struct sz_translate *t, const nir_variable *var,
                          uint32_t id)
{
   if (!t->so || t->so->num_outputs == 0 ||
       t->nir->info.stage != MESA_SHADER_VERTEX ||
       var->data.mode != nir_var_shader_out)
      return;

   const unsigned rank = sz_output_rank(t, (unsigned)var->data.location);

   for (unsigned i = 0; i < t->so->num_outputs; i++) {
      const struct pipe_stream_output *o = &t->so->output[i];
      if (o->register_index != rank)
         continue;

      sz_spv_decorate(t->m, id, SpvDecorationXfbBuffer, o->output_buffer);
      /* Stride and offset are in dwords on the Gallium side and in bytes in
       * SPIR-V. */
      sz_spv_decorate(t->m, id, SpvDecorationXfbStride,
                      t->so->stride[o->output_buffer] * 4);
      sz_spv_decorate(t->m, id, SpvDecorationOffset, o->dst_offset * 4);
   }
}

static bool
sz_declare_variable(struct sz_translate *t, nir_variable *var)
{
   if (t->num_vars == SZ_MAX_VARIABLES) {
      sz_fail(t, "too many shader variables");
      return false;
   }

   SpvStorageClass storage;
   SpvBuiltIn builtin = (SpvBuiltIn)0;
   bool has_builtin = false;
   unsigned location = 0;

   switch (var->data.mode) {
   case nir_var_shader_in:
      storage = SpvStorageClassInput;
      break;
   case nir_var_shader_out:
      storage = SpvStorageClassOutput;
      break;
   case nir_var_function_temp:
      /* nir_lower_vars_to_ssa() should have removed these; a leftover means the
       * pass bailed out, and pointing at it beats translating it wrong. */
      sz_fail(t, "local variable `%s` survived vars-to-SSA", var->name);
      return false;
   case nir_var_uniform: {
      /* A sampler uniform survives the lowering as a declaration with no
       * remaining user: the tex instructions name their unit directly. Its
       * type is what the translator cannot express here -- a sampler has no
       * SPIR-V representation outside a descriptor -- so it is skipped rather
       * than declared, and sz_sampler_get() declares the descriptor when a tex
       * instruction first names the unit. */
      if (glsl_type_is_sampler(glsl_without_array(var->type)))
         return true;
      sz_fail(t, "uniform `%s` survived nir_lower_uniforms_to_ubo", var->name);
      return false;
   }
   default:
      /* Images and atomic counters need descriptors that are not in the layout
       * yet. */
      sz_fail(t, "variable `%s` in mode 0x%x is not supported yet", var->name,
              (unsigned)var->data.mode);
      return false;
   }

   struct sz_dtype type;
   if (!sz_dtype_from_glsl(t, var->type, &type))
      return false;

   if (storage != SpvStorageClassFunction) {
      if (!sz_varying_location(t, var, &builtin, &has_builtin, &location))
         return false;
   }

   const uint32_t pointee = sz_dtype_type(t, &type);
   const uint32_t ptr_type = sz_spv_type_ptr(t->m, storage, pointee);
   const uint32_t id = sz_spv_variable(t->m, ptr_type, storage);

   if (has_builtin) {
      sz_spv_decorate_builtin(t->m, id, builtin);
   } else {
      sz_spv_decorate(t->m, id, SpvDecorationLocation, location);
      /* The GLSL linker packs several small varyings into one location, each at
       * its own component (`location_frac`). Without the Component decoration
       * two variables would claim the same Location, which is invalid SPIR-V on
       * the output side and reads the wrong channel on the input side. */
      if (var->data.location_frac != 0)
         sz_spv_decorate(t->m, id, SpvDecorationComponent,
                         var->data.location_frac);
   }

   sz_decorate_stream_output(t, var, id);

   /* Integer varyings have to be flat: the Vulkan rule is unconditional, and a
    * missing decoration is a validation error rather than a performance hint.
    * Flat is only meaningful -- and only legal -- on fragment stage inputs and
    * on outputs, so a vertex input never gets it. */
   const bool flat_stage = storage == SpvStorageClassOutput ||
      (storage == SpvStorageClassInput &&
       t->nir->info.stage == MESA_SHADER_FRAGMENT);
   if (flat_stage && type.base != SZ_BASE_FLOAT && type.base != SZ_BASE_BOOL)
      sz_spv_decorate_bare(t->m, id, SpvDecorationFlat);
   else if (flat_stage && storage == SpvStorageClassInput &&
            var->data.interpolation == INTERP_MODE_FLAT)
      sz_spv_decorate_bare(t->m, id, SpvDecorationFlat);

   t->vars[t->num_vars].var = var;
   t->vars[t->num_vars].id = id;
   t->vars[t->num_vars].pointee = pointee;
   t->num_vars++;

   return true;
}

static int
sz_find_variable(struct sz_translate *t, const nir_variable *var)
{
   for (unsigned i = 0; i < t->num_vars; i++)
      if (t->vars[i].var == var)
         return i;

   return -1;
}

/* ------------------------------------------------------------------------- */
/* Derefs                                                                   */
/* ------------------------------------------------------------------------- */

/* Returns the SPIR-V id of the pointer `deref` denotes, or 0 on failure. */
static uint32_t
sz_emit_deref(struct sz_translate *t, nir_deref_instr *deref)
{
   switch (deref->deref_type) {
   case nir_deref_type_var: {
      const int idx = sz_find_variable(t, deref->var);
      if (idx < 0) {
         sz_fail(t, "deref of undeclared variable `%s`", deref->var->name);
         return 0;
      }
      return t->vars[idx].id;
   }

   case nir_deref_type_cast: {
      /* The SPIR-V type system has no bitcasts on pointers, so a cast is only
       * free if the pointee already matches; anything else needs a copy. */
      nir_deref_instr *parent = nir_src_as_deref(deref->parent);
      if (!parent) {
         sz_fail(t, "deref cast without a deref parent");
         return 0;
      }
      return sz_emit_deref(t, parent);
   }

   default:
      sz_fail(t, "unsupported deref type %d (arrays and structs come with "
                 "descriptors)",
              (int)deref->deref_type);
      return 0;
   }
}

/* ------------------------------------------------------------------------- */
/* ALU                                                                      */
/* ------------------------------------------------------------------------- */

struct sz_alu_map {
   nir_op op;
   SpvOp spirv;
};

static const struct sz_alu_map sz_alu_maps[] = {
   { nir_op_fadd, SpvOpFAdd },
   { nir_op_fsub, SpvOpFSub },
   { nir_op_fmul, SpvOpFMul },
   { nir_op_fdiv, SpvOpFDiv },
   { nir_op_fmod, SpvOpFMod },
   { nir_op_frem, SpvOpFRem },
   { nir_op_fneg, SpvOpFNegate },
   { nir_op_iadd, SpvOpIAdd },
   { nir_op_isub, SpvOpISub },
   { nir_op_imul, SpvOpIMul },
   { nir_op_idiv, SpvOpSDiv },
   { nir_op_udiv, SpvOpUDiv },
   { nir_op_imod, SpvOpSMod },
   { nir_op_umod, SpvOpUMod },
   { nir_op_irem, SpvOpSRem },
   { nir_op_ineg, SpvOpSNegate },
   { nir_op_iand, SpvOpBitwiseAnd },
   { nir_op_ior, SpvOpBitwiseOr },
   { nir_op_ixor, SpvOpBitwiseXor },
   { nir_op_inot, SpvOpNot },
   { nir_op_ishl, SpvOpShiftLeftLogical },
   { nir_op_ishr, SpvOpShiftRightArithmetic },
   { nir_op_ushr, SpvOpShiftRightLogical },
   { nir_op_feq, SpvOpFOrdEqual },
   { nir_op_fneu, SpvOpFUnordNotEqual },
   { nir_op_flt, SpvOpFOrdLessThan },
   { nir_op_fge, SpvOpFOrdGreaterThanEqual },
   { nir_op_fequ, SpvOpFUnordEqual },
   { nir_op_fltu, SpvOpFUnordLessThan },
   { nir_op_fgeu, SpvOpFUnordGreaterThanEqual },
   { nir_op_fneo, SpvOpFOrdNotEqual },
   { nir_op_ieq, SpvOpIEqual },
   { nir_op_ine, SpvOpINotEqual },
   { nir_op_ilt, SpvOpSLessThan },
   { nir_op_ige, SpvOpSGreaterThanEqual },
   { nir_op_ult, SpvOpULessThan },
   { nir_op_uge, SpvOpUGreaterThanEqual },
   { nir_op_i2f32, SpvOpConvertSToF },
   { nir_op_u2f32, SpvOpConvertUToF },
   { nir_op_f2i32, SpvOpConvertFToS },
   { nir_op_f2u32, SpvOpConvertFToU },
   { nir_op_f2f32, SpvOpFConvert },
   { nir_op_i2i32, SpvOpSConvert },
   { nir_op_u2u32, SpvOpUConvert },
};

/* Opcodes that exist in GLSL.std.450 rather than in core SPIR-V. */
struct sz_glsl_map {
   nir_op op;
   uint32_t ext;
};

static const struct sz_glsl_map sz_glsl_maps[] = {
   { nir_op_fabs, GLSLstd450FAbs },
   { nir_op_iabs, GLSLstd450SAbs },
   { nir_op_fmin, GLSLstd450FMin },
   { nir_op_fmax, GLSLstd450FMax },
   { nir_op_imin, GLSLstd450SMin },
   { nir_op_imax, GLSLstd450SMax },
   { nir_op_umin, GLSLstd450UMin },
   { nir_op_umax, GLSLstd450UMax },
   { nir_op_ffloor, GLSLstd450Floor },
   { nir_op_fceil, GLSLstd450Ceil },
   { nir_op_ftrunc, GLSLstd450Trunc },
   { nir_op_fround_even, GLSLstd450RoundEven },
   { nir_op_ffract, GLSLstd450Fract },
   { nir_op_fsqrt, GLSLstd450Sqrt },
   { nir_op_frsq, GLSLstd450InverseSqrt },
   { nir_op_fsign, GLSLstd450FSign },
   { nir_op_flrp, GLSLstd450FMix },
   { nir_op_fexp2, GLSLstd450Exp2 },
   { nir_op_flog2, GLSLstd450Log2 },
   { nir_op_fpow, GLSLstd450Pow },
   { nir_op_fsin, GLSLstd450Sin },
   { nir_op_fcos, GLSLstd450Cos },
   { nir_op_ftanh, GLSLstd450Tanh },
   { nir_op_ffma, GLSLstd450Fma },
};

static uint32_t sz_glsl450;

/* Applies a source swizzle, producing a value with `ncomp` components. */
static uint32_t
sz_const_as(struct sz_translate *t, const nir_load_const_instr *lconst,
            const struct sz_dtype *want);

static uint32_t
sz_alu_source_as(struct sz_translate *t, const nir_alu_src *src,
                  const struct sz_dtype *want, unsigned ncomp)
{
   const nir_def *def = src->src.ssa;
   if (def == NULL) {
      sz_fail(t, "ALU source has no defining instruction");
      return 0;
   }
   const uint32_t base = sz_scalar_type(t, t->def_types[def->index].base,
                                        t->def_types[def->index].bits);
   const unsigned n = def->num_components;

   /* A constant has no type in NIR, only a bit pattern: the type it was given
    * when it was emitted is a guess made from its uses. When the use this call
    * serves wants a different one -- the same 0 used as the int arm of one select
    * and the float arm of another -- the guess is the wrong side, and emitting it
    * unchanged produces OpSelect with operands of two types, which SPIR-V
    * rejects. */
   const nir_load_const_instr *lconst = t->consts[def->index];
   uint32_t id = t->defs_id[def->index];
   if (want && lconst && t->def_types[def->index].base != want->base &&
       t->def_types[def->index].bits == want->bits &&
       n == want->ncomp && want->ncomp == ncomp)
      id = sz_const_as(t, lconst, want);

   bool identity = n == ncomp;
   for (unsigned i = 0; identity && i < n; i++)
      if (src->swizzle[i] != i)
         identity = false;

   if (identity)
      return id;

   uint32_t comps[4];
   for (unsigned i = 0; i < ncomp; i++) {
      uint32_t comp = id;
      if (n > 1) {
         if (n != ncomp || src->swizzle[i] != i) {
            const uint32_t indices[] = { src->swizzle[i] };
            comp = sz_spv_composite_extract(t->m, base, id, indices, 1);
         }
      }

      comps[i] = comp;
   }

   if (ncomp == 1)
      return comps[0];

   return sz_spv_composite_construct(t->m,
                                     sz_dtype_type(t, &t->def_types[def->index]),
                                     comps, ncomp);
}

static uint32_t
sz_alu_source(struct sz_translate *t, const nir_alu_src *src, unsigned ncomp)
{
   return sz_alu_source_as(t, src, NULL, ncomp);
}

static uint32_t
sz_emit_ext(struct sz_translate *t, uint32_t ext_op, const struct sz_dtype *type,
            const uint32_t *ops, unsigned count)
{
   return sz_spv_ext_inst(t->m, sz_glsl450, ext_op, sz_dtype_type(t, type), ops,
                          count);
}

/* Multiplies a bool by picking between a type's zero and one. SPIR-V has no
 * direct bool-to-number conversion; OpSelect on a zero/one pair is the idiom. */
static uint32_t
sz_emit_b2x(struct sz_translate *t, const struct sz_dtype *dst,
            uint32_t condition)
{
   const uint32_t zero = sz_spv_const_bits(
      t->m, sz_scalar_type(t, dst->base, dst->bits), 0);
   const uint32_t one = sz_spv_const_bits(
      t->m, sz_scalar_type(t, dst->base, dst->bits),
      dst->base == SZ_BASE_FLOAT ? 0x3f800000 : 1);

   uint32_t conds[4], zeros[4], ones[4];
   for (unsigned i = 0; i < dst->ncomp; i++) {
      if (dst->ncomp > 1) {
         const uint32_t idx[] = { i };
         const uint32_t bool_type = sz_spv_type_bool(t->m);
         conds[i] = sz_spv_composite_extract(t->m, bool_type, condition, idx, 1);
      } else {
         conds[i] = condition;
      }
      zeros[i] = zero;
      ones[i] = one;
   }

   if (dst->ncomp == 1) {
      const uint32_t ops[] = { condition, one, zero };
      return sz_spv_alu(t->m, SpvOpSelect, sz_dtype_type(t, dst), ops, 3);
   }

   const uint32_t z = sz_spv_composite_construct(
      t->m, sz_dtype_type(t, dst), zeros, dst->ncomp);
   const uint32_t o = sz_spv_composite_construct(
      t->m, sz_dtype_type(t, dst), ones, dst->ncomp);

   const uint32_t bool_vec = sz_spv_type_vec(
      t->m, sz_spv_type_bool(t->m), dst->ncomp);
   const uint32_t cond = sz_spv_composite_construct(t->m, bool_vec, conds,
                                                   dst->ncomp);
   const uint32_t ops[] = { cond, o, z };
   return sz_spv_alu(t->m, SpvOpSelect, sz_dtype_type(t, dst), ops, 3);
}

/* Splats a float constant over the destination type. Most ALU sources are
 * scalars here, but fsat/clamp and friends take the same operand list whether
 * the value is a float or a vec4, so the constant has to match. */
static uint32_t
sz_splat_float32(struct sz_translate *t, const struct sz_dtype *dst,
                 float value)
{
   const uint32_t scalar = sz_spv_const_float32(t->m, value);
   if (dst->ncomp == 1)
      return scalar;

   uint32_t comps[4] = { scalar, scalar, scalar, scalar };
   return sz_spv_composite_construct(t->m, sz_dtype_type(t, dst), comps,
                                     dst->ncomp);
}

static void
sz_emit_alu(struct sz_translate *t, nir_alu_instr *alu)
{
   const nir_op op = alu->op;
   struct sz_dtype dst;
   dst.bits = alu->def.bit_size;
   dst.ncomp = alu->def.num_components;

   /* A polymorphic op's type is its operand's, not the one the opcode table
    * names: mov, vecN and bcsel all record nir_type_uint as a placeholder, so
    * reading the table types a float move as unsigned -- and then a store of that
    * value into a float variable is invalid SPIR-V, which is a crash inside
    * RADV's parser rather than a validation error. */
   if (sz_alu_is_passthrough(op)) {
      const nir_def *typed = alu->src[op == nir_op_bcsel ? 1 : 0].src.ssa;
      dst.base = typed ? t->def_types[typed->index].base : SZ_BASE_FLOAT;
   } else if (!sz_base_from_alu_type(nir_op_infos[op].output_type, &dst.base)) {
      sz_fail(t, "NIR op `%s` has no SPIR-V mapping", nir_op_infos[op].name);
      return;
   }

   /* A one-bit result is a bool, whatever the opcode table calls it: the
    * comparison ops declare nir_type_uint, and the logical ops on bools come
    * from GLSL's && and ||. Deciding here rather than per op is what keeps
    * OpBitwiseAnd/OpNot away from bool operands, which SPIR-V rejects. */
   if (dst.bits == 1)
      dst.base = SZ_BASE_BOOL;

   /* Pure aliases. */
   if (op == nir_op_mov) {
      const uint32_t src = sz_alu_source_as(t, &alu->src[0], &dst, dst.ncomp);
      t->defs_id[alu->def.index] = src;
      t->def_types[alu->def.index] = dst;
      return;
   }

   if (op == nir_op_vec2 || op == nir_op_vec3 || op == nir_op_vec4 ||
       op == nir_op_vec8 || op == nir_op_vec16) {
      uint32_t comps[16];

      for (unsigned i = 0; i < dst.ncomp; i++)
         comps[i] = sz_alu_source_as(t, &alu->src[i], &dst, 1);

      const uint32_t vec = sz_spv_composite_construct(
         t->m, sz_dtype_type(t, &dst), comps, dst.ncomp);
      t->defs_id[alu->def.index] = vec;
      t->def_types[alu->def.index] = dst;
      return;
   }

   if (op == nir_op_bcsel) {
      const uint32_t cond = sz_alu_source(t, &alu->src[0], dst.ncomp);
      const uint32_t a = sz_alu_source_as(t, &alu->src[1], &dst, dst.ncomp);
      const uint32_t b = sz_alu_source_as(t, &alu->src[2], &dst, dst.ncomp);
      const uint32_t ops[] = { cond, a, b };
      t->defs_id[alu->def.index] =
         sz_spv_alu(t->m, SpvOpSelect, sz_dtype_type(t, &dst), ops, 3);
      t->def_types[alu->def.index] = dst;
      return;
   }

   if (op == nir_op_b2i32 || op == nir_op_b2f32) {
      const uint32_t cond = sz_alu_source(t, &alu->src[0], dst.ncomp);
      t->defs_id[alu->def.index] = sz_emit_b2x(t, &dst, cond);
      t->def_types[alu->def.index] = dst;
      return;
   }

   /* slt/sge/seq/sne are the "soft bool" comparisons the frontend emits when the
    * driver has no native integers: they take floats and return 1.0f or 0.0f
    * rather than a bool, so the translation is the SPIR-V float comparison
    * followed by the same select the other bool-to-number paths use. */
   {
      static const struct {
         nir_op op;
         SpvOp spirv;
      } soft_cmps[] = {
         { nir_op_slt, SpvOpFOrdLessThan },
         { nir_op_sge, SpvOpFOrdGreaterThanEqual },
         { nir_op_seq, SpvOpFOrdEqual },
         { nir_op_sne, SpvOpFOrdNotEqual },
      };

      for (unsigned i = 0; i < ARRAY_SIZE(soft_cmps); i++) {
         if (soft_cmps[i].op != op)
            continue;

         const uint32_t ops[] = {
            sz_alu_source(t, &alu->src[0], dst.ncomp),
            sz_alu_source(t, &alu->src[1], dst.ncomp),
         };
         const struct sz_dtype cmp = {
            .base = SZ_BASE_BOOL,
            .bits = 1,
            .ncomp = dst.ncomp,
         };
         const uint32_t cond = sz_spv_alu(t->m, soft_cmps[i].spirv,
                                          sz_dtype_type(t, &cmp), ops, 2);
         t->defs_id[alu->def.index] = sz_emit_b2x(t, &dst, cond);
         t->def_types[alu->def.index] = dst;
         return;
      }
   }

   if (op == nir_op_frcp) {
      /* GLSL.std.450 has no reciprocal, so it is spelled 1/x. x == 0 gives the
       * correctly-signed infinity from OpFDiv, which is what GL promises. */
      const uint32_t src = sz_alu_source(t, &alu->src[0], dst.ncomp);
      const uint32_t one = sz_splat_float32(t, &dst, 1.0f);
      const uint32_t ops[] = { one, src };
      t->defs_id[alu->def.index] =
         sz_spv_alu(t->m, SpvOpFDiv, sz_dtype_type(t, &dst), ops, 2);
      t->def_types[alu->def.index] = dst;
      return;
   }

   if (op == nir_op_fsat) {
      const uint32_t src = sz_alu_source(t, &alu->src[0], dst.ncomp);
      const uint32_t ops[] = {
         src,
         sz_splat_float32(t, &dst, 0.0f),
         sz_splat_float32(t, &dst, 1.0f),
      };
      t->defs_id[alu->def.index] = sz_emit_ext(t, GLSLstd450FClamp, &dst, ops, 3);
      t->def_types[alu->def.index] = dst;
      return;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(sz_alu_maps); i++) {
      if (sz_alu_maps[i].op != op)
         continue;

      /* GLSL's && and || lower to the *bitwise* ops on bool operands, and
       * SPIR-V has no bitwise operation on a bool: OpBitwiseAnd needs an
       * integer type. The logical set is the same truth table. */
      SpvOp spirv = sz_alu_maps[i].spirv;
      if (dst.base == SZ_BASE_BOOL) {
         switch (op) {
         case nir_op_iand:
            spirv = SpvOpLogicalAnd;
            break;
         case nir_op_ior:
            spirv = SpvOpLogicalOr;
            break;
         case nir_op_inot:
            spirv = SpvOpLogicalNot;
            break;
         case nir_op_ixor:
            /* There is no OpLogicalXor; "not equal" is the same truth table. */
            spirv = SpvOpLogicalNotEqual;
            break;
         default:
            break;
         }
      }

      /* Only the sources the op declares exist: the rest of nir_alu_src[] is
       * uninitialised, and reading it dereferences a NULL nir_def. */
      const unsigned nsrcs = MIN2(nir_op_infos[op].num_inputs, 3);
      uint32_t ops[3];
      for (unsigned j = 0; j < nsrcs; j++)
         ops[j] = sz_alu_source(t, &alu->src[j], dst.ncomp);

      if (op == nir_op_f2i1 || op == nir_op_i2i1) {
         /* "not equal to zero": a comparison, so the operand list is (a, 0). */
         const uint32_t zero = sz_spv_const_bits(
            t->m, sz_scalar_type(t, dst.base, dst.bits), 0);
         const uint32_t cmp[] = { ops[0], zero };
         t->defs_id[alu->def.index] =
            sz_spv_alu(t->m, spirv, sz_dtype_type(t, &dst), cmp, 2);
      } else {
         t->defs_id[alu->def.index] =
            sz_spv_alu(t->m, spirv, sz_dtype_type(t, &dst), ops, nsrcs);
      }
      t->def_types[alu->def.index] = dst;
      return;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(sz_glsl_maps); i++) {
      if (sz_glsl_maps[i].op != op)
         continue;

      const unsigned nsrcs = MIN2(nir_op_infos[op].num_inputs, 3);
      uint32_t ops[3];
      for (unsigned j = 0; j < nsrcs; j++)
         ops[j] = sz_alu_source(t, &alu->src[j], dst.ncomp);

      struct sz_dtype work = dst;
      /* FSqrt and friends are defined on floats; keep the declared type. */
      t->defs_id[alu->def.index] =
         sz_emit_ext(t, sz_glsl_maps[i].ext, &work, ops, nsrcs);
      t->def_types[alu->def.index] = dst;
      return;
   }

   sz_fail(t, "NIR op `%s` has no SPIR-V mapping", nir_op_infos[op].name);
}

/* ------------------------------------------------------------------------- */
/* Instructions                                                             */
/* ------------------------------------------------------------------------- */

static void
sz_emit_load_const(struct sz_translate *t, nir_load_const_instr *lconst)
{
   struct sz_dtype type;
   sz_guess_def_type(t, &lconst->def, &type);

   const uint32_t scalar = sz_scalar_type(t, type.base, type.bits);
   uint32_t comps[4];
   for (unsigned i = 0; i < type.ncomp; i++) {
      uint32_t bits;
      switch (type.bits) {
      case 64:
         bits = (uint32_t)lconst->value[i].u64;
         break;
      default:
         bits = lconst->value[i].u32;
         break;
      }
      comps[i] = sz_spv_const_bits(t->m, scalar, bits);
   }

   t->defs_id[lconst->def.index] =
      type.ncomp == 1 ? comps[0] :
                        sz_spv_const_composite(t->m, sz_dtype_type(t, &type),
                                               comps, type.ncomp);
   t->def_types[lconst->def.index] = type;
   t->consts[lconst->def.index] = lconst;
}

/* Declares (once) the uniform block for a gallium constant buffer slot and
 * returns a pointer to its word array.
 *
 * The block is deliberately untyped -- one runtime array of uint -- because
 * load_ubo addresses it by byte offset, not by member. Typing it as the NIR
 * variable's struct would mean reproducing std140 member offsets here, and the
 * load would still have to be lowered to per-component reads. */
static uint32_t
sz_ubo_array_ptr(struct sz_translate *t, unsigned index)
{
   for (unsigned i = 0; i < t->num_ubos; i++) {
      if (t->ubos[i].index == index)
         return t->ubos[i].array_ptr;
   }

   if (t->num_ubos == SZ_MAX_UBOS) {
      sz_fail(t, "more than %d uniform buffers in one shader", SZ_MAX_UBOS);
      return 0;
   }

   const uint32_t uint_type = sz_spv_type_uint32(t->m);
   const uint32_t array_type =
      sz_spv_type_array(t->m, uint_type, SZ_MAX_CONSTBUF_WORDS);
   /* Every load below works in 32-bit words, so the stride is the scalar size
    * rather than anything derived from the block's GLSL type. */
   sz_spv_decorate(t->m, array_type, SpvDecorationArrayStride, 4);

   const uint32_t block_type = sz_spv_type_struct(t->m, &array_type, 1);
   sz_spv_decorate_bare(t->m, block_type, SpvDecorationBlock);
   sz_spv_member_decorate(t->m, block_type, 0, SpvDecorationOffset, 0);

   const uint32_t block_ptr =
      sz_spv_type_ptr(t->m, SpvStorageClassUniform, block_type);
   const uint32_t var = sz_spv_variable(t->m, block_ptr, SpvStorageClassUniform);

   sz_spv_decorate(t->m, var, SpvDecorationDescriptorSet, SZ_DESCRIPTOR_SET);
   sz_spv_decorate(t->m, var, SpvDecorationBinding,
                   sz_descriptor_binding(t->nir->info.stage, index));

   /* Member 0 of the block, then the array itself: OpAccessChain cannot skip
    * the struct level, because the first index into a struct pointer selects a
    * member rather than an element. */
   const uint32_t member0 = sz_spv_const_uint32(t->m, 0);
   const uint32_t array_ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassUniform, array_type);
   const uint32_t array_ptr =
      sz_spv_access_chain(t->m, array_ptr_type, var, &member0, 1);

   t->ubos[t->num_ubos].index = index;
   t->ubos[t->num_ubos].array_ptr = array_ptr;
   t->num_ubos++;

   /* SPIR-V 1.4 and later want every global the entry point reaches in its
    * interface list, uniform blocks included. */
   t->interfaces[t->num_interfaces++] = var;

   return array_ptr;
}

/* Reinterprets a loaded 32-bit word as the load's destination type. The block is
 * untyped, so the type is applied here instead of by the load. */
static uint32_t
sz_word_as_type(struct sz_translate *t, uint32_t word, enum sz_base base)
{
   const uint32_t uint_type = sz_spv_type_uint32(t->m);

   switch (base) {
   case SZ_BASE_UINT:
      return word;
   case SZ_BASE_INT: {
      const uint32_t int_type = sz_spv_type_int32(t->m);
      return sz_spv_alu(t->m, SpvOpBitcast, int_type, &word, 1);
   }
   case SZ_BASE_FLOAT: {
      const uint32_t float_type = sz_spv_type_float32(t->m);
      return sz_spv_alu(t->m, SpvOpBitcast, float_type, &word, 1);
   }
   case SZ_BASE_BOOL: {
      const uint32_t zero = sz_spv_const_uint32(t->m, 0);
      const uint32_t ops[] = { word, zero };
      return sz_spv_alu(t->m, SpvOpINotEqual, sz_spv_type_bool(t->m), ops, 2);
   }
   }
   return 0;
}

/* `in_vec4_units` selects between the two addressing modes NIR has for uniform
 * buffers: load_ubo counts bytes, load_ubo_vec4 counts vec4s. Both are supported
 * because the frontend picks one or the other from the shader's GLSL version
 * (nir_lower_uniforms_to_ubo's load_vec4 argument comes from
 * NativeIntegers, which needs GLSL 1.30), so the choice is not ours to make. */
static void
sz_emit_load_ubo(struct sz_translate *t, nir_intrinsic_instr *intrin,
                 bool in_vec4_units)
{
   const nir_def *def = &intrin->def;
   const unsigned ncomp = def->num_components;
   const unsigned unit_bytes = in_vec4_units ? 16 : 1;

   if (def->bit_size != 32) {
      sz_fail(t, "uniform load of %u-bit values is not supported",
              def->bit_size);
      return;
   }

   /* A dynamic descriptor index would mean an array of uniform buffers bound at
    * one binding; GL binds each one separately, so the index is always the
    * constant slot st chose. */
   const nir_const_value *index_value = nir_src_as_const_value(intrin->src[0]);
   if (!index_value) {
      sz_fail(t, "indirect uniform buffer selection is not supported");
      return;
   }

   const unsigned ubo = index_value[0].u32;
   const uint32_t array_ptr = sz_ubo_array_ptr(t, ubo);
   if (!array_ptr)
      return;

   /* nir_lower_uniforms_to_ubo() puts the whole byte offset in src[1] and leaves
    * base at zero. load_ubo has no BASE index at all -- only load_ubo_vec4 does
    * -- and nir_intrinsic_base() asserts on that in a debug build and reads
    * const_index[-1], the previous intrinsic's metadata, in a release build. A
    * garbage base of one word is enough to make a shader read the wrong uniform
    * and, for gl_FbWposYTransform, flip gl_FragCoord.y into nonsense. */
   const unsigned base =
      nir_intrinsic_has_base(intrin) ? (unsigned)nir_intrinsic_base(intrin) : 0u;

   const uint32_t uint_type = sz_spv_type_uint32(t->m);
   const uint32_t word_ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassUniform, uint_type);

   uint32_t first_word = 0;
   const nir_const_value *offset_value = nir_src_as_const_value(intrin->src[1]);
   if (offset_value) {
      const uint64_t bytes = (base + (uint64_t)offset_value[0].u32) * unit_bytes;
      if (bytes % 4) {
         sz_fail(t, "uniform load at byte offset %" PRIu64 " is not word "
                    "aligned", bytes);
         return;
      }
      first_word = sz_spv_const_uint32(t->m, (uint32_t)(bytes / 4));
   } else {
      /* std140 keeps every member 4-byte aligned and the vec4 mode is 16-byte
       * aligned by construction, so a shift is exact in both directions. */
      const uint32_t dynamic = t->defs_id[intrin->src[1].ssa->index];
      const uint32_t base_id = sz_spv_const_uint32(t->m, base);
      const uint32_t two = sz_spv_const_uint32(t->m, 2);
      const uint32_t sum_ops[] = { dynamic, base_id };
      const uint32_t sum = sz_spv_alu(t->m, SpvOpIAdd, uint_type, sum_ops, 2);
      const uint32_t shift_ops[] = { sum, two };
      first_word = sz_spv_alu(t->m,
                              in_vec4_units ? SpvOpShiftLeftLogical
                                            : SpvOpShiftRightLogical,
                              uint_type, shift_ops, 2);
   }

   struct sz_dtype type = {
      .base = SZ_BASE_FLOAT,
      .bits = def->bit_size,
      .ncomp = ncomp,
   };
   /* load_ubo carries the type it wants the bytes read as, which is the only
    * reliable source here: the block itself is untyped. */
   if (!sz_base_from_alu_type(nir_intrinsic_instr_dest_type(intrin),
                              &type.base))
      sz_guess_def_type(t, def, &type);

   uint32_t comps[4];
   const bool constant_index = offset_value != NULL;

   for (unsigned i = 0; i < ncomp; i++) {
      uint32_t word_index = first_word;
      if (i > 0 || !constant_index) {
         const uint32_t add = sz_spv_const_uint32(t->m, i);
         const uint32_t ops[] = { first_word, add };
         word_index = sz_spv_alu(t->m, SpvOpIAdd, uint_type, ops, 2);
      }

      const uint32_t elem =
         sz_spv_access_chain(t->m, word_ptr_type, array_ptr, &word_index, 1);
      const uint32_t word = sz_spv_load(t->m, uint_type, elem);
      comps[i] = sz_word_as_type(t, word, type.base);
   }

   t->def_types[def->index] = type;
   t->defs_id[def->index] =
      ncomp == 1 ? comps[0]
                 : sz_spv_composite_construct(t->m, sz_dtype_type(t, &type),
                                              comps, ncomp);
}

static void
sz_emit_load_deref(struct sz_translate *t, nir_intrinsic_instr *intrin)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (!deref) {
      sz_fail(t, "load_deref without a deref source");
      return;
   }

   const uint32_t ptr = sz_emit_deref(t, deref);
   if (!ptr)
      return;

   struct sz_dtype type;
   if (!sz_dtype_from_glsl(t, deref->type, &type))
      return;

   const uint32_t loaded = sz_spv_load(t->m, sz_dtype_type(t, &type), ptr);
   t->defs_id[intrin->def.index] = loaded;
   t->def_types[intrin->def.index] = type;
}

static const char *
sz_base_name(enum sz_base base)
{
   switch (base) {
   case SZ_BASE_FLOAT:
      return "float";
   case SZ_BASE_INT:
      return "int";
   case SZ_BASE_UINT:
      return "uint";
   case SZ_BASE_BOOL:
      return "bool";
   }
   return "unknown";
}

static void
sz_emit_store_deref(struct sz_translate *t, nir_intrinsic_instr *intrin)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (!deref) {
      sz_fail(t, "store_deref without a deref source");
      return;
   }

   const uint32_t ptr = sz_emit_deref(t, deref);
   if (!ptr)
      return;

   struct sz_dtype dst_type;
   if (!sz_dtype_from_glsl(t, deref->type, &dst_type))
      return;

   const nir_def *value = intrin->src[1].ssa;
   const unsigned ncomp = intrin->num_components;
   const uint32_t value_id = t->defs_id[value->index];

   if (ncomp == dst_type.ncomp) {
      /* SPIR-V requires the stored value's type to match the pointee exactly, and
       * a mismatch is not a validation error the loader reports: it makes the
       * driver's SPIR-V parser read the wrong type. Nothing here converts between
       * bases, so a mismatch is a bug in the translator and is reported as one. */
      if (t->def_types[value->index].base != dst_type.base) {
         sz_fail(t, "store_deref of a %s value into a %s variable",
                 sz_base_name(t->def_types[value->index].base),
                 sz_base_name(dst_type.base));
         return;
      }
      sz_spv_store(t->m, ptr, value_id);
      return;
   }

   /* A partial write -- `gl_Position.xy = ...` -- has to become one store per
    * component, because SPIR-V cannot store into a subset of a pointee. */
   const uint32_t scalar = sz_scalar_type(t, dst_type.base, dst_type.bits);
   const uint32_t ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassOutput, scalar);

   for (unsigned i = 0; i < ncomp; i++) {
      uint32_t comp = value_id;
      if (ncomp > 1) {
         const uint32_t idx[] = { i };
         comp = sz_spv_composite_extract(t->m, scalar, value_id, idx, 1);
      }

      const uint32_t index = sz_spv_const_uint32(t->m, i);
      const uint32_t elem = sz_spv_access_chain(t->m, ptr_type, ptr, &index, 1);
      sz_spv_store(t->m, elem, comp);
   }
}

static void
sz_emit_jump(struct sz_translate *t, nir_jump_instr *jump, nir_block *block)
{
   switch (jump->type) {
   case nir_jump_return:
      sz_spv_return(t->m);
      t->block_open = false;
      return;

   case nir_jump_goto: {
      const uint32_t target = t->labels[jump->target->index];
      sz_spv_branch(t->m, target);
      t->block_open = false;
      return;
   }

   case nir_jump_goto_if: {
      const uint32_t cond = t->defs_id[jump->condition.ssa->index];
      const uint32_t ops[] = {
         cond,
         t->labels[jump->target->index],
         t->labels[jump->else_target ? jump->else_target->index : block->index],
      };
      sz_spv_branch_conditional(t->m, ops[0], ops[1], ops[2]);
      t->block_open = false;
      return;
   }

   case nir_jump_halt:
   case nir_jump_abort:
      /* Both terminate the invocation; the driver has no discards yet, so the
       * closest legal SPIR-V is a return. */
      sz_spv_return(t->m);
      t->block_open = false;
      return;

   case nir_jump_break:
      if (!t->loop_break) {
         sz_fail(t, "break outside of a loop");
         return;
      }
      sz_spv_branch(t->m, t->loop_break);
      t->block_open = false;
      return;

   case nir_jump_continue: {
      if (!t->loop_continue) {
         sz_fail(t, "continue outside of a loop");
         return;
      }
      const uint32_t target = t->loop_continue == t->labels[block->index]
                                 ? t->loop_header
                                 : t->loop_continue;
      sz_spv_branch(t->m, target);
      t->block_open = false;
      return;
   }
   }

   sz_fail(t, "unknown jump type %d", (int)jump->type);
}

/* ------------------------------------------------------------------------- */
/* Texture sampling                                                         */
/* ------------------------------------------------------------------------- */

/* Vulkan's image dimensionality, from the NIR sampler dimension. RECT and
 * EXTERNAL both map onto 2D: Vulkan has no separate rectangle sampler, and a
 * texture that is addressed with normalised coordinates anyway is exactly what
 * a 2D image is. */
static bool
sz_sampler_dim(struct sz_translate *t, enum glsl_sampler_dim dim, SpvDim *out)
{
   switch (dim) {
   case GLSL_SAMPLER_DIM_1D:
      *out = SpvDim1D;
      return true;
   case GLSL_SAMPLER_DIM_2D:
   case GLSL_SAMPLER_DIM_RECT:
   case GLSL_SAMPLER_DIM_EXTERNAL:
      *out = SpvDim2D;
      return true;
   case GLSL_SAMPLER_DIM_3D:
      *out = SpvDim3D;
      return true;
   case GLSL_SAMPLER_DIM_CUBE:
      *out = SpvDimCube;
      return true;
   case GLSL_SAMPLER_DIM_MS:
      *out = SpvDim2D;
      return true;
   case GLSL_SAMPLER_DIM_BUF:
      *out = SpvDimBuffer;
      return true;
   default:
      sz_fail(t, "sampler dimension %d is not supported", (int)dim);
      return false;
   }
}

/* The value a def was translated into, or zero if it was never translated. */
static uint32_t
sz_def_value(struct sz_translate *t, const nir_def *def)
{
   if (def == NULL) {
      sz_fail(t, "texture instruction is missing a source");
      return 0;
   }

   const uint32_t id = t->defs_id[def->index];
   if (id == 0)
      sz_fail(t, "texture source was never translated");

   return id;
}

/* The sampled image value for `tex`'s texture unit, declaring the descriptor
 * variable the first time the unit appears in the shader. */
static uint32_t
sz_sampler_get(struct sz_translate *t, const nir_tex_instr *tex, SpvDim dim)
{
   const unsigned index = tex->sampler_index;
   const bool arrayed = tex->is_array;
   const bool depth = tex->is_shadow;

   for (unsigned i = 0; i < t->num_samplers; i++) {
      if (t->samplers[i].index != index)
         continue;

      /* The same unit used with two different types cannot be described by one
       * binding: a VkDescriptorSetLayoutBinding has a single descriptor type and
       * a SPIR-V variable a single pointee type. GL allows the aliasing, so say
       * so instead of sampling through the wrong declaration. */
      if (t->samplers[i].dim != dim || t->samplers[i].arrayed != arrayed ||
          t->samplers[i].depth != depth) {
         sz_fail(t, "texture unit %u is sampled as two different texture types "
                 "in one shader", index);
         return 0;
      }

      return t->samplers[i].value;
   }

   if (index >= SZ_MAX_SAMPLERS) {
      sz_fail(t, "texture unit %u is past the %d the descriptor layout holds",
              index, SZ_MAX_SAMPLERS);
      return 0;
   }

   /* The texel component type: GLSL samplers are float, integer and unsigned
    * integer textures, and SPIR-V needs the matching scalar type because it is
    * what decides how the sampled values are converted. */
   enum sz_base base;
   if (!sz_base_from_alu_type(tex->dest_type, &base)) {
      sz_fail(t, "unsupported texture result type %d", (int)tex->dest_type);
      return 0;
   }

   const uint32_t sampled_type = sz_scalar_type(t, base, 32);
   const uint32_t image_type = sz_spv_type_image(
      t->m, sampled_type, dim, depth, arrayed, false, true);
   const uint32_t sampled_image_type = sz_spv_type_sampled_image(t->m, image_type);
   const uint32_t ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassUniformConstant, sampled_image_type);
   const uint32_t var =
      sz_spv_variable(t->m, ptr_type, SpvStorageClassUniformConstant);

   sz_spv_decorate(t->m, var, SpvDecorationDescriptorSet, SZ_DESCRIPTOR_SET);
   sz_spv_decorate(t->m, var, SpvDecorationBinding,
                   sz_sampler_binding(t->nir->info.stage, index));

   /* A combined image sampler is read as a whole, so one OpLoad yields both
    * halves; the load is emitted once and its result reused by every instruction
    * that samples this unit. */
   const uint32_t value = sz_spv_load(t->m, sampled_image_type, var);

   t->samplers[t->num_samplers].index = index;
   t->samplers[t->num_samplers].dim = dim;
   t->samplers[t->num_samplers].arrayed = arrayed;
   t->samplers[t->num_samplers].depth = depth;
   t->samplers[t->num_samplers].image_type = image_type;
   t->samplers[t->num_samplers].value = value;
   t->num_samplers++;

   t->interfaces[t->num_interfaces++] = var;

   return value;
}

/* SPIR-V has no projective sampling instruction, so the divide that GL's
 * texture2DProj implies happens here. The layer component of an arrayed
 * coordinate is not part of the projection. */
static uint32_t
sz_emit_projector(struct sz_translate *t, uint32_t coord, unsigned ncomp,
                  bool arrayed, uint32_t projector)
{
   const unsigned projected = arrayed ? ncomp - 1 : ncomp;
   const uint32_t float_type = sz_spv_type_float32(t->m);
   const uint32_t vec_type = sz_spv_type_vec(t->m, float_type, ncomp);

   uint32_t comps[4];
   for (unsigned i = 0; i < ncomp; i++) {
      uint32_t comp;
      if (ncomp > 1) {
         const uint32_t index[] = { i };
         comp = sz_spv_composite_extract(t->m, float_type, coord, index, 1);
      } else {
         comp = coord;
      }

      if (i < projected) {
         const uint32_t ops[] = { comp, projector };
         comp = sz_spv_alu(t->m, SpvOpFDiv, float_type, ops, 2);
      }

      comps[i] = comp;
   }

   if (ncomp == 1)
      return comps[0];

   return sz_spv_composite_construct(t->m, vec_type, comps, ncomp);
}

static void
sz_emit_tex(struct sz_translate *t, nir_tex_instr *tex)
{
   if (tex->is_sparse) {
      sz_fail(t, "sparse texture reads are not supported");
      return;
   }

   const nir_def *coord = NULL, *bias = NULL, *lod = NULL, *dref = NULL;
   const nir_def *ddx = NULL, *ddy = NULL, *projector = NULL;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      const nir_tex_src *src = &tex->src[i];
      switch (src->src_type) {
      case nir_tex_src_coord:
         coord = src->src.ssa;
         break;
      case nir_tex_src_bias:
         bias = src->src.ssa;
         break;
      case nir_tex_src_lod:
         lod = src->src.ssa;
         break;
      case nir_tex_src_comparator:
         dref = src->src.ssa;
         break;
      case nir_tex_src_ddx:
         ddx = src->src.ssa;
         break;
      case nir_tex_src_ddy:
         ddy = src->src.ssa;
         break;
      case nir_tex_src_projector:
         projector = src->src.ssa;
         break;
      default:
         /* Sampler arrays arrive as an offset added to sampler_index, which the
          * descriptor layout cannot express: a dynamic index would need
          * VK_EXT_descriptor_indexing. GLSL 1.20 has no sampler arrays, so this
          * is a shader this driver should not have accepted. */
         sz_fail(t, "texture source type %d is not supported yet",
                 (int)src->src_type);
         return;
      }
   }

   if (coord == NULL) {
      sz_fail(t, "texture instruction has no coordinate");
      return;
   }

   SpvDim dim;
   if (!sz_sampler_dim(t, tex->sampler_dim, &dim))
      return;

   const unsigned ncomp = nir_tex_instr_dest_size(tex);
   struct sz_dtype result_type = {
      .bits = 32,
      .ncomp = ncomp,
   };
   if (!sz_base_from_alu_type(tex->dest_type, &result_type.base)) {
      sz_fail(t, "unsupported texture result type %d", (int)tex->dest_type);
      return;
   }
   const uint32_t type = sz_dtype_type(t, &result_type);

   /* Query operations read the image and not the sampler, but the descriptor is
    * a combined one, so it has to be declared either way. The depth flag is
    * cleared for them: a shadow sample is the only thing that sets Depth. */
   const bool needs_sampler =
      tex->op != nir_texop_txs && tex->op != nir_texop_query_levels &&
      tex->op != nir_texop_texture_samples;
   const uint32_t sampled_image =
      sz_sampler_get(t, tex, dim);
   if (sampled_image == 0)
      return;

   unsigned slot = 0;
   for (; slot < t->num_samplers; slot++)
      if (t->samplers[slot].index == tex->sampler_index)
         break;
   const uint32_t image_type = t->samplers[slot].image_type;

   uint32_t coord_value = sz_def_value(t, coord);
   if (coord_value == 0)
      return;

   if (projector != NULL) {
      const uint32_t projector_value = sz_def_value(t, projector);
      if (projector_value == 0)
         return;
      coord_value = sz_emit_projector(t, coord_value, tex->coord_components,
                                      tex->is_array, projector_value);
   }

   /* Extra operands after the coordinate: for the sampled ops these are the
    * ImageOperands mask followed by one value per bit, in bit order. */
   uint32_t image_operands[3];
   unsigned extra = 0;

   SpvOp opcode;
   switch (tex->op) {
   case nir_texop_tex:
   case nir_texop_txb:
      /* A bias is part of implicit-LOD sampling: GL's texture2D's optional bias
       * argument and texture2DProj's, plus implicit-LOD texture2D in a fragment
       * shader that asked for it. */
      opcode = tex->is_shadow ? SpvOpImageSampleDrefImplicitLod
                              : SpvOpImageSampleImplicitLod;
      if (bias != NULL) {
         const uint32_t value = sz_def_value(t, bias);
         if (value == 0)
            return;
         image_operands[extra++] = SpvImageOperandsBiasMask;
         image_operands[extra++] = value;
      }
      break;
   case nir_texop_txl:
      opcode = tex->is_shadow ? SpvOpImageSampleDrefExplicitLod
                              : SpvOpImageSampleExplicitLod;
      if (lod == NULL) {
         sz_fail(t, "explicit-LOD sample has no LOD");
         return;
      }
      image_operands[extra++] = SpvImageOperandsLodMask;
      image_operands[extra++] = sz_def_value(t, lod);
      break;
   case nir_texop_txd:
      opcode = tex->is_shadow ? SpvOpImageSampleDrefExplicitLod
                              : SpvOpImageSampleExplicitLod;
      if (ddx == NULL || ddy == NULL) {
         sz_fail(t, "gradient sample has no gradients");
         return;
      }
      image_operands[extra++] = SpvImageOperandsGradMask;
      image_operands[extra++] = sz_def_value(t, ddx);
      image_operands[extra++] = sz_def_value(t, ddy);
      break;
   case nir_texop_txf:
   case nir_texop_txf_ms: {
      /* Fetch takes a bare image rather than a sampled one: the sampler state is
       * not consulted at all, so the combined sampler is unwrapped. */
      const uint32_t image = sz_spv_alu(t->m, SpvOpImage, image_type,
                                        &sampled_image, 1);
      uint32_t ops[4];
      unsigned n = 0;
      ops[n++] = image;
      ops[n++] = coord_value;
      if (tex->is_array) {
         /* The array index of a fetch is an integer operand, not part of the
          * coordinate vector. */
         sz_fail(t, "arrayed texel fetches are not supported yet");
         return;
      }
      const uint32_t value =
         sz_spv_alu(t->m, SpvOpImageFetch, type, ops, n);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   case nir_texop_txs: {
      const uint32_t image = needs_sampler
         ? sz_spv_alu(t->m, SpvOpImage, image_type, &sampled_image, 1)
         : sampled_image;
      const uint32_t zero = sz_spv_const_int32(t->m, 0);
      const uint32_t ops[] = { image, zero };
      const uint32_t value =
         sz_spv_alu(t->m, dim == SpvDimBuffer ? SpvOpImageQuerySize
                                              : SpvOpImageQuerySizeLod,
                    type, ops, dim == SpvDimBuffer ? 1 : 2);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   case nir_texop_query_levels: {
      const uint32_t image = sz_spv_alu(t->m, SpvOpImage, image_type,
                                        &sampled_image, 1);
      const uint32_t value =
         sz_spv_alu(t->m, SpvOpImageQueryLevels, type, &image, 1);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   case nir_texop_texture_samples: {
      const uint32_t image = sz_spv_alu(t->m, SpvOpImage, image_type,
                                        &sampled_image, 1);
      const uint32_t value =
         sz_spv_alu(t->m, SpvOpImageQuerySamples, type, &image, 1);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   case nir_texop_tg4: {
      const uint32_t image = sz_spv_alu(t->m, SpvOpImage, image_type,
                                        &sampled_image, 1);
      const uint32_t component = sz_spv_const_int32(t->m, tex->component);
      uint32_t ops[3];
      unsigned n = 0;
      ops[n++] = image;
      ops[n++] = coord_value;
      ops[n++] = component;
      const uint32_t value = sz_spv_alu(
         t->m, tex->is_shadow ? SpvOpImageDrefGather : SpvOpImageGather, type,
         ops, n);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   case nir_texop_lod: {
      const uint32_t ops[] = { sampled_image, coord_value };
      const uint32_t value =
         sz_spv_alu(t->m, SpvOpImageQueryLod,
                    sz_spv_type_vec(t->m, sz_spv_type_float32(t->m), 2), ops, 2);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   default:
      sz_fail(t, "texture opcode %d is not supported yet", (int)tex->op);
      return;
   }

   uint32_t ops[5];
   unsigned n = 0;
   ops[n++] = sampled_image;
   ops[n++] = coord_value;
   if (tex->is_shadow) {
      if (dref == NULL) {
         sz_fail(t, "shadow sample has no comparator");
         return;
      }
      ops[n++] = sz_def_value(t, dref);
   }
   for (unsigned i = 0; i < extra; i++)
      ops[n++] = image_operands[i];

   const uint32_t value = sz_spv_alu(t->m, opcode, type, ops, n);
   t->defs_id[tex->def.index] = value;
   t->def_types[tex->def.index] = result_type;
}

/* Screen-space derivatives. They are intrinsics rather than ALU ops in this
 * Mesa, and the fine/coarse forms need the DerivativeControl capability. */
static void
sz_emit_derivative(struct sz_translate *t, nir_intrinsic_instr *intrin)
{
   SpvOp op;
   bool needs_capability = false;

   switch (intrin->intrinsic) {
   case nir_intrinsic_ddx:
      op = SpvOpDPdx;
      break;
   case nir_intrinsic_ddy:
      op = SpvOpDPdy;
      break;
   case nir_intrinsic_ddx_fine:
      op = SpvOpDPdxFine;
      needs_capability = true;
      break;
   case nir_intrinsic_ddy_fine:
      op = SpvOpDPdyFine;
      needs_capability = true;
      break;
   case nir_intrinsic_ddx_coarse:
      op = SpvOpDPdxCoarse;
      needs_capability = true;
      break;
   case nir_intrinsic_ddy_coarse:
      op = SpvOpDPdyCoarse;
      needs_capability = true;
      break;
   default:
      UNREACHABLE("not a derivative intrinsic");
   }

   /* Emitted once: a module may contain any number of derivatives, and a
    * repeated OpCapability is at best redundant. */
   if (needs_capability && !t->derivative_control) {
      sz_spv_capability(t->m, SpvCapabilityDerivativeControl);
      t->derivative_control = true;
   }

   const struct sz_dtype dst = {
      .base = SZ_BASE_FLOAT,
      .bits = intrin->def.bit_size,
      .ncomp = intrin->def.num_components,
   };
   const uint32_t src = sz_def_value(t, intrin->src[0].ssa);
   t->defs_id[intrin->def.index] =
      sz_spv_alu(t->m, op, sz_dtype_type(t, &dst), &src, 1);
   t->def_types[intrin->def.index] = dst;
}

static bool
sz_emit_intrinsic(struct sz_translate *t, nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
   case nir_intrinsic_ddx:
   case nir_intrinsic_ddy:
   case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy_fine:
   case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy_coarse:
      sz_emit_derivative(t, intrin);
      return true;
   case nir_intrinsic_load_ubo:
      sz_emit_load_ubo(t, intrin, false);
      return true;
   case nir_intrinsic_load_ubo_vec4:
      sz_emit_load_ubo(t, intrin, true);
      return true;
   case nir_intrinsic_load_uniform:
      /* The state tracker is supposed to have lowered these away; reaching here
       * means nir_options went back to leaving the default block as derefs. */
      sz_fail(t, "load_uniform survived nir_lower_uniforms_to_ubo");
      return false;
   case nir_intrinsic_load_deref:
      sz_emit_load_deref(t, intrin);
      return true;
   case nir_intrinsic_store_deref:
      sz_emit_store_deref(t, intrin);
      return true;
   default:
      sz_fail(t, "intrinsic `%s` is not supported yet",
              nir_intrinsic_infos[intrin->intrinsic].name);
      return false;
   }
}

static bool
sz_emit_instr(struct sz_translate *t, nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      sz_emit_alu(t, nir_instr_as_alu(instr));
      return true;
   case nir_instr_type_deref: {
      nir_deref_instr *deref = nir_instr_as_deref(instr);
      const uint32_t ptr = sz_emit_deref(t, deref);
      if (!ptr)
         return false;
      t->defs_id[deref->def.index] = ptr;
      /* A pointer has no value type; only the id is read, by nothing but a
       * later deref of this one. */
      t->def_types[deref->def.index] = (struct sz_dtype){ SZ_BASE_INT, 64, 1 };
      return true;
   }
   case nir_instr_type_intrinsic:
      return sz_emit_intrinsic(t, nir_instr_as_intrinsic(instr));
   case nir_instr_type_tex:
      sz_emit_tex(t, nir_instr_as_tex(instr));
      return !t->failed;
   case nir_instr_type_load_const:
      sz_emit_load_const(t, nir_instr_as_load_const(instr));
      return true;
   case nir_instr_type_undef: {
      nir_undef_instr *undef = nir_instr_as_undef(instr);
      struct sz_dtype type;
      sz_guess_def_type(t, &undef->def, &type);
      t->defs_id[undef->def.index] = sz_spv_undef(t->m, sz_dtype_type(t, &type));
      t->def_types[undef->def.index] = type;
      return true;
   }
   case nir_instr_type_jump:
      /* Terminators are emitted with the block they end. */
      return true;
   case nir_instr_type_phi:
      return sz_emit_phi(t, nir_instr_as_phi(instr));
   default:
      sz_fail(t, "NIR instruction type %d is not supported yet",
              (int)instr->type);
      return false;
   }
}

/* ------------------------------------------------------------------------- */
/* Phi                                                                       */
/* ------------------------------------------------------------------------- */

/* Records a phi operand whose value has not been translated yet. A loop header's
 * phi always has one, because the value that arrives over the back edge is
 * defined inside the body, which is emitted after the header. */
static bool
sz_phi_fixup_add(struct sz_translate *t, size_t offset, unsigned def)
{
   if (t->num_phi_fixups == t->phi_fixups_capacity) {
      const unsigned capacity = MAX2(t->phi_fixups_capacity * 2, 16);
      void *fixups = realloc(t->phi_fixups, capacity * sizeof(*t->phi_fixups));
      if (!fixups) {
         sz_fail(t, "out of memory tracking phi operands");
         return false;
      }
      t->phi_fixups = fixups;
      t->phi_fixups_capacity = capacity;
   }

   t->phi_fixups[t->num_phi_fixups].offset = offset;
   t->phi_fixups[t->num_phi_fixups].def = def;
   t->num_phi_fixups++;
   return true;
}

/* NIR constants are untyped bit patterns, so one `load_const` can be the float
 * 0.0 for one phi and the integer 0 for another -- which is exactly what an
 * induction variable's initial value and an accumulator look like. SPIR-V
 * constants are typed, so a use that disagrees with the type the def was emitted
 * with gets a constant of its own from the same bits. A conversion instruction
 * would not do here: the OpBitcast would have to sit in the predecessor block to
 * dominate the phi, but the phi is emitted first. */
static uint32_t
sz_const_as(struct sz_translate *t, const nir_load_const_instr *lconst,
            const struct sz_dtype *want)
{
   const uint32_t scalar = sz_scalar_type(t, want->base, want->bits);
   uint32_t comps[4];
   for (unsigned i = 0; i < want->ncomp; i++) {
      const uint32_t bits = want->bits == 64 ? (uint32_t)lconst->value[i].u64
                                             : lconst->value[i].u32;
      comps[i] = sz_spv_const_bits(t->m, scalar, bits);
   }

   return want->ncomp == 1
             ? comps[0]
             : sz_spv_const_composite(t->m, sz_dtype_type(t, want), comps,
                                      want->ncomp);
}

/* A phi becomes OpPhi verbatim: NIR is in SSA form and already names the
 * predecessor block each operand arrives from, so there is nothing to lower. */
static bool
sz_emit_phi(struct sz_translate *t, nir_phi_instr *phi)
{
   struct sz_dtype type;
   sz_guess_def_type(t, &phi->def, &type);
   const uint32_t type_id = sz_dtype_type(t, &type);

   unsigned entries = 0;
   nir_foreach_phi_src (src, phi)
      entries++;

   uint32_t *operands = calloc(MAX2(entries, 1) * 2, sizeof(*operands));
   if (!operands) {
      sz_fail(t, "out of memory translating a phi");
      return false;
   }

   unsigned unresolved = 0;
   unsigned i = 0;
   nir_foreach_phi_src (src, phi) {
      /* An operand may legitimately be nir_ssa_undef, which has no id yet; it is
       * resolved by the same fixup pass as a forward reference. */
      uint32_t value = t->defs_id[src->src.ssa->index];
      const nir_load_const_instr *lconst = t->consts[src->src.ssa->index];
      if (value && lconst && t->def_types[src->src.ssa->index].base != type.base)
         value = sz_const_as(t, lconst, &type);
      operands[2 * i] = value;
      operands[2 * i + 1] = src->pred == t->back_edge_pred
                               ? t->back_edge_label
                               : t->labels[src->pred->index];
      if (!value)
         unresolved++;
      i++;
   }

   size_t offset = 0;
   const uint32_t id =
      sz_spv_phi(t->m, type_id, operands, entries * 2, &offset);
   free(operands);

   t->defs_id[phi->def.index] = id;
   t->def_types[phi->def.index] = type;

   if (unresolved) {
      unsigned entry = 0;
      nir_foreach_phi_src (src, phi) {
         if (!t->defs_id[src->src.ssa->index]) {
            if (!sz_phi_fixup_add(t, offset + 2 * entry, src->src.ssa->index))
               return false;
         }
         entry++;
      }
   }

   return true;
}

/* ------------------------------------------------------------------------- */
/* Control flow                                                             */
/* ------------------------------------------------------------------------- */

/* NIR keeps its if/loop structure, so the emitter mirrors it directly: no
 * relooper, no block reordering. The one thing that needs care is that SPIR-V
 * wants each block to end in an instruction that says where control goes next,
 * while NIR simply lets one block fall into the next. A block is therefore left
 * "open" after its instructions and terminated when the next one is opened --
 * which is also what lets a branch reach the right label when the two arms of an
 * if are involved. */

static void sz_emit_cf_list(struct sz_translate *t, struct exec_list *list,
                            uint32_t fallthrough, nir_loop *loop);

/* The first control-flow node of a list, or NULL when the list is empty. */
static nir_cf_node *
sz_cf_list_head(struct exec_list *list)
{
   struct exec_node *head = exec_list_get_head(list);
   return head ? exec_node_data(nir_cf_node, head, node) : NULL;
}

/* The block control reaches after `node`, which is the next sibling block, or
 * `fallthrough` when the construct is the last node of its list. A nested if or
 * loop in between does not change the answer: control still ends up at the same
 * place after it. */
static uint32_t
sz_next_label(struct sz_translate *t, nir_cf_node *node, uint32_t fallthrough)
{
   for (nir_cf_node *next = nir_cf_node_next(node); next;
        next = nir_cf_node_next(next)) {
      if (next->type == nir_cf_node_block)
         return t->labels[nir_cf_node_as_block(next)->index];
   }

   return fallthrough;
}

/* Emits the terminator of the block that is currently open. */
static void
sz_emit_branch(struct sz_translate *t, uint32_t target)
{
   sz_spv_branch(t->m, target);
   t->block_open = false;
}

static void
sz_emit_branch_conditional(struct sz_translate *t, uint32_t condition,
                           uint32_t then_id, uint32_t else_id)
{
   sz_spv_branch_conditional(t->m, condition, then_id, else_id);
   t->block_open = false;
}

static void
sz_open_block(struct sz_translate *t, uint32_t label)
{
   /* Falling out of the previous block is a branch to this one; that is the only
    * place a NIR block's implicit successor is turned into SPIR-V. */
   if (t->block_open)
      sz_emit_branch(t, label);

   sz_spv_label_id(t->m, label);
   t->block_open = true;
}

/* `with_phis` is false only for a loop's first block: its phis belong to the loop
 * header block, which the emitter creates itself. */
static void
sz_emit_block_body(struct sz_translate *t, nir_block *block, bool with_phis)
{   nir_foreach_instr (instr, block) {
      if (instr->type == nir_instr_type_phi && !with_phis)
         continue;
      if (instr->type == nir_instr_type_jump) {
         sz_emit_jump(t, nir_instr_as_jump(instr), block);
         return;
      }
      if (!sz_emit_instr(t, instr))
         return;
   }
}

static void
sz_emit_block(struct sz_translate *t, nir_block *block)
{
   sz_open_block(t, t->labels[block->index]);
   sz_emit_block_body(t, block, true);
}

/* The phis of a loop's first block belong to the header block the emitter
 * creates; this pushes them there before the merge instruction. */
static void
sz_emit_phis(struct sz_translate *t, nir_block *block)
{
   nir_foreach_instr (instr, block) {
      if (instr->type != nir_instr_type_phi)
         return;
      if (!sz_emit_instr(t, instr))
         return;
   }
}

static void
sz_emit_if(struct sz_translate *t, nir_if *nif, uint32_t fallthrough,
           nir_loop *loop)
{
   if (!t->block_open) {
      sz_fail(t, "if without a preceding block");
      return;
   }
   const uint32_t merge_id = sz_next_label(t, &nif->cf_node, fallthrough);
   const uint32_t then_id = t->labels[nir_if_first_then_block(nif)->index];
   const bool has_else = !exec_list_is_empty(&nif->else_list);
   const uint32_t else_id =
      has_else ? t->labels[nir_if_first_else_block(nif)->index] : merge_id;
   const uint32_t condition = t->defs_id[nif->condition.ssa->index];

   sz_spv_selection_merge(t->m, merge_id, SpvSelectionControlMaskNone);
   sz_emit_branch_conditional(t, condition, then_id, else_id);

   /* Each arm falls through to the merge block, never into the other arm. */
   sz_emit_cf_list(t, &nif->then_list, merge_id, loop);
   if (t->block_open)
      sz_emit_branch(t, merge_id);

   if (has_else) {
      sz_emit_cf_list(t, &nif->else_list, merge_id, loop);
      if (t->block_open)
         sz_emit_branch(t, merge_id);
   }
}

/* The loop header is a block of the emitter's own, not the loop's first NIR block.
 * Two reasons, both of them structural rather than stylistic:
 *
 *   - NIR folds the loop test into the first block of the body, and that block may
 *     also be where a nested if's condition is tested. A SPIR-V block may carry
 *     only one merge instruction, so the loop merge and the selection merge cannot
 *     live in the same block.
 *
 *   - A loop's phis live in its header, and every phi operand names the block it
 *     arrives from. Giving the header its own block makes those predecessors
 *     exactly the two SPIR-V provides: the block before the loop and the continue
 *     block. */
static void
sz_emit_loop(struct sz_translate *t, nir_loop *loop, uint32_t fallthrough)
{
   nir_block *begin = nir_loop_first_block(loop);
   const bool has_continue = nir_loop_has_continue_construct(loop);
   const uint32_t header_id = sz_spv_id(t->m);
   const uint32_t merge_id = sz_next_label(t, &loop->cf_node, fallthrough);
   const uint32_t begin_id = t->labels[begin->index];
   /* The continue target has to be a block of the continue construct, not the
    * loop's last block: constructs in the body routinely merge into that one, and
    * a merge block may not sit outside the construct its header belongs to.
    * Naming the header makes the whole body the continue construct, which then
    * has to be post dominated by the back edge -- impossible as soon as the body
    * contains a break. So a continue target of our own it is. */
   nir_block *back_edge = has_continue ? NULL : nir_loop_last_block(loop);
   const uint32_t continue_id =
      has_continue ? t->labels[nir_loop_first_continue_block(loop)->index]
                   : sz_spv_id(t->m);

   const uint32_t save_break = t->loop_break;
   const uint32_t save_continue = t->loop_continue;
   const uint32_t save_header = t->loop_header;
   nir_block *save_pred = t->back_edge_pred;
   const uint32_t save_pred_label = t->back_edge_label;
   t->loop_break = merge_id;
   t->loop_continue = continue_id;
   t->loop_header = header_id;
   t->back_edge_pred = back_edge;
   t->back_edge_label = continue_id;

   sz_open_block(t, header_id);
   sz_emit_phis(t, begin);
   t->back_edge_pred = save_pred;
   t->back_edge_label = save_pred_label;

   sz_spv_loop_merge(t->m, merge_id, continue_id, SpvLoopControlMaskNone);
   sz_emit_branch(t, begin_id);

   sz_open_block(t, begin_id);
   sz_emit_block_body(t, begin, false);

   /* The rest of the body, starting after the block just emitted. */
   sz_emit_cf_list_from(t, &loop->body, &begin->cf_node, continue_id, loop);
   if (t->block_open)
      sz_emit_branch(t, continue_id);

   t->loop_break = save_break;
   t->loop_continue = save_continue;
   t->loop_header = save_header;

   if (has_continue) {
      sz_emit_cf_list(t, &loop->continue_list, header_id, loop);
      if (t->block_open)
         sz_emit_branch(t, header_id);
      return;
   }

   /* The continue target: every continue in the body lands here, and this is the
    * only back edge into the header. */
   sz_spv_label_id(t->m, continue_id);
   sz_spv_branch(t->m, header_id);
   t->block_open = false;
}

static void
sz_emit_cf_list_from(struct sz_translate *t, struct exec_list *list,
                     nir_cf_node *after, uint32_t fallthrough, nir_loop *loop)
{
   /* nir_cf_node_next() is what knows where a list ends: the tail sentinel is a
    * real node whose `next` is NULL, so a loop that stops on a NULL node pointer
    * would hand the sentinel to the emitters as if it were a block. */
   for (nir_cf_node *cf = after ? nir_cf_node_next(after)
                                : sz_cf_list_head(list);
        cf; cf = nir_cf_node_next(cf)) {
      if (t->failed)
         return;

      switch (cf->type) {
      case nir_cf_node_block:
         sz_emit_block(t, nir_cf_node_as_block(cf));
         break;
      case nir_cf_node_if:
         sz_emit_if(t, nir_cf_node_as_if(cf), fallthrough, loop);
         break;
      case nir_cf_node_loop:
         sz_emit_loop(t, nir_cf_node_as_loop(cf), fallthrough);
         break;
      case nir_cf_node_function:
         sz_fail(t, "a nested function is not supported");
         return;
      }
   }
}

static void
sz_emit_cf_list(struct sz_translate *t, struct exec_list *list,
                uint32_t fallthrough, nir_loop *loop)
{
   sz_emit_cf_list_from(t, list, NULL, fallthrough, loop);
}

/* ------------------------------------------------------------------------- */
/* Entry point                                                              */
/* ------------------------------------------------------------------------- */

static bool
sz_emit_shader(struct sz_translate *t, uint32_t function)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(t->nir);

   /* One extra slot: nir_index_blocks() gives end_block the index num_blocks, so
    * any reference to it would otherwise be one past the end. */
   t->labels = calloc(impl->num_blocks + 1, sizeof(*t->labels));
   if (!t->labels) {
      sz_fail(t, "out of memory");
      return false;
   }

   nir_foreach_block (block, impl)
      t->labels[block->index] = sz_spv_id(t->m);

   /* Only allocated when something actually needs to merge into the end of the
    * function. */
   t->end_label = sz_spv_id(t->m);

   const uint32_t void_type = sz_spv_type_void(t->m);
   const uint32_t fn_type = sz_spv_type_function(t->m, void_type, NULL, 0);
   sz_spv_function(t->m, function, void_type, fn_type);

   /* The function's first block is the NIR entry block, which is the first node
    * of the body, so the structured walk starts there. */
   sz_emit_cf_list(t, &impl->body, t->end_label, NULL);

   if (t->block_open)
      sz_spv_return(t->m);

   /* The end block is only branched to when a construct sits at the end of the
    * body; emitting it unconditionally is one empty block and keeps the rule
    * "every label a branch names is defined" free of exceptions. */
   sz_spv_label_id(t->m, t->end_label);
   sz_spv_return(t->m);

   sz_spv_function_end(t->m);

   /* Phi operands that were not defined yet when the phi was emitted are filled
    * in now that the whole function has been translated. */
   for (unsigned i = 0; i < t->num_phi_fixups; i++) {
      const unsigned def = t->phi_fixups[i].def;
      const uint32_t id = t->defs_id[def];
      if (!id) {
         sz_fail(t, "phi operand %u was never translated", def);
         break;
      }
      sz_spv_patch_function_word(t->m, t->phi_fixups[i].offset, id);
   }

   free(t->phi_fixups);
   t->phi_fixups = NULL;
   t->num_phi_fixups = 0;
   t->phi_fixups_capacity = 0;
   free(t->labels);
   t->labels = NULL;

   return !t->failed;
}

bool
sz_nir_to_spirv(struct nir_shader *nir,
                const struct pipe_stream_output_info *so, uint32_t **words,
                size_t *word_count, char **error)
{
   struct sz_translate t = {
      .nir = nir,
      .so = so,
      .error = error,
   };

   *words = NULL;
   *word_count = 0;
   if (error)
      *error = NULL;

   t.m = sz_spv_module_create();
   if (!t.m) {
      sz_fail(&t, "could not create a SPIR-V module");
      return false;
   }

   /* nir_shader_clone() copies each def's index but never rebuilds the counter,
    * so a cloned shader reports ssa_alloc == 0 while its defs still carry indices
    * from the original. Renumbering is the fix: it makes both agree and keeps the
    * indices dense, which is what the two per-def tables rely on. */
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   const unsigned num_defs = impl->ssa_alloc;
   /* num_blocks is only valid once the blocks are indexed, and it is stale (zero)
    * on a freshly cloned shader. */
   nir_index_blocks(impl);
   nir_foreach_block (dbg_block, impl)
   t.defs_id = calloc(num_defs, sizeof(*t.defs_id));
   t.def_types = calloc(num_defs, sizeof(*t.def_types));
   t.consts = calloc(num_defs, sizeof(*t.consts));
   t.interfaces = calloc(SZ_MAX_VARIABLES, sizeof(*t.interfaces));
   if (!t.defs_id || !t.def_types || !t.consts || !t.interfaces) {
      sz_fail(&t, "out of memory");
      goto fail;
   }

   SpvExecutionModel model;
   switch (nir->info.stage) {
   case MESA_SHADER_VERTEX:
      model = SpvExecutionModelVertex;
      break;
   case MESA_SHADER_FRAGMENT:
      model = SpvExecutionModelFragment;
      break;
   default:
      sz_fail(&t, "shader stage %d is not supported yet", (int)nir->info.stage);
      goto fail;
   }

   sz_spv_capability(t.m, SpvCapabilityShader);
   /* Emitted unconditionally: the texture queries are reachable from any GLSL
    * 1.20 shader through textureSize() and textureQueryLevels(), and a capability
    * the module does not use costs nothing. */
   sz_spv_capability(t.m, SpvCapabilityImageQuery);
   sz_spv_memory_model(t.m, SpvAddressingModelLogical, SpvMemoryModelGLSL450);
   sz_glsl450 = sz_spv_ext_inst_import(t.m, "GLSL.std.450");

   nir_foreach_variable_in_shader (var, nir) {
      /* Only inputs and outputs are declared from the NIR variable list. Blocks
       * are descriptors: they are declared on first access instead, which is
       * what lets the load decide the layout, and their variables stay in the
       * NIR after being lowered away and are simply unused. */
      if (var->data.mode != nir_var_shader_in &&
          var->data.mode != nir_var_shader_out)
         continue;

      if (!sz_declare_variable(&t, var))
         goto fail;
      t.interfaces[t.num_interfaces++] = t.vars[t.num_vars - 1].id;
   }

   const uint32_t function = sz_spv_id(t.m);
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      sz_spv_execution_mode(t.m, function, SpvExecutionModeOriginUpperLeft);
   /* Vulkan refuses to begin transform feedback for a pipeline whose last
    * pre-rasterization stage does not declare the Xfb execution mode
    * (VUID-vkCmdBeginTransformFeedbackEXT-None-04128). The per-variable
    * XfbBuffer/XfbStride/Offset decorations alone are not enough. */
   if (nir->info.stage == MESA_SHADER_VERTEX && t.so && t.so->num_outputs > 0) {
      /* The Xfb execution mode is only valid with this capability declared. */
      sz_spv_capability(t.m, SpvCapabilityTransformFeedback);
      sz_spv_execution_mode(t.m, function, SpvExecutionModeXfb);
   }

   if (!sz_emit_shader(&t, function))
      goto fail;

   /* Emitted last because SPIR-V 1.4 and later require every global variable the
    * entry point touches to be listed here, including the uniform blocks that
    * are only declared once a load asks for one. The module reorders its
    * sections, so the instruction still lands before the functions. */
   sz_spv_entry_point(t.m, model, function, "main", t.interfaces,
                      t.num_interfaces);

   size_t bytes = 0;
   const uint32_t *module = sz_spv_finalize(t.m, &bytes);

   /* The caller owns the words: the module is gone by the time it is uploaded.
    * sz_spv_finalize() reports bytes, not words -- mixing the two over-reads the
    * module fourfold. */
   uint32_t *copy = malloc(bytes);
   if (!copy) {
      sz_fail(&t, "out of memory");
      goto fail;
   }
   memcpy(copy, module, bytes);
   *words = copy;
   *word_count = bytes / sizeof(*copy);

   free(t.defs_id);
   free(t.def_types);
   free(t.consts);
   free(t.interfaces);
   sz_spv_module_destroy(t.m);
   return true;

fail:
   free(t.defs_id);
   free(t.def_types);
   free(t.consts);
   free(t.interfaces);
   sz_spv_module_destroy(t.m);

   if (!t.failed)
      sz_fail(&t, "translation failed");

   if (error && *error)
      mesa_loge("SZ: %s shader translation failed: %s",
                nir->info.stage == MESA_SHADER_VERTEX ? "vertex" :
                nir->info.stage == MESA_SHADER_FRAGMENT ? "fragment" : "unknown",
                *error);

   return false;
}
