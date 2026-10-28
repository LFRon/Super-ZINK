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

/* The legacy varying slots (COL0, TEX0..TEX7, ...) are numbers 0..15 and the
 * generic ones start at VARYING_SLOT_VAR0, so both families fit in the same
 * 0..15 location range as long as one shader does not use both. A location may
 * not be large: Vulkan limits a vertex stage's outputs to
 * maxVertexOutputComponents / 4 = 16 locations, so a scheme that puts one
 * family above the other loses half the range to the other half's reservations
 * and then fails the limit outright -- a geometry shader reading four generic
 * varyings would need locations 16..19 and be rejected.
 *
 * Both stages derive the location from the slot with the same arithmetic, which
 * is what interface matching needs. A shader that mixes the two families has no
 * collision-free mapping without per-pipeline slot assignment, and is reported
 * instead of silently aliasing two varyings onto one location. */
#define SZ_LOC_LEGACY_MAX 16

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
   /* Non-zero when this is an array of `ncomp`-wide values. Only one level is
    * ever needed: the arrays that reach the translator are a geometry shader's
    * per-vertex inputs, which GL itself limits to one array dimension. */
   unsigned array_len;
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

   /* The screen has no 1D depth images, so a 1D shadow sampler is declared as 2D
    * and sampled with a zero second coordinate. See sz_image_vk_type(). */
   bool alias_1d_shadow;
   /* VkPhysicalDeviceFeatures::shaderClipDistance / shaderCullDistance. */
   bool clip_distance;
   bool cull_distance;

   /* Indexed by nir_def::index. */
   struct sz_dtype *def_types;
   uint32_t *defs_id;
   /* A base type a def must be emitted with, because a phi already claimed it.
    * A phi is emitted at the head of its block, so the value arriving on a loop's
    * back edge has no type yet; without a hint the two sides guess independently
    * and can disagree, which OpPhi rejects. 0xff means "no hint". */
   uint8_t *base_hint;
   /* Constants are untyped in NIR and typed in SPIR-V, so a def that reaches a
    * phi as a different base type needs its own typed constant; remembering the
    * instruction here is what makes that possible. */
   nir_load_const_instr **consts;

   /* Set once OpCapability DerivativeControl has been emitted. */
   bool derivative_control;

   /* Which varying families this shader declares, so a shader that would alias
    * a legacy slot onto a generic one is reported rather than mis-linked. */
   bool used_legacy_varying;
   bool used_generic_varying;

   struct {
      const nir_variable *var;
      uint32_t id;
      uint32_t pointee;
      SpvStorageClass storage;
      /* The type the variable was declared with. A load has to match its
       * pointee, and the deref's own type can disagree with it: the frontend
       * declares varyings as `vec4` and reinterprets them as `ivec4` on use,
       * and it declares gl_FrontFacing as a 32-bit integer while SPIR-V wants a
       * bool. */
      struct sz_dtype type;
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
      bool ms;
      /* OpTypeImage, kept so OpImageFetch and the query ops can name it. */
      uint32_t image_type;
      /* What the OpLoad of this unit produced. A buffer texture loads its image
       * directly; every other dimension loads a sampled image that still has to
       * be split with OpImage. */
      uint32_t loaded_type;
      /* The value OpLoad produced from the variable, which is what the sampling
       * ops take as their image operand. */
      uint32_t value;
   } samplers[SZ_MAX_SAMPLERS];
   unsigned num_samplers;

   uint32_t *interfaces;
   unsigned num_interfaces;

   /* Scalar integer builtin inputs (gl_VertexID, gl_InstanceID), declared on
    * first use: a shader that does not read one must not declare it, because an
    * unused builtin input is still a pipeline interface. */
   struct {
      SpvBuiltIn builtin;
      uint32_t var;
   } int_builtins[4];
   unsigned num_int_builtins;

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
static uint32_t sz_def_value(struct sz_translate *t, const nir_def *def);
static uint32_t sz_bitcast_to(struct sz_translate *t, uint32_t id,
                              const struct sz_dtype *from,
                              const struct sz_dtype *to);
static bool sz_emit_phi(struct sz_translate *t, nir_phi_instr *phi);
static uint32_t sz_intrin_source_as(struct sz_translate *t, const nir_src *src,
                                    const struct sz_dtype *want,
                                    unsigned ncomp);
static void sz_retype_for_plan(struct sz_translate *t, nir_def *def, uint32_t *id,
                               struct sz_dtype *type);
static nir_def *sz_instr_def(nir_instr *instr);
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

/* A scalar constant of the given base and width.
 *
 * A one-bit value is a bool, and a bool has no OpConstant form: SPIR-V spells it
 * OpConstantTrue/OpConstantFalse, and `OpConstant %bool 1` is rejected with
 * "Type Id ... is not a scalar numeric type" -- which the validation layer
 * reports and a driver's parser is free to do anything with. Routing through
 * here is what keeps every constant site honest. */
static uint32_t
sz_const_scalar(struct sz_translate *t, enum sz_base base, unsigned bits,
                uint32_t bits_value)
{
   if (base == SZ_BASE_BOOL || bits == 1)
      return sz_spv_const_bool(t->m, bits_value != 0);

   return sz_spv_const_bits(t->m, sz_scalar_type(t, base, bits), bits_value);
}

static uint32_t
sz_dtype_type(struct sz_translate *t, const struct sz_dtype *type)
{
   const uint32_t scalar = sz_scalar_type(t, type->base, type->bits);
   const uint32_t element = type->ncomp == 1
                               ? scalar
                               : sz_spv_type_vec(t->m, scalar, type->ncomp);

   if (type->array_len > 0)
      return sz_spv_type_array(t->m, element, type->array_len);

   return element;
}

static bool
sz_dtype_from_glsl(struct sz_translate *t, const struct glsl_type *glsl,
                   struct sz_dtype *out)
{
   out->array_len = 0;

   /* One level of array, which is what a geometry shader's per-vertex inputs
    * are. Nested arrays and structs would need the OpTypeArray recursion and a
    * member-offset scheme that nothing here asks for yet. */
   if (glsl_type_is_array(glsl)) {
      const struct glsl_type *elem = glsl_get_array_element(glsl);
      if (glsl_type_is_array(elem) || glsl_type_is_struct_or_ifc(elem) ||
          glsl_type_is_matrix(elem)) {
         sz_fail(t, "unsupported array element type in `%s`",
                 glsl_get_type_name(glsl));
         return false;
      }
      out->array_len = glsl_get_length(glsl);
      glsl = elem;
   }

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

         /* Both spellings write through a deref to a variable and both keep
          * their value in src[1]. store_output is the one that matters for a
          * fragment colour: it is what `icolor = uvec4(r)` becomes, and without
          * it the only statement about r's base type is lost. */
         if ((intrin->intrinsic != nir_intrinsic_store_deref &&
              intrin->intrinsic != nir_intrinsic_store_output) ||
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

/* The base type of a definition, read from the instruction that produced it.
 *
 * The alternative -- the base recorded when the def was emitted -- is empty for a
 * def that has not been emitted yet, and the guess made from its uses is circular
 * when the only use is the phi being asked about. The producer's own declared
 * output type has neither problem. */
static bool
sz_def_base_from_producer(const nir_def *def, enum sz_base *out)
{
   /* Some opcodes take their type from an operand rather than declaring one:
    * OpSelect, the vector constructors and the plain moves. Follow the operand
    * that carries it -- the condition of a select carries a bool, not the
    * result's type -- and give up rather than guess after a few steps. */
   for (unsigned depth = 0; depth < 4; depth++) {
      const nir_instr *producer = nir_def_instr_const(def);
      if (!producer || producer->type != nir_instr_type_alu)
         return false;

      const nir_alu_instr *alu = nir_instr_as_alu(producer);
      if (sz_base_from_alu_type(nir_op_infos[alu->op].output_type, out))
         return true;

      const unsigned index = alu->op == nir_op_bcsel ? 1 : 0;
      if (nir_op_infos[alu->op].num_inputs <= index || !alu->src[index].src.ssa)
         return false;
      def = alu->src[index].src.ssa;
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

   /* A phi already decided this def's base type. */
   if (t->base_hint && t->base_hint[def->index] != 0xff)
      out->base = (enum sz_base)t->base_hint[def->index];

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

   const bool vertex = t->nir->info.stage == MESA_SHADER_VERTEX;

   if (var->data.mode == nir_var_shader_in) {
      /* A geometry shader's inputs are the previous stage's *outputs*, not
       * vertex attributes, so they follow the varying rules below. Ranking them
       * among the shader's inputs -- the vertex rule -- would declare
       * gl_in[].gl_Position at Location 0 and break the interface with the
       * vertex stage. */
      if (!frag && vertex) {
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
         /* gl_FragCoord in a fragment shader, gl_in[].gl_Position in a geometry
          * shader: the same slot, two different builtins. */
         *builtin = frag ? SpvBuiltInFragCoord : SpvBuiltInPosition;
         *has_builtin = true;
         return true;
      }

      if (slot == VARYING_SLOT_FACE && frag) {
         *builtin = SpvBuiltInFrontFacing;
         *has_builtin = true;
         return true;
      }

      /* gl_PrimitiveIDIn in a geometry shader, gl_PrimitiveID in a fragment
       * shader: the same builtin, and both need the Geometry capability. */
      if (slot == VARYING_SLOT_PRIMITIVE_ID) {
         *builtin = SpvBuiltInPrimitiveId;
         *has_builtin = true;
         return true;
      }

      /* Clip and cull distances are arrays of floats with a builtin, not
       * locations. The frontend keeps them merged into one array per family, so
       * only the first slot of each is ever declared. */
      if (slot == VARYING_SLOT_CLIP_DIST0 || slot == VARYING_SLOT_CLIP_DIST1) {
         if (slot != VARYING_SLOT_CLIP_DIST0) {
            sz_fail(t, "%s is not the first clip distance slot", var->name);
            return false;
         }
         if (!t->clip_distance) {
            sz_fail(t, "%s needs the shaderClipDistance feature, which this "
                       "device does not report", var->name);
            return false;
         }
         *builtin = SpvBuiltInClipDistance;
         *has_builtin = true;
         return true;
      }

      if (slot == VARYING_SLOT_CULL_DIST0 || slot == VARYING_SLOT_CULL_DIST1) {
         if (slot != VARYING_SLOT_CULL_DIST0) {
            sz_fail(t, "%s is not the first cull distance slot", var->name);
            return false;
         }
         if (!t->cull_distance) {
            sz_fail(t, "%s needs the shaderCullDistance feature, which this "
                       "device does not report", var->name);
            return false;
         }
         *builtin = SpvBuiltInCullDistance;
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
         if (slot == VARYING_SLOT_PRIMITIVE_ID) {
            /* Writing gl_PrimitiveID is a geometry stage operation. */
            if (!vertex) {
               *builtin = SpvBuiltInPrimitiveId;
               *has_builtin = true;
               return true;
            }
         }

         if (slot == VARYING_SLOT_CLIP_DIST0) {
            if (!t->clip_distance) {
               sz_fail(t, "%s needs the shaderClipDistance feature, which this "
                          "device does not report", var->name);
               return false;
            }
            *builtin = SpvBuiltInClipDistance;
            *has_builtin = true;
            return true;
         }

         if (slot == VARYING_SLOT_CULL_DIST0) {
            if (!t->cull_distance) {
               sz_fail(t, "%s needs the shaderCullDistance feature, which this "
                          "device does not report", var->name);
               return false;
            }
            *builtin = SpvBuiltInCullDistance;
            *has_builtin = true;
            return true;
         }
      } else {
         /* gl_Layer and gl_ViewportIndex: a geometry or vertex shader writes
          * them to say which layer and which viewport its primitives go to, and
          * Vulkan has a builtin for each. Without them the capability that
          * promises layered rendering cannot work. */
         if (slot == VARYING_SLOT_LAYER) {
            *builtin = SpvBuiltInLayer;
            *has_builtin = true;
            return true;
         }
         if (slot == VARYING_SLOT_VIEWPORT) {
            *builtin = SpvBuiltInViewportIndex;
            *has_builtin = true;
            return true;
         }
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
      *location = slot - VARYING_SLOT_VAR0;
      t->used_generic_varying = true;
      return true;
   }

   if (slot >= 0 && slot < SZ_LOC_LEGACY_MAX) {
      *location = slot;
      t->used_legacy_varying = true;
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

   /* gl_ClipVertex exists only to feed the fixed-function clip planes, and this
    * driver has none (see caps->clip_planes). The state tracker rewrites it into
    * ClipDistance stores whenever a plane is enabled, so what reaches here is the
    * variant where the value is never read -- but it still has stores, and the
    * interface has no location to give it: the legacy slots already reach 16 and
    * a vertex stage has 16 locations. Private storage takes the stores somewhere
    * harmless: unlike Function it is legal at module scope, and unlike Output it
    * needs no decoration and is not part of the entry point's interface. */
   /* The name is what distinguishes it from a user varying that shares the
    * number: a vertex attribute's location is a VERT_ATTRIB_* value, and
    * VERT_ATTRIB_GENERIC0 is 16 -- the same number as VARYING_SLOT_CLIP_VERTEX, so
    * a shader whose second attribute lives there would otherwise be mangled. GL's
    * own rule is the same one the linker uses: a builtin's name starts with
    * "gl_". */
   const bool dead_clip_vertex =
      var->data.location == VARYING_SLOT_CLIP_VERTEX &&
      t->nir->info.stage != MESA_SHADER_FRAGMENT && var->name &&
      strncmp(var->name, "gl_", 3) == 0;

   switch (var->data.mode) {
   case nir_var_shader_in:
      /* A geometry shader receives it as an input; the value is dead there too,
       * and an input with no interface location is just as invalid. */
      storage = dead_clip_vertex ? SpvStorageClassPrivate
                                 : SpvStorageClassInput;
      break;
   case nir_var_shader_out:
      storage = dead_clip_vertex ? SpvStorageClassPrivate
                                 : SpvStorageClassOutput;
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

   struct sz_dtype type = { 0 };
   if (!sz_dtype_from_glsl(t, var->type, &type))
      return false;

   if (storage != SpvStorageClassPrivate) {
      if (!sz_varying_location(t, var, &builtin, &has_builtin, &location))
         return false;
   }

   /* gl_FrontFacing is a bool in SPIR-V whatever the frontend calls it. */
   if (has_builtin && builtin == SpvBuiltInFrontFacing) {
      type.base = SZ_BASE_BOOL;
      type.bits = 1;
      type.ncomp = 1;
   }

   const uint32_t pointee = sz_dtype_type(t, &type);
   const uint32_t ptr_type = sz_spv_type_ptr(t->m, storage, pointee);
   const uint32_t id = sz_spv_variable(t->m, ptr_type, storage);

   if (has_builtin) {
      sz_spv_decorate_builtin(t->m, id, builtin);

      /* A builtin is not usable on its own: PrimitiveId exists only with the
       * Geometry capability, the two distance arrays only with theirs. Emitting
       * the capability where the builtin is declared keeps the two from
       * drifting. */
      switch (builtin) {
      case SpvBuiltInPrimitiveId:
         sz_spv_capability(t->m, SpvCapabilityGeometry);
         break;
      case SpvBuiltInClipDistance:
         sz_spv_capability(t->m, SpvCapabilityClipDistance);
         break;
      case SpvBuiltInCullDistance:
         sz_spv_capability(t->m, SpvCapabilityCullDistance);
         break;
      case SpvBuiltInLayer:
         sz_spv_capability(t->m, SpvCapabilityGeometry);
         break;
      case SpvBuiltInViewportIndex:
         sz_spv_capability(t->m, SpvCapabilityMultiViewport);
         break;
      default:
         break;
      }
   } else if (storage != SpvStorageClassPrivate) {
      sz_spv_decorate(t->m, id, SpvDecorationLocation, location);
      /* The GLSL linker packs several small varyings into one location, each at
       * its own component (`location_frac`). Without the Component decoration
       * two variables would claim the same Location, which is invalid SPIR-V on
       * the output side and reads the wrong channel on the input side. */
      if (var->data.location_frac != 0)
         sz_spv_decorate(t->m, id, SpvDecorationComponent,
                         var->data.location_frac);

      /* Dual-source blending: the second colour output of a location is the
       * same Location with Index 1, which is how `layout(index = 1)` and
       * glBindFragDataLocationIndexed reach Vulkan. Without it both outputs
       * claim location 0 outright and the module is rejected. */
      if (var->data.index != 0)
         sz_spv_decorate(t->m, id, SpvDecorationIndex, var->data.index);
   }

   if (storage == SpvStorageClassOutput)
      sz_decorate_stream_output(t, var, id);

   /* Flat is meaningful only on fragment stage inputs: an integer varying
    * carries no interpolation because it cannot be blended, and SPIR-V requires
    * the decoration there. On an output the same decoration is *invalid*
    * (VUID-StandaloneSpirv-Flat-06201), which is why the check is on the storage
    * class rather than on the type alone. */
   if (!has_builtin && storage == SpvStorageClassInput &&
       t->nir->info.stage == MESA_SHADER_FRAGMENT &&
       (type.base != SZ_BASE_FLOAT || var->data.interpolation == INTERP_MODE_FLAT))
      sz_spv_decorate_bare(t->m, id, SpvDecorationFlat);

   t->vars[t->num_vars].var = var;
   t->vars[t->num_vars].id = id;
   t->vars[t->num_vars].pointee = pointee;
   t->vars[t->num_vars].storage = storage;
   t->vars[t->num_vars].type = type;
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

/* The storage class a deref chain reads through, from the mode of the variable
 * at its root. */
static SpvStorageClass
sz_deref_storage(const nir_deref_instr *deref)
{
   if (deref->modes & nir_var_shader_in)
      return SpvStorageClassInput;
   if (deref->modes & nir_var_shader_out)
      return SpvStorageClassOutput;
   if (deref->modes & (nir_var_uniform | nir_var_mem_ubo |
                       nir_var_mem_ssbo | nir_var_mem_push_const))
      return SpvStorageClassUniform;
   return SpvStorageClassFunction;
}

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

   case nir_deref_type_array: {
      /* One element of an array variable: OpAccessChain with the index. The
       * result pointer type comes from the deref itself, which NIR has already
       * computed, so nothing here has to re-derive the element type. */
      nir_deref_instr *parent = nir_src_as_deref(deref->parent);
      if (!parent) {
         sz_fail(t, "array deref without a deref parent");
         return 0;
      }
      const uint32_t base = sz_emit_deref(t, parent);
      if (!base)
         return 0;

      struct sz_dtype elem_type = { 0 };
      if (!sz_dtype_from_glsl(t, deref->type, &elem_type))
         return 0;
      elem_type.array_len = 0;

      uint32_t index = sz_def_value(t, deref->arr.index.ssa);
      if (index == 0)
         return 0;

      /* The index must be a 32-bit int; an index built from a float or a
       * differently-typed constant is reinterpreted rather than trusted. */
      const struct sz_dtype want = { SZ_BASE_INT, 32, 1, 0 };
      const uint32_t cast =
         sz_bitcast_to(t, index, &t->def_types[deref->arr.index.ssa->index],
                       &want);
      if (cast)
         index = cast;

      /* The storage class follows the parent variable, not the deref: a chain
       * into a uniform or an input keeps that class. Getting it wrong is a
       * validation error -- an OpAccessChain whose result pointer class does not
       * match its base pointer's. */
      const uint32_t ptr_type =
         sz_spv_type_ptr(t->m, sz_deref_storage(deref),
                         sz_dtype_type(t, &elem_type));

      return sz_spv_access_chain(t->m, ptr_type, base, &index, 1);
   }

   default:
      sz_fail(t, "unsupported deref type %d (structs come with descriptors)",
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

   /* GLSL 3.30's integer bit operations: bitfieldExtract/Insert/Reverse,
    * bitCount, findLSB and findMSB. */
   { nir_op_ubitfield_extract, SpvOpBitFieldUExtract },
   { nir_op_ibitfield_extract, SpvOpBitFieldSExtract },
   { nir_op_bitfield_reverse, SpvOpBitReverse },
   { nir_op_bit_count, SpvOpBitCount },
};

/* Opcodes that exist in GLSL.std.450 rather than in core SPIR-V. */
struct sz_glsl_map {
   nir_op op;
   uint32_t ext;
};

static const struct sz_glsl_map sz_glsl_maps[] = {
   { nir_op_fabs, GLSLstd450FAbs },
   { nir_op_iabs, GLSLstd450SAbs },
   /* sign() on integers; the float form is below. */
   { nir_op_isign, GLSLstd450SSign },
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
   /* GLSL 3.30 floating-point additions. */
   { nir_op_ldexp, GLSLstd450Ldexp },
   /* SPIR-V dropped the findLSB/findMSB core instructions before 1.0; the
    * GLSL dialect kept them, with exactly the -1-for-no-bit result GL wants. */
   { nir_op_find_lsb, GLSLstd450FindILsb },
   { nir_op_ufind_msb, GLSLstd450FindUMsb },
   { nir_op_ifind_msb, GLSLstd450FindSMsb },
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
   const unsigned n = def->num_components;

   /* The base a source is *declared* with need not be the base the use is
    * defined on. Two cases produce that:
    *
    *  - NIR constants are untyped bit patterns, so the type picked when the
    *    constant was emitted is a guess from its other uses. The same 0 used as
    *    the int arm of one select and the float arm of another is right in one
    *    of them and wrong in the other, and OpSelect with operands of two types
    *    does not validate.
    *  - Varyings are declared as `vec4` even when the shader declared them as
    *    `ivec4`: the frontend keeps the IO declaration float and reinterprets on
    *    use (`i2f32` of a float load). Feeding such a load to OpConvertSToF
    *    unchanged is rejected ("Expected input to be int scalar or vector").
    *
    * A constant is re-materialised in the wanted base -- its stored bits are the
    * number, so a bitcast would produce the float whose bits spell 3. A value is
    * bitcast, which is what the frontend's reinterpretation means. */
   const struct sz_dtype *from = &t->def_types[def->index];
   const nir_load_const_instr *lconst = t->consts[def->index];
   struct sz_dtype typed = { 0 };
   uint32_t id = t->defs_id[def->index];

   if (want && from->base != want->base && from->bits == want->bits) {
      if (lconst) {
         typed = *want;
         typed.ncomp = n;
         id = sz_const_as(t, lconst, &typed);
         from = &typed;
      }
   }

   const uint32_t base = sz_scalar_type(t, from->base, from->bits);

   bool identity = n == ncomp;
   for (unsigned i = 0; identity && i < n; i++)
      if (src->swizzle[i] != i)
         identity = false;

   uint32_t result = id;
   if (!identity) {
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

      result = ncomp == 1
                  ? comps[0]
                  : sz_spv_composite_construct(t->m, sz_dtype_type(t, from), comps,
                                               ncomp);
   }

   if (want && !lconst && from->base != want->base && from->bits == want->bits) {
      struct sz_dtype have = *from;
      have.ncomp = ncomp;
      struct sz_dtype to = *want;
      to.ncomp = ncomp;
      const uint32_t cast = sz_bitcast_to(t, result, &have, &to);
      if (cast)
         result = cast;
   }

   return result;
}

static uint32_t
sz_alu_source(struct sz_translate *t, const nir_alu_src *src, unsigned ncomp)
{
   return sz_alu_source_as(t, src, NULL, ncomp);
}

/* Operand `j` of `op`, typed the way the opcode defines it.
 *
 * NIR gives every opcode an operand type: OpConvertSToF's operand is an integer
 * even though its result is a float. A source whose emitted type disagrees is
 * almost always a constant whose base was guessed from a different use, and
 * feeding it in unchanged makes spirv-val reject the instruction ("Expected
 * input to be int scalar or vector"). Only the base is corrected -- the width
 * comes from the definition itself, because the alu type does not always carry
 * one (comparison operands are declared `nir_type_int` with no size). */
static uint32_t
sz_alu_source_op(struct sz_translate *t, nir_op op, const nir_alu_src *src,
                 unsigned j, const struct sz_dtype *dst, unsigned ncomp)
{
   const nir_def *def = src->src.ssa;
   if (!def)
      return sz_alu_source(t, src, ncomp);

   struct sz_dtype want = *dst;
   want.ncomp = ncomp;
   want.bits = t->def_types[def->index].bits;
   if (want.base == SZ_BASE_BOOL)
      want.bits = 1;

   switch (nir_alu_type_get_base_type(nir_op_infos[op].input_types[j])) {
   case nir_type_int:
      want.base = SZ_BASE_INT;
      break;
   case nir_type_uint:
      want.base = SZ_BASE_UINT;
      break;
   case nir_type_bool:
      want.base = SZ_BASE_BOOL;
      want.bits = 1;
      break;
   default:
      want.base = SZ_BASE_FLOAT;
      break;
   }

   return sz_alu_source_as(t, src, &want, ncomp);
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
   struct sz_dtype dst = { 0 };
   dst.bits = alu->def.bit_size;
   dst.ncomp = alu->def.num_components;

   /* A polymorphic op's type is its operand's, not the one the opcode table
    * names: mov, vecN and bcsel all record nir_type_uint as a placeholder, so
    * reading the table types a float move as unsigned -- and then a store of that
    * value into a float variable is invalid SPIR-V, which is a crash inside
    * RADV's parser rather than a validation error. */
   if (sz_alu_is_passthrough(op)) {
      const nir_def *typed = alu->src[op == nir_op_bcsel ? 1 : 0].src.ssa;
      enum sz_base base = typed ? t->def_types[typed->index].base : SZ_BASE_FLOAT;

      /* These three carry no type of their own. NIR spells a bitcast as a plain
       * mov, so what the value *is* comes from whoever consumes it: a phi that
       * already decided, and otherwise the uses. The operand is only the last
       * resort, for a value nothing typed consumes. */
      if (t->base_hint && t->base_hint[alu->def.index] != 0xff) {
         base = (enum sz_base)t->base_hint[alu->def.index];
      } else {
         enum sz_base inferred;
         if (sz_infer_def_base(&alu->def, 0, &inferred))
            base = inferred;
      }

      dst.base = base;
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
   /* b2b1 truncates a bool to one bit, which in SPIR-V is already the only
    * bool there is: the value passes through unchanged. */
   if (op == nir_op_b2b1) {
      t->defs_id[alu->def.index] = sz_alu_source(t, &alu->src[0], dst.ncomp);
      t->def_types[alu->def.index] = dst;
      return;
   }

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
            sz_alu_source_op(t, op, &alu->src[0], 0, &dst, dst.ncomp),
            sz_alu_source_op(t, op, &alu->src[1], 1, &dst, dst.ncomp),
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

   /* bitfieldExtract()/bitfieldInsert() reach the SPIR-V bitfield opcodes, and
    * those want Offset/Count in the result type -- a uint result takes uint
    * offsets, even though NIR declares them int. bitfieldInsert also takes four
    * operands, one more than the generic loop below reads. */
   if (op == nir_op_ubitfield_extract || op == nir_op_ibitfield_extract ||
       op == nir_op_bitfield_insert) {
      const SpvOp spirv = op == nir_op_bitfield_insert ? SpvOpBitFieldInsert
                           : op == nir_op_ibitfield_extract
                              ? SpvOpBitFieldSExtract
                              : SpvOpBitFieldUExtract;
      enum sz_base base;
      if (sz_base_from_alu_type(nir_op_infos[op].output_type, &base))
         dst.base = base;

      const struct sz_dtype want = {
         .base = dst.base,
         .bits = dst.bits,
         .ncomp = dst.ncomp,
      };
      const unsigned nsrcs = nir_op_infos[op].num_inputs;
      uint32_t ops[4];
      for (unsigned j = 0; j < nsrcs; j++)
         ops[j] = sz_alu_source_as(t, &alu->src[j], &want, dst.ncomp);

      t->defs_id[alu->def.index] =
         sz_spv_alu(t->m, spirv, sz_dtype_type(t, &dst), ops, nsrcs);
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

      /* The opcode declares the type of its result, and that beats the guess
       * made from the def's uses: a bitfield extract is unsigned whatever its
       * consumer looks like, and OpBitFieldUExtract rejects a float result type.
       * A bool result is left alone -- it has no declared type of its own here,
       * and its bit width is one. */
      if (dst.bits > 1) {
         enum sz_base declared;
         if (sz_base_from_alu_type(nir_op_infos[op].output_type, &declared) &&
             declared != SZ_BASE_BOOL)
            dst.base = declared;
      }

      /* Equality on bools arrives as `ieq`/`ine` -- GLSL's `==` on bvec lowers
       * to the integer comparison -- but SPIR-V has no bitwise or integer
       * comparison on a bool, only OpLogicalEqual/NotEqual. */
      const bool bool_compare =
         (op == nir_op_ieq || op == nir_op_ine) && alu->src[0].src.ssa &&
         t->def_types[alu->src[0].src.ssa->index].base == SZ_BASE_BOOL;
      if (bool_compare) {
         spirv = op == nir_op_ieq ? SpvOpLogicalEqual : SpvOpLogicalNotEqual;
      }

      /* Only the sources the op declares exist: the rest of nir_alu_src[] is
       * uninitialised, and reading it dereferences a NULL nir_def. */
      const unsigned nsrcs = MIN2(nir_op_infos[op].num_inputs, 3);
      uint32_t ops[3];
      for (unsigned j = 0; j < nsrcs; j++)
         ops[j] = bool_compare
                     ? sz_alu_source(t, &alu->src[j], dst.ncomp)
                     : sz_alu_source_op(t, op, &alu->src[j], j, &dst, dst.ncomp);

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
         ops[j] = sz_alu_source_op(t, op, &alu->src[j], j, &dst, dst.ncomp);

      struct sz_dtype work = dst;
      /* FSqrt and friends are defined on floats; keep the declared type. */
      t->defs_id[alu->def.index] =
         sz_emit_ext(t, sz_glsl_maps[i].ext, &work, ops, nsrcs);
      t->def_types[alu->def.index] = dst;
      return;
   }

   sz_fail(t, "NIR op `%s` has no SPIR-V mapping", nir_op_infos[op].name);
}


/* The base type an OpPhi arm has to be phrased in, decided before anything is
 * emitted.
 *
 * An OpPhi cannot convert its arms, and an arm's block is gone by the time the
 * phi is reached, so the only place an arm can be converted is its own producer
 * -- which means the phi's type has to be known *before* the producer runs. The
 * blocks are emitted in layout order and SSA destruction builds a `switch`'s arm
 * chain in that same order, so "wait and see which arm arrives first" does not
 * work: the float arm is translated long before the integer arms meant to
 * outvote it exist.
 *
 * So the decision is taken here instead, as a fixpoint over the whole function.
 * Evidence, strongest first: what consumes the phi (the destination variable
 * states the type outright, and `uvec4(r)` written to a uvec4 output is the
 * usual case), then an arm that already has an answer, then an arm whose
 * producer declares one. Chains of phis collapse because each round writes the
 * answer back into base_hint[], which every producer consults.
 */
static bool
sz_arm_declares_base(const struct sz_translate *t, const nir_def *arm,
                     enum sz_base *out)
{
   if (!arm || arm->bit_size <= 1)
      return false;

   const uint8_t hint = t->base_hint[arm->index];
   if (hint != 0xff && (enum sz_base)hint != SZ_BASE_BOOL) {
      *out = (enum sz_base)hint;
      return true;
   }

   const nir_instr *producer = nir_def_instr_const(arm);
   if (!producer || producer->type != nir_instr_type_alu)
      return false;

   /* mov, vecN and bcsel declare nothing of their own. */
   const nir_op op = nir_instr_as_alu(producer)->op;
   if (sz_alu_is_passthrough(op))
      return false;

   enum sz_base declared;
   if (!sz_base_from_alu_type(nir_op_infos[op].output_type, &declared) ||
       declared == SZ_BASE_BOOL)
      return false;

   *out = declared;
   return true;
}

static void
sz_plan_phi_arms(struct sz_translate *t, nir_function_impl *impl)
{
   for (unsigned iter = 0; iter < 16; iter++) {
      bool changed = false;

      nir_foreach_block (block, impl) {
         nir_foreach_instr (instr, block) {
            if (instr->type != nir_instr_type_phi)
               continue;
            nir_phi_instr *phi = nir_instr_as_phi(instr);

            enum sz_base base;
            if (!sz_infer_def_base(&phi->def, 0, &base) ||
                base == SZ_BASE_BOOL) {
               bool have = false;
               nir_foreach_phi_src (src, phi) {
                  if (sz_arm_declares_base(t, src->src.ssa, &base)) {
                     have = true;
                     break;
                  }
               }
               if (!have)
                  continue;
            }

            if (t->base_hint[phi->def.index] != (uint8_t)base) {
               t->base_hint[phi->def.index] = (uint8_t)base;
               changed = true;
            }

            nir_foreach_phi_src (src, phi) {
               const nir_def *arm = src->src.ssa;
               if (!arm || arm->bit_size <= 1 || t->consts[arm->index])
                  continue;
               if (t->base_hint[arm->index] == 0xff) {
                  t->base_hint[arm->index] = (uint8_t)base;
                  changed = true;
               }
            }
         }
      }

      if (!changed)
         break;
   }
}

/* Phrase a finished value in the base type the plan asked for.
 *
 * The value's own instruction cannot produce another base and its block closes
 * as soon as it is emitted, so a bitcast here is the last chance. */
static void
sz_retype_for_plan(struct sz_translate *t, nir_def *def, uint32_t *id,
                   struct sz_dtype *type)
{
   if (!*id || def->bit_size <= 1 || type->base == SZ_BASE_BOOL)
      return;

   const uint8_t hint = t->base_hint[def->index];
   if (hint == 0xff || (enum sz_base)hint == type->base)
      return;

   struct sz_dtype to = *type;
   to.base = (enum sz_base)hint;
   const uint32_t cast = sz_bitcast_to(t, *id, type, &to);
   if (!cast)
      return;

   *id = cast;
   *type = to;
}

/* The def an instruction produces, if any. A phi is left out: its own emitter
 * already places it in the plan, and retyping it would be circular. */
static nir_def *
sz_instr_def(nir_instr *instr)
{
   switch (instr->type) {
   case nir_instr_type_alu:
      return &nir_instr_as_alu(instr)->def;
   case nir_instr_type_intrinsic:
      return &nir_instr_as_intrinsic(instr)->def;
   case nir_instr_type_tex:
      return &nir_instr_as_tex(instr)->def;
   default:
      return NULL;
   }
}

/* ------------------------------------------------------------------------- */
/* Instructions                                                             */
/* ------------------------------------------------------------------------- */

static void
sz_emit_load_const(struct sz_translate *t, nir_load_const_instr *lconst)
{
   struct sz_dtype type = { 0 };
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
      comps[i] = sz_const_scalar(t, type.base, type.bits, bits);
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

   sz_spv_decorate(t->m, var, SpvDecorationDescriptorSet,
                   SZ_DESCRIPTOR_SET_UBO);
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

   struct sz_dtype type = { 0 };
   if (!sz_dtype_from_glsl(t, deref->type, &type))
      return;

   if (deref->deref_type == nir_deref_type_var) {
      const int index = sz_find_variable(t, deref->var);
      if (index >= 0)
         type = t->vars[index].type;
   }

   uint32_t loaded = sz_spv_load(t->m, sz_dtype_type(t, &type), ptr);

   /* A varying declared `vec4` whose value the shader treats as `ivec4` loads as
    * float and is consumed as int. Left as it is, the first instruction that
    * needs an integer operand rejects it, and a phi that mixes the two has no
    * single type its operands satisfy -- OpPhi's incoming values must all match
    * its result type. The reinterpretation happens where the value is produced,
    * which is also where a bitcast dominates every use. */
   struct sz_dtype guess = { 0 };
   sz_guess_def_type(t, &intrin->def, &guess);
   if (guess.base != type.base && guess.bits == type.bits &&
       guess.ncomp == type.ncomp) {
      const uint32_t cast = sz_bitcast_to(t, loaded, &type, &guess);
      if (cast) {
         loaded = cast;
         type = guess;
      }
   }

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

/* Reinterprets a value as another base type of the same width.
 *
 * NIR values carry no type: a def's recorded base is inferred from its uses, and
 * the same bits are legitimately stored into a float variable here and compared
 * as an int there. SPIR-V is typed and wants the pointee and the value to agree,
 * so a same-width difference becomes an OpBitcast instead of a refusal --
 * refusing would reject shaders every other driver compiles, and the bits are
 * exactly what GL means anyway. Bool is excluded: it has no defined bit pattern
 * in a 32-bit register, so a mismatch involving one is a translator bug. */
static uint32_t
sz_bitcast_to(struct sz_translate *t, uint32_t id, const struct sz_dtype *from,
              const struct sz_dtype *to)
{
   if (from->base == to->base)
      return id;

   if (from->base == SZ_BASE_BOOL || to->base == SZ_BASE_BOOL ||
       from->bits != to->bits || from->ncomp != to->ncomp)
      return 0;

   return sz_spv_alu(t->m, SpvOpBitcast, sz_dtype_type(t, to), &id, 1);
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

   struct sz_dtype dst_type = { 0 };
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
         const uint32_t cast =
            sz_bitcast_to(t, value_id, &t->def_types[value->index], &dst_type);
         if (!cast) {
            sz_fail(t, "store_deref of a %s value into a %s variable",
                    sz_base_name(t->def_types[value->index].base),
                    sz_base_name(dst_type.base));
            return;
         }
         sz_spv_store(t->m, ptr, cast);
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
   /* A sample index is only meaningful on a multisampled image, so the presence
    * of one is what makes the declared OpTypeImage carry MS = 1. Without it the
    * module is invalid: an ImageOperand of Sample requires the image to be
    * multisampled (spirv-val rejects the mismatch). */
   const bool ms = tex->op == nir_texop_txf_ms;

   for (unsigned i = 0; i < t->num_samplers; i++) {
      if (t->samplers[i].index != index)
         continue;

      /* The same unit used with two different types cannot be described by one
       * binding: a VkDescriptorSetLayoutBinding has a single descriptor type and
       * a SPIR-V variable a single pointee type. GL allows the aliasing, so say
       * so instead of sampling through the wrong declaration. */
      if (t->samplers[i].dim != dim || t->samplers[i].arrayed != arrayed ||
          t->samplers[i].depth != depth || t->samplers[i].ms != ms) {
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
      t->m, sampled_type, dim, depth, arrayed, ms, true);

   /* A buffer texture is not a combined image sampler. Its Vulkan descriptor is
    * a UNIFORM_TEXEL_BUFFER -- a VkBufferView, with no sampler beside it -- and
    * the SPIR-V side is therefore a bare OpTypeImage that OpImageFetch reads.
    * Declaring a sampled image here would ask the descriptor set for a sampler
    * the binding does not carry. */
   const bool is_buffer = dim == SpvDimBuffer;
   const uint32_t pointee_type = is_buffer
                                    ? image_type
                                    : sz_spv_type_sampled_image(t->m, image_type);
   const uint32_t ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassUniformConstant, pointee_type);
   const uint32_t var =
      sz_spv_variable(t->m, ptr_type, SpvStorageClassUniformConstant);

   if (is_buffer) {
      /* Dim Buffer is only legal with this capability. */
      sz_spv_capability(t->m, SpvCapabilitySampledBuffer);
      sz_spv_decorate(t->m, var, SpvDecorationDescriptorSet,
                      SZ_DESCRIPTOR_SET_TEXEL_BUFFERS);
      sz_spv_decorate(t->m, var, SpvDecorationBinding,
                      sz_texel_binding(t->nir->info.stage, index));
   } else {
      sz_spv_decorate(t->m, var, SpvDecorationDescriptorSet,
                      SZ_DESCRIPTOR_SET_SAMPLERS);
      sz_spv_decorate(t->m, var, SpvDecorationBinding,
                      sz_sampler_binding(t->nir->info.stage, index));
   }

   /* A combined image sampler is read as a whole, so one OpLoad yields both
    * halves; the load is emitted once and its result reused by every instruction
    * that samples this unit. A buffer texture loads its image and nothing else. */
   const uint32_t value = sz_spv_load(t->m, pointee_type, var);

   t->samplers[t->num_samplers].index = index;
   t->samplers[t->num_samplers].dim = dim;
   t->samplers[t->num_samplers].arrayed = arrayed;
   t->samplers[t->num_samplers].depth = depth;
   t->samplers[t->num_samplers].ms = ms;
   t->samplers[t->num_samplers].image_type = image_type;
   t->samplers[t->num_samplers].loaded_type = pointee_type;
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

/* The integer texel offset of a textureOffset/texelFetchOffset instruction.
 *
 * SPIR-V takes it as an ImageOperands operand rather than as part of the
 * coordinate: a constant one becomes ConstOffset, which every implementation
 * folds into the addressing, and a computed one becomes Offset. The value has to
 * be an integer vector, and a constant here carries no type of its own, so it is
 * materialised with the one the operand needs. */
static uint32_t
sz_tex_offset_value(struct sz_translate *t, const nir_def *offset,
                    bool *is_constant)
{
   struct sz_dtype want = {
      .base = SZ_BASE_INT,
      .bits = 32,
      .ncomp = offset->num_components,
   };

   const nir_load_const_instr *lconst = t->consts[offset->index];
   if (lconst) {
      *is_constant = true;
      return sz_const_as(t, lconst, &want);
   }

   *is_constant = false;
   const struct sz_dtype *have = &t->def_types[offset->index];
   return sz_bitcast_to(t, t->defs_id[offset->index], have, &want);
}

static void
sz_emit_tex(struct sz_translate *t, nir_tex_instr *tex)
{
   if (tex->is_sparse) {
      sz_fail(t, "sparse texture reads are not supported");
      return;
   }

   const nir_def *coord = NULL, *bias = NULL, *lod = NULL, *dref = NULL;
   const nir_src *coord_src = NULL;
   const nir_def *ddx = NULL, *ddy = NULL, *projector = NULL;
   const nir_def *offset = NULL, *ms_index = NULL;

   for (unsigned i = 0; i < tex->num_srcs; i++) {
      const nir_tex_src *src = &tex->src[i];
      switch (src->src_type) {
      case nir_tex_src_coord:
         coord = src->src.ssa;
         coord_src = &src->src;
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
      case nir_tex_src_offset:
         offset = src->src.ssa;
         break;
      case nir_tex_src_ms_index:
         ms_index = src->src.ssa;
         break;
      default:
         /* Sampler arrays arrive as an offset added to sampler_index, which the
          * descriptor layout cannot express: a dynamic index would need
          * VK_EXT_descriptor_indexing. */
         sz_fail(t, "texture source type %d is not supported yet",
                 (int)src->src_type);
         return;
      }
   }

   SpvDim dim;
   if (!sz_sampler_dim(t, tex->sampler_dim, &dim))
      return;

   /* A 1D shadow texture is a 2D image one texel high on a device without 1D
    * depth, so the declared image dimension has to say 2D or the descriptor and
    * the shader disagree. The coordinate gains a zero second component below. */
   const bool alias_2d = t->alias_1d_shadow && tex->is_shadow &&
                         tex->sampler_dim == GLSL_SAMPLER_DIM_1D;
   if (alias_2d)
      dim = SpvDim2D;

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
   const uint32_t sampled_image = sz_sampler_get(t, tex, dim);
   if (sampled_image == 0)
      return;

   unsigned slot = 0;
   for (; slot < t->num_samplers; slot++)
      if (t->samplers[slot].index == tex->sampler_index)
         break;
   const uint32_t image_type = t->samplers[slot].image_type;

   /* A buffer texture's load already yielded the image; everything else holds a
    * sampled image and extracts the image from it. */
   const uint32_t image =
      image_type == t->samplers[slot].loaded_type
         ? sampled_image
         : sz_spv_alu(t->m, SpvOpImage, image_type, &sampled_image, 1);

   /* Everything that is not a query needs a coordinate; the queries are the
    * reason this check cannot be at the top. */
   uint32_t coord_value = 0;
   if (coord != NULL) {
      coord_value = sz_def_value(t, coord);
      if (coord_value == 0)
         return;

      if (projector != NULL) {
         const uint32_t projector_value = sz_def_value(t, projector);
         if (projector_value == 0)
            return;
         coord_value = sz_emit_projector(t, coord_value, tex->coord_components,
                                         tex->is_array, projector_value);
      }

      /* The aliased 1D shadow coordinate is (s), or (s, layer) for an array;
       * the 2D image wants (s, 0) and (s, 0, layer). So a zero goes in at
       * index 1, not on the end -- appending it would put the layer where the
       * row belongs and the shader would sample the wrong texel. The image is
       * one texel high and clamp-addressed, so a row of 0 is the only one. */
      if (alias_2d) {
         const unsigned n = tex->coord_components;
         const uint32_t float_type = sz_spv_type_float32(t->m);
         const uint32_t zero = sz_spv_const_float32(t->m, 0.0f);
         uint32_t comps[5];

         unsigned out = 0;
         for (unsigned i = 0; i < n; i++) {
            if (i == 1)
               comps[out++] = zero;
            comps[out++] = n == 1 ? coord_value
                                  : sz_spv_composite_extract(
                                       t->m, float_type, coord_value, &i, 1);
         }
         if (n == 1)
            comps[out++] = zero;

         coord_value = sz_spv_composite_construct(
            t->m, sz_spv_type_vec(t->m, float_type, out), comps, out);
      }
   }

   uint32_t offset_value = 0;
   bool offset_constant = false;
   if (offset != NULL) {
      offset_value = sz_tex_offset_value(t, offset, &offset_constant);
      if (offset_value == 0)
         return;
   }

   /* The image operands: a bit mask followed by one value per set bit, in the
    * order the bits are numbered (Bias, Lod, Grad, ConstOffset, Offset, Sample).
    * Building them in that order is what keeps a sample with both a bias and an
    * offset legal -- the operand list is positional, not named. */
   uint32_t mask = 0;
   uint32_t values[6];
   unsigned num_values = 0;

   switch (tex->op) {
   case nir_texop_txs: {
      uint32_t level = sz_spv_const_int32(t->m, 0);
      if (lod) {
         level = sz_def_value(t, lod);
         if (level == 0)
            return;
         /* The Lod operand must be a 32-bit int scalar; a value loaded from the
          * untyped uniform block carries whatever base the translator guessed,
          * so it is reinterpreted here rather than trusted. */
         const struct sz_dtype want = { SZ_BASE_INT, 32, 1, 0 };
         const uint32_t cast =
            sz_bitcast_to(t, level, &t->def_types[lod->index], &want);
         if (cast)
            level = cast;
      }
      const uint32_t ops[] = { image, level };

      /* An aliased 1D shadow texture is a 2D image, so the query answers with
       * one more component than the shader asked for: (width, height) instead
       * of (width), and (width, height, layers) instead of (width, layers). The
       * extra component is the one-texel row, so it is dropped -- asking with
       * the shader's own narrow type instead would be a type mismatch between
       * the instruction's result and the query's. */
      if (alias_2d) {
         const unsigned wide = tex->is_array ? 3 : 2;
         const uint32_t int_type = sz_spv_type_int32(t->m);
         const uint32_t wide_type =
            sz_spv_type_vec(t->m, int_type, wide);

         const uint32_t queried =
            sz_spv_alu(t->m, SpvOpImageQuerySizeLod, wide_type, ops, 2);

         /* OpCompositeExtract's indices are literals, not ids. */
         const unsigned zero_idx = 0;
         const unsigned last_idx = wide - 1;

         uint32_t comps[2];
         unsigned n = 0;
         comps[n++] =
            sz_spv_composite_extract(t->m, int_type, queried, &zero_idx, 1);
         if (tex->is_array) {
            /* The layer count is the last component; index 1 is the height. */
            comps[n++] =
               sz_spv_composite_extract(t->m, int_type, queried, &last_idx, 1);
         }

         t->defs_id[tex->def.index] =
            n == 1 ? comps[0]
                   : sz_spv_composite_construct(
                        t->m, sz_spv_type_vec(t->m, int_type, n), comps, n);
         t->def_types[tex->def.index] = (struct sz_dtype){ SZ_BASE_INT, 32, n, 0 };
         return;
      }

      const uint32_t value =
         sz_spv_alu(t->m,
                    dim == SpvDimBuffer ? SpvOpImageQuerySize
                                        : SpvOpImageQuerySizeLod,
                    type, ops, dim == SpvDimBuffer ? 1 : 2);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   case nir_texop_query_levels: {
      const uint32_t value =
         sz_spv_alu(t->m, SpvOpImageQueryLevels, type, &image, 1);
      t->defs_id[tex->def.index] = value;
      t->def_types[tex->def.index] = result_type;
      return;
   }
   case nir_texop_texture_samples: {
      const uint32_t value =
         sz_spv_alu(t->m, SpvOpImageQuerySamples, type, &image, 1);
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
      break;
   }

   if (coord_value == 0) {
      sz_fail(t, "texture instruction has no coordinate");
      return;
   }

   /* Fetch: a bare image, the coordinate, and the level or sample index as the
    * image operand. The sampler state is not consulted at all. */
   if (tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms) {
      uint32_t ops[6];
      unsigned n = 0;
      ops[n++] = image;

      /* OpImageFetch wants a 32-bit *integer* coordinate, and NIR hands a fetch
       * a float one (GL's texelFetch takes an int/ivec, and the frontend keeps
       * the varying float-shaped). A buffer texture makes that visible: every
       * other dimension accepts a float vector too, Dim Buffer does not. */
      {
         const unsigned ncomp = tex->coord_components;
         struct sz_dtype want = { SZ_BASE_INT, 32, ncomp };
         const uint32_t as_int =
            sz_intrin_source_as(t, coord_src, &want, ncomp);
         ops[n++] = as_int ? as_int : coord_value;
      }

      if (tex->op == nir_texop_txf_ms) {
         const struct sz_dtype want = { SZ_BASE_INT, 32, 1 };
         uint32_t sample;

         if (ms_index) {
            /* The Sample operand must be a 32-bit *int* scalar. A uniform we
             * load out of the untyped block carries whatever base the
             * translator guessed from its uses, so it is reinterpreted here
             * rather than relied on. */
            sample = sz_def_value(t, ms_index);
            if (sample == 0)
               return;
            const uint32_t cast =
               sz_bitcast_to(t, sample, &t->def_types[ms_index->index], &want);
            if (cast)
               sample = cast;
         } else {
            sample = sz_spv_const_int32(t->m, 0);
         }

         mask |= SpvImageOperandsSampleMask;
         values[num_values++] = sample;
      } else if (lod != NULL) {
         const nir_load_const_instr *lconst = t->consts[lod->index];
         /* Level zero is what an omitted Lod operand means, so a constant zero
          * costs nothing to leave out -- and leaving it out keeps the common
          * texelFetch(tex, coord, 0) down to two operands. */
         const bool zero_lod = lconst && lconst->value[0].u32 == 0;
         if (!zero_lod) {
            const uint32_t value = sz_def_value(t, lod);
            if (value == 0)
               return;
            mask |= SpvImageOperandsLodMask;
            values[num_values++] = value;
         }
      }

      if (offset != NULL) {
         if (offset_constant) {
            mask |= SpvImageOperandsConstOffsetMask;
         } else {
            /* A computed offset is the non-constant form of the same operand and
             * needs this capability; the constant form does not. */
            sz_spv_capability(t->m, SpvCapabilityImageGatherExtended);
            mask |= SpvImageOperandsOffsetMask;
         }
         values[num_values++] = offset_value;
      }

      if (mask != 0)
         ops[n++] = mask;
      for (unsigned i = 0; i < num_values; i++)
         ops[n++] = values[i];

      t->defs_id[tex->def.index] =
         sz_spv_alu(t->m, SpvOpImageFetch, type, ops, n);
      t->def_types[tex->def.index] = result_type;
      return;
   }

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
         mask |= SpvImageOperandsBiasMask;
         values[num_values++] = value;
      }
      break;
   case nir_texop_txl: {
      opcode = tex->is_shadow ? SpvOpImageSampleDrefExplicitLod
                              : SpvOpImageSampleExplicitLod;
      if (lod == NULL) {
         sz_fail(t, "explicit-LOD sample has no LOD");
         return;
      }
      const uint32_t value = sz_def_value(t, lod);
      if (value == 0)
         return;
      mask |= SpvImageOperandsLodMask;
      values[num_values++] = value;
      break;
   }
   case nir_texop_txd: {
      opcode = tex->is_shadow ? SpvOpImageSampleDrefExplicitLod
                              : SpvOpImageSampleExplicitLod;
      if (ddx == NULL || ddy == NULL) {
         sz_fail(t, "gradient sample has no gradients");
         return;
      }
      const uint32_t dx = sz_def_value(t, ddx);
      const uint32_t dy = sz_def_value(t, ddy);
      if (dx == 0 || dy == 0)
         return;
      mask |= SpvImageOperandsGradMask;
      values[num_values++] = dx;
      values[num_values++] = dy;
      break;
   }
   case nir_texop_tg4: {
      opcode = tex->is_shadow ? SpvOpImageDrefGather : SpvOpImageGather;
      /* The component is the last positional operand of a gather, before the
       * image operands. */
      if (tex->is_shadow) {
         if (dref == NULL) {
            sz_fail(t, "shadow gather has no comparator");
            return;
         }
      }
      break;
   }
   default:
      sz_fail(t, "texture opcode %d is not supported yet", (int)tex->op);
      return;
   }

   if (offset != NULL) {
      if (offset_constant) {
         mask |= SpvImageOperandsConstOffsetMask;
      } else {
         sz_spv_capability(t->m, SpvCapabilityImageGatherExtended);
         mask |= SpvImageOperandsOffsetMask;
      }
      values[num_values++] = offset_value;
   }

   uint32_t ops[8];
   unsigned n = 0;
   ops[n++] = sampled_image;
   ops[n++] = coord_value;
   if (tex->is_shadow)
      ops[n++] = sz_def_value(t, dref);
   if (tex->op == nir_texop_tg4)
      ops[n++] = sz_spv_const_int32(t->m, tex->component);
   if (mask != 0)
      ops[n++] = mask;
   for (unsigned i = 0; i < num_values; i++)
      ops[n++] = values[i];

   const uint32_t value = sz_spv_alu(t->m, opcode, type, ops, n);
   t->defs_id[tex->def.index] = value;
   t->def_types[tex->def.index] = result_type;
}


/* interpolateAtCentroid/Sample/Offset -- GLSL 4.00's interpolateAt* family.
 *
 * The GLSL dialect carries all three, and each takes a pointer to the fragment
 * input rather than its value: that is the whole point of the feature, since the
 * value a plain load reads is the one interpolated at the pixel centre. A
 * backend that cannot interpolate at a chosen point cannot fake these.
 *
 * The shapes SPIR-V wants are narrower than NIR's: the sample index must be a
 * 32-bit int and the offset a vec2 of 32-bit floats, so anything else is
 * reinterpreted on the way in.
 */
/* sz_alu_source_as() takes an nir_alu_src, which carries a component swizzle;
 * an intrinsic's source does not, and a zero-initialised swizzle reads as "take
 * component 0 four times" -- which silently turns a vec2 into (vec2, float). */
static uint32_t
sz_intrin_source_as(struct sz_translate *t, const nir_src *src,
                    const struct sz_dtype *want, unsigned ncomp)
{
   nir_alu_src alu_src = { .src = *src };
   for (unsigned i = 0; i < 4; i++)
      alu_src.swizzle[i] = i;
   return sz_alu_source_as(t, &alu_src, want, ncomp);
}

static void
sz_emit_interpolate(struct sz_translate *t, nir_intrinsic_instr *intrin)
{
   nir_deref_instr *deref = nir_src_as_deref(intrin->src[0]);
   if (!deref) {
      sz_fail(t, "interp_deref without a deref source");
      return;
   }

   const uint32_t ptr = sz_emit_deref(t, deref);
   if (!ptr)
      return;

   struct sz_dtype type = { 0 };
   if (!sz_dtype_from_glsl(t, deref->type, &type))
      return;

   sz_spv_capability(t->m, SpvCapabilityInterpolationFunction);

   switch (intrin->intrinsic) {
   case nir_intrinsic_interp_deref_at_centroid: {
      const uint32_t ops[] = { ptr };
      t->defs_id[intrin->def.index] =
         sz_emit_ext(t, GLSLstd450InterpolateAtCentroid, &type, ops, 1);
      break;
   }
   case nir_intrinsic_interp_deref_at_sample: {
      struct sz_dtype want = { SZ_BASE_INT, 32, 1 };
      const uint32_t sample =
         sz_intrin_source_as(t, &intrin->src[1], &want, 1);
      const uint32_t ops[] = { ptr, sample };
      t->defs_id[intrin->def.index] =
         sz_emit_ext(t, GLSLstd450InterpolateAtSample, &type, ops, 2);
      break;
   }
   case nir_intrinsic_interp_deref_at_offset: {
      struct sz_dtype want = { SZ_BASE_FLOAT, 32, 2 };
      const uint32_t offset =
         sz_intrin_source_as(t, &intrin->src[1], &want, 2);
      const uint32_t ops[] = { ptr, offset };
      t->defs_id[intrin->def.index] =
         sz_emit_ext(t, GLSLstd450InterpolateAtOffset, &type, ops, 2);
      break;
   }
   default:
      sz_fail(t, "unsupported interpolate intrinsic");
      return;
   }

   t->def_types[intrin->def.index] = type;
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

/* gl_SampleMaskIn is an array of int in SPIR-V, one element per 32 samples, and
 * GL exposes it as a single int. Reading element zero is what gl_SampleMaskIn
 * means on every device this driver runs on: Vulkan only reaches past 32 samples
 * with the coverage reductions GL has no equivalent for. */
static uint32_t
sz_load_mask_element(struct sz_translate *t, uint32_t var, uint32_t array_type,
                     uint32_t int_type)
{
   const uint32_t arr = sz_spv_load(t->m, array_type, var);
   const uint32_t index = sz_spv_const_int32(t->m, 0);
   return sz_spv_composite_extract(t->m, int_type, arr, &index, 1);
}

/* gl_SamplePosition and gl_SampleMask[] are the two remaining sample builtins:
 * the first is a vec2 float input, the second an int array output that the
 * fragment shader may write to kill individual samples. */
static uint32_t
sz_load_sample_position(struct sz_translate *t)
{
   const uint32_t float_type = sz_spv_type_float32(t->m);
   const uint32_t vec2 = sz_spv_type_vec(t->m, float_type, 2);

   sz_spv_capability(t->m, SpvCapabilitySampleRateShading);

   const uint32_t ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassInput, vec2);
   const uint32_t var = sz_spv_variable(t->m, ptr_type, SpvStorageClassInput);
   sz_spv_decorate(t->m, var, SpvDecorationBuiltIn, SpvBuiltInSamplePosition);

   t->interfaces[t->num_interfaces++] = var;
   return sz_spv_load(t->m, vec2, var);
}

static uint32_t
sz_load_sample_mask_in(struct sz_translate *t)
{
   const uint32_t int_type = sz_spv_type_int32(t->m);
   const uint32_t array_type = sz_spv_type_array(t->m, int_type, 1);

   /* Both sample builtins require this capability, and SPIR-V wants it declared
    * before the decoration that names them. */
   sz_spv_capability(t->m, SpvCapabilitySampleRateShading);

   for (unsigned i = 0; i < t->num_int_builtins; i++) {
      if (t->int_builtins[i].builtin == SpvBuiltInSampleMask)
         return sz_load_mask_element(t, t->int_builtins[i].var, array_type,
                                     int_type);
   }

   if (t->num_int_builtins == ARRAY_SIZE(t->int_builtins)) {
      sz_fail(t, "too many integer builtins in one shader");
      return 0;
   }

   const uint32_t ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassInput, array_type);
   const uint32_t var = sz_spv_variable(t->m, ptr_type, SpvStorageClassInput);
   sz_spv_decorate(t->m, var, SpvDecorationBuiltIn, SpvBuiltInSampleMask);
   /* A fragment input of integer type has to be Flat whatever it is, builtin or
    * not (VUID-StandaloneSpirv-Flat-04744). */
   sz_spv_decorate_bare(t->m, var, SpvDecorationFlat);

   t->interfaces[t->num_interfaces++] = var;
   t->int_builtins[t->num_int_builtins].builtin = SpvBuiltInSampleMask;
   t->int_builtins[t->num_int_builtins].var = var;
   t->num_int_builtins++;

   return sz_load_mask_element(t, var, array_type, int_type);
}

/* Loads a scalar integer builtin input, declaring it on first use.
 *
 * gl_VertexID and gl_InstanceID are the two GL builtins that need no state at
 * all: Vulkan's VertexIndex already carries firstVertex and a vertexOffset, and
 * InstanceIndex carries firstInstance, which is what the instanced attribute
 * fetch uses in GL too. */
static uint32_t
sz_load_builtin_int(struct sz_translate *t, SpvBuiltIn builtin)
{
   const uint32_t int_type = sz_spv_type_int32(t->m);

   for (unsigned i = 0; i < t->num_int_builtins; i++) {
      if (t->int_builtins[i].builtin == builtin)
         return sz_spv_load(t->m, int_type, t->int_builtins[i].var);
   }

   if (t->num_int_builtins == ARRAY_SIZE(t->int_builtins)) {
      sz_fail(t, "too many integer builtins in one shader");
      return 0;
   }

   const uint32_t ptr_type =
      sz_spv_type_ptr(t->m, SpvStorageClassInput, int_type);
   const uint32_t var = sz_spv_variable(t->m, ptr_type, SpvStorageClassInput);
   sz_spv_decorate(t->m, var, SpvDecorationBuiltIn, builtin);
   /* An integer fragment input has to be Flat whatever it names, gl_SampleID
    * included (VUID-StandaloneSpirv-Flat-04744). */
   if (t->nir->info.stage == MESA_SHADER_FRAGMENT)
      sz_spv_decorate_bare(t->m, var, SpvDecorationFlat);

   t->interfaces[t->num_interfaces++] = var;
   t->int_builtins[t->num_int_builtins].builtin = builtin;
   t->int_builtins[t->num_int_builtins].var = var;
   t->num_int_builtins++;

   return sz_spv_load(t->m, int_type, var);
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
   case nir_intrinsic_emit_vertex:
      /* OpEmitVertex takes no operands: it emits whatever the output variables
       * currently hold, which is exactly GL's EmitVertex(). */
      sz_spv_emit_vertex(t->m);
      return true;

   case nir_intrinsic_end_primitive:
      sz_spv_end_primitive(t->m);
      return true;

   case nir_intrinsic_vote_any:
   case nir_intrinsic_vote_all: {
      /* ARB_shader_group_vote's anyInvocation() and allInvocations(). Both need
       * the GroupNonUniform capability, and the subgroup size Vulkan works with
       * is the device's -- GL's "all invocations in the current set" is the
       * same set, since the extension leaves the set implementation-defined. */
      sz_spv_capability(t->m, SpvCapabilityGroupNonUniform);

      const struct sz_dtype bool_type = { SZ_BASE_BOOL, 1, 1 };
      const uint32_t cond = sz_intrin_source_as(t, &intrin->src[0], &bool_type, 1);
      if (cond == 0)
         return false;

      const uint32_t scope =
         sz_spv_const_bits(t->m, sz_spv_type_int32(t->m), SpvScopeSubgroup);
      const uint32_t zero =
         sz_spv_const_bits(t->m, sz_spv_type_int32(t->m), SpvGroupOperationReduce);
      const SpvOp op = intrin->intrinsic == nir_intrinsic_vote_any
                          ? SpvOpGroupNonUniformAny
                          : SpvOpGroupNonUniformAll;
      const uint32_t ops[] = { scope, cond };
      const uint32_t value =
         sz_spv_alu(t->m, op, sz_spv_type_bool(t->m), ops, 2);
      (void)zero;
      t->defs_id[intrin->def.index] = value;
      t->def_types[intrin->def.index] = (struct sz_dtype){ SZ_BASE_BOOL, 1, 1 };
      return true;
   }
   case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_base_instance: {
      /* ARB_shader_draw_parameters' gl_BaseVertex and gl_BaseInstance. Vulkan
       * carries them as the BaseVertex and BaseInstance builtins, which
       * vkCmdDraw* already receives from pipe_draw_info. */
      const SpvBuiltIn builtin =
         intrin->intrinsic == nir_intrinsic_load_base_vertex
            ? SpvBuiltInBaseVertex
            : SpvBuiltInBaseInstance;
      const uint32_t value = sz_load_builtin_int(t, builtin);
      if (value == 0)
         return false;
      t->defs_id[intrin->def.index] = value;
      t->def_types[intrin->def.index] = (struct sz_dtype){ SZ_BASE_INT, 32, 1 };
      return true;
   }
   case nir_intrinsic_load_sample_id: {
      /* gl_SampleID. Reading the sample index is what makes per-sample shading
       * meaningful, which is why the capability is gated on this working. */
      sz_spv_capability(t->m, SpvCapabilitySampleRateShading);
      const uint32_t value = sz_load_builtin_int(t, SpvBuiltInSampleId);
      if (value == 0)
         return false;
      t->defs_id[intrin->def.index] = value;
      t->def_types[intrin->def.index] = (struct sz_dtype){ SZ_BASE_INT, 32, 1 };
      return true;
   }
   case nir_intrinsic_load_sample_pos_from_id:
   case nir_intrinsic_load_sample_pos: {
      const uint32_t value = sz_load_sample_position(t);
      if (value == 0)
         return false;
      t->defs_id[intrin->def.index] = value;
      t->def_types[intrin->def.index] = (struct sz_dtype){ SZ_BASE_FLOAT, 32, 2 };
      return true;
   }
   case nir_intrinsic_load_sample_mask_in: {
      const uint32_t value = sz_load_sample_mask_in(t);
      if (value == 0)
         return false;
      t->defs_id[intrin->def.index] = value;
      t->def_types[intrin->def.index] = (struct sz_dtype){ SZ_BASE_INT, 32, 1 };
      return true;
   }
   case nir_intrinsic_load_primitive_id: {
      /* gl_PrimitiveIDIn, which Vulkan spells PrimitiveId. Declared on first
       * use for the same reason the other builtins are. */
      const uint32_t value = sz_load_builtin_int(t, SpvBuiltInPrimitiveId);
      if (value == 0)
         return false;
      t->defs_id[intrin->def.index] = value;
      t->def_types[intrin->def.index] = (struct sz_dtype){ SZ_BASE_INT, 32, 1, 0 };
      return true;
   }

   case nir_intrinsic_load_vertex_id:
   case nir_intrinsic_load_instance_id: {
      const SpvBuiltIn builtin = intrin->intrinsic == nir_intrinsic_load_vertex_id
                                    ? SpvBuiltInVertexIndex
                                    : SpvBuiltInInstanceIndex;
      const uint32_t value = sz_load_builtin_int(t, builtin);
      if (value == 0)
         return false;
      t->defs_id[intrin->def.index] = value;
      t->def_types[intrin->def.index] = (struct sz_dtype){ SZ_BASE_INT, 32, 1 };
      return true;
   }
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
   case nir_intrinsic_interp_deref_at_centroid:
   case nir_intrinsic_interp_deref_at_sample:
   case nir_intrinsic_interp_deref_at_offset:
      sz_emit_interpolate(t, intrin);
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
sz_emit_instr_inner(struct sz_translate *t, nir_instr *instr)
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
      struct sz_dtype type = { 0 };
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

static bool
sz_emit_instr(struct sz_translate *t, nir_instr *instr)
{
   if (!sz_emit_instr_inner(t, instr))
      return false;

   nir_def *def = sz_instr_def(instr);
   if (def)
      sz_retype_for_plan(t, def, &t->defs_id[def->index],
                         &t->def_types[def->index]);
   return true;
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
   uint32_t comps[4];
   for (unsigned i = 0; i < want->ncomp; i++) {
      const uint32_t bits = want->bits == 64 ? (uint32_t)lconst->value[i].u64
                                             : lconst->value[i].u32;
      comps[i] = sz_const_scalar(t, want->base, want->bits, bits);
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
   struct sz_dtype type = { 0 };
   sz_guess_def_type(t, &phi->def, &type);

   /* The guess is made from the phi's uses, and a phi that merges an induction
    * variable with an unsigned modulation has uses that cannot agree: the guess
    * said int while the value coming round the back edge is the uint that
    * OpUMod produced, and OpPhi's incoming values must all match its result type.
    * A value's own recorded type is the better evidence -- take it from the first
    * operand that is not a constant, and let the constants be re-created in that
    * type below. */
   nir_foreach_phi_src (src, phi) {
      if (src->src.ssa == NULL || t->consts[src->src.ssa->index])
         continue;

      /* Asked of the NIR definition rather than of the type recorded when it was
       * emitted: a phi's incoming value routinely comes from later in the shader
       * (the back edge of a loop), so nothing has been recorded for it yet. */
      struct sz_dtype have = t->def_types[src->src.ssa->index];
      if (have.bits == 0) {
         /* Not emitted yet: it will be, later in the shader (a loop's back edge
          * is defined in the body), so ask the NIR side and remember the answer
          * for when it is. */
         sz_guess_def_type(t, src->src.ssa, &have);
         enum sz_base produced;
         if (sz_def_base_from_producer(src->src.ssa, &produced))
            have.base = produced;
      }
      if (have.base != type.base && have.bits == type.bits &&
          have.ncomp == type.ncomp) {
         if (t->def_types[src->src.ssa->index].bits == 0) {
            /* The phi decides, and the operand has to agree when it is finally
             * emitted. */
            t->base_hint[src->src.ssa->index] = type.base;
         } else {
            /* The operand is already in the module, so its type wins. */
            type.base = have.base;
         }
      }
      break;
   }

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
         break;
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

/* Checks the module's control-flow cross references before it is handed over.
 *
 * A phi names two ids that are not results of its own: the value on each
 * incoming edge, and the block that edge comes from. Both are written as raw
 * numbers during emission, so a bookkeeping mistake produces a module that is
 * merely *wrong* rather than one that fails to build -- and the consumer of a
 * wrong module is a driver's SPIR-V parser, which on RADV means a segfault
 * inside vkCreateGraphicsPipelines. The translator's whole design is that a
 * refusal is cheaper than a wrong shader, so the same rule applies here: walk
 * the result, and if a phi points at something that is not a label or not a
 * defined value, say so and fail.
 *
 * This costs one pass over the words at shader-compile time, which is nothing
 * next to the compile itself, and it turns a class of "the driver crashed"
 * reports into a legible translation error. */
static bool
sz_spv_verify_cfg(struct sz_translate *t, const uint32_t *words, size_t count,
                  char **error)
{
   const size_t n = count / 4;
   if (n == 0)
      return true;

   const uint32_t bound = words[3];
   if (bound == 0 || bound > 4u * 1024 * 1024) {
      if (error)
         *error = strdup("the emitted module has an implausible id bound");
      return false;
   }

   uint8_t *kind = calloc(bound, 1);
   if (!kind)
      return true;

   /* 1 = a defined value id, 2 = an OpLabel id. */
   size_t i = 5;
   while (i < n) {
      const uint16_t wc = words[i] >> 16;
      const uint16_t op = words[i] & 0xFFFF;
      if (wc == 0)
         break;

      uint32_t result = 0;
      bool has_result = false;

      if (op == SpvOpLabel && wc >= 2) {
         result = words[i + 1];
         has_result = true;
         if (result < bound)
            kind[result] = 2;
      } else if ((op >= SpvOpTypeVoid && op <= SpvOpTypeForwardPointer) ||
                 op == SpvOpVariable || op == SpvOpFunction ||
                 op == SpvOpFunctionParameter || op == SpvOpConstant ||
                 op == SpvOpConstantComposite || op == SpvOpConstantTrue ||
                 op == SpvOpConstantFalse || op == SpvOpUndef ||
                 op == SpvOpExtInst || op == SpvOpExtInstImport ||
                 op == SpvOpSampledImage || op == SpvOpImage ||
                 op == SpvOpCompositeConstruct || op == SpvOpCompositeExtract ||
                 op == SpvOpCompositeInsert || op == SpvOpVectorShuffle ||
                 op == SpvOpAccessChain || op == SpvOpInBoundsAccessChain ||
                 op == SpvOpPtrAccessChain || op == SpvOpLoad ||
                 op == SpvOpSelect || op == SpvOpPhi || op == SpvOpBitcast ||
                 op == SpvOpIAdd || op == SpvOpISub || op == SpvOpIMul ||
                 op == SpvOpSDiv || op == SpvOpUDiv || op == SpvOpFAdd ||
                 op == SpvOpFSub || op == SpvOpFMul || op == SpvOpFDiv ||
                 op == SpvOpFNegate || op == SpvOpSNegate || op == SpvOpNot ||
                 op == SpvOpBitwiseAnd || op == SpvOpBitwiseOr ||
                 op == SpvOpBitwiseXor || op == SpvOpLogicalAnd ||
                 op == SpvOpLogicalOr || op == SpvOpLogicalNot ||
                 op == SpvOpLogicalEqual || op == SpvOpLogicalNotEqual ||
                 op == SpvOpShiftLeftLogical || op == SpvOpShiftRightLogical ||
                 op == SpvOpShiftRightArithmetic || op == SpvOpIEqual ||
                 op == SpvOpINotEqual || op == SpvOpSLessThan ||
                 op == SpvOpSGreaterThan || op == SpvOpSGreaterThanEqual ||
                 op == SpvOpULessThan || op == SpvOpUGreaterThan ||
                 op == SpvOpUGreaterThanEqual || op == SpvOpFOrdEqual ||
                 op == SpvOpFOrdLessThan || op == SpvOpFUnordNotEqual ||
                 op == SpvOpFOrdGreaterThanEqual ||
                 op == SpvOpFUnordGreaterThanEqual ||
                 op == SpvOpFUnordLessThan || op == SpvOpFOrdNotEqual ||
                 op == SpvOpConvertFToS || op == SpvOpConvertFToU ||
                 op == SpvOpConvertSToF || op == SpvOpConvertUToF ||
                 op == SpvOpFConvert || op == SpvOpSConvert ||
                 op == SpvOpUConvert || op == SpvOpImageFetch ||
                 op == SpvOpImageQuerySize || op == SpvOpImageQuerySizeLod ||
                 op == SpvOpImageQueryLevels || op == SpvOpImageQuerySamples ||
                 op == SpvOpImageQueryLod || op == SpvOpImageGather ||
                 op == SpvOpImageDrefGather || op == SpvOpDPdx ||
                 op == SpvOpDPdy || op == SpvOpFwidth || op == SpvOpDPdxFine ||
                 op == SpvOpDPdyFine || op == SpvOpDPdxCoarse ||
                 op == SpvOpDPdyCoarse ||
                 (op >= SpvOpImageSampleImplicitLod &&
                  op <= SpvOpImageSparseDrefGather)) {
         if (wc >= 3) {
            result = words[i + 2];
            has_result = true;
         }
      }

      if (has_result && result < bound && kind[result] == 0)
         kind[result] = 1;

      i += wc;
   }

   /* 2 = labels, 1 = values. Now check the phis. */
   bool ok = true;
   i = 5;
   while (i < n && ok) {
      const uint16_t wc = words[i] >> 16;
      const uint16_t op = words[i] & 0xFFFF;
      if (wc == 0)
         break;

      if (op == SpvOpPhi && wc >= 5) {
         for (unsigned k = 2; k + 1 < wc; k += 2) {
            const uint32_t value = words[i + 1 + k];
            const uint32_t parent = words[i + 2 + k];
            if (value >= bound || kind[value] == 0 || kind[value] == 2) {
               if (error && !*error)
               {
                  char buf[160];
                  snprintf(buf, sizeof(buf),
                           "a phi references value id %u, which the module never "
                           "defines (id bound %u)", value, bound);
                  *error = strdup(buf);
               }
               ok = false;
               break;
            }
            if (parent >= bound || kind[parent] != 2) {
               if (error && !*error)
               {
                  char buf[160];
                  snprintf(buf, sizeof(buf),
                           "a phi references parent block id %u, which is not a "
                           "label the module emits", parent);
                  *error = strdup(buf);
               }
               ok = false;
               break;
            }
         }
      }

      i += wc;
   }

   free(kind);
   return ok;
}

bool
sz_nir_to_spirv(struct nir_shader *nir,
                const struct pipe_stream_output_info *so,
                const struct sz_translate_options *opts, uint32_t **words,
                size_t *word_count, char **error)
{
   struct sz_translate t = {
      .nir = nir,
      .so = so,
      .alias_1d_shadow = opts->alias_1d_shadow,
      .clip_distance = opts->clip_distance,
      .cull_distance = opts->cull_distance,
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
   t.base_hint = malloc(num_defs);
   if (t.base_hint)
      memset(t.base_hint, 0xff, num_defs);
   t.interfaces = calloc(SZ_MAX_VARIABLES, sizeof(*t.interfaces));
   if (!t.defs_id || !t.def_types || !t.consts || !t.base_hint || !t.interfaces) {
      sz_fail(&t, "out of memory");
      goto fail;
   }

   sz_plan_phi_arms(&t, impl);

   SpvExecutionModel model;
   switch (nir->info.stage) {
   case MESA_SHADER_VERTEX:
      model = SpvExecutionModelVertex;
      break;
   case MESA_SHADER_GEOMETRY:
      model = SpvExecutionModelGeometry;
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
      /* Every global variable the entry point can reach belongs in its
       * interface list: SPIR-V 1.4 made that the rule for all storage classes,
       * not just Input and Output, and leaving a Private one out is a validation
       * error ("is used by entry point but is not listed as an interface"). */
      t.interfaces[t.num_interfaces++] = t.vars[t.num_vars - 1].id;
   }

   if (t.used_legacy_varying && t.used_generic_varying) {
      sz_fail(&t, "mixes legacy (`gl_TexCoord` style) and generic varyings, "
                  "which would share locations");
      goto fail;
   }

   const uint32_t function = sz_spv_id(t.m);
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      sz_spv_execution_mode(t.m, function, SpvExecutionModeOriginUpperLeft);

   /* A geometry shader has to declare both of its primitives and how many
    * vertices it may emit, because the pipeline is compiled against them: Vulkan
    * has no default and a missing mode is a validation error, while a wrong one
    * silently assembles the wrong primitive. */
   if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      /* The Geometry execution model is only valid with this capability. */
      sz_spv_capability(t.m, SpvCapabilityGeometry);

      SpvExecutionMode input, output;

      switch (nir->info.gs.input_primitive) {
      case MESA_PRIM_POINTS:
         input = SpvExecutionModeInputPoints;
         break;
      case MESA_PRIM_LINES:
         input = SpvExecutionModeInputLines;
         break;
      case MESA_PRIM_LINES_ADJACENCY:
         input = SpvExecutionModeInputLinesAdjacency;
         break;
      case MESA_PRIM_TRIANGLES:
         input = SpvExecutionModeTriangles;
         break;
      case MESA_PRIM_TRIANGLES_ADJACENCY:
         input = SpvExecutionModeInputTrianglesAdjacency;
         break;
      default:
         sz_fail(&t, "geometry input primitive %d is not supported",
                 (int)nir->info.gs.input_primitive);
         goto fail;
      }

      switch (nir->info.gs.output_primitive) {
      case MESA_PRIM_POINTS:
         output = SpvExecutionModeOutputPoints;
         break;
      case MESA_PRIM_LINE_STRIP:
         output = SpvExecutionModeOutputLineStrip;
         break;
      case MESA_PRIM_TRIANGLE_STRIP:
         output = SpvExecutionModeOutputTriangleStrip;
         break;
      default:
         sz_fail(&t, "geometry output primitive %d is not supported",
                 (int)nir->info.gs.output_primitive);
         goto fail;
      }

      sz_spv_execution_mode(t.m, function, input);
      sz_spv_execution_mode(t.m, function, output);
      sz_spv_execution_mode_literal(t.m, function, SpvExecutionModeOutputVertices,
                                    nir->info.gs.vertices_out);

      /* Invocations is not optional: the pipeline creation rule is "the entry
       * point must specify an invocation count greater than zero", so a shader
       * that says nothing is rejected rather than defaulted to one
       * (VUID-VkPipelineShaderStageCreateInfo-stage-00715). */
      sz_spv_execution_mode_literal(
         t.m, function, SpvExecutionModeInvocations,
         MAX2(nir->info.gs.invocations, 1));
   }
   /* Vulkan refuses to begin transform feedback for a pipeline whose last
    * pre-rasterization stage does not declare the Xfb execution mode
    * (VUID-vkCmdBeginTransformFeedbackEXT-None-04128). The per-variable
    * XfbBuffer/XfbStride/Offset decorations alone are not enough. */
   if (nir->info.stage != MESA_SHADER_FRAGMENT && t.so &&
       t.so->num_outputs > 0) {
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

   /* Before handing the words over: a wrong module is a driver crash, not an
    * error the consumer can report. */
   if (module && !sz_spv_verify_cfg(&t, module, bytes, error))
      goto fail;

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
   free(t.base_hint);
   free(t.def_types);
   free(t.consts);
   free(t.interfaces);
   sz_spv_module_destroy(t.m);
   return true;

fail:
   free(t.defs_id);
   free(t.base_hint);
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
