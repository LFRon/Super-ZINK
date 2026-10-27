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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "compiler/glsl_types.h"
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

   /* Indexed by nir_def::index. */
   struct sz_dtype *def_types;
   uint32_t *defs_id;

   struct {
      const nir_variable *var;
      uint32_t id;
      uint32_t pointee;
   } vars[SZ_MAX_VARIABLES];
   unsigned num_vars;

   uint32_t *interfaces;
   unsigned num_interfaces;
};

static bool sz_emit_instr(struct sz_translate *t, nir_instr *instr);

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

/* NIR does not record a base type on defs, so instructions that do not carry one
 * -- load_const, undef -- are typed by looking at their uses. GLSL's default
 * for an otherwise ambiguous constant is float, and that is also what the
 * frontend produces for the cases this driver can reach, so it is the fallback. */
static void
sz_guess_def_type(struct sz_translate *t, nir_def *def, struct sz_dtype *out)
{
   out->base = SZ_BASE_FLOAT;
   out->bits = def->bit_size;
   out->ncomp = def->num_components;

   if (def->bit_size == 1) {
      out->base = SZ_BASE_BOOL;
      return;
   }

   nir_foreach_use (src, def) {
      nir_instr *parent = nir_src_use_instr(src);

      if (parent->type == nir_instr_type_alu) {
         nir_alu_instr *alu = nir_instr_as_alu(parent);
         const nir_op_info *info = &nir_op_infos[alu->op];

         for (unsigned i = 0; i < info->num_inputs; i++) {
            if (&alu->src[i].src != src)
               continue;
            enum sz_base base;
            if (sz_base_from_alu_type(info->input_types[i], &base)) {
               out->base = base;
               return;
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
            out->base = base;
            return;
         }
      }
   }
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
         if (slot < VERT_ATTRIB_GENERIC0) {
            sz_fail(t, "vertex attribute `%s` uses the legacy slot %d, which "
                        "only the fixed-function paths need",
                    var->name, slot);
            return false;
         }
         *location = slot - VERT_ATTRIB_GENERIC0;
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
   default:
      /* Uniforms, UBOs and samplers need descriptors, which is Phase 2. */
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

   if (has_builtin)
      sz_spv_decorate_builtin(t->m, id, builtin);
   else
      sz_spv_decorate(t->m, id, SpvDecorationLocation, location);

   /* Integer varyings have to be flat: the Vulkan rule is unconditional, and a
    * missing decoration is a validation error rather than a performance hint. */
   if ((storage == SpvStorageClassInput || storage == SpvStorageClassOutput) &&
       type.base != SZ_BASE_FLOAT && type.base != SZ_BASE_BOOL)
      sz_spv_decorate_bare(t->m, id, SpvDecorationFlat);
   else if (storage == SpvStorageClassInput &&
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
   { nir_op_seq, SpvOpIEqual },
   { nir_op_sne, SpvOpINotEqual },
   { nir_op_ilt, SpvOpSLessThan },
   { nir_op_slt, SpvOpSLessThan },
   { nir_op_ige, SpvOpSGreaterThanEqual },
   { nir_op_sge, SpvOpSGreaterThanEqual },
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
   { nir_op_fsin, GLSLstd450Sin },
   { nir_op_fcos, GLSLstd450Cos },
   { nir_op_ftanh, GLSLstd450Tanh },
   { nir_op_ffma, GLSLstd450Fma },
};

static uint32_t sz_glsl450;

/* Applies a source swizzle, producing a value with `ncomp` components. */
static uint32_t
sz_alu_source(struct sz_translate *t, const nir_alu_src *src, unsigned ncomp)
{
   const nir_def *def = src->src.ssa;
   const uint32_t base = sz_scalar_type(t, t->def_types[def->index].base,
                                        t->def_types[def->index].bits);
   const unsigned n = def->num_components;
   const uint32_t id = t->defs_id[def->index];

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

   const uint32_t z = sz_spv_composite_construct(
      t->m, sz_dtype_type(t, dst), zeros, dst->ncomp);
   const uint32_t o = sz_spv_composite_construct(
      t->m, sz_dtype_type(t, dst), ones, dst->ncomp);

   if (dst->ncomp == 1) {
      const uint32_t ops[] = { condition, o, z };
      return sz_spv_alu(t->m, SpvOpSelect, sz_dtype_type(t, dst), ops, 3);
   }

   const uint32_t bool_vec = sz_spv_type_vec(
      t->m, sz_spv_type_bool(t->m), dst->ncomp);
   const uint32_t cond = sz_spv_composite_construct(t->m, bool_vec, conds,
                                                   dst->ncomp);
   const uint32_t ops[] = { cond, o, z };
   return sz_spv_alu(t->m, SpvOpSelect, sz_dtype_type(t, dst), ops, 3);
}

static void
sz_emit_alu(struct sz_translate *t, nir_alu_instr *alu)
{
   const nir_op op = alu->op;
   struct sz_dtype dst;
   dst.bits = alu->def.bit_size;
   dst.ncomp = alu->def.num_components;

   enum sz_base out_base;
   if (!sz_base_from_alu_type(nir_op_infos[op].output_type, &out_base)) {
      sz_fail(t, "NIR op `%s` has no SPIR-V mapping", nir_op_infos[op].name);
      return;
   }
   dst.base = out_base;

   /* Pure aliases. */
   if (op == nir_op_mov) {
      const uint32_t src = sz_alu_source(t, &alu->src[0], dst.ncomp);
      t->defs_id[alu->def.index] = src;
      t->def_types[alu->def.index] = dst;
      return;
   }

   if (op == nir_op_vec2 || op == nir_op_vec3 || op == nir_op_vec4 ||
       op == nir_op_vec8 || op == nir_op_vec16) {
      uint32_t comps[16];
      const struct sz_dtype *first = &t->def_types[alu->src[0].src.ssa->index];
      dst.base = first->base;
      dst.bits = first->bits;

      for (unsigned i = 0; i < dst.ncomp; i++)
         comps[i] = sz_alu_source(t, &alu->src[i], 1);

      const uint32_t vec = sz_spv_composite_construct(
         t->m, sz_dtype_type(t, &dst), comps, dst.ncomp);
      t->defs_id[alu->def.index] = vec;
      t->def_types[alu->def.index] = dst;
      return;
   }

   if (op == nir_op_bcsel) {
      const uint32_t cond = sz_alu_source(t, &alu->src[0], dst.ncomp);
      const uint32_t a = sz_alu_source(t, &alu->src[1], dst.ncomp);
      const uint32_t b = sz_alu_source(t, &alu->src[2], dst.ncomp);
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

   if (op == nir_op_fsat) {
      const uint32_t src = sz_alu_source(t, &alu->src[0], dst.ncomp);
      const uint32_t zero = sz_spv_const_float32(t->m, 0.0f);
      const uint32_t one = sz_spv_const_float32(t->m, 1.0f);
      const uint32_t ops[] = { src, zero, one };
      t->defs_id[alu->def.index] = sz_emit_ext(t, GLSLstd450FClamp, &dst, ops, 3);
      t->def_types[alu->def.index] = dst;
      return;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(sz_alu_maps); i++) {
      if (sz_alu_maps[i].op != op)
         continue;

      const uint32_t a = sz_alu_source(t, &alu->src[0], dst.ncomp);
      const uint32_t ops[] = {
         a,
         sz_alu_source(t, &alu->src[1], dst.ncomp),
         sz_alu_source(t, &alu->src[2], dst.ncomp),
      };

      if (op == nir_op_f2i1 || op == nir_op_i2i1) {
         /* "not equal to zero": a comparison, so the operand list is (a, 0). */
         const uint32_t zero = sz_spv_const_bits(
            t->m, sz_scalar_type(t, out_base, dst.bits), 0);
         const uint32_t cmp[] = { a, zero };
         t->defs_id[alu->def.index] =
            sz_spv_alu(t->m, sz_alu_maps[i].spirv, sz_dtype_type(t, &dst), cmp,
                       2);
      } else {
         t->defs_id[alu->def.index] = sz_spv_alu(
            t->m, sz_alu_maps[i].spirv, sz_dtype_type(t, &dst), ops,
            MIN2(nir_op_infos[op].num_inputs, 3));
      }
      t->def_types[alu->def.index] = dst;
      return;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(sz_glsl_maps); i++) {
      if (sz_glsl_maps[i].op != op)
         continue;

      const uint32_t a = sz_alu_source(t, &alu->src[0], dst.ncomp);
      const uint32_t ops[] = {
         a,
         sz_alu_source(t, &alu->src[1], dst.ncomp),
         sz_alu_source(t, &alu->src[2], dst.ncomp),
      };

      struct sz_dtype work = dst;
      /* FSqrt and friends are defined on floats; keep the declared type. */
      t->defs_id[alu->def.index] = sz_emit_ext(
         t, sz_glsl_maps[i].ext, &work, ops,
         MIN2(nir_op_infos[op].num_inputs, 3));
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
sz_emit_jump(struct sz_translate *t, nir_jump_instr *jump, nir_block *block,
             const uint32_t *labels)
{
   switch (jump->type) {
   case nir_jump_return:
      sz_spv_return(t->m);
      return;

   case nir_jump_goto: {
      const uint32_t target = labels[jump->target->index];
      sz_spv_raw_function_op(t->m, SpvOpBranch, &target, 1);
      return;
   }

   case nir_jump_goto_if: {
      const uint32_t cond = t->defs_id[jump->condition.ssa->index];
      const uint32_t ops[] = {
         cond,
         labels[jump->target->index],
         labels[jump->else_target ? jump->else_target->index : block->index],
      };
      sz_spv_raw_function_op(t->m, SpvOpBranchConditional, ops, 3);
      return;
   }

   case nir_jump_halt:
   case nir_jump_abort:
      /* Both terminate the invocation; the driver has no discards yet, so the
       * closest legal SPIR-V is a return. */
      sz_spv_return(t->m);
      return;

   case nir_jump_break:
   case nir_jump_continue:
      /* nir_foreach_block_unstructured flattens loops, so a break is a branch to
       * a flattened block that the structured iteration would have hidden. */
      sz_fail(t, "jump type %d is not handled yet", (int)jump->type);
      return;
   }

   sz_fail(t, "unknown jump type %d", (int)jump->type);
}

static bool
sz_emit_intrinsic(struct sz_translate *t, nir_intrinsic_instr *intrin)
{
   switch (intrin->intrinsic) {
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
   default:
      sz_fail(t, "NIR instruction type %d is not supported yet",
              (int)instr->type);
      return false;
   }
}

/* ------------------------------------------------------------------------- */
/* Entry point                                                              */
/* ------------------------------------------------------------------------- */

static bool
sz_emit_shader(struct sz_translate *t, uint32_t function)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(t->nir);
   nir_block *entry = nir_start_block(impl);

   /* One extra slot: nir_index_blocks() gives end_block the index num_blocks, so
    * any reference to it would otherwise be one past the end. */
   uint32_t *labels = calloc(impl->num_blocks + 1, sizeof(*labels));
   if (!labels) {
      sz_fail(t, "out of memory");
      return false;
   }

   nir_foreach_block_unstructured (block, impl)
      labels[block->index] = sz_spv_id(t->m);

   const uint32_t void_type = sz_spv_type_void(t->m);
   const uint32_t fn_type = sz_spv_type_function(t->m, void_type, NULL, 0);
   sz_spv_function(t->m, function, void_type, fn_type);

   /* SPIR-V requires the function's first block to be the one the entry point
    * executes, so it cannot go through the generic loop below. */
   sz_spv_label_id(t->m, labels[entry->index]);
   bool entry_terminated = false;
   nir_foreach_instr (instr, entry) {
      if (instr->type == nir_instr_type_jump) {
         sz_emit_jump(t, nir_instr_as_jump(instr), entry, labels);
         entry_terminated = true;
         break;
      }
      if (!sz_emit_instr(t, instr))
         break;
   }
   /* A block with no terminator instruction is malformed NIR, but emitting the
    * return keeps the module valid so the failure names the real problem. */
   if (!entry_terminated && !t->failed)
      sz_spv_return(t->m);

   nir_foreach_block_unstructured (block, impl) {
      if (block == entry || t->failed)
         continue;

      sz_spv_label_id(t->m, labels[block->index]);
      bool terminated = false;
      nir_foreach_instr (instr, block) {
         if (instr->type == nir_instr_type_jump) {
            sz_emit_jump(t, nir_instr_as_jump(instr), block, labels);
            terminated = true;
            break;
         }
         if (!sz_emit_instr(t, instr)) {
            terminated = true;
            break;
         }
      }

      /* A block with no terminator is malformed NIR, but a validator error that
       * points at the missing branch is more useful than a crash. */
      if (!terminated && !t->failed)
         sz_spv_return(t->m);
   }

   sz_spv_function_end(t->m);
   free(labels);
   return !t->failed;
}

bool
sz_nir_to_spirv(struct nir_shader *nir, uint32_t **words, size_t *word_count,
                char **error)
{
   struct sz_translate t = {
      .nir = nir,
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
   t.defs_id = calloc(num_defs, sizeof(*t.defs_id));
   t.def_types = calloc(num_defs, sizeof(*t.def_types));
   t.interfaces = calloc(SZ_MAX_VARIABLES, sizeof(*t.interfaces));
   if (!t.defs_id || !t.def_types || !t.interfaces) {
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
   sz_spv_memory_model(t.m, SpvAddressingModelLogical, SpvMemoryModelGLSL450);
   sz_glsl450 = sz_spv_ext_inst_import(t.m, "GLSL.std.450");

   nir_foreach_variable_in_shader (var, nir) {
      if (!sz_declare_variable(&t, var))
         goto fail;
      t.interfaces[t.num_interfaces++] = t.vars[t.num_vars - 1].id;
   }

   const uint32_t function = sz_spv_id(t.m);
   sz_spv_entry_point(t.m, model, function, "main", t.interfaces,
                      t.num_interfaces);
   if (nir->info.stage == MESA_SHADER_FRAGMENT)
      sz_spv_execution_mode(t.m, function, SpvExecutionModeOriginUpperLeft);

   if (!sz_emit_shader(&t, function))
      goto fail;

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
   free(t.interfaces);
   sz_spv_module_destroy(t.m);
   return true;

fail:
   free(t.defs_id);
   free(t.def_types);
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
