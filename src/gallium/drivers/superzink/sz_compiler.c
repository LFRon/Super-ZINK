/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_compiler.h"

#include "sz_debug.h"

#include <stdio.h>

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_legacy.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/ralloc.h"

/* One options block shared by every stage.
 *
 * The rule for every field: true means "the frontend lowers this before I see
 * it", so a true here is a promise the translator does not have to keep, and a
 * false is one it does.
 *
 * The shader stages we advertise are vertex and fragment with GLSL 1.20, so the
 * set below is what GLSL 1.20 can actually produce plus what the state tracker
 * synthesises for fixed-function state. */
static const struct nir_shader_compiler_options sz_nir_options_default = {
   /* 32-bit only. Anything narrower or wider is lowered here because SPIR-V
    * types have to be declared per width, and a half-done 16-bit path is a
    * correctness bug waiting for the first shader that uses it. */
   .support_16bit_alu = false,
   .lower_int64_options = ~0,
   .lower_flrp64 = true,
   .lower_doubles_options = 0,

   /* SPIR-V has OpFma and GLSL fma() maps onto it directly. */
   .float_mul_add32 = nir_float_muladd_support_has_ffma,

   /* Ops with no SPIR-V equivalent, or whose expansion is two instructions. */
   .lower_mul_high = true,
   .lower_uadd_carry = true,
   .lower_usub_borrow = true,
   .lower_uadd_sat = true,
   .lower_usub_sat = true,
   .lower_bitfield_extract = true,
   .lower_bitfield_insert = true,
   .lower_bitfield_reverse = true,
   .lower_bit_count = true,
   .lower_ifind_msb = true,
   .lower_ufind_msb = true,
   .lower_find_lsb = true,
   /* `(x & 0xff)` becomes nir_op_extract_u8 in the algebraic pass, and a byte
    * mask is everywhere -- packUnorm's expansion alone emits four. They lower
    * to a shift and an and, which the translator already speaks. */
   .lower_extract_byte = true,
   .lower_extract_word = true,
   .lower_insert_byte = true,
   .lower_insert_word = true,
   .lower_pack_half_2x16 = true,
   .lower_unpack_half_2x16 = true,
   .lower_pack_split = true,

   /* Arrays are translated as arrays, not flattened, so the frontend does not
    * have to compact them. */
   .compact_arrays = true,

   /* Scalar NIR is what the translator emits one component at a time; keeping
    * vectors around would need a second path through every op for no benefit,
    * because the SPIR-V consumer scalarises anyway. */
   .lower_to_scalar = true,

   /* The default uniform block becomes a UBO, so the translator only ever sees
    * load_ubo and never a nir_var_uniform deref. The frontend also shifts the
    * bindings of real UBOs up by one, which makes the NIR index equal the
    * gallium constant buffer slot: slot 0 is the default block, slot i+1 is
    * uniform block i. */
   .lower_uniforms_to_ubo = true,

   /* GLSL has real subtraction instructions and SPIR-V has OpFSub/OpISub. */
   .has_fsub = true,
   .has_isub = true,

   /* Zero frontend optimisation: D8. */
   .max_unroll_iterations = 0,

   /* Indirect indexing of an interface array is translated as an SPIR-V access
    * chain with a dynamic index, which is exactly what SPIR-V allows. Telling
    * the frontend so keeps it from rewriting each of those loads and stores into
    * an if-else tree over the whole array: a tessellation control shader reads
    * `gl_in[gl_InvocationID]`, and lowering that to a chain of comparisons turns
    * one access chain into one branch per patch vertex. The set below is the
    * same one zink reports, and the stages it names are the ones whose inputs or
    * outputs GL indexes indirectly in practice -- tessellation (gl_in/gl_out by
    * invocation) and the fragment stage (input arrays by index). */
   .support_indirect_inputs = BITFIELD_BIT(MESA_SHADER_TESS_CTRL) |
                              BITFIELD_BIT(MESA_SHADER_TESS_EVAL) |
                              BITFIELD_BIT(MESA_SHADER_FRAGMENT),
   .support_indirect_outputs = (uint8_t)BITFIELD_MASK(MESA_SHADER_STAGES),
};

const struct nir_shader_compiler_options *
sz_nir_options(mesa_shader_stage stage)
{
   /* Every stage gets the same block today. It is a function rather than a plain
    * array so that the moment geometry or tessellation appears, the divergence
    * has somewhere to live instead of being sprinkled through the screen code. */
   (void)stage;
   return &sz_nir_options_default;
}

void
sz_compiler_dump_spirv(const uint32_t *words, size_t word_count,
                       const char *suffix)
{
   if (!sz_debug_enabled(SZ_DEBUG_SPIRV) || !words)
      return;

   static unsigned counter;
   char name[64];
   snprintf(name, sizeof(name), "dump%02u_%s.spv", counter++, suffix);

   FILE *f = fopen(name, "wb");
   if (!f) {
      mesa_logw("SZ: could not open %s for the SPIR-V dump", name);
      return;
   }
   fwrite(words, sizeof(*words), word_count, f);
   fclose(f);

   mesa_logi("SZ: dumped SPIR-V %s (%zu words)", name, word_count);
}

void
sz_lower_nir(struct nir_shader *nir)
{
   /* GLSL lets a shader call functions, and NIR keeps them: ANGLE's translated
    * shaders (which is what a browser hands us) are full of helpers, and Skia
    * puts a call to its own entry wrapper around the whole body. Translating a
    * call properly means a second function emitter -- parameters, per-function
    * bodies, OpFunctionCall -- for no gain, because a call site is exactly as
    * inlinable here as it is in the GLSL compiler that produced it. Inline them
    * instead, following the sequence nir_functions.c documents.
    *
    * Cost is code size at the call site, which matters only for a shader that
    * calls the same helper many times in a loop body; the SPIR-V consumer
    * optimises afterwards either way. */
   unsigned num_impls = 0;
   nir_foreach_function_impl (impl, nir) {
      (void)impl;
      if (++num_impls > 1)
         break;
   }
   if (num_impls > 1) {
      nir_lower_variable_initializers(nir, nir_var_function_temp);
      nir_lower_returns(nir);
      nir_inline_functions(nir);
      nir_opt_deref(nir);
      nir_remove_non_entrypoints(nir);
   }

   /* Locals become SSA. This is a lowering, not an optimisation: SPIR-V variables
    * in Function storage have to be declared in the entry block, and an SSA
    * emitter has no reason to track them. Doing it here keeps the translator free
    * of a whole class of deref. Running it after inlining is what turns the
    * callee's locals -- now the caller's locals -- into SSA as well. */
   nir_lower_vars_to_ssa(nir);

   /* frexp() is two NIR ops (a significand and an exponent) and there is no
    * SPIR-V pair that returns them separately; the GLSL dialect's FrexpStruct
    * returns a struct, which this translator has no reason to grow a struct type
    * for. The pass rewrites both ops into arithmetic instead. */
   nir_lower_frexp(nir);

   /* GL clip space has z in [-1, 1] and Vulkan's viewport expects [0, 1]. The Y
    * axis is handled by the viewport's negative height, but z has no viewport
    * equivalent, so the last pre-rasterisation stage rescales it: z' = (z + w) /
    * 2, which is exactly what this pass emits. Without it everything behind z = 0
    * is clipped away, so half the scene disappears and the depth values of what
    * remains are squeezed into the far half of the depth buffer.
    *
    * This is the same trade ANGLE makes in its Vulkan backend, and for the same
    * reason: a shader-side transform cannot be forgotten by a state object.
    *
    * glClipControl(GL_ZERO_TO_ONE) -- reported as pipe_rasterizer_state::
    * clip_halfz -- asks for Vulkan's own convention, and a shader lowered here
    * would end up with its depth in [0.5, 1]. Honouring it needs a second shader
    * variant per program, which is the shader-cache work of a later phase; until
    * then the pipeline layer says so once instead of rendering a subtly wrong
    * depth range in silence. */
   switch (nir->info.stage) {
   case MESA_SHADER_VERTEX:
   case MESA_SHADER_TESS_EVAL:
   case MESA_SHADER_GEOMETRY:
      nir_lower_clip_halfz(nir);
      break;
   default:
      break;
   }

   /* Only a leftover local is a problem; the pass returning false just means it
    * found nothing to do, which is the common case. */
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_function_temp_variable (var, impl) {
      mesa_logw("SZ: local variable `%s` survived vars-to-SSA; translation will "
                "reject this shader", var->name);
   }
}
