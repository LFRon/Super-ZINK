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
   .lower_doubles_options = ~0,

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

   /* GLSL has real subtraction instructions and SPIR-V has OpFSub/OpISub. */
   .has_fsub = true,
   .has_isub = true,

   /* Zero frontend optimisation: D8. */
   .max_unroll_iterations = 0,
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
   /* Locals become SSA. This is a lowering, not an optimisation: SPIR-V variables
    * in Function storage have to be declared in the entry block, and an SSA
    * emitter has no reason to track them. Doing it here keeps the translator free
    * of a whole class of deref. */
   nir_lower_vars_to_ssa(nir);

   /* Only a leftover local is a problem; the pass returning false just means it
    * found nothing to do, which is the common case. */
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_foreach_function_temp_variable (var, impl) {
      mesa_logw("SZ: local variable `%s` survived vars-to-SSA; translation will "
                "reject this shader", var->name);
   }
}
