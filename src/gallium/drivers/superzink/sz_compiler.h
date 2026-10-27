/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Shader compilation.
 *
 * Two things live here that have to agree with each other: the
 * nir_shader_compiler_options we hand the state tracker, and the set of NIR we
 * can translate. The options are not tuning knobs -- each one tells the frontend
 * "lower this for me", and every option left off is a promise that the
 * translator handles it. Getting that wrong produces a shader that links and then
 * translates into nonsense, so the options describe exactly the capabilities this
 * backend has and nothing more.
 *
 * Following decision D8, no nir_opt_* pass runs here. Every Vulkan driver turns
 * SPIR-V back into NIR and optimises it again, so optimising here is work whose
 * result is discarded -- and it costs compile time on the critical path.
 */

#ifndef SZ_COMPILER_H
#define SZ_COMPILER_H

#include <stddef.h>
#include <stdint.h>

#include "compiler/nir/nir.h"

/* Number of constant buffer slots a stage can bind, which is also how many
 * uniform buffer bindings that stage owns in the descriptor set layout. */
#define SZ_MAX_CONST_BUFFERS 8

/* Combined image samplers per stage, i.e. GL's texture image units after the
 * driver had its say. The binding budget below is what fixes this number: a
 * conformant device may push as few as 32 descriptors, and every slot of every
 * stage has to fit in that. */
#define SZ_MAX_SAMPLERS 8

/* Size of the array the translator declares inside a uniform block, in 32-bit
 * words. The block has no declared member types -- the load applies one -- so
 * the array exists only to give SPIR-V a type with an ArrayStride. It cannot be
 * a runtime array: that needs VK_EXT_shader_uniform_buffer_unsized_array, which
 * Turnip does not advertise. 64 KiB is both what shader_caps reports as
 * max_const_buffer0_size and the GLSL maximum block size, and every Vulkan
 * device allows at least a quarter of it. */
#define SZ_MAX_CONSTBUF_WORDS 16384

/* One push descriptor set holds every stage's resources. The spec allows at
 * most one push descriptor set per pipeline layout (VUID 00293), so a set per
 * stage -- the obvious mapping -- is not legal. Each stage therefore owns a
 * disjoint range of bindings inside the single set, and the binding number a
 * shader decorates is stage_base + gallium constant buffer slot.
 *
 * The layout may not declare more than maxPushDescriptors bindings in total
 * (VUID 00281), and the spec's minimum for that limit is 32 -- a limit a
 * conformant device is allowed to report. Each stage owns SZ_DESCRIPTOR_SLOTS
 * of them (constant buffers first, then samplers), and two stages fill the 32
 * exactly, which is also why only vertex and fragment are compiled: a stage
 * registered beyond these gets a binding number outside the layout, which is
 * why sz_descriptor_push() asserts on it.
 *
 * Wanting more of either is a signal to move to VK_EXT_descriptor_buffer,
 * where descriptors live in a plain buffer and the count is bounded by memory
 * rather than by a per-device limit. */
#define SZ_DESCRIPTOR_SET 0
#define SZ_DESCRIPTOR_STAGES 2
#define SZ_DESCRIPTOR_SLOTS (SZ_MAX_CONST_BUFFERS + SZ_MAX_SAMPLERS)
#define SZ_DESCRIPTOR_BINDINGS (SZ_DESCRIPTOR_STAGES * SZ_DESCRIPTOR_SLOTS)

/* VK_KHR_push_descriptor's own floor for maxPushDescriptors. A device that
 * reports less is out of spec, but the decode is cheap and the alternative is
 * creating a layout the driver is allowed to reject. */
#define SZ_MIN_PUSH_DESCRIPTORS 32

static inline unsigned
sz_descriptor_stage_base(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return 0;
   case MESA_SHADER_FRAGMENT:
      return SZ_DESCRIPTOR_SLOTS;
   default:
      return SZ_DESCRIPTOR_BINDINGS;
   }
}

/* Binding of gallium constant buffer `slot` for `stage`. */
static inline unsigned
sz_descriptor_binding(mesa_shader_stage stage, unsigned slot)
{
   return sz_descriptor_stage_base(stage) + slot;
}

/* Binding of texture unit `slot` for `stage`: the samplers of a stage follow its
 * constant buffers in one contiguous run. */
static inline unsigned
sz_sampler_binding(mesa_shader_stage stage, unsigned slot)
{
   return sz_descriptor_stage_base(stage) + SZ_MAX_CONST_BUFFERS + slot;
}

/* The options for one shader stage. The returned pointer stays valid for the
 * lifetime of the process. */
const struct nir_shader_compiler_options *sz_nir_options(mesa_shader_stage stage);

/* Correctness-only lowering, applied to the driver's own copy of the NIR before
 * translation. Idempotent; safe to call on already-lowered shaders. */
void sz_lower_nir(struct nir_shader *nir);

/* Translates a lowered NIR shader into a SPIR-V module. On success *out_words is
 * a malloc'd array of *out_size bytes, owned by the caller.
 *
 * Returns false and leaves *out_words NULL for anything outside the implemented
 * subset. Every rejection names the NIR instruction it could not handle: a
 * translator that silently skips an instruction produces a shader that renders
 * the wrong thing, which is far more expensive to debug than a refusal. */
bool sz_nir_to_spirv(struct nir_shader *nir, uint32_t **out_words,
                     size_t *out_size, char **error);

/* SZ_DEBUG=spirv: writes a module to dumpNN_<suffix>.spv in the working
 * directory, next to the bootstrap shaders' dumps. */
void sz_compiler_dump_spirv(const uint32_t *words, size_t word_count,
                            const char *suffix);

#endif /* SZ_COMPILER_H */
