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
 * uniform buffer bindings that stage owns in the descriptor set layout.
 *
 * 14 rather than a rounder number because st_init_limits() withdraws
 * GL_ARB_uniform_buffer_object -- and with it the whole of GL 3.1 -- from any
 * driver whose per-stage uniform block count is below 12, and the count it
 * reads is this minus the first slot, which holds the default uniform block.
 * With the samplers moved to their own descriptor set, two stages of 14 still
 * fit inside the 32 push descriptors a conformant device guarantees. */
#define SZ_MAX_CONST_BUFFERS 14

/* Combined image samplers per stage, i.e. GL's texture image units after the
 * driver had its say. The binding budget below is what fixes this number: a
 * conformant device may push as few as 32 descriptors, and every slot of every
 * stage has to fit in that. */
#define SZ_MAX_SAMPLERS 16

/* Storage images per stage. GL guarantees eight image units, which is what
 * GL_MAX_IMAGE_UNITS reports; they live in a descriptor set of their own because
 * VK_DESCRIPTOR_TYPE_STORAGE_IMAGE is a third descriptor type. */
#define SZ_MAX_IMAGES 8

/* Size of the array the translator declares inside a uniform block, in 32-bit
 * words. The block has no declared member types -- the load applies one -- so
 * the array exists only to give SPIR-V a type with an ArrayStride. It cannot be
 * a runtime array: that needs VK_EXT_shader_uniform_buffer_unsized_array, which
 * Turnip does not advertise. 64 KiB is both what shader_caps reports as
 * max_const_buffer0_size and the GLSL maximum block size, and every Vulkan
 * device allows at least a quarter of it. */
#define SZ_MAX_CONSTBUF_WORDS 16384

/* Two descriptor sets, because a pipeline layout may contain at most one push
 * descriptor set (VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293) and one
 * push set cannot hold both kinds: SZ_MAX_CONST_BUFFERS + SZ_MAX_SAMPLERS per
 * stage, times two stages, is more than the 32 bindings maxPushDescriptors is
 * guaranteed to allow.
 *
 * Set 0 is the push set and holds every stage's uniform buffers: it is rewritten
 * per state change, which is the hot path and what push descriptors are for.
 * Set 1 is a plain descriptor set and holds the combined image samplers, which
 * are updated and bound only when the sampler state itself changes.
 *
 * Three stages, because GL 3.2's geometry stage binds its own constant buffers
 * and samples its own textures. That puts the two sets at 42 and 48 bindings,
 * which is past the 32 a push descriptor set is guaranteed to hold -- so a
 * device without VK_EXT_descriptor_buffer can no longer be driven, and says so
 * at screen creation instead of failing at the first draw.
 *
 * A fourth kind (shader buffers or images) takes a third set. */
#define SZ_DESCRIPTOR_SET_UBO 0
#define SZ_DESCRIPTOR_SET_SAMPLERS 1
/* Set 2 holds the texel buffers. They get a set of their own because
 * VkDescriptorSetLayoutBinding has exactly one descriptor type, and a buffer
 * texture is a VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER -- a VkBufferView, with no
 * sampler and no image view, which a COMBINED_IMAGE_SAMPLER binding cannot hold. */
#define SZ_DESCRIPTOR_SET_TEXEL_BUFFERS 2
/* Set 3 holds the storage images: the images a shader reads and writes through
 * imageLoad/imageStore/imageAtomic. VK_DESCRIPTOR_TYPE_STORAGE_IMAGE is a third
 * descriptor type, so it needs a third set. */
#define SZ_DESCRIPTOR_SET_IMAGES 3
#define SZ_DESCRIPTOR_SET_COUNT 4
#define SZ_DESCRIPTOR_STAGES 5
#define SZ_DESCRIPTOR_UBO_BINDINGS (SZ_DESCRIPTOR_STAGES * SZ_MAX_CONST_BUFFERS)
#define SZ_DESCRIPTOR_SAMPLER_BINDINGS (SZ_DESCRIPTOR_STAGES * SZ_MAX_SAMPLERS)
#define SZ_DESCRIPTOR_TEXEL_BINDINGS (SZ_DESCRIPTOR_STAGES * SZ_MAX_SAMPLERS)
#define SZ_DESCRIPTOR_IMAGE_BINDINGS (SZ_DESCRIPTOR_STAGES * SZ_MAX_IMAGES)

/* How many distinct sampler-set contents to cache. Every draw that rebinds the
 * same textures and samplers hits an existing entry, so this is the number of
 * *different* combinations an application uses, not the number of draws. A slot
 * is only reused once the submission that last bound it has completed, so a
 * single command buffer that binds more distinct combinations than this cannot
 * make progress: that is the known limit of a pool-based sampler set, and the
 * reason the next step is VK_EXT_descriptor_buffer, where descriptors live in a
 * buffer and are bump-allocated per draw. */
#define SZ_SAMPLER_SET_POOL 1024

/* VK_KHR_push_descriptor's own floor for maxPushDescriptors. A device that
 * reports less is out of spec, but the decode is cheap and the alternative is
 * creating a layout the driver is allowed to reject. */
#define SZ_MIN_PUSH_DESCRIPTORS 32

static inline unsigned
sz_descriptor_stage_index(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return 0;
   case MESA_SHADER_TESS_CTRL:
      return 1;
   case MESA_SHADER_TESS_EVAL:
      return 2;
   case MESA_SHADER_GEOMETRY:
      return 3;
   case MESA_SHADER_FRAGMENT:
      return 4;
   default:
      return SZ_DESCRIPTOR_STAGES;
   }
}

/* Binding of gallium constant buffer `slot` for `stage`. */
static inline unsigned
sz_descriptor_binding(mesa_shader_stage stage, unsigned slot)
{
   return sz_descriptor_stage_index(stage) * SZ_MAX_CONST_BUFFERS + slot;
}

/* Binding of texture unit `slot` for `stage`: the samplers of a stage follow its
 * constant buffers in one contiguous run. */
static inline unsigned
sz_sampler_binding(mesa_shader_stage stage, unsigned slot)
{
   return sz_descriptor_stage_index(stage) * SZ_MAX_SAMPLERS + slot;
}

/* The same numbering, in the texel-buffer set. */
static inline unsigned
sz_texel_binding(mesa_shader_stage stage, unsigned slot)
{
   return sz_descriptor_stage_index(stage) * SZ_MAX_SAMPLERS + slot;
}

/* The same numbering, in the storage-image set. */
static inline unsigned
sz_image_binding(mesa_shader_stage stage, unsigned slot)
{
   return sz_descriptor_stage_index(stage) * SZ_MAX_IMAGES + slot;
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
 * the wrong thing, which is far more expensive to debug than a refusal.
 *
 * `alias_1d_shadow` says the screen has no 1D depth images, so a 1D shadow
 * sampler's texture is really a 2D image one texel high: the declared image
 * dimension becomes 2D and the coordinate gains a zero second component. */
struct pipe_stream_output_info;

/* What the translator needs to know about the device beyond the NIR itself.
 *
 * These are flags rather than the screen so the translator keeps no dependency
 * on the driver's state, and so a shader can be translated in a test without a
 * Vulkan device. */
struct sz_translate_options {
   /* No 1D depth images: a 1D shadow sampler is declared as 2D and sampled with
    * a zero second coordinate. See sz_image_vk_type(). */
   bool alias_1d_shadow;
   /* VkPhysicalDeviceFeatures::shaderClipDistance / shaderCullDistance. Without
    * them the ClipDistance / CullDistance builtins do not exist. */
   bool clip_distance;
   bool cull_distance;
   /* VkPhysicalDeviceVulkan11Features::shaderDrawParameters: the BaseInstance
    * builtin exists, so gl_InstanceID can subtract it. */
   bool base_instance;
};

bool sz_nir_to_spirv(struct nir_shader *nir,
                     const struct pipe_stream_output_info *so,
                     const struct sz_translate_options *opts,
                     uint32_t **out_words, size_t *out_size, char **error);

/* SZ_DEBUG=spirv: writes a module to dumpNN_<suffix>.spv in the working
 * directory, next to the bootstrap shaders' dumps. */
void sz_compiler_dump_spirv(const uint32_t *words, size_t word_count,
                            const char *suffix);

#endif /* SZ_COMPILER_H */
