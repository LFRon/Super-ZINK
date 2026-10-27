/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Graphics pipelines and the shaders they consume.
 *
 * Two rules keep the draw path free of translation work (ADR-005b/005c):
 *
 *  - The draw path only compares the current state epoch against the epoch the
 *    bound state was resolved for. Nothing is translated per draw.
 *  - Every screen owns one pipeline layout and one VkPipelineCache, and every
 *    context keeps SZ_MAX_CACHED_PIPELINES of them. Layouts used to be created
 *    per rebuild and the pipeline cache was VK_NULL_HANDLE, which made RADV
 *    recompile the shaders on every state change (7-18 us measured); a cache hit
 *    is now a hash lookup plus one key comparison.
 *
 * The shaders are built with the SPIR-V emitter rather than through a compiler.
 * A vertex shader that transforms one attribute and a fragment shader that
 * writes a constant colour are enough to exercise vertex fetch, rasterisation,
 * blending and the attachment path -- and they are the smallest thing that can
 * prove the render path works. They are also a regression test for the emitter
 * that the real NIR translator will use.
 */

#ifndef SZ_PIPELINE_H
#define SZ_PIPELINE_H

#include "sz_context.h"
#include "sz_types.h"

/* The pipeline key records everything that is baked into the VkPipeline. The
 * state objects are compared by identity because the state tracker allocates a
 * new object per distinct state and cso_context caches them by content, so two
 * logically identical states are also the same pointer. The derived vertex
 * layout is compared by value: a buffer whose stride changes without a new
 * element object would otherwise reuse a pipeline that fetches the wrong rows.
 *
 * A 64-bit hash of this struct is only a lookup accelerator. Correctness comes
 * from the full comparison, so a collision costs a comparison, not a wrong
 * pipeline. */
struct sz_pipeline_key {
   VkFormat color[PIPE_MAX_COLOR_BUFS];
   unsigned color_count;
   VkFormat depth;
   VkPrimitiveTopology topology;
   VkSampleCountFlagBits samples;
   /* Primitive restart is per draw in Gallium, but Vulkan bakes it into the
    * input assembly state, so it belongs in the key. */
   bool primitive_restart;
   unsigned vertex_binding_count;
   VkVertexInputBindingDescription vertex_bindings[PIPE_MAX_ATTRIBS];
   unsigned vertex_attribute_count;
   VkVertexInputAttributeDescription vertex_attributes[PIPE_MAX_ATTRIBS];
   /* Unique ids, not addresses: state objects are freed and their addresses
    * reused, so a cached key holding a dangling pointer could match a different
    * state. */
   uint64_t blend;
   uint64_t rasterizer;
   uint64_t dsa;
   uint64_t vs;
   uint64_t fs;
};

struct sz_gfx_pipeline {
   VkShaderModule vs;
   VkShaderModule fs;
   VkPipeline pipeline;

   uint64_t hash;
   uint64_t last_use;
   struct sz_pipeline_key key;
};

/* FNV-1a. Exposed because the state objects hash their derived Vulkan state with
 * it; keeping one implementation means a pipeline key and a state hash can never
 * disagree about how bytes turn into a number. */
uint64_t sz_hash_bytes(uint64_t hash, const void *data, size_t size);

/* Gallium's PIPE_FUNC_* to VkCompareOp. Depth state and sampler state both need
 * it, and the two orderings are not the same enum, so it lives in one place. */
VkCompareOp sz_compare_op(unsigned func);

/* Screen-owned objects: the pipeline layout, the VkPipelineCache and the
 * bootstrap shader modules. Created after the descriptor set layout, destroyed
 * before the device. */
bool sz_pipeline_screen_init(struct sz_screen *screen);
void sz_pipeline_screen_finish(struct sz_screen *screen);

/* Returns a pipeline usable for the current state, or NULL. A cache hit costs
 * one hash comparison; a miss builds one pipeline and evicts the
 * least-recently-used entry that is not currently bound. */
struct sz_gfx_pipeline *sz_pipeline_get(struct sz_context *ctx,
                                        VkPrimitiveTopology topology);

/* Frees the retired pipelines whose submission has completed. Called before a
 * cache entry is evicted and on context teardown. */
void sz_pipeline_reclaim(struct sz_context *ctx);

/* Timestamps the pipelines retired while the batch that just went out was being
 * recorded. Called right after the context submits, before anything can look at
 * sz_pipeline_reclaim() again. */
void sz_pipeline_note_submit(struct sz_context *ctx);

void sz_pipeline_finish(struct sz_context *ctx);

#endif /* SZ_PIPELINE_H */
