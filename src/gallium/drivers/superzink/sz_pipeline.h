/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Graphics pipelines and the shaders they consume.
 *
 * Phase 1 builds one pipeline at a time and rebuilds it whenever the vertex
 * layout or the attachment formats change. That is deliberately wasteful: there
 * is no pipeline cache, no state key and no deduplication, because the render
 * path has to be proven before it is worth optimising, and a wrong pipeline key
 * is a rendering bug while a missing cache is only a stutter.
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

struct sz_gfx_pipeline {
   VkShaderModule vs;
   VkShaderModule fs;
   VkPipelineLayout layout;
   VkPipeline pipeline;

   /* What the pipeline was built for. A draw whose state disagrees gets a
    * rebuild; comparing these is how a stale pipeline is caught. */
   struct {
      VkFormat color[PIPE_MAX_COLOR_BUFS];
      unsigned color_count;
      VkFormat depth;
      VkPrimitiveTopology topology;
      /* The full vertex fetch layout. Buffers, strides, formats and offsets all
       * have to match: reusing a pipeline whose second attribute reads a
       * different buffer produces garbage positions with no Vulkan error. */
      unsigned vertex_binding_count;
      VkVertexInputBindingDescription vertex_bindings[PIPE_MAX_ATTRIBS];
      unsigned vertex_attribute_count;
      VkVertexInputAttributeDescription vertex_attributes[PIPE_MAX_ATTRIBS];
      VkSampleCountFlagBits samples;
      /* The pipe state objects the depth, cull, front-face and blend fields were
       * derived from. Identity, not contents: the state tracker builds a new
       * object whenever any of them changes. */
      const void *blend;
      const void *rasterizer;
      const void *dsa;
      /* Which shaders the pipeline was built from. Two programs with the same
       * layout are still different pipelines. */
      const void *vs;
      const void *fs;
   } key;
};

/* Gallium's PIPE_FUNC_* to VkCompareOp. Depth state and sampler state both need
 * it, and the two orderings are not the same enum, so it lives in one place. */
VkCompareOp sz_compare_op(unsigned func);

/* Frees the retired pipelines whose submission has completed. Called before a
 * pipeline is replaced and on context teardown. */
void sz_pipeline_reclaim(struct sz_context *ctx);

/* Timestamps the pipelines retired while the batch that just went out was being
 * recorded. Called right after the context submits, before anything can look at
 * sz_pipeline_reclaim() again. */
void sz_pipeline_note_submit(struct sz_context *ctx);

/* Returns a pipeline usable for the current state, or NULL. Rebuilds when the
 * recorded state no longer matches. */
struct sz_gfx_pipeline *sz_pipeline_get(struct sz_context *ctx,
                                        VkPrimitiveTopology topology);

void sz_pipeline_finish(struct sz_context *ctx);

#endif /* SZ_PIPELINE_H */
