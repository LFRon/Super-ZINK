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
      unsigned vertex_element_count;
      unsigned stride;
      /* The format of the attribute the bootstrap shader reads. Same stride and
       * count with a different format is a different pipeline, and reusing the
       * old one reads the vertices as the wrong type without any Vulkan error. */
      VkFormat vertex_format;
      VkSampleCountFlagBits samples;
   } key;
};

/* Returns a pipeline usable for the current state, or NULL. Rebuilds when the
 * recorded state no longer matches. */
struct sz_gfx_pipeline *sz_pipeline_get(struct sz_context *ctx,
                                        VkPrimitiveTopology topology,
                                        unsigned vertex_element_count,
                                        unsigned stride);

void sz_pipeline_finish(struct sz_context *ctx);

#endif /* SZ_PIPELINE_H */
