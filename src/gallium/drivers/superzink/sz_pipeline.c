/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_pipeline.h"

#include "sz_debug.h"
#include "sz_format.h"
#include "sz_spirv.h"

#include "vk_format.h"

#include "util/log.h"
#include "util/u_memory.h"
#include "util/macros.h"
#include "util/u_inlines.h"

#include <stdio.h>
#include <string.h>

/* The fragment colour the bootstrap pipeline writes. A value that is not
 * representable as a "nice" byte on purpose: 0.5 would round, 0.25 would not
 * catch a channel swap. */
static const float sz_pipeline_color[4] = { 0.25f, 0.50f, 0.75f, 1.0f };

/* SZ_DEBUG=spirv: write the module out so spirv-val / spirv-dis can check it.
 *
 * Same convention as zink's ZINK_DEBUG=spirv: dumpNN_<stage>.spv in the working
 * directory. A module the driver accepts but a validator rejects is a bug that
 * only manifests on somebody else's GPU, so the dump is worth having. */
static void
sz_pipeline_dump(const uint32_t *words, size_t size, const char *suffix)
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
   fwrite(words, 1, size, f);
   fclose(f);

   mesa_logi("SZ: dumped SPIR-V %s (%zu bytes)", name, size);
}

/* ------------------------------------------------------------------------- */
/* Bootstrap shaders                                                          */
/* ------------------------------------------------------------------------- */

/* Vertex shader: gl_Position = the attribute bound at location 0.
 *
 * Nothing else. There is no uniform, no matrix and no varying, so a failure can
 * only be in vertex fetch, the pipeline or the attachment path -- never in
 * algebra the test does not need. */
static VkShaderModule
sz_pipeline_bootstrap_vs(struct sz_screen *screen)
{
   struct sz_spv_module *m = sz_spv_module_create();
   if (!m)
      return VK_NULL_HANDLE;

   sz_spv_capability(m, SpvCapabilityShader);
   sz_spv_memory_model(m, SpvAddressingModelLogical, SpvMemoryModelGLSL450);

   const uint32_t f32 = sz_spv_type_float32(m);
   const uint32_t v4 = sz_spv_type_vec(m, f32, 4);

   const uint32_t input = sz_spv_variable(m,
                                          sz_spv_type_ptr(m, SpvStorageClassInput, v4),
                                          SpvStorageClassInput);
   const uint32_t position =
      sz_spv_variable(m, sz_spv_type_ptr(m, SpvStorageClassOutput, v4),
                      SpvStorageClassOutput);

   sz_spv_decorate(m, input, SpvDecorationLocation, 0);
   sz_spv_decorate_builtin(m, position, SpvBuiltInPosition);

   const uint32_t interface[] = { input, position };
   const uint32_t main_fn = sz_spv_id(m);
   sz_spv_entry_point(m, SpvExecutionModelVertex, main_fn, "main", interface,
                      ARRAY_SIZE(interface));

   sz_spv_function(m, main_fn, sz_spv_type_void(m),
                   sz_spv_type_function(m, sz_spv_type_void(m), NULL, 0));
   sz_spv_label(m);
   const uint32_t value = sz_spv_id(m);
   sz_spv_load(m, v4, value, input);
   sz_spv_store(m, position, value);
   sz_spv_return(m);
   sz_spv_function_end(m);

   size_t size = 0;
   const uint32_t *words = sz_spv_finalize(m, &size);
   sz_pipeline_dump(words, size, "vs");

   VkShaderModule module = VK_NULL_HANDLE;
   if (words) {
      const VkShaderModuleCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = size,
         .pCode = words,
      };
      VkResult result = screen->vk.CreateShaderModule(screen->dev, &ci, NULL,
                                                      &module);
      if (result != VK_SUCCESS) {
         mesa_loge("SZ: vkCreateShaderModule(vertex) failed: %d", result);
         module = VK_NULL_HANDLE;
      }
   }

   sz_spv_module_destroy(m);
   return module;
}

/* Fragment shader: writes one constant colour.
 *
 * A SPIR-V validation error here is worth more than a passing render test: it
 * means the emitter produces something no driver will accept, and every later
 * shader would inherit the bug. */
static VkShaderModule
sz_pipeline_bootstrap_fs(struct sz_screen *screen)
{
   struct sz_spv_module *m = sz_spv_module_create();
   if (!m)
      return VK_NULL_HANDLE;

   sz_spv_capability(m, SpvCapabilityShader);
   sz_spv_memory_model(m, SpvAddressingModelLogical, SpvMemoryModelGLSL450);

   const uint32_t f32 = sz_spv_type_float32(m);
   const uint32_t v4 = sz_spv_type_vec(m, f32, 4);

   uint32_t components[4];
   for (unsigned i = 0; i < 4; i++)
      components[i] = sz_spv_const_float32(m, sz_pipeline_color[i]);

   const uint32_t color = sz_spv_const_composite(m, v4, components, 4);

   const uint32_t out =
      sz_spv_variable(m, sz_spv_type_ptr(m, SpvStorageClassOutput, v4),
                      SpvStorageClassOutput);
   sz_spv_decorate(m, out, SpvDecorationLocation, 0);

   const uint32_t interface[] = { out };
   const uint32_t main_fn = sz_spv_id(m);
   sz_spv_entry_point(m, SpvExecutionModelFragment, main_fn, "main", interface,
                      ARRAY_SIZE(interface));
   sz_spv_execution_mode(m, main_fn, SpvExecutionModeOriginUpperLeft);

   sz_spv_function(m, main_fn, sz_spv_type_void(m),
                   sz_spv_type_function(m, sz_spv_type_void(m), NULL, 0));
   sz_spv_label(m);
   sz_spv_store(m, out, color);
   sz_spv_return(m);
   sz_spv_function_end(m);

   size_t size = 0;
   const uint32_t *words = sz_spv_finalize(m, &size);
   sz_pipeline_dump(words, size, "fs");

   VkShaderModule module = VK_NULL_HANDLE;
   if (words) {
      const VkShaderModuleCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
         .codeSize = size,
         .pCode = words,
      };
      VkResult result = screen->vk.CreateShaderModule(screen->dev, &ci, NULL,
                                                       &module);
      if (result != VK_SUCCESS) {
         mesa_loge("SZ: vkCreateShaderModule(fragment) failed: %d", result);
         module = VK_NULL_HANDLE;
      }
   }

   sz_spv_module_destroy(m);
   return module;
}

/* ------------------------------------------------------------------------- */
/* Pipeline                                                                   */
/* ------------------------------------------------------------------------- */

static void
sz_pipeline_reset(struct sz_screen *screen, struct sz_gfx_pipeline *p)
{
   if (p->pipeline != VK_NULL_HANDLE)
      screen->vk.DestroyPipeline(screen->dev, p->pipeline, NULL);
   if (p->layout != VK_NULL_HANDLE)
      screen->vk.DestroyPipelineLayout(screen->dev, p->layout, NULL);
   if (p->vs != VK_NULL_HANDLE)
      screen->vk.DestroyShaderModule(screen->dev, p->vs, NULL);
   if (p->fs != VK_NULL_HANDLE)
      screen->vk.DestroyShaderModule(screen->dev, p->fs, NULL);
   memset(p, 0, sizeof(*p));
}

/* Builds a pipeline for the current framebuffer formats and vertex layout.
 *
 * `elements` describes the attribute layout; the bootstrap shader only reads
 * location 0, but the binding description has to be right for every attribute
 * anyway, because a wrong stride makes every vertex wrong and Vulkan will not
 * complain. */
static bool
sz_pipeline_build(struct sz_context *ctx, struct sz_gfx_pipeline *p,
                  VkPrimitiveTopology topology, unsigned element_count,
                  unsigned stride)
{
   struct sz_screen *screen = ctx->screen;
   struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   p->vs = sz_pipeline_bootstrap_vs(screen);
   p->fs = sz_pipeline_bootstrap_fs(screen);
   if (p->vs == VK_NULL_HANDLE || p->fs == VK_NULL_HANDLE)
      goto fail;

   const VkPipelineLayoutCreateInfo layout_ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkResult result = screen->vk.CreatePipelineLayout(screen->dev, &layout_ci,
                                                      NULL, &p->layout);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreatePipelineLayout failed: %d", result);
      goto fail;
   }

   const VkPipelineShaderStageCreateInfo stages[] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = p->vs,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = p->fs,
         .pName = "main",
      },
   };

   /* One binding, because the bootstrap shader reads one attribute. Attributes
    * beyond location 0 would need shader input variables that do not exist yet,
    * and declaring them would make the pipeline fail validation for an interface
    * the shader never uses. */
   const VkVertexInputBindingDescription binding = {
      .binding = 0,
      .stride = stride,
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };

   const struct sz_state *ve = ctx->state.vertex_elements;

   VkVertexInputAttributeDescription attributes[PIPE_MAX_ATTRIBS];
   unsigned attribute_count =
      ve ? MIN2(MIN2(element_count, ve->u.ve.count), 1) : 0;

   for (unsigned i = 0; i < attribute_count; i++) {
      const struct pipe_vertex_element *elem = &ve->u.ve.elements[i];
      attributes[i] = (VkVertexInputAttributeDescription) {
         .location = i,
         .binding = 0,
         .format = vk_format_from_pipe_format(elem->src_format),
         .offset = elem->src_offset,
      };
   }

   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = attribute_count ? 1 : 0,
      .pVertexBindingDescriptions = attribute_count ? &binding : NULL,
      .vertexAttributeDescriptionCount = attribute_count,
      .pVertexAttributeDescriptions = attribute_count ? attributes : NULL,
   };

   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topology,
      .primitiveRestartEnable = VK_FALSE,
   };

   /* Viewport and scissor are dynamic: GL changes them per draw and per viewport
    * index, and baking them into the pipeline would rebuild it constantly. */
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .scissorCount = 1,
   };

   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .depthBiasEnable = VK_FALSE,
      .lineWidth = 1.0f,
   };

   const VkSampleCountFlagBits samples =
      sz_vk_sample_count(fb->nr_cbufs ? fb->cbufs[0].nr_samples : 0);

   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = samples,
      .sampleShadingEnable = VK_FALSE,
   };

   const bool has_depth = fb->zsbuf.texture != NULL;

   const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_FALSE,
      .depthWriteEnable = has_depth,
      .depthCompareOp = VK_COMPARE_OP_ALWAYS,
      .stencilTestEnable = VK_FALSE,
   };

   VkPipelineColorBlendAttachmentState blend_attachments[PIPE_MAX_COLOR_BUFS] = { 0 };
   const unsigned color_count = MIN2(fb->nr_cbufs, PIPE_MAX_COLOR_BUFS);

   for (unsigned i = 0; i < color_count; i++) {
      blend_attachments[i] = (VkPipelineColorBlendAttachmentState) {
         .blendEnable = VK_FALSE,
         .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
      };
   }

   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = color_count,
      .pAttachments = color_count ? blend_attachments : NULL,
   };

   const VkDynamicState dynamic[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
   };

   const VkPipelineDynamicStateCreateInfo dynamic_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = ARRAY_SIZE(dynamic),
      .pDynamicStates = dynamic,
   };

   /* Dynamic rendering: the attachment formats are part of the pipeline, not of
    * a VkRenderPass. This is the whole reason the driver does not need a render
    * pass cache. */
   VkFormat color_formats[PIPE_MAX_COLOR_BUFS] = { 0 };
   for (unsigned i = 0; i < color_count; i++)
      color_formats[i] = vk_format_from_pipe_format(fb->cbufs[i].format);

   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = color_count,
      .pColorAttachmentFormats = color_count ? color_formats : NULL,
      .depthAttachmentFormat =
         has_depth ? vk_format_from_pipe_format(fb->zsbuf.format)
                   : VK_FORMAT_UNDEFINED,
      .stencilAttachmentFormat =
         has_depth ? vk_format_from_pipe_format(fb->zsbuf.format)
                   : VK_FORMAT_UNDEFINED,
   };

   const VkGraphicsPipelineCreateInfo pipeline_ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = ARRAY_SIZE(stages),
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &blend,
      .pDynamicState = &dynamic_state,
      .layout = p->layout,
   };

   result = screen->vk.CreateGraphicsPipelines(screen->dev, VK_NULL_HANDLE, 1,
                                                &pipeline_ci, NULL, &p->pipeline);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateGraphicsPipelines failed: %d", result);
      goto fail;
   }

   p->key.topology = topology;
   p->key.vertex_element_count = element_count;
   p->key.stride = stride;
   p->key.vertex_format = attribute_count ? attributes[0].format : VK_FORMAT_UNDEFINED;
   p->key.samples = samples;
   p->key.color_count = color_count;
   for (unsigned i = 0; i < color_count; i++)
      p->key.color[i] = color_formats[i];
   p->key.depth = rendering.depthAttachmentFormat;

   return true;

fail:
   sz_pipeline_reset(screen, p);
   return false;
}

static bool
sz_pipeline_key_matches(struct sz_gfx_pipeline *p, struct sz_context *ctx,
                        VkPrimitiveTopology topology, unsigned element_count,
                        unsigned stride)
{
   struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   if (p->pipeline == VK_NULL_HANDLE)
      return false;
   if (p->key.topology != topology || p->key.vertex_element_count != element_count ||
       p->key.stride != stride)
      return false;

   const struct sz_state *ve = ctx->state.vertex_elements;
   const VkFormat vertex_format =
      ve && ve->u.ve.count
         ? vk_format_from_pipe_format(ve->u.ve.elements[0].src_format)
         : VK_FORMAT_UNDEFINED;
   if (p->key.vertex_format != vertex_format)
      return false;

   const VkSampleCountFlagBits samples =
      sz_vk_sample_count(fb->nr_cbufs ? fb->cbufs[0].nr_samples : 0);
   if (p->key.samples != samples)
      return false;

   if (p->key.color_count != MIN2(fb->nr_cbufs, PIPE_MAX_COLOR_BUFS))
      return false;

   for (unsigned i = 0; i < p->key.color_count; i++) {
      if (p->key.color[i] != vk_format_from_pipe_format(fb->cbufs[i].format))
         return false;
   }

   const VkFormat depth = fb->zsbuf.texture
                             ? vk_format_from_pipe_format(fb->zsbuf.format)
                             : VK_FORMAT_UNDEFINED;
   return p->key.depth == depth;
}

struct sz_gfx_pipeline *
sz_pipeline_get(struct sz_context *ctx, VkPrimitiveTopology topology,
                unsigned element_count, unsigned stride)
{
   if (!ctx->pipeline) {
      ctx->pipeline = CALLOC_STRUCT(sz_gfx_pipeline);
      if (!ctx->pipeline)
         return NULL;
   }

   if (sz_pipeline_key_matches(ctx->pipeline, ctx, topology, element_count,
                               stride))
      return ctx->pipeline;

   /* Rebuilding destroys the old pipeline immediately, including when the batch
    * being recorded already references it. Vulkan allows this: a pipeline is
    * consumed at record time and the driver defers the free until the work that
    * uses it retires, so no reference count of our own is needed. */
   sz_pipeline_reset(ctx->screen, ctx->pipeline);

   if (!sz_pipeline_build(ctx, ctx->pipeline, topology, element_count, stride))
      return NULL;

   return ctx->pipeline;
}

void
sz_pipeline_finish(struct sz_context *ctx)
{
   if (!ctx->pipeline)
      return;

   sz_pipeline_reset(ctx->screen, ctx->pipeline);
   free(ctx->pipeline);
   ctx->pipeline = NULL;
}
