/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_pipeline.h"

#include "sz_batch.h"
#include "sz_debug.h"
#include "sz_format.h"
#include "sz_spirv.h"

#include "vk_format.h"

#include "util/format/u_format.h"
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
   const uint32_t value = sz_spv_load(m, v4, input);
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
sz_pipeline_destroy_objects(struct sz_screen *screen, struct sz_gfx_pipeline *p)
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

void
sz_pipeline_reclaim(struct sz_context *ctx)
{
   struct sz_queue_state *qs = &ctx->screen->batch_state;

   for (unsigned i = 0; i < ctx->num_retired; ) {
      if (ctx->retired[i].retire_value == SZ_RETIRE_PENDING ||
          ctx->retired[i].retire_value > qs->completed_value) {
         i++;
         continue;
      }

      sz_pipeline_destroy_objects(ctx->screen, ctx->retired[i].pipeline);
      free(ctx->retired[i].pipeline);
      ctx->num_retired--;
      ctx->retired[i] = ctx->retired[ctx->num_retired];
   }
}

/* A pipeline that is still bound in the command buffer being recorded may not be
 * destroyed: Vulkan keeps it alive only until the recording that references it
 * ends, and the validation layer reports "pipeline was destroyed" for the
 * command buffer otherwise -- RADV turns that into a device loss. Retiring it
 * until its submission has retired is the whole fix. */
static void
sz_pipeline_retire(struct sz_context *ctx, struct sz_gfx_pipeline *p)
{
   struct sz_queue_state *qs = &ctx->screen->batch_state;

   if (ctx->num_retired == SZ_MAX_RETIRED_PIPELINES) {
      /* Nothing else has completed, so waiting is the only way to make room.
       * The queue is idle afterwards, which reclaims every entry. */
      sz_queue_wait_idle(qs);
      sz_pipeline_reclaim(ctx);
   }

   if (ctx->num_retired < SZ_MAX_RETIRED_PIPELINES) {
      struct sz_gfx_pipeline *keep = MALLOC_STRUCT(sz_gfx_pipeline);
      if (keep) {
         *keep = *p;
         memset(p, 0, sizeof(*p));
         ctx->retired[ctx->num_retired].pipeline = keep;
         /* Recording: the batch being recorded is the one that has the pipeline
          * bound, and its timeline value is only handed out on submission, so
          * the entry stays pending until sz_pipeline_note_submit() runs. Only
          * an already submitted stream can use next_value+1 as a bound. */
         ctx->retired[ctx->num_retired].retire_value =
            ctx->batch ? SZ_RETIRE_PENDING : qs->next_value + 1;
         ctx->num_retired++;
         return;
      }
   }

   sz_pipeline_destroy_objects(ctx->screen, p);
   memset(p, 0, sizeof(*p));
}

void
sz_pipeline_note_submit(struct sz_context *ctx)
{
   struct sz_queue_state *qs = &ctx->screen->batch_state;

   /* The submission that just went out is the one that could still have been
    * referencing the pipelines retired while it was being recorded, and
    * next_value is now its value, so "one past it" is the first value that
    * proves the command buffer that bound them has finished. */
   for (unsigned i = 0; i < ctx->num_retired; i++) {
      if (ctx->retired[i].retire_value == SZ_RETIRE_PENDING)
         ctx->retired[i].retire_value = qs->next_value + 1;
   }
}

/* Returns a module for `stage`: the translated shader when one is bound, the
 * bootstrap shader otherwise. The bootstrap pair stays in use for the driver's
 * own selftest, which draws without a shader bound. */
static VkShaderModule
sz_pipeline_module(struct sz_context *ctx, mesa_shader_stage stage)
{
   struct sz_screen *screen = ctx->screen;
   const struct sz_state *shader = ctx->state.shaders[stage];

   if (!shader || !shader->u.shader.spirv) {
      return stage == MESA_SHADER_VERTEX ? sz_pipeline_bootstrap_vs(screen)
                                         : sz_pipeline_bootstrap_fs(screen);
   }

   VkShaderModule module = VK_NULL_HANDLE;
   const VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = shader->u.shader.spirv_words * sizeof(uint32_t),
      .pCode = shader->u.shader.spirv,
   };
   VkResult result = screen->vk.CreateShaderModule(screen->dev, &ci, NULL,
                                                  &module);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateShaderModule(%s) failed: %d",
                mesa_shader_stage_name(stage), result);
      return VK_NULL_HANDLE;
   }

   return module;
}

/* ── pipe state -> Vulkan state ──────────────────────────────────────────── */

/* The recorded rasterizer/blend/depth-stencil objects, or NULL when the state
 * tracker has not bound one. A NULL means "GL defaults", which is what
 * cso_context binds at context creation anyway; treating it as defaults rather
 * than as "no state" keeps a draw from being silently discarded. */
static const struct pipe_depth_stencil_alpha_state *
sz_dsa(const struct sz_context *ctx)
{
   const struct sz_state *state = ctx->state.depth_stencil_alpha;
   return state ? &state->u.dsa : NULL;
}

static const struct pipe_rasterizer_state *
sz_raster(const struct sz_context *ctx)
{
   const struct sz_state *state = ctx->state.rasterizer;
   return state ? &state->u.rasterizer : NULL;
}

static const struct pipe_blend_state *
sz_blend(const struct sz_context *ctx)
{
   const struct sz_state *state = ctx->state.blend;
   return state ? &state->u.blend : NULL;
}

static VkCompareOp
sz_compare_op(unsigned func)
{
   switch (func) {
   case PIPE_FUNC_NEVER:
      return VK_COMPARE_OP_NEVER;
   case PIPE_FUNC_LESS:
      return VK_COMPARE_OP_LESS;
   case PIPE_FUNC_EQUAL:
      return VK_COMPARE_OP_EQUAL;
   case PIPE_FUNC_LEQUAL:
      return VK_COMPARE_OP_LESS_OR_EQUAL;
   case PIPE_FUNC_GREATER:
      return VK_COMPARE_OP_GREATER;
   case PIPE_FUNC_NOTEQUAL:
      return VK_COMPARE_OP_NOT_EQUAL;
   case PIPE_FUNC_GEQUAL:
      return VK_COMPARE_OP_GREATER_OR_EQUAL;
   case PIPE_FUNC_ALWAYS:
   default:
      return VK_COMPARE_OP_ALWAYS;
   }
}

static VkBlendFactor
sz_blend_factor(unsigned factor)
{
   switch (factor) {
   case PIPE_BLENDFACTOR_ONE:
      return VK_BLEND_FACTOR_ONE;
   case PIPE_BLENDFACTOR_SRC_COLOR:
      return VK_BLEND_FACTOR_SRC_COLOR;
   case PIPE_BLENDFACTOR_SRC_ALPHA:
      return VK_BLEND_FACTOR_SRC_ALPHA;
   case PIPE_BLENDFACTOR_DST_ALPHA:
      return VK_BLEND_FACTOR_DST_ALPHA;
   case PIPE_BLENDFACTOR_DST_COLOR:
      return VK_BLEND_FACTOR_DST_COLOR;
   case PIPE_BLENDFACTOR_SRC_ALPHA_SATURATE:
      return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
   case PIPE_BLENDFACTOR_CONST_COLOR:
      return VK_BLEND_FACTOR_CONSTANT_COLOR;
   case PIPE_BLENDFACTOR_CONST_ALPHA:
      return VK_BLEND_FACTOR_CONSTANT_ALPHA;
   case PIPE_BLENDFACTOR_SRC1_COLOR:
      return VK_BLEND_FACTOR_SRC1_COLOR;
   case PIPE_BLENDFACTOR_SRC1_ALPHA:
      return VK_BLEND_FACTOR_SRC1_ALPHA;
   case PIPE_BLENDFACTOR_ZERO:
      return VK_BLEND_FACTOR_ZERO;
   case PIPE_BLENDFACTOR_INV_SRC_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
   case PIPE_BLENDFACTOR_INV_SRC_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
   case PIPE_BLENDFACTOR_INV_DST_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
   case PIPE_BLENDFACTOR_INV_DST_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
   case PIPE_BLENDFACTOR_INV_CONST_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
   case PIPE_BLENDFACTOR_INV_CONST_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
   case PIPE_BLENDFACTOR_INV_SRC1_COLOR:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
   case PIPE_BLENDFACTOR_INV_SRC1_ALPHA:
      return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
   default:
      return VK_BLEND_FACTOR_ONE;
   }
}

static VkBlendOp
sz_blend_op(unsigned func)
{
   switch (func) {
   case PIPE_BLEND_SUBTRACT:
      return VK_BLEND_OP_SUBTRACT;
   case PIPE_BLEND_REVERSE_SUBTRACT:
      return VK_BLEND_OP_REVERSE_SUBTRACT;
   case PIPE_BLEND_MIN:
      return VK_BLEND_OP_MIN;
   case PIPE_BLEND_MAX:
      return VK_BLEND_OP_MAX;
   case PIPE_BLEND_ADD:
   default:
      return VK_BLEND_OP_ADD;
   }
}

/* Gallium's logic op enum is ordered like the GL one, Vulkan's is not, so the
 * two cannot be cast into each other. */
static VkLogicOp
sz_logic_op(unsigned op)
{
   switch (op) {
   case PIPE_LOGICOP_CLEAR:
      return VK_LOGIC_OP_CLEAR;
   case PIPE_LOGICOP_NOR:
      return VK_LOGIC_OP_NOR;
   case PIPE_LOGICOP_AND_INVERTED:
      return VK_LOGIC_OP_AND_INVERTED;
   case PIPE_LOGICOP_COPY_INVERTED:
      return VK_LOGIC_OP_COPY_INVERTED;
   case PIPE_LOGICOP_AND_REVERSE:
      return VK_LOGIC_OP_AND_REVERSE;
   case PIPE_LOGICOP_INVERT:
      return VK_LOGIC_OP_INVERT;
   case PIPE_LOGICOP_XOR:
      return VK_LOGIC_OP_XOR;
   case PIPE_LOGICOP_NAND:
      return VK_LOGIC_OP_NAND;
   case PIPE_LOGICOP_AND:
      return VK_LOGIC_OP_AND;
   case PIPE_LOGICOP_EQUIV:
      return VK_LOGIC_OP_EQUIVALENT;
   case PIPE_LOGICOP_NOOP:
      return VK_LOGIC_OP_NO_OP;
   case PIPE_LOGICOP_OR_INVERTED:
      return VK_LOGIC_OP_OR_INVERTED;
   case PIPE_LOGICOP_COPY:
      return VK_LOGIC_OP_COPY;
   case PIPE_LOGICOP_OR_REVERSE:
      return VK_LOGIC_OP_OR_REVERSE;
   case PIPE_LOGICOP_OR:
      return VK_LOGIC_OP_OR;
   case PIPE_LOGICOP_SET:
   default:
      return VK_LOGIC_OP_SET;
   }
}

static VkPolygonMode
sz_polygon_mode(unsigned mode)
{
   switch (mode) {
   case PIPE_POLYGON_MODE_LINE:
      return VK_POLYGON_MODE_LINE;
   case PIPE_POLYGON_MODE_POINT:
      return VK_POLYGON_MODE_POINT;
   case PIPE_POLYGON_MODE_FILL:
   default:
      return VK_POLYGON_MODE_FILL;
   }
}

static VkColorComponentFlags
sz_color_mask(unsigned mask)
{
   VkColorComponentFlags out = 0;
   if (mask & PIPE_MASK_R)
      out |= VK_COLOR_COMPONENT_R_BIT;
   if (mask & PIPE_MASK_G)
      out |= VK_COLOR_COMPONENT_G_BIT;
   if (mask & PIPE_MASK_B)
      out |= VK_COLOR_COMPONENT_B_BIT;
   if (mask & PIPE_MASK_A)
      out |= VK_COLOR_COMPONENT_A_BIT;
   return out;
}

/* Fills in the vertex fetch layout the current state needs.
 *
 * Two rules make this less obvious than "one binding per draw":
 *
 *  - An attribute reads from the buffer the frontend bound for it
 *    (pipe_vertex_element::vertex_buffer_index) at that element's own stride. A
 *    fixed-function draw of a client array set gets one buffer per enabled
 *    array, so assuming a single shared stride reads every attribute from the
 *    wrong row and the draw silently produces nothing.
 *  - The location of an attribute is its *rank* in the vertex shader's read set,
 *    not its gallium slot: st_atom_array orders the element array that way, and
 *    sz_varying_location() in the translator assigns locations from the same
 *    order. A location derived from VERT_ATTRIB_GENERIC0 instead would only be
 *    right for shaders that read POS and nothing else.
 */
static void
sz_pipeline_vertex_input(struct sz_context *ctx,
                         VkVertexInputBindingDescription *bindings,
                         unsigned *out_binding_count,
                         VkVertexInputAttributeDescription *attributes,
                         unsigned *out_attribute_count)
{
   const struct sz_state *ve = ctx->state.vertex_elements;
   const unsigned element_count =
      ve ? MIN2(ve->u.ve.count, PIPE_MAX_ATTRIBS) : 0;

   unsigned binding_count = 0;
   for (unsigned i = 0; i < element_count; i++) {
      const struct pipe_vertex_element *elem = &ve->u.ve.elements[i];
      const unsigned vb = elem->vertex_buffer_index;

      unsigned b = 0;
      for (; b < binding_count; b++) {
         if (bindings[b].binding == vb)
            break;
      }
      if (b == binding_count) {
         /* The stride lives on the element, not on the buffer: every element
          * that reads one buffer is required to report the same stride, so the
          * first one seen is the binding's stride. */
         bindings[binding_count++] = (VkVertexInputBindingDescription) {
            .binding = vb,
            .stride = elem->src_stride,
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
         };
      }

      attributes[i] = (VkVertexInputAttributeDescription) {
         .location = i,
         .binding = vb,
         .format = vk_format_from_pipe_format(elem->src_format),
         .offset = elem->src_offset,
      };
   }

   *out_binding_count = binding_count;
   *out_attribute_count = element_count;
}

/* Builds a pipeline for the current framebuffer formats and vertex layout. */
static bool
sz_pipeline_build(struct sz_context *ctx, struct sz_gfx_pipeline *p,
                  VkPrimitiveTopology topology)
{
   struct sz_screen *screen = ctx->screen;
   struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   p->vs = sz_pipeline_module(ctx, MESA_SHADER_VERTEX);
   p->fs = sz_pipeline_module(ctx, MESA_SHADER_FRAGMENT);
   if (p->vs == VK_NULL_HANDLE || p->fs == VK_NULL_HANDLE)
      goto fail;

   /* ADR-008: the pipeline layout is the push-descriptor layout, so a draw
    * never allocates a set. Every stage's bindings are declared even when a
    * shader uses none: a layout with extra bindings is legal, a push against an
    * undeclared binding is not. */
   const VkDescriptorSetLayout set_layouts[] = {
      screen->descriptor_set_layout,
   };

   /* No VkPipelineLayoutCreateFlagBits is needed: the push-descriptor bit lives
    * on the descriptor set layout, and Vulkan 1.4 removed the pipeline layout
    * requirement that used to exist in VK_KHR_push_descriptor. */
   const VkPipelineLayoutCreateInfo layout_ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = ARRAY_SIZE(set_layouts),
      .pSetLayouts = set_layouts,
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

   /* Every element gets a description: a shader reading a location the pipeline
    * does not declare gets an undefined value with no error. */
   VkVertexInputBindingDescription bindings[PIPE_MAX_ATTRIBS];
   VkVertexInputAttributeDescription attributes[PIPE_MAX_ATTRIBS];
   unsigned binding_count = 0, attribute_count = 0;
   sz_pipeline_vertex_input(ctx, bindings, &binding_count, attributes,
                            &attribute_count);

   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = binding_count,
      .pVertexBindingDescriptions = binding_count ? bindings : NULL,
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

   const struct pipe_rasterizer_state *rast_state = sz_raster(ctx);

   /* sz_lower_nir() always translates GL clip space to Vulkan's, so a context
    * that asked for GL_ZERO_TO_ONE depth through glClipControl needs a shader
    * variant the driver does not build yet. Say so rather than render a depth
    * range that silently starts at 0.5. */
   if (rast_state && rast_state->clip_halfz)
      sz_warn_once("SZ: glClipControl(GL_ZERO_TO_ONE) is not implemented; "
                   "depth will span [0.5, 1]");
   const struct pipe_depth_stencil_alpha_state *dsa = sz_dsa(ctx);
   const struct pipe_blend_state *blend_state = sz_blend(ctx);

   /* Gallium's cull flags are the GL ones, and so are Vulkan's. */
   VkCullModeFlags cull_mode = VK_CULL_MODE_NONE;
   if (rast_state) {
      if (rast_state->cull_face & PIPE_FACE_FRONT)
         cull_mode |= VK_CULL_MODE_FRONT_BIT;
      if (rast_state->cull_face & PIPE_FACE_BACK)
         cull_mode |= VK_CULL_MODE_BACK_BIT;
   }

   /* The viewport keeps a positive height (see sz_draw_viewport), so neither NDC
    * axis is mirrored and GL's winding order carries over unchanged. */
   const VkFrontFace front_face =
      (!rast_state || rast_state->front_ccw) ? VK_FRONT_FACE_COUNTER_CLOCKWISE
                                             : VK_FRONT_FACE_CLOCKWISE;

   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .depthClampEnable = rast_state ? rast_state->depth_clamp : VK_FALSE,
      .rasterizerDiscardEnable = VK_FALSE,
      .polygonMode = rast_state ? sz_polygon_mode(rast_state->fill_front)
                                : VK_POLYGON_MODE_FILL,
      .cullMode = cull_mode,
      .frontFace = front_face,
      .depthBiasEnable = rast_state ? rast_state->offset_tri : VK_FALSE,
      .depthBiasConstantFactor = rast_state ? rast_state->offset_units : 0.0f,
      .depthBiasClamp = rast_state ? rast_state->offset_clamp : 0.0f,
      .depthBiasSlopeFactor = rast_state ? rast_state->offset_scale : 0.0f,
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

   /* GL's depth test is off by default; a draw that enables it without a depth
    * attachment is a state-tracker bug, not something to honour.
    *
    * Depth writes without a depth test are what a "renders but looks wrong"
    * picture is made of: without the compare, later primitives overwrite earlier
    * ones regardless of distance, so a scene comes out with its back faces on
    * top. Both bits come from the same Gallium state. */
   const bool depth_enabled = has_depth && dsa && dsa->depth_enabled;
   const bool depth_write = depth_enabled && dsa->depth_writemask;

   /* Stencil is not translated yet. Saying so once is better than silently
    * dropping it, because a stencil-masked effect renders as if the mask were
    * always zero. */
   if (dsa && (dsa->stencil[0].enabled || dsa->stencil[1].enabled))
      sz_warn_once("SZ: stencil testing is not implemented; ignoring it");

   const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = depth_enabled,
      .depthWriteEnable = depth_write,
      .depthCompareOp = dsa ? sz_compare_op(dsa->depth_func)
                            : VK_COMPARE_OP_ALWAYS,
      .stencilTestEnable = VK_FALSE,
   };

   VkPipelineColorBlendAttachmentState blend_attachments[PIPE_MAX_COLOR_BUFS] = { 0 };
   const unsigned color_count = MIN2(fb->nr_cbufs, PIPE_MAX_COLOR_BUFS);

   /* Logic op and blending are mutually exclusive in Vulkan, and GL says the
    * same: with GL_COLOR_LOGIC_OP enabled the blend equation is bypassed. */
   const bool logicop_enable = blend_state && blend_state->logicop_enable;

   for (unsigned i = 0; i < color_count; i++) {
      const struct pipe_rt_blend_state *rt = NULL;
      if (blend_state) {
         rt = &blend_state->rt[blend_state->independent_blend_enable ? i : 0];
      }

      const bool blend_enable = rt && rt->blend_enable && !logicop_enable;

      blend_attachments[i] = (VkPipelineColorBlendAttachmentState) {
         .blendEnable = blend_enable,
         .srcColorBlendFactor = sz_blend_factor(rt ? rt->rgb_src_factor
                                                   : PIPE_BLENDFACTOR_ONE),
         .dstColorBlendFactor = sz_blend_factor(rt ? rt->rgb_dst_factor
                                                   : PIPE_BLENDFACTOR_ZERO),
         .colorBlendOp = sz_blend_op(rt ? rt->rgb_func : PIPE_BLEND_ADD),
         .srcAlphaBlendFactor = sz_blend_factor(rt ? rt->alpha_src_factor
                                                   : PIPE_BLENDFACTOR_ONE),
         .dstAlphaBlendFactor = sz_blend_factor(rt ? rt->alpha_dst_factor
                                                   : PIPE_BLENDFACTOR_ZERO),
         .alphaBlendOp = sz_blend_op(rt ? rt->alpha_func : PIPE_BLEND_ADD),
         .colorWriteMask = sz_color_mask(rt ? rt->colormask
                                            : (PIPE_MASK_R | PIPE_MASK_G |
                                               PIPE_MASK_B | PIPE_MASK_A)),
      };
   }

   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = logicop_enable,
      .logicOp = sz_logic_op(blend_state ? blend_state->logicop_func
                                         : PIPE_LOGICOP_COPY),
      .attachmentCount = color_count,
      .pAttachments = color_count ? blend_attachments : NULL,
      /* Set per draw through the dynamic state below: glBlendColor is GL state
       * that changes without a pipeline rebind. */
   };

   const VkDynamicState dynamic[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
      VK_DYNAMIC_STATE_BLEND_CONSTANTS,
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
      /* A depth format without a stencil aspect has to report UNDEFINED here:
       * naming a stencil-less format makes the pipeline invalid rather than
       * stencil-less. */
      .stencilAttachmentFormat =
         has_depth && util_format_has_stencil(util_format_description(fb->zsbuf.format))
            ? vk_format_from_pipe_format(fb->zsbuf.format)
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
   p->key.vertex_binding_count = binding_count;
   memcpy(p->key.vertex_bindings, bindings,
          binding_count * sizeof(bindings[0]));
   p->key.vertex_attribute_count = attribute_count;
   memcpy(p->key.vertex_attributes, attributes,
          attribute_count * sizeof(attributes[0]));
   p->key.samples = samples;
   p->key.blend = ctx->state.blend;
   p->key.rasterizer = ctx->state.rasterizer;
   p->key.dsa = ctx->state.depth_stencil_alpha;
   p->key.vs = ctx->state.shaders[MESA_SHADER_VERTEX];
   p->key.fs = ctx->state.shaders[MESA_SHADER_FRAGMENT];
   p->key.color_count = color_count;
   for (unsigned i = 0; i < color_count; i++)
      p->key.color[i] = color_formats[i];
   p->key.depth = rendering.depthAttachmentFormat;

   return true;

fail:
   sz_pipeline_destroy_objects(screen, p);
   return false;
}

static bool
sz_pipeline_key_matches(struct sz_gfx_pipeline *p, struct sz_context *ctx,
                        VkPrimitiveTopology topology)
{
   struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   if (p->pipeline == VK_NULL_HANDLE)
      return false;
   if (p->key.topology != topology)
      return false;

   /* Depth, cull, front face and blend are baked into the pipeline object, so a
    * change to any of them needs a new one. The state tracker allocates a fresh
    * object per distinct state, which makes identity comparison enough. */
   if (p->key.blend != ctx->state.blend ||
       p->key.rasterizer != ctx->state.rasterizer ||
       p->key.dsa != ctx->state.depth_stencil_alpha)
      return false;

   if (p->key.vs != ctx->state.shaders[MESA_SHADER_VERTEX] ||
       p->key.fs != ctx->state.shaders[MESA_SHADER_FRAGMENT])
      return false;

   /* The whole vertex fetch layout, not just the first attribute: a draw whose
    * second buffer has a different stride or format needs a different pipeline,
    * and reusing the old one reads the vertices as the wrong type without any
    * Vulkan error. */
   VkVertexInputBindingDescription bindings[PIPE_MAX_ATTRIBS];
   VkVertexInputAttributeDescription attributes[PIPE_MAX_ATTRIBS];
   unsigned binding_count = 0, attribute_count = 0;
   sz_pipeline_vertex_input(ctx, bindings, &binding_count, attributes,
                            &attribute_count);

   if (p->key.vertex_binding_count != binding_count ||
       p->key.vertex_attribute_count != attribute_count)
      return false;
   if (memcmp(p->key.vertex_bindings, bindings,
              binding_count * sizeof(bindings[0])) != 0)
      return false;
   if (memcmp(p->key.vertex_attributes, attributes,
              attribute_count * sizeof(attributes[0])) != 0)
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
sz_pipeline_get(struct sz_context *ctx, VkPrimitiveTopology topology)
{
   if (!ctx->pipeline) {
      ctx->pipeline = CALLOC_STRUCT(sz_gfx_pipeline);
      if (!ctx->pipeline)
         return NULL;
   }

   if (sz_pipeline_key_matches(ctx->pipeline, ctx, topology))
      return ctx->pipeline;

   sz_pipeline_reclaim(ctx);
   sz_pipeline_retire(ctx, ctx->pipeline);

   if (!sz_pipeline_build(ctx, ctx->pipeline, topology))
      return NULL;

   return ctx->pipeline;
}

void
sz_pipeline_finish(struct sz_context *ctx)
{
   /* Everything the context retires is dropped here; the context is going away,
    * so waiting for the queue is both allowed and required. */
   for (unsigned i = 0; i < ctx->num_retired; i++) {
      sz_pipeline_destroy_objects(ctx->screen, ctx->retired[i].pipeline);
      free(ctx->retired[i].pipeline);
   }
   ctx->num_retired = 0;

   if (!ctx->pipeline)
      return;

   sz_pipeline_destroy_objects(ctx->screen, ctx->pipeline);
   free(ctx->pipeline);
   ctx->pipeline = NULL;
}
