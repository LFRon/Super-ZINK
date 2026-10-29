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
sz_pipeline_bootstrap_vs(struct sz_screen *screen, bool with_input)
{
   /* Created once: the self-tests draw without a program bound, and a module per
    * such draw would be pure churn. */
   VkShaderModule *cached =
      with_input ? &screen->bootstrap_vs : &screen->bootstrap_vs_empty;
   if (*cached != VK_NULL_HANDLE)
      return *cached;

   struct sz_spv_module *m = sz_spv_module_create();
   if (!m)
      return VK_NULL_HANDLE;

   sz_spv_capability(m, SpvCapabilityShader);
   sz_spv_memory_model(m, SpvAddressingModelLogical, SpvMemoryModelGLSL450);

   const uint32_t f32 = sz_spv_type_float32(m);
   const uint32_t v4 = sz_spv_type_vec(m, f32, 4);

   uint32_t input = 0;
   if (with_input) {
      input = sz_spv_variable(m,
                              sz_spv_type_ptr(m, SpvStorageClassInput, v4),
                              SpvStorageClassInput);
      sz_spv_decorate(m, input, SpvDecorationLocation, 0);
   }
   const uint32_t position =
      sz_spv_variable(m, sz_spv_type_ptr(m, SpvStorageClassOutput, v4),
                      SpvStorageClassOutput);

   sz_spv_decorate_builtin(m, position, SpvBuiltInPosition);

   const uint32_t interface[] = { input, position };
   const uint32_t main_fn = sz_spv_id(m);
   sz_spv_entry_point(m, SpvExecutionModelVertex, main_fn, "main", interface,
                      with_input ? ARRAY_SIZE(interface) : 1);

   sz_spv_function(m, main_fn, sz_spv_type_void(m),
                   sz_spv_type_function(m, sz_spv_type_void(m), NULL, 0));
   sz_spv_label(m);
   uint32_t value = 0;
   if (with_input) {
      value = sz_spv_load(m, v4, input);
   } else {
      /* A draw with no vertex elements still needs a position for the stages
       * after it; a geometry-only program never looks at one. */
      const uint32_t zero = sz_spv_const_float32(m, 0.0f);
      const uint32_t one = sz_spv_const_float32(m, 1.0f);
      const uint32_t pos[4] = {zero, zero, zero, one};
      value = sz_spv_const_composite(m, v4, pos, 4);
   }
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
   *cached = module;
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
   if (screen->bootstrap_fs != VK_NULL_HANDLE)
      return screen->bootstrap_fs;

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
   screen->bootstrap_fs = module;
   return module;
}

/* ------------------------------------------------------------------------- */
/* Pipeline                                                                   */
/* ------------------------------------------------------------------------- */

/* The modules a pipeline built without a bound shader names are the screen's
 * bootstrap ones, not copies: destroying them with the pipeline would leave the
 * next pipeline holding a handle that is already gone, and would free them a
 * second time when the screen is torn down. */
static bool
sz_pipeline_module_is_shared(const struct sz_screen *screen, VkShaderModule module)
{
   return module == screen->bootstrap_vs ||
          module == screen->bootstrap_vs_empty ||
          module == screen->bootstrap_fs;
}

static void
sz_pipeline_destroy_objects(struct sz_screen *screen, struct sz_gfx_pipeline *p)
{
   if (p->pipeline != VK_NULL_HANDLE)
      screen->vk.DestroyPipeline(screen->dev, p->pipeline, NULL);
   /* The pipeline layout is shared by every pipeline on the screen, so it is
    * destroyed by sz_pipeline_screen_finish(), not here. Modules and the
    * pipeline itself belong to the one pipeline that was built with them. */
   if (p->vs != VK_NULL_HANDLE && !sz_pipeline_module_is_shared(screen, p->vs))
      screen->vk.DestroyShaderModule(screen->dev, p->vs, NULL);
   if (p->tcs != VK_NULL_HANDLE && !sz_pipeline_module_is_shared(screen, p->tcs))
      screen->vk.DestroyShaderModule(screen->dev, p->tcs, NULL);
   if (p->tes != VK_NULL_HANDLE && !sz_pipeline_module_is_shared(screen, p->tes))
      screen->vk.DestroyShaderModule(screen->dev, p->tes, NULL);
   if (p->gs != VK_NULL_HANDLE && !sz_pipeline_module_is_shared(screen, p->gs))
      screen->vk.DestroyShaderModule(screen->dev, p->gs, NULL);
   if (p->fs != VK_NULL_HANDLE && !sz_pipeline_module_is_shared(screen, p->fs))
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

/* Whether the draw that created this pipeline reads any vertex attribute.
 *
 * A pipeline whose vertex input state is empty may not consume a vertex input
 * builtin: validation rejects a vertex shader that reads a location the
 * pipeline does not describe, so the bootstrap vertex shader has to come in a
 * variant without one. */
static bool
sz_pipeline_has_vertex_input(const struct sz_context *ctx)
{
   const struct sz_state *ve = ctx->state.vertex_elements;
   return ve && ve->u.ve.count > 0;
}

/* Returns a module for `stage`: the translated shader when one is bound, the
 * bootstrap shader otherwise. The bootstrap pair stays in use for the driver's
 * own selftest, which draws without a shader bound. */
VkShaderModule
sz_pipeline_module(struct sz_context *ctx, mesa_shader_stage stage)
{
   struct sz_screen *screen = ctx->screen;
   const struct sz_state *shader = ctx->state.shaders[stage];

   if (!shader || !shader->u.shader.spirv) {
      /* No fallback for geometry: the caller only asks for a GS module when one
       * is bound, so reaching here means the binding is broken. */
      if (stage == MESA_SHADER_GEOMETRY)
         return VK_NULL_HANDLE;
      return stage == MESA_SHADER_VERTEX
                ? sz_pipeline_bootstrap_vs(screen, sz_pipeline_has_vertex_input(ctx))
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

VkCompareOp
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

static VkStencilOp
sz_stencil_op(unsigned op)
{
   switch (op) {
   case PIPE_STENCIL_OP_KEEP:
      return VK_STENCIL_OP_KEEP;
   case PIPE_STENCIL_OP_ZERO:
      return VK_STENCIL_OP_ZERO;
   case PIPE_STENCIL_OP_REPLACE:
      return VK_STENCIL_OP_REPLACE;
   case PIPE_STENCIL_OP_INCR:
      return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
   case PIPE_STENCIL_OP_DECR:
      return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
   case PIPE_STENCIL_OP_INCR_WRAP:
      return VK_STENCIL_OP_INCREMENT_AND_WRAP;
   case PIPE_STENCIL_OP_DECR_WRAP:
      return VK_STENCIL_OP_DECREMENT_AND_WRAP;
   case PIPE_STENCIL_OP_INVERT:
      return VK_STENCIL_OP_INVERT;
   default:
      return VK_STENCIL_OP_KEEP;
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
            /* glVertexAttribDivisor(0) is a per-vertex attribute and anything
             * else is per-instance; Vulkan has exactly those two rates, so a
             * divisor above one is not distinguishable from one -- see the
             * limitation noted in sz_screen.c. */
            .inputRate = elem->instance_divisor
                            ? VK_VERTEX_INPUT_RATE_INSTANCE
                            : VK_VERTEX_INPUT_RATE_VERTEX,
         };
      }

      attributes[i] = (VkVertexInputAttributeDescription) {
         .location = i,
         .binding = vb,
         .format = sz_vk_format(ctx->screen, elem->src_format),
         .offset = elem->src_offset,
      };
   }

   *out_binding_count = binding_count;
   *out_attribute_count = element_count;
}

/* Sample count of the framebuffer.
 *
 * pipe_surface carries no sample count of its own -- the resource does, and the
 * "no attachment" framebuffer is the only case that uses fb->samples. Reading
 * pipe_surface::nr_samples returns 1 for a multisampled renderbuffer surface,
 * which builds a pipeline for the wrong sample count: Vulkan then either
 * rejects the draw (VUID-vkCmdDraw-multisampledRenderToSingleSampled-07286) or,
 * on drivers that do not check, renders into a mismatched attachment.
 *
 * All attachments of a framebuffer share one sample count by the GL rules, so
 * the maximum is the framebuffer's count; a disagreement means the state
 * tracker built an invalid framebuffer and is reported once. */
static VkSampleCountFlagBits
sz_framebuffer_samples(const struct pipe_framebuffer_state *fb)
{
   unsigned samples = fb->samples;

   for (unsigned i = 0; i < fb->nr_cbufs; i++) {
      if (fb->cbufs[i].texture)
         samples = MAX2(samples, fb->cbufs[i].texture->nr_samples);
   }
   if (fb->zsbuf.texture)
      samples = MAX2(samples, fb->zsbuf.texture->nr_samples);

   if (samples == 0)
      samples = 1;

   return sz_vk_sample_count(samples);
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

   /* Tessellation: both stages are bound or neither is, because GL links them
    * as a pair. */
   p->tcs = ctx->state.shaders[MESA_SHADER_TESS_CTRL]
                ? sz_pipeline_module(ctx, MESA_SHADER_TESS_CTRL)
                : VK_NULL_HANDLE;
   p->tes = ctx->state.shaders[MESA_SHADER_TESS_EVAL]
                ? sz_pipeline_module(ctx, MESA_SHADER_TESS_EVAL)
                : VK_NULL_HANDLE;
   if ((p->tcs == VK_NULL_HANDLE) != (p->tes == VK_NULL_HANDLE))
      goto fail;

   /* A geometry stage is part of the pipeline only when one is bound. There is
    * no bootstrap fallback for it: an empty GS is exactly "no GS", and GL 3.2
    * apps are free not to use one. */
   p->gs = ctx->state.shaders[MESA_SHADER_GEOMETRY]
              ? sz_pipeline_module(ctx, MESA_SHADER_GEOMETRY)
              : VK_NULL_HANDLE;
   if (p->gs == VK_NULL_HANDLE && ctx->state.shaders[MESA_SHADER_GEOMETRY])
      goto fail;

   /* ADR-008/005c: the pipeline layout is the push-descriptor layout and is
    * created once per screen. It depends only on the descriptor set layout, so
    * a per-pipeline copy was pure overhead -- and creating one per rebuild was
    * a measurable part of the cost of a state change. */
   VkResult result;

   VkPipelineShaderStageCreateInfo stages[5];
   unsigned stage_count = 0;

   stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = p->vs,
      .pName = "main",
   };

   if (p->tcs != VK_NULL_HANDLE) {
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
         .module = p->tcs,
         .pName = "main",
      };
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
         .module = p->tes,
         .pName = "main",
      };
   }

   if (p->gs != VK_NULL_HANDLE) {
      stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_GEOMETRY_BIT,
         .module = p->gs,
         .pName = "main",
      };
   }

   stages[stage_count++] = (VkPipelineShaderStageCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = p->fs,
      .pName = "main",
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

   /* Vulkan restarts on the largest index of the index type, which is exactly
    * GL's GL_PRIMITIVE_RESTART_FIXED_INDEX. An application that chose its own
    * index through glPrimitiveRestartIndex is handled above the driver: cso's
    * vbuf rewrites the buffer when caps->emulate_nonfixed_primitive_restart is
    * set, so what arrives here is always the fixed index. */
   /* GL restarts a list topology on the restart index like any other, Vulkan does
    * not: for POINT_LIST, LINE_LIST, TRIANGLE_LIST and the two list-with-adjacency
    * topologies it requires primitiveRestartEnable to be VK_FALSE unless
    * primitiveTopologyListRestart is enabled
    * (VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252). Turning it off
    * there is the only legal answer when the device cannot do it. */
   const bool list_topology =
      topology == VK_PRIMITIVE_TOPOLOGY_POINT_LIST ||
      topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
      topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST ||
      topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY ||
      topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;

   const bool restart =
      ctx->state.primitive_restart &&
      (!list_topology || ctx->screen->caps.list_restart);

   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topology,
      .primitiveRestartEnable = restart ? VK_TRUE : VK_FALSE,
   };

   /* Viewport and scissor are dynamic: GL changes them per draw and per viewport
    * index, and baking them into the pipeline would rebuild it constantly. */
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = PIPE_MAX_VIEWPORTS,
      .scissorCount = PIPE_MAX_VIEWPORTS,
   };

   const struct pipe_rasterizer_state *rast_state = sz_raster(ctx);

   /* sz_lower_nir() always translates GL clip space to Vulkan's, so a context
    * that asked for GL_ZERO_TO_ONE depth through glClipControl needs a shader
    * variant the driver does not build yet. Say so rather than render a depth
    * range that silently starts at 0.5. */
   if (rast_state && rast_state->clip_halfz)
      sz_warn_once("glClipControl(GL_ZERO_TO_ONE) is not implemented; "
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

   const VkSampleCountFlagBits samples = sz_framebuffer_samples(fb);

   /* Per-sample shading: glMinSampleShading reaches the driver as
    * pipe_rasterizer_state::force_persample_interp, which st sets when
    * GL_SAMPLE_SHADING is on and MinSampleShadingValue * samples > 1. Vulkan
    * wants a fraction as well; Gallium does not carry one, so the rate is 1.0 --
    * shade every sample. That is a superset of a lower request (the image is
    * right, only slower), and it is exactly what GL asks for at the default
    * value. A single-sampled target cannot shade per sample at all, and Vulkan
    * rejects the combination, so the request is dropped there. */
   const bool sample_shading =
      rast_state && rast_state->force_persample_interp && samples > 1;

   /* The sample mask is a pipeline property in Vulkan and a per-draw state in
    * Gallium. Without it glSampleCoverage and util_blitter's per-sample draws
    * (one draw per destination sample, each masked to that sample) both write
    * every sample, so an MSAA blit ends up with whichever sample was drawn last
    * in all of them. */
   const VkSampleMask sample_mask = (VkSampleMask)ctx->state.sample_mask;

   /* Both are GL pixel operations that turn coverage into a per-sample mask;
    * Vulkan has the same two switches and nothing else to map them onto. */
   const bool alpha_to_coverage =
      blend_state && blend_state->alpha_to_coverage;

   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = samples,
      .sampleShadingEnable = sample_shading ? VK_TRUE : VK_FALSE,
      .minSampleShading = 1.0f,
      /* The mask and alpha-to-coverage are mutually exclusive: coverage derived
       * from alpha replaces the mask, so naming both is invalid
       * (VUID-VkPipelineMultisampleStateCreateInfo-pSampleMask-09427) and the
       * fixed mask wins over the computed coverage on RADV. */
      .pSampleMask = alpha_to_coverage ? NULL : &sample_mask,
      .alphaToCoverageEnable = alpha_to_coverage ? VK_TRUE : VK_FALSE,
      .alphaToOneEnable =
         blend_state && blend_state->alpha_to_one ? VK_TRUE : VK_FALSE,
   };

   const bool has_depth = fb->zsbuf.texture != NULL;

   /* The attachment formats belong to the pipeline: dynamic rendering has no
    * render pass object to take them from. They are computed once, here, because
    * the stencil decision below needs to know whether a stencil aspect exists at
    * all before the pipeline is described. */
   const VkFormat depth_format =
      has_depth ? sz_vk_format(screen, fb->zsbuf.format) : VK_FORMAT_UNDEFINED;
   /* A depth format without a stencil aspect has to report UNDEFINED: naming a
    * stencil-less format makes the pipeline invalid rather than stencil-less. */
   const VkFormat stencil_format =
      has_depth &&
         util_format_has_stencil(util_format_description(fb->zsbuf.format))
            ? sz_vk_format(screen, fb->zsbuf.format)
            : VK_FORMAT_UNDEFINED;

   /* GL's depth test is off by default; a draw that enables it without a depth
    * attachment is a state-tracker bug, not something to honour.
    *
    * Depth writes without a depth test are what a "renders but looks wrong"
    * picture is made of: without the compare, later primitives overwrite earlier
    * ones regardless of distance, so a scene comes out with its back faces on
    * top. Both bits come from the same Gallium state. */
   const bool depth_enabled = has_depth && dsa && dsa->depth_enabled;
   const bool depth_write = depth_enabled && dsa->depth_writemask;

   /* Stencil, both faces.
    *
    * Gallium reports the back face in stencil[1] and sets its `enabled` bit only
    * when two-sided testing is on (GL_EXT_stencil_two_side); a single-sided
    * state must apply the front face's function and ops to back faces too, which
    * is what Vulkan's VkPipelineDepthStencilStateCreateInfo.back is for.
    *
    * The reference value is not part of the state object -- glStencilFunc
    * changes it without touching anything else -- so it is dynamic and comes
    * from vkCmdSetStencilReference per draw. The masks are part of the state
    * object, and the state tracker already folds them into a new pipe object
    * when they change, so they travel with the pipeline. */
   const bool has_stencil_aspect = stencil_format != VK_FORMAT_UNDEFINED;
   const bool stencil_enabled =
      has_stencil_aspect && dsa && dsa->stencil[0].enabled;
   const struct pipe_stencil_state *front_state =
      dsa ? &dsa->stencil[0] : NULL;
   const struct pipe_stencil_state *back_state =
      (dsa && dsa->stencil[1].enabled) ? &dsa->stencil[1] : front_state;

   if (dsa && dsa->stencil[0].enabled && !has_stencil_aspect)
      sz_warn_once("stencil testing enabled without a stencil attachment; "
                   "ignoring it");

   const VkStencilOpState stencil_front = {
      .failOp = sz_stencil_op(front_state ? front_state->fail_op
                                          : PIPE_STENCIL_OP_KEEP),
      .passOp = sz_stencil_op(front_state ? front_state->zpass_op
                                          : PIPE_STENCIL_OP_KEEP),
      .depthFailOp = sz_stencil_op(front_state ? front_state->zfail_op
                                               : PIPE_STENCIL_OP_KEEP),
      .compareOp = sz_compare_op(front_state ? front_state->func
                                             : PIPE_FUNC_ALWAYS),
      .compareMask = front_state ? front_state->valuemask : 0xff,
      .writeMask = front_state ? front_state->writemask : 0xff,
   };
   const VkStencilOpState stencil_back = {
      .failOp = sz_stencil_op(back_state ? back_state->fail_op
                                          : PIPE_STENCIL_OP_KEEP),
      .passOp = sz_stencil_op(back_state ? back_state->zpass_op
                                          : PIPE_STENCIL_OP_KEEP),
      .depthFailOp = sz_stencil_op(back_state ? back_state->zfail_op
                                               : PIPE_STENCIL_OP_KEEP),
      .compareOp = sz_compare_op(back_state ? back_state->func
                                             : PIPE_FUNC_ALWAYS),
      .compareMask = back_state ? back_state->valuemask : 0xff,
      .writeMask = back_state ? back_state->writemask : 0xff,
   };

   const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = depth_enabled,
      .depthWriteEnable = depth_write,
      .depthCompareOp = dsa ? sz_compare_op(dsa->depth_func)
                            : VK_COMPARE_OP_ALWAYS,
      .stencilTestEnable = stencil_enabled,
      .front = stencil_front,
      .back = stencil_back,
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
      /* glStencilFunc's reference value changes without a pipeline rebind. */
      VK_DYNAMIC_STATE_STENCIL_REFERENCE,
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
      color_formats[i] = sz_vk_format(screen, fb->cbufs[i].format);

   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = color_count,
      .pColorAttachmentFormats = color_count ? color_formats : NULL,
      .depthAttachmentFormat = depth_format,
      .stencilAttachmentFormat = stencil_format,
   };

      /* Vulkan requires this structure when and only when a tessellation stage is
    * present, and it has no default: the input patch size is a pipeline
    * property, not a shader one. */
   const VkPipelineTessellationStateCreateInfo tessellation = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
      .patchControlPoints = MAX2(ctx->state.patch_vertices, 1),
   };

   const VkGraphicsPipelineCreateInfo pipeline_ci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      /* A pipeline whose descriptors come from a descriptor buffer has to say so
       * at creation time, or binding a descriptor buffer and drawing from it is
       * invalid (VUID-vkCmdDraw-None-08117). The flag is a promise that the
       * pipeline is *only* drawn with descriptor buffers, which is exactly the
       * case whenever the screen took that path. */
      .flags = screen->db.enabled
                  ? VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
                  : 0,
      .pNext = &rendering,
      .stageCount = stage_count,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pTessellationState = p->tcs != VK_NULL_HANDLE ? &tessellation : NULL,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pDepthStencilState = &depth_stencil,
      .pColorBlendState = &blend,
      .pDynamicState = &dynamic_state,
      .layout = screen->pipeline_layout,
   };

   /* The pipeline cache is what stops RADV from recompiling the shaders for a
    * state combination it has already seen. It is the single cheapest thing in
    * this file: a driver-level cache is keyed by the whole create info. */
   result = screen->vk.CreateGraphicsPipelines(screen->dev,
                                                screen->pipeline_cache, 1,
                                                &pipeline_ci, NULL, &p->pipeline);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateGraphicsPipelines failed: %d", result);
      goto fail;
   }

   return true;

fail:
   sz_pipeline_destroy_objects(screen, p);
   return false;
}

/* Fills `key` from the current state. Called once per state change, never per
 * draw: that is the whole point of the epoch. */
static void
sz_pipeline_key_from_state(struct sz_context *ctx, VkPrimitiveTopology topology,
                           struct sz_pipeline_key *key)
{
   const struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   memset(key, 0, sizeof(*key));
   key->topology = topology;
   /* Ids, not pointers: see struct sz_pipeline_key. A state that is not bound
    * is id 0, which never collides because ids start at 1. */
#define SZ_STATE_ID(ptr)                                                       \
   ((ptr) ? ((const struct sz_state *)(ptr))->id : 0)
   key->blend = SZ_STATE_ID(ctx->state.blend);
   key->rasterizer = SZ_STATE_ID(ctx->state.rasterizer);
   key->dsa = SZ_STATE_ID(ctx->state.depth_stencil_alpha);
   key->vs = SZ_STATE_ID(ctx->state.shaders[MESA_SHADER_VERTEX]);
   key->tcs = SZ_STATE_ID(ctx->state.shaders[MESA_SHADER_TESS_CTRL]);
   key->tes = SZ_STATE_ID(ctx->state.shaders[MESA_SHADER_TESS_EVAL]);
   /* patchControlPoints is baked into the pipeline, and GL lets the application
    * change it between draws with glPatchParameteri. */
   key->patch_vertices = ctx->state.patch_vertices;
   key->gs = SZ_STATE_ID(ctx->state.shaders[MESA_SHADER_GEOMETRY]);
   key->fs = SZ_STATE_ID(ctx->state.shaders[MESA_SHADER_FRAGMENT]);
#undef SZ_STATE_ID

   sz_pipeline_vertex_input(ctx, key->vertex_bindings,
                            &key->vertex_binding_count, key->vertex_attributes,
                            &key->vertex_attribute_count);

   key->samples = sz_framebuffer_samples(fb);
   key->sample_mask = ctx->state.sample_mask;
   key->color_count = MIN2(fb->nr_cbufs, PIPE_MAX_COLOR_BUFS);
   for (unsigned i = 0; i < key->color_count; i++)
      key->color[i] = sz_vk_format(ctx->screen, fb->cbufs[i].format);
   key->depth = fb->zsbuf.texture
                   ? sz_vk_format(ctx->screen, fb->zsbuf.format)
                   : VK_FORMAT_UNDEFINED;
}

/* The hash is only a lookup accelerator; this is the authoritative comparison. */
static bool
sz_pipeline_key_equal(const struct sz_pipeline_key *a,
                      const struct sz_pipeline_key *b)
{
   return a->blend == b->blend && a->rasterizer == b->rasterizer &&
          a->dsa == b->dsa && a->vs == b->vs && a->fs == b->fs &&
          a->tcs == b->tcs && a->tes == b->tes &&
          a->patch_vertices == b->patch_vertices &&
          a->topology == b->topology && a->samples == b->samples &&
          a->sample_mask == b->sample_mask &&
          a->gs == b->gs &&
          a->color_count == b->color_count && a->depth == b->depth &&
          a->vertex_binding_count == b->vertex_binding_count &&
          a->vertex_attribute_count == b->vertex_attribute_count &&
          memcmp(a->color, b->color,
                 b->color_count * sizeof(b->color[0])) == 0 &&
          memcmp(a->vertex_bindings, b->vertex_bindings,
                 b->vertex_binding_count * sizeof(b->vertex_bindings[0])) == 0 &&
          memcmp(a->vertex_attributes, b->vertex_attributes,
                 b->vertex_attribute_count *
                    sizeof(b->vertex_attributes[0])) == 0;
}

struct sz_gfx_pipeline *
sz_pipeline_get(struct sz_context *ctx, VkPrimitiveTopology topology)
{
   struct sz_pipeline_cache *cache = &ctx->pipelines;
   struct sz_pipeline_key key;

   sz_pipeline_key_from_state(ctx, topology, &key);
   const uint64_t hash = sz_hash_bytes(0, &key, sizeof(key));

   unsigned free_slot = SZ_MAX_CACHED_PIPELINES;
   struct sz_gfx_pipeline *victim = NULL;
   unsigned victim_slot = 0;

   for (unsigned i = 0; i < SZ_MAX_CACHED_PIPELINES; i++) {
      struct sz_gfx_pipeline *p = cache->entries[i];
      if (!p) {
         if (free_slot == SZ_MAX_CACHED_PIPELINES)
            free_slot = i;
         continue;
      }

      if (p->hash == hash && sz_pipeline_key_equal(&p->key, &key)) {
         p->last_use = ++cache->clock;
         return p;
      }

      /* Never evict the pipeline the command buffer has bound. Everything else
       * is safe to retire: the destroy is deferred until the submission that
       * may reference it has completed. */
      if (p != ctx->pipeline && (!victim || p->last_use < victim->last_use)) {
         victim = p;
         victim_slot = i;
      }
   }

   if (free_slot != SZ_MAX_CACHED_PIPELINES) {
      victim = NULL;
      victim_slot = free_slot;
   } else if (!victim) {
      /* Every slot holds the bound pipeline. Cannot happen with a cache larger
       * than one, but returning NULL beats destroying something in use. */
      return NULL;
   }

   if (victim) {
      sz_pipeline_reclaim(ctx);
      sz_pipeline_retire(ctx, victim);
      cache->entries[victim_slot] = NULL;
      cache->count--;
      free(victim);
   }

   struct sz_gfx_pipeline *p = CALLOC_STRUCT(sz_gfx_pipeline);
   if (!p)
      return NULL;

   if (!sz_pipeline_build(ctx, p, topology)) {
      free(p);
      return NULL;
   }

   ctx->screen->stats.pipeline_builds++;

   p->key = key;
   p->hash = hash;
   p->last_use = ++cache->clock;
   cache->entries[victim_slot] = p;
   cache->count++;

   return p;
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

   for (unsigned i = 0; i < SZ_MAX_CACHED_PIPELINES; i++) {
      struct sz_gfx_pipeline *p = ctx->pipelines.entries[i];
      if (!p)
         continue;
      sz_pipeline_destroy_objects(ctx->screen, p);
      free(p);
      ctx->pipelines.entries[i] = NULL;
   }
   ctx->pipelines.count = 0;
   ctx->pipeline = NULL;
}

/* A 64-bit FNV-1a over 64-bit words. Byte-at-a-time would spend ~300 ns on a
 * pipeline key; the keys are zero-padded by their designated initialisers, so
 * reading them as words is well defined. */
uint64_t
sz_hash_bytes(uint64_t hash, const void *data, size_t size)
{
   const uint8_t *bytes = data;

   if (hash == 0)
      hash = 0xcbf29ce484222325ull;

   while (size >= sizeof(uint64_t)) {
      uint64_t word;
      memcpy(&word, bytes, sizeof(word));
      hash ^= word;
      hash *= 0x100000001b3ull;
      bytes += sizeof(word);
      size -= sizeof(word);
   }

   while (size--) {
      hash ^= *bytes++;
      hash *= 0x100000001b3ull;
   }

   return hash;
}

bool
sz_pipeline_screen_init(struct sz_screen *screen)
{
   const VkDescriptorSetLayout set_layouts[SZ_DESCRIPTOR_SET_COUNT] = {
      screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_UBO],
      screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_SAMPLERS],
      screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_TEXEL_BUFFERS],
      screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_IMAGES],
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
                                                     NULL,
                                                     &screen->pipeline_layout);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreatePipelineLayout failed: %d", result);
      return false;
   }

   const VkPipelineCacheCreateInfo cache_ci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
   };
   result = screen->vk.CreatePipelineCache(screen->dev, &cache_ci, NULL,
                                           &screen->pipeline_cache);
   if (result != VK_SUCCESS) {
      /* Not fatal: a driver may refuse a cache, and Vulkan explicitly allows
       * VK_NULL_HANDLE to mean "no cache". Every pipeline still works, it just
       * costs a recompile per unseen state. */
      mesa_logw("SZ: vkCreatePipelineCache failed: %d; continuing without one",
                result);
      screen->pipeline_cache = VK_NULL_HANDLE;
   }

   return true;
}

void
sz_pipeline_screen_finish(struct sz_screen *screen)
{
   if (screen->bootstrap_vs != VK_NULL_HANDLE)
      screen->vk.DestroyShaderModule(screen->dev, screen->bootstrap_vs, NULL);
   if (screen->bootstrap_vs_empty != VK_NULL_HANDLE)
      screen->vk.DestroyShaderModule(screen->dev, screen->bootstrap_vs_empty,
                                     NULL);
   if (screen->bootstrap_fs != VK_NULL_HANDLE)
      screen->vk.DestroyShaderModule(screen->dev, screen->bootstrap_fs, NULL);
   if (screen->pipeline_cache != VK_NULL_HANDLE)
      screen->vk.DestroyPipelineCache(screen->dev, screen->pipeline_cache, NULL);
   if (screen->pipeline_layout != VK_NULL_HANDLE)
      screen->vk.DestroyPipelineLayout(screen->dev, screen->pipeline_layout,
                                       NULL);

   screen->bootstrap_vs = VK_NULL_HANDLE;
   screen->bootstrap_vs_empty = VK_NULL_HANDLE;
   screen->bootstrap_fs = VK_NULL_HANDLE;
   screen->pipeline_cache = VK_NULL_HANDLE;
   screen->pipeline_layout = VK_NULL_HANDLE;
}
