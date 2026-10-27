/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * The Gallium context.
 *
 * The state tracker calls a large part of this interface unconditionally --
 * cso_unbind_context alone rebinds every state slot -- so the full *shape* has
 * to exist before any of the behaviour does. State objects are recorded into
 * sz_state and created objects are placeholders. What is real here is command
 * submission and the transfer/clear commands, because those need no shaders and
 * are therefore the first part of the driver that can be verified end to end.
 */

#include "sz_context.h"

#include "sz_descriptor.h"

#include "sz_batch.h"
#include "sz_compiler.h"
#include "sz_debug.h"
#include "sz_format.h"
#include "sz_pipeline.h"
#include "sz_renderpass.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "compiler/nir/nir.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"
#include "util/u_surface.h"
#include "util/u_upload_mgr.h"

#include <string.h>

static inline struct sz_context *
sz_context(struct pipe_context *pipe)
{
   return (struct sz_context *)pipe;
}

/* Builds a VkClearColorValue from an unpacked texel.
 *
 * The conversion has to follow the format's channel type, not the C type of the
 * source pointer: a UNORM format wants floats in [0,1], an integer format wants
 * the raw integers, and a float format wants the bits reinterpreted. Passing the
 * wrong union member clears with a value that looks plausible and is wrong. */
static void
sz_clear_value_from_pixels(VkClearColorValue *out,
                           const struct util_format_description *desc,
                           const void *data)
{
   union {
      uint32_t u32[4];
      int32_t i32[4];
      float f32[4];
   } v = { { 0, 0, 0, 0 } };

   if (!data) {
      *out = (VkClearColorValue) { .uint32 = { 0, 0, 0, 0 } };
      return;
   }

   const unsigned bytes = util_format_get_blocksize(desc->format);

   for (unsigned c = 0; c < 4; c++) {
      const unsigned chan = c < desc->nr_channels ? c : desc->nr_channels - 1;
      const struct util_format_channel_description *ch = &desc->channel[chan];

      switch (ch->type) {
      case UTIL_FORMAT_TYPE_FLOAT:
         v.f32[c] = ch->size == 32 ? ((const float *)data)[chan]
                                   : (float)((const uint16_t *)data)[chan];
         break;
      case UTIL_FORMAT_TYPE_SIGNED:
         v.i32[c] = ch->size == 32 ? ((const int32_t *)data)[chan]
                                   : (int32_t)((const int16_t *)data)[chan];
         break;
      default:
         v.u32[c] = ch->size == 32 ? ((const uint32_t *)data)[chan]
                                   : (uint32_t)((const uint8_t *)data)[chan];
         break;
      }
   }

   (void)bytes;

   if (desc->channel[0].type == UTIL_FORMAT_TYPE_FLOAT)
      *out = (VkClearColorValue) { .float32 = { v.f32[0], v.f32[1], v.f32[2],
                                                v.f32[3] } };
   else if (desc->channel[0].type == UTIL_FORMAT_TYPE_SIGNED)
      *out = (VkClearColorValue) { .int32 = { v.i32[0], v.i32[1], v.i32[2],
                                              v.i32[3] } };
   else
      *out = (VkClearColorValue) { .uint32 = { v.u32[0], v.u32[1], v.u32[2],
                                               v.u32[3] } };
}

static void
sz_clear_value_depth_stencil(VkClearDepthStencilValue *out,
                             const struct util_format_description *desc,
                             const void *data, VkImageAspectFlags aspect)
{
   *out = (VkClearDepthStencilValue) { .depth = 0.0f, .stencil = 0 };

   if (!data)
      return;

   /* Depth is always a float in Vulkan regardless of the source encoding;
    * stencil is always an integer. */
   for (unsigned c = 0; c < desc->nr_channels; c++) {
      const struct util_format_channel_description *ch = &desc->channel[c];

      if (!(aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || ch->type != UTIL_FORMAT_TYPE_FLOAT)
         continue;

      out->depth = ch->size == 32 ? ((const float *)data)[c]
                                 : (float)((const uint16_t *)data)[c];
      break;
   }

   for (unsigned c = 0; c < desc->nr_channels; c++) {
      const struct util_format_channel_description *ch = &desc->channel[c];

      if (!(aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ||
          ch->type == UTIL_FORMAT_TYPE_FLOAT)
         continue;

      out->stencil = ((const uint8_t *)data)[c];
      break;
   }
}

/* ------------------------------------------------------------------------- */
/* Command stream                                                             */
/* ------------------------------------------------------------------------- */

/* Starts recording on first use and returns the live command buffer. Every
 * recorder goes through here, so no call site needs to know whether a batch is
 * already open. */
static VkCommandBuffer
sz_context_cmd(struct sz_context *ctx)
{
   if (!ctx->batch) {
      ctx->batch = sz_batch_begin(&ctx->screen->batch_state);
      if (!ctx->batch)
         return VK_NULL_HANDLE;
   }
   return ctx->batch->cmd;
}

/* For commands Vulkan forbids inside a render pass: clears, copies and
 * pipeline barriers. Ends the pass first, and the next draw starts a fresh one --
 * which costs a begin/end pair per clear/copy interleaved with draws, and is the
 * price of not having to prove that a clear never lands between two draws in
 * Phase 1. */
static VkCommandBuffer
sz_context_cmd_no_pass(struct sz_context *ctx)
{
   VkCommandBuffer cmd = sz_context_cmd(ctx);
   if (cmd != VK_NULL_HANDLE && ctx->rendering)
      sz_renderpass_end(ctx, cmd);
   return cmd;
}

static void
sz_context_submit(struct sz_context *ctx)
{
   if (!ctx->batch)
      return;

   /* A render pass cannot outlive the command buffer's recording. Ending it
    * here rather than in every caller means no path can forget. */
   sz_renderpass_end(ctx, ctx->batch->cmd);

   struct sz_batch *batch = ctx->batch;
   ctx->batch = NULL;

   if (!sz_batch_end(batch))
      return;

   if (sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
      sz_pipeline_note_submit(ctx);
}

static void
sz_context_flush(struct pipe_context *pipe, struct pipe_fence_handle **fence,
                 unsigned flags)
{
   struct sz_context *ctx = sz_context(pipe);

   /* There is no fence type yet. Returning NULL is honest: the batch is handed
    * to the queue before this returns, so the caller cannot observe it as
    * "not yet submitted". A real timeline-backed fence arrives in Phase 2,
    * where something actually needs to wait on GPU completion. */
   if (fence)
      *fence = NULL;

   sz_context_submit(ctx);

   /* The winsys asks for this before handing a buffer to the display engine.
    * Without it the compositor can sample a surface the GPU is still writing. */
   if (flags & PIPE_FLUSH_END_OF_FRAME) {
      sz_queue_wait_idle(&ctx->screen->batch_state);
      /* The queue just went idle, so every retired resource whose submission has
       * completed can be torn down. Doing it here rather than only on the next
       * destroy keeps the table from filling up during a long run. */
      sz_resource_reclaim(ctx->screen);
   }
}

/* ------------------------------------------------------------------------- */
/* State objects                                                              */
/* ------------------------------------------------------------------------- */

static struct sz_state *
sz_state_alloc(enum sz_state_kind kind)
{
   struct sz_state *state = CALLOC_STRUCT(sz_state);
   if (state)
      state->kind = kind;
   return state;
}

static void
sz_state_free(void *opaque)
{
   struct sz_state *state = opaque;
   if (!state)
      return;
   if (state->kind == SZ_STATE_SHADER) {
      free(state->u.shader.spirv);
      ralloc_free(state->u.shader.mem_ctx);
   }
   else if (state->kind == SZ_STATE_VERTEX_ELEMENTS)
      free(state->u.ve.elements);
   free(state);
}

static void *
sz_create_blend_state(struct pipe_context *pipe,
                      const struct pipe_blend_state *templ)
{
   struct sz_state *state = sz_state_alloc(SZ_STATE_BLEND);
   if (state)
      state->u.blend = *templ;
   return state;
}

static void
sz_delete_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_state *state = opaque;

   if (!state) {
      sz_state_free(opaque);
      return;
   }

   /* The state tracker deletes a state object as soon as it replaces it, which
    * can be while it is still the bound one. Leaving the shadow pointer dangling
    * would make the next draw read freed memory. */
   switch (state->kind) {
   case SZ_STATE_BLEND:
      if (ctx->state.blend == state)
         ctx->state.blend = NULL;
      break;
   case SZ_STATE_RASTERIZER:
      if (ctx->state.rasterizer == state)
         ctx->state.rasterizer = NULL;
      break;
   case SZ_STATE_DEPTH_STENCIL_ALPHA:
      if (ctx->state.depth_stencil_alpha == state)
         ctx->state.depth_stencil_alpha = NULL;
      break;
   case SZ_STATE_VERTEX_ELEMENTS:
      if (ctx->state.vertex_elements == state)
         ctx->state.vertex_elements = NULL;
      break;
   case SZ_STATE_SAMPLER:
      /* Sampler objects and sampler views share these arrays, so both are
       * cleared by identity rather than by position. */
      for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
         for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; i++) {
            if (ctx->state.sampler_states[s][i] == state)
               ctx->state.sampler_states[s][i] = NULL;
         }
      }
      break;
   case SZ_STATE_SHADER:
      for (unsigned s = 0; s < MESA_SHADER_STAGES; s++) {
         if (ctx->state.shaders[s] == state)
            ctx->state.shaders[s] = NULL;
      }
      break;
   }

   sz_state_free(opaque);
}

static void
sz_bind_typed(struct sz_context *ctx, struct sz_state *state)
{
   switch (state->kind) {
   case SZ_STATE_BLEND:
      ctx->state.blend = state;
      break;
   case SZ_STATE_RASTERIZER:
      ctx->state.rasterizer = state;
      break;
   case SZ_STATE_DEPTH_STENCIL_ALPHA:
      ctx->state.depth_stencil_alpha = state;
      break;
   case SZ_STATE_VERTEX_ELEMENTS:
      ctx->state.vertex_elements = state;
      break;
   default:
      break;
   }
}

/* One entry point per slot, because NULL carries no kind and therefore cannot say
 * which slot to clear. cso_context really does pass NULL when a state is unbound,
 * so "cannot happen" would be a crash rather than a missing draw. */
static void
sz_bind_blend_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.blend = NULL;
}

static void
sz_bind_rasterizer_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.rasterizer = NULL;
}

static void
sz_bind_depth_stencil_alpha_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.depth_stencil_alpha = NULL;
}

static void
sz_bind_vertex_elements_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.vertex_elements = NULL;
}

static void *
sz_create_sampler_state(struct pipe_context *pipe,
                        const struct pipe_sampler_state *templ)
{
   struct sz_state *state = sz_state_alloc(SZ_STATE_SAMPLER);
   if (state)
      state->u.sampler = *templ;
   return state;
}

static void *
sz_create_rasterizer_state(struct pipe_context *pipe,
                           const struct pipe_rasterizer_state *templ)
{
   struct sz_state *state = sz_state_alloc(SZ_STATE_RASTERIZER);
   if (state)
      state->u.rasterizer = *templ;
   return state;
}

static void *
sz_create_depth_stencil_alpha_state(
   struct pipe_context *pipe, const struct pipe_depth_stencil_alpha_state *templ)
{
   struct sz_state *state = sz_state_alloc(SZ_STATE_DEPTH_STENCIL_ALPHA);
   if (state)
      state->u.dsa = *templ;
   return state;
}

static void *
sz_create_vertex_elements_state(struct pipe_context *pipe, unsigned count,
                                const struct pipe_vertex_element *elements)
{
   struct sz_state *state = sz_state_alloc(SZ_STATE_VERTEX_ELEMENTS);
   if (!state)
      return NULL;

   if (count) {
      state->u.ve.elements = malloc(count * sizeof(*elements));
      if (!state->u.ve.elements) {
         free(state);
         return NULL;
      }
      memcpy(state->u.ve.elements, elements, count * sizeof(*elements));
   }
   state->u.ve.count = count;

   return state;
}

static void *
sz_create_shader_state(struct pipe_context *pipe,
                       mesa_shader_stage stage,
                       const struct pipe_shader_state *templ)
{
   struct sz_state *state = sz_state_alloc(SZ_STATE_SHADER);
   if (!state)
      return NULL;

   state->stage = stage;
   state->u.shader.base = *templ;

   /* The IR belongs to the state tracker, which frees it independently of us, so
    * the driver needs its own copy: a shader object outlives the one call that
    * created it by design (cso_context keeps the bound object and rebinds it).
    * Pointing at the caller's copy works until the caller frees it, which happens
    * on the very next program change. */
   if (templ->type != PIPE_SHADER_IR_NIR || !templ->ir.nir) {
      sz_warn_once("SZ: only NIR shaders are supported");
      state->u.shader.base.ir.nir = NULL;
      return state;
   }

   state->u.shader.mem_ctx = ralloc_context(NULL);
   if (!state->u.shader.mem_ctx) {
      free(state);
      return NULL;
   }

   state->u.shader.nir = nir_shader_clone(state->u.shader.mem_ctx, templ->ir.nir);
   state->u.shader.base.ir.nir = state->u.shader.nir;

   sz_lower_nir(state->u.shader.nir);

   if (sz_debug_enabled(SZ_DEBUG_NIR)) {
      mesa_logi("SZ: NIR %s shader:",
                mesa_shader_stage_name(state->stage));
      nir_print_shader(state->u.shader.nir, stderr);
   }

   char *error = NULL;
   if (!sz_nir_to_spirv(state->u.shader.nir, &state->u.shader.spirv,
                        &state->u.shader.spirv_words, &error)) {
      /* The caller reads error_message only when it asks for compile errors, and
       * frees it with free(), which is why this is strdup and not ralloc. */
      ((struct pipe_shader_state *)templ)->error_message =
         error ? error : strdup("superzink: shader translation failed");
      ralloc_free(state->u.shader.mem_ctx);
      free(state);
      return NULL;
   }

   sz_compiler_dump_spirv(state->u.shader.spirv, state->u.shader.spirv_words,
                          mesa_shader_stage_name(state->stage));

   return state;
}

static void *
sz_create_fs_state(struct pipe_context *pipe,
                   const struct pipe_shader_state *templ)
{
   return sz_create_shader_state(pipe, MESA_SHADER_FRAGMENT, templ);
}

static void *
sz_create_vs_state(struct pipe_context *pipe,
                   const struct pipe_shader_state *templ)
{
   return sz_create_shader_state(pipe, MESA_SHADER_VERTEX, templ);
}

static void *
sz_create_gs_state(struct pipe_context *pipe,
                   const struct pipe_shader_state *templ)
{
   return sz_create_shader_state(pipe, MESA_SHADER_GEOMETRY, templ);
}

static void *
sz_create_tcs_state(struct pipe_context *pipe,
                    const struct pipe_shader_state *templ)
{
   return sz_create_shader_state(pipe, MESA_SHADER_TESS_CTRL, templ);
}

static void *
sz_create_tes_state(struct pipe_context *pipe,
                    const struct pipe_shader_state *templ)
{
   return sz_create_shader_state(pipe, MESA_SHADER_TESS_EVAL, templ);
}

static void *
sz_create_ts_state(struct pipe_context *pipe,
                   const struct pipe_shader_state *templ)
{
   return sz_create_shader_state(pipe, MESA_SHADER_TASK, templ);
}

static void *
sz_create_ms_state(struct pipe_context *pipe,
                   const struct pipe_shader_state *templ)
{
   return sz_create_shader_state(pipe, MESA_SHADER_MESH, templ);
}

/* ------------------------------------------------------------------------- */
/* Sampler views                                                              */
/* ------------------------------------------------------------------------- */

static void
sz_sampler_view_destroy(struct pipe_context *pipe,
                        struct pipe_sampler_view *view)
{
   struct sz_sampler_view *sv = (struct sz_sampler_view *)view;
   pipe_resource_reference(&sv->base.texture, NULL);
   free(sv);
}

static void
sz_sampler_view_release(struct pipe_context *pipe,
                        struct pipe_sampler_view *view)
{
   /* The frontend signalling that it dropped its reference does not mean the
    * driver may free the object: sampler_view_destroy is the only release point.
    * Destroying here would double-free on the next unbind. */
}

static struct pipe_sampler_view *
sz_create_sampler_view(struct pipe_context *pipe, struct pipe_resource *texture,
                       const struct pipe_sampler_view *templ)
{
   struct sz_sampler_view *sv = CALLOC_STRUCT(sz_sampler_view);
   if (!sv)
      return NULL;

   sv->base = *templ;
   sv->base.context = pipe;
   pipe_resource_reference(&sv->base.texture, texture);

   return &sv->base;
}

/* ------------------------------------------------------------------------- */
/* Clears                                                                     */
/* ------------------------------------------------------------------------- */

/* True when `box` covers the whole of `level`'s extent, which is the only case
 * vkCmdClearColorImage can express: it clears a whole subresource. Anything
 * smaller needs a render pass, which needs pipelines, which is Phase 2. */
static bool
sz_box_covers_surface(const struct pipe_resource *res, unsigned level,
                      unsigned x, unsigned y, unsigned width, unsigned height)
{
   return x == 0 && y == 0 &&
          width >= u_minify(res->width0, level) &&
          height >= u_minify(res->height0, level);
}

static void
sz_clear_render_target(struct pipe_context *pipe, struct pipe_surface *dst,
                       const union pipe_color_union *color, unsigned dstx,
                       unsigned dsty, unsigned width, unsigned height,
                       bool render_condition_enabled)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_resource *res = sz_resource(dst->texture);

   if (!sz_box_covers_surface(&res->base, dst->level, dstx, dsty, width, height)) {
      /* Silently dropping this would show up as a stale region on screen, so say
       * so once rather than render the wrong thing quietly. */
      static bool warned;
      if (!warned) {
         mesa_logw("SZ: partial surface clears need a render pass; ignoring "
                   "the clear instead of corrupting the surface");
         warned = true;
      }
      return;
   }

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   sz_resource_access(ctx->screen, cmd, res, SZ_ACCESS_TRANSFER_DST);

   const VkClearColorValue clear = {
      .float32 = { color->f[0], color->f[1], color->f[2], color->f[3] },
   };

   const VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = dst->level,
      .levelCount = 1,
      .baseArrayLayer = dst->first_layer,
      .layerCount = dst->last_layer - dst->first_layer + 1,
   };

   ctx->screen->vk.CmdClearColorImage(cmd, res->image,
                                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      &clear, 1, &range);
}

static void
sz_clear_depth_stencil(struct pipe_context *pipe, struct pipe_surface *dst,
                       unsigned clear_flags, double depth, unsigned stencil,
                       unsigned dstx, unsigned dsty, unsigned width,
                       unsigned height, bool render_condition_enabled)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_resource *res = sz_resource(dst->texture);

   const struct util_format_description *desc =
      util_format_description(res->base.format);
   if (!desc)
      return;

   if (!sz_box_covers_surface(&res->base, dst->level, dstx, dsty, width, height))
      return;

   VkImageAspectFlags aspect = 0;
   if ((clear_flags & PIPE_CLEAR_DEPTH) && util_format_has_depth(desc))
      aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
   if ((clear_flags & PIPE_CLEAR_STENCIL) && util_format_has_stencil(desc))
      aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

   /* Vulkan cannot clear depth without the stencil aspect being present in the
    * same range when the format has one, so widen rather than skip. */
   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT && util_format_has_stencil(desc))
      aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;

   if (aspect == 0)
      return;

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   sz_resource_access(ctx->screen, cmd, res, SZ_ACCESS_TRANSFER_DST);

   const VkClearDepthStencilValue clear = {
      .depth = (float)depth,
      .stencil = stencil,
   };

   const VkImageSubresourceRange range = {
      .aspectMask = aspect,
      .baseMipLevel = dst->level,
      .levelCount = 1,
      .baseArrayLayer = dst->first_layer,
      .layerCount = dst->last_layer - dst->first_layer + 1,
   };

   ctx->screen->vk.CmdClearDepthStencilImage(
      cmd, res->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &range);
}

static void
sz_clear(struct pipe_context *pipe, unsigned buffers, uint32_t color_clear_mask,
         uint8_t stencil_clear_mask,
         const struct pipe_scissor_state *scissor_state,
         const union pipe_color_union *color, double depth, unsigned stencil)
{
   struct sz_context *ctx = sz_context(pipe);
   struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   /* glClear honours the scissor, and st splits the clear itself rather than
    * passing one down, so a scissor here means the caller expects per-buffer
    * handling we cannot express with vkCmdClear*. Refusing loudly beats silently
    * clearing the whole surface. */
   if (scissor_state) {
      static bool warned;
      if (!warned) {
         mesa_logw("SZ: scissored clears need a render pass; Phase 1 only "
                   "clears whole surfaces");
         warned = true;
      }
      return;
   }

   /* Mesa 26.3 embeds pipe_surface by value in pipe_framebuffer_state, so an
    * unattached slot is one whose texture is NULL rather than a NULL pointer. */
   if (buffers & PIPE_CLEAR_COLOR) {
      for (unsigned i = 0; i < fb->nr_cbufs; i++) {
         if (!fb->cbufs[i].texture)
            continue;
         sz_clear_render_target(pipe, &fb->cbufs[i], color, 0, 0,
                                fb->width, fb->height, false);
      }
   }

   if ((buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) &&
       fb->zsbuf.texture) {
      sz_clear_depth_stencil(pipe, &fb->zsbuf,
                             buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL),
                             depth, stencil, 0, 0, fb->width, fb->height, false);
   }
}

static void
sz_clear_texture(struct pipe_context *pipe, struct pipe_resource *res,
                 unsigned level, const struct pipe_box *box, const void *data)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_resource *sres = sz_resource(res);
   const struct util_format_description *desc =
      util_format_description(res->format);
   if (!desc || !box)
      return;

   /* Only whole-subresource clears are expressible; a partial box has to wait
    * for a render pass or a compute path. */
   if (box->x || box->y || box->z ||
       box->width < u_minify(res->width0, level) ||
       box->height < u_minify(res->height0, level)) {
      static bool warned;
      if (!warned) {
         mesa_logw("SZ: partial clear_texture is not implemented yet");
         warned = true;
      }
      return;
   }

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   sz_resource_access(ctx->screen, cmd, sres, SZ_ACCESS_TRANSFER_DST);

   VkImageAspectFlags aspect;
   if (util_format_has_depth(desc) && util_format_has_stencil(desc))
      aspect = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   else if (util_format_has_depth(desc))
      aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
   else if (util_format_has_stencil(desc))
      aspect = VK_IMAGE_ASPECT_STENCIL_BIT;
   else
      aspect = VK_IMAGE_ASPECT_COLOR_BIT;

   const VkImageSubresourceRange range = {
      .aspectMask = aspect,
      .baseMipLevel = level,
      .levelCount = 1,
      .baseArrayLayer = box->z,
      .layerCount = box->depth,
   };

   if (aspect == VK_IMAGE_ASPECT_COLOR_BIT) {
      VkClearColorValue clear;
      sz_clear_value_from_pixels(&clear, desc, data);
      ctx->screen->vk.CmdClearColorImage(cmd, sres->image,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                         &clear, 1, &range);
   } else {
      VkClearDepthStencilValue clear;
      sz_clear_value_depth_stencil(&clear, desc, data, aspect);
      ctx->screen->vk.CmdClearDepthStencilImage(
         cmd, sres->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
         &range);
   }
}

/* ------------------------------------------------------------------------- */
/* Copies                                                                     */
/* ------------------------------------------------------------------------- */

static void
sz_resource_copy_region(struct pipe_context *pipe, struct pipe_resource *dst,
                        unsigned dst_level, unsigned dstx, unsigned dsty,
                        unsigned dstz, struct pipe_resource *src,
                        unsigned src_level, const struct pipe_box *src_box)
{
   /* Same-format only, which is what Gallium promises for this entry point. */
   struct sz_context *ctx = sz_context(pipe);
   struct sz_resource *sdst = sz_resource(dst);
   struct sz_resource *ssrc = sz_resource(src);

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   const VkImageSubresourceLayers layers = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .mipLevel = 0,
      .baseArrayLayer = 0,
      .layerCount = 1,
   };

   if (sdst->buffer != VK_NULL_HANDLE && ssrc->buffer != VK_NULL_HANDLE) {
      const VkBufferCopy region = {
         .srcOffset = sz_bo_offset(ssrc->bo) + src_box->x,
         .dstOffset = sz_bo_offset(sdst->bo) + dstx,
         .size = src_box->width,
      };
      if (region.size)
         ctx->screen->vk.CmdCopyBuffer(cmd, ssrc->buffer, sdst->buffer, 1,
                                        &region);
      return;
   }

   if (sdst->image == VK_NULL_HANDLE || ssrc->image == VK_NULL_HANDLE) {
      mesa_logw("SZ: resource_copy_region between a buffer and an image is not "
                "implemented yet");
      return;
   }

   sz_resource_access(ctx->screen, cmd, ssrc, SZ_ACCESS_TRANSFER_SRC);
   sz_resource_access(ctx->screen, cmd, sdst, SZ_ACCESS_TRANSFER_DST);

   VkImageCopy region = {
      .srcSubresource = layers,
      .srcOffset = { src_box->x, src_box->y, src_box->z },
      .dstSubresource = layers,
      .dstOffset = { dstx, dsty, dstz },
      .extent = { src_box->width, src_box->height, src_box->depth },
   };
   region.srcSubresource.mipLevel = src_level;
   region.dstSubresource.mipLevel = dst_level;

   ctx->screen->vk.CmdCopyImage(cmd, ssrc->image,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                sdst->image,
                                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                &region);
}

/* ------------------------------------------------------------------------- */
/* Remaining interface                                                        */
/* ------------------------------------------------------------------------- */

/* Phase 1 stubs. They exist so the state tracker's unconditional calls have
 * somewhere to land; their bodies arrive with the subsystem each one belongs to.
 * Leaving a pointer NULL would turn a missing feature into a crash in code that
 * has nothing to do with it. */
static void sz_stub_void(void) {}

#define SZ_STUB(name)                              \
   static void name(void) { sz_stub_void(); }

static void
sz_set_blend_color(struct pipe_context *pipe, const struct pipe_blend_color *c)
{
   sz_context(pipe)->state.blend_color = *c;
}

static void
sz_set_stencil_ref(struct pipe_context *pipe, const struct pipe_stencil_ref ref)
{
   sz_context(pipe)->state.stencil_ref = ref;
}

static void
sz_set_sample_mask(struct pipe_context *pipe, unsigned mask)
{
   sz_context(pipe)->state.sample_mask = mask;
}

static void
sz_set_framebuffer_state(struct pipe_context *pipe,
                         const struct pipe_framebuffer_state *fb)
{
   struct sz_context *ctx = sz_context(pipe);
   const struct pipe_framebuffer_state *cur = &ctx->state.framebuffer;

   /* A dynamic rendering instance names concrete views, so it cannot be
    * retargeted: changing the attachments mid-pass has to close it. */
   bool changed = cur->nr_cbufs != fb->nr_cbufs || cur->width != fb->width ||
                  cur->height != fb->height ||
                  cur->zsbuf.texture != fb->zsbuf.texture;

   for (unsigned i = 0; !changed && i < fb->nr_cbufs; i++)
      changed = cur->cbufs[i].texture != fb->cbufs[i].texture ||
                cur->cbufs[i].level != fb->cbufs[i].level ||
                cur->cbufs[i].first_layer != fb->cbufs[i].first_layer;

   if (changed && ctx->rendering) {
      VkCommandBuffer cmd = sz_context_cmd(ctx);
      if (cmd != VK_NULL_HANDLE)
         sz_renderpass_end(ctx, cmd);
   }

   /* pipe_framebuffer_state holds the surfaces by value in this Mesa version, so
    * a struct copy is the whole update. */
   ctx->state.framebuffer = *fb;
}

static void
sz_set_viewport_states(struct pipe_context *pipe, unsigned start_slot,
                       unsigned num_viewports,
                       const struct pipe_viewport_state *viewports)
{
   struct sz_context *ctx = sz_context(pipe);

   for (unsigned i = 0; i < num_viewports; i++) {
      const unsigned slot = start_slot + i;
      if (slot >= PIPE_MAX_VIEWPORTS)
         break;
      ctx->state.viewports[slot] = viewports[i];
   }
   ctx->state.num_viewports = MAX2(ctx->state.num_viewports,
                                   start_slot + num_viewports);
}

static void
sz_set_scissor_states(struct pipe_context *pipe, unsigned start_slot,
                      unsigned num_scissors,
                      const struct pipe_scissor_state *scissors)
{
   struct sz_context *ctx = sz_context(pipe);

   for (unsigned i = 0; i < num_scissors; i++) {
      const unsigned slot = start_slot + i;
      if (slot >= PIPE_MAX_VIEWPORTS)
         break;
      ctx->state.scissors[slot] = scissors[i];
   }
   ctx->state.num_scissors = MAX2(ctx->state.num_scissors,
                                  start_slot + num_scissors);
}

static void
sz_set_sampler_views(struct pipe_context *pipe, mesa_shader_stage shader,
                     unsigned start_slot, unsigned num_views,
                     unsigned unbind_num_trailing_slots,
                     struct pipe_sampler_view **views)
{
   struct sz_context *ctx = sz_context(pipe);
   if (shader >= MESA_SHADER_STAGES)
      return;

   for (unsigned i = 0; i < num_views; i++) {
      const unsigned slot = start_slot + i;
      if (slot >= PIPE_MAX_SAMPLERS)
         break;
      ctx->state.sampler_views[shader][slot] = views ? views[i] : NULL;
   }
}

static void
sz_bind_sampler_states(struct pipe_context *pipe, mesa_shader_stage shader,
                       unsigned start_slot, unsigned num_samplers,
                       void **samplers)
{
   struct sz_context *ctx = sz_context(pipe);
   if (shader >= MESA_SHADER_STAGES)
      return;

   for (unsigned i = 0; i < num_samplers; i++) {
      const unsigned slot = start_slot + i;
      if (slot >= PIPE_MAX_SAMPLERS)
         break;
      if (samplers)
         ctx->state.sampler_states[shader][slot] = samplers[i];
   }
}

static void
sz_bind_shader_state(struct pipe_context *pipe, mesa_shader_stage stage,
                     void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   if (stage < MESA_SHADER_STAGES)
      ctx->state.shaders[stage] = opaque;
}

static void
sz_bind_fs_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_FRAGMENT, s);
}

static void
sz_bind_vs_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_VERTEX, s);
}

static void
sz_bind_gs_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_GEOMETRY, s);
}

static void
sz_bind_tcs_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_TESS_CTRL, s);
}

static void
sz_bind_tes_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_TESS_EVAL, s);
}

static void
sz_bind_ts_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_TASK, s);
}

static void
sz_bind_ms_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_MESH, s);
}

static void
sz_bind_compute_state(struct pipe_context *pipe, void *s)
{
   sz_bind_shader_state(pipe, MESA_SHADER_COMPUTE, s);
}

static void
sz_set_constant_buffer(struct pipe_context *pipe, mesa_shader_stage shader,
                       uint index, const struct pipe_constant_buffer *buf)
{
   struct sz_context *ctx = sz_context(pipe);

   if (shader >= MESA_SHADER_STAGES || index >= SZ_MAX_CONST_BUFFERS) {
      sz_warn_once("SZ: ignoring a constant buffer outside the supported range");
      return;
   }

   struct pipe_constant_buffer *dst = &ctx->state.constant_buffers[shader][index];

   if (!buf || (!buf->buffer && !buf->user_buffer)) {
      pipe_resource_reference(&dst->buffer, NULL);
      dst->buffer_offset = 0;
      dst->buffer_size = 0;
      dst->user_buffer = NULL;
      return;
   }

   /* st is asked for real buffers (caps->prefer_real_buffer_in_constbuf0), so a
    * client-memory pointer here means some other path -- a draw module or a
    * blit shader -- and copying it would need an uploader this phase does not
    * have yet. Say so once instead of reading stale descriptors. */
   if (!buf->buffer) {
      sz_warn_once("SZ: constant buffers in client memory are not implemented, "
                   "ignoring");
      return;
   }

   pipe_resource_reference(&dst->buffer, buf->buffer);
   dst->buffer_offset = buf->buffer_offset;
   dst->buffer_size = buf->buffer_size;
   dst->user_buffer = NULL;
}

static void
sz_set_inlinable_constants(struct pipe_context *pipe, mesa_shader_stage shader,
                           uint num_values, uint32_t *values)
{
   /* Inlinable constants are the ones a driver may bake into a shader variant.
    * This one always reads uniforms from the buffer, so the values are already
    * where the shader looks for them and there is nothing to recompile. The
    * entry point still has to exist: st calls it unconditionally, and leaving it
    * NULL is a null-dereference in the middle of a draw. */
}

static void
sz_set_vertex_buffers(struct pipe_context *pipe, unsigned count,
                      const struct pipe_vertex_buffer *buffers)
{
   struct sz_context *ctx = sz_context(pipe);

   const unsigned n = MIN2(count, PIPE_MAX_ATTRIBS);
   if (n)
      memcpy(ctx->state.vertex_buffers, buffers, n * sizeof(*buffers));
   ctx->state.num_vertex_buffers = n;
}

static void
sz_set_shader_buffers(struct pipe_context *pipe, mesa_shader_stage shader,
                      unsigned start_slot, unsigned count,
                      const struct pipe_shader_buffer *buffers,
                      unsigned writable_bitmask)
{
}

static void
sz_set_shader_images(struct pipe_context *pipe, mesa_shader_stage shader,
                     unsigned start_slot, unsigned count,
                     unsigned unbind_num_trailing_slots,
                     const struct pipe_image_view *images)
{
}

static void
sz_set_clip_state(struct pipe_context *pipe, const struct pipe_clip_state *c)
{
   sz_context(pipe)->state.clip = *c;
}

static void
sz_set_polygon_stipple(struct pipe_context *pipe,
                       const struct pipe_poly_stipple *stipple)
{
}

static void
sz_set_tess_state(struct pipe_context *pipe,
                  const float default_outer_level[4],
                  const float default_inner_level[2])
{
   struct sz_context *ctx = sz_context(pipe);
   memcpy(ctx->state.default_outer_level, default_outer_level,
          sizeof(ctx->state.default_outer_level));
   memcpy(ctx->state.default_inner_level, default_inner_level,
          sizeof(ctx->state.default_inner_level));
}

static void
sz_set_patch_vertices(struct pipe_context *pipe, uint8_t patch_vertices)
{
   sz_context(pipe)->state.patch_vertices = patch_vertices;
}

static void
sz_set_stream_output_targets(struct pipe_context *pipe, unsigned num_targets,
                             struct pipe_stream_output_target **targets,
                             const unsigned *offsets,
                             enum mesa_prim output_prim)
{
}

/* Records the viewport and scissor for slot 0.
 *
 * GL's viewport is in window coordinates with y pointing up from the bottom,
 * Vulkan's viewport is in framebuffer coordinates with y pointing down. The
 * negative height flips it, which is exactly what GL expects and what keeps
 * gl_Position's -1..1 range landing on the right pixels.
 *
 * Clamping happens in template: Gallium's viewport z is 0..1 already (GL's
 * glDepthRange is folded in by the state tracker), so no remap is needed. */
static void
sz_draw_viewport(struct sz_context *ctx, VkCommandBuffer cmd)
{
   const struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   /* Gallium viewport state is in GL window coordinates: y grows upwards from
    * the bottom-left corner. Vulkan's framebuffer origin is the top-left corner,
    * so a positive height with y at `translate - scale` maps NDC +1 to row 0 --
    * which is where GL puts the top of the window. Keeping the height positive is
    * what keeps the rendered image the right way up; a negative height (the
    * alternative, used when the shader does not flip y itself) would mirror the
    * frame vertically on screen. */
   VkViewport viewport = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)fb->width,
      .height = (float)fb->height,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };

   if (ctx->state.num_viewports > 0) {
      const struct pipe_viewport_state *vp = &ctx->state.viewports[0];
      viewport.x = vp->translate[0] - vp->scale[0];
      viewport.y = vp->translate[1] - vp->scale[1];
      viewport.width = 2.0f * vp->scale[0];
      viewport.height = 2.0f * vp->scale[1];
      viewport.minDepth = vp->translate[2] - vp->scale[2];
      viewport.maxDepth = vp->translate[2] + vp->scale[2];
   }

   VkRect2D scissor = {
      .offset = { 0, 0 },
      .extent = { fb->width, fb->height },
   };

   if (ctx->state.num_scissors > 0) {
      const struct pipe_scissor_state *sc = &ctx->state.scissors[0];
      scissor.offset.x = sc->minx;
      scissor.offset.y = sc->miny;
      scissor.extent.width = sc->maxx - sc->minx;
      scissor.extent.height = sc->maxy - sc->miny;
   }

   if (scissor.extent.width == 0 || scissor.extent.height == 0)
      return;

   ctx->screen->vk.CmdSetViewport(cmd, 0, 1, &viewport);
   ctx->screen->vk.CmdSetScissor(cmd, 0, 1, &scissor);

   /* glBlendColor is dynamic in Vulkan, so it is set per draw rather than baked
    * into the pipeline. */
   if (ctx->screen->vk.CmdSetBlendConstants) {
      const struct pipe_blend_color *bc = &ctx->state.blend_color;
      const float constants[4] = { bc->color[0], bc->color[1], bc->color[2],
                                   bc->color[3] };
      ctx->screen->vk.CmdSetBlendConstants(cmd, constants);
   }
}

/* Gallium primitive -> Vulkan topology.
 *
 * LINE_LOOP has no Vulkan equivalent; Gallium sets was_line_loop and rewrites the
 * indices, so treating it as a strip is correct for the draws the state tracker
 * already expanded. */
static VkPrimitiveTopology
sz_topology(enum mesa_prim mode)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
      return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
   case MESA_PRIM_LINES:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
   case MESA_PRIM_LINE_LOOP:
   case MESA_PRIM_LINE_STRIP:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   case MESA_PRIM_TRIANGLE_STRIP:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
   case MESA_PRIM_LINES_ADJACENCY:
      return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
   case MESA_PRIM_PATCHES:
      return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
   default:
      return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
   }
}

static void
sz_draw_vbo(struct pipe_context *pipe, const struct pipe_draw_info *info,
            unsigned drawid_offset,
            const struct pipe_draw_indirect_info *indirect,
            const struct pipe_draw_start_count_bias *draws, unsigned num_draws)
{
   struct sz_context *ctx = sz_context(pipe);

   if (indirect || num_draws == 0) {
      sz_warn_once("SZ: indirect draws are not implemented, ignoring");
      return;
   }

   const struct sz_state *ve = ctx->state.vertex_elements;
   if (!ve || ve->u.ve.count == 0) {
      sz_warn_once("SZ: draw without vertex elements, ignoring");
      return;
   }

   if (ctx->state.num_vertex_buffers == 0) {
      sz_warn_once("SZ: draw without vertex buffers, ignoring");
      return;
   }

   VkCommandBuffer cmd = sz_context_cmd(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   if (!ctx->rendering && !sz_renderpass_begin(ctx, cmd))
      return;

   /* The index buffer is part of pipe_draw_info, not context state, so it is
    * bound per draw. Binding it when it has not changed is what Phase 3's state
    * tracker removes; Phase 1 correctness first. */
   struct sz_resource *index = NULL;
   if (info->index_size) {
      if (info->has_user_indices) {
         sz_warn_once("SZ: user index buffers are not implemented, ignoring");
         return;
      }
      index = sz_resource(info->index.resource);
      if (!index || index->buffer == VK_NULL_HANDLE)
         return;

      sz_resource_access(ctx->screen, cmd, index, SZ_ACCESS_INDEX_BUFFER);

      const VkIndexType type = info->index_size == 4 ? VK_INDEX_TYPE_UINT32
                                                     : VK_INDEX_TYPE_UINT16;
      ctx->screen->vk.CmdBindIndexBuffer(cmd, index->buffer, 0, type);
   }

   const VkPrimitiveTopology topology = sz_topology(info->mode);

   struct sz_gfx_pipeline *pipeline = sz_pipeline_get(ctx, topology);
   if (!pipeline)
      return;

   ctx->screen->vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    pipeline->pipeline);

   /* Constant buffers are pushed per draw, not per state change: a push goes
    * into the command stream, so the cost is stream space and the correctness is
    * trivial -- the descriptors a draw sees are the ones bound when it was
    * recorded. */
   sz_descriptor_push(ctx, cmd);

   sz_draw_viewport(ctx, cmd);

   /* Every slot the frontend bound, not just slot 0: a shader reading two
    * attributes has two vertex buffers, and binding only the first leaves the
    * rest undefined. A slot with no resource behind it is left unbound, which is
    * legal as long as no shader input reads it. */
   for (unsigned i = 0; i < ctx->state.num_vertex_buffers; i++) {
      const struct pipe_vertex_buffer *vb = &ctx->state.vertex_buffers[i];
      if (vb->is_user_buffer || !vb->buffer.resource)
         continue;

      struct sz_resource *vres = sz_resource(vb->buffer.resource);
      if (vres->buffer == VK_NULL_HANDLE)
         continue;

      sz_resource_access(ctx->screen, cmd, vres, SZ_ACCESS_VERTEX_BUFFER);

      const VkDeviceSize offset = vb->buffer_offset;
      ctx->screen->vk.CmdBindVertexBuffers(cmd, i, 1, &vres->buffer, &offset);
   }

   for (unsigned i = 0; i < num_draws; i++) {
      if (draws[i].count == 0)
         continue;

      if (info->index_size) {
         ctx->screen->vk.CmdDrawIndexed(cmd, draws[i].count, info->instance_count,
                                        draws[i].start, (int32_t)draws[i].index_bias,
                                        info->start_instance);
      } else {
         ctx->screen->vk.CmdDraw(cmd, draws[i].count, info->instance_count,
                                 draws[i].start, info->start_instance);
      }
   }
}

static void
sz_draw_vbo_buffers(struct pipe_context *pipe, const struct pipe_draw_info *info,
                    const struct pipe_vertex_buffer *buffers,
                    unsigned buffer_count,
                    const struct pipe_draw_start_count_bias *draws,
                    unsigned num_draws)
{
}

static void
sz_draw_vertex_state(struct pipe_context *pipe,
                     struct pipe_vertex_state *state,
                     uint32_t partial_velem_mask,
                     struct pipe_draw_vertex_state_info info,
                     const struct pipe_draw_start_count_bias *draws,
                     unsigned num_draws)
{
}

static void
sz_launch_grid(struct pipe_context *pipe, const struct pipe_grid_info *info)
{
}

static void
sz_blit(struct pipe_context *pipe, const struct pipe_blit_info *info)
{
}

static void
sz_flush_resource(struct pipe_context *pipe, struct pipe_resource *res)
{
}

static void
sz_clear_buffer(struct pipe_context *pipe, struct pipe_resource *res,
                unsigned offset, unsigned size, const void *clear_value,
                int clear_value_size)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_resource *sres = sz_resource(res);

   if (sres->buffer == VK_NULL_HANDLE || clear_value_size != 4)
      return;

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   uint32_t value;
   memcpy(&value, clear_value, sizeof(value));

   const uint64_t base = sz_bo_offset(sres->bo);
   const uint64_t dst = base + offset;
   const uint64_t end = MIN2(dst + (uint64_t)size, base + sz_bo_size(sres->bo));
   if (end <= dst)
      return;

   ctx->screen->vk.CmdFillBuffer(cmd, sres->buffer, dst, end - dst, value);
}

static void
sz_clear_texture_entry(struct pipe_context *pipe, struct pipe_resource *res,
                       unsigned level, const struct pipe_box *box,
                       const void *data)
{
   sz_clear_texture(pipe, res, level, box, data);
}

static void *sz_transfer_map(struct pipe_context *pipe, struct pipe_resource *res,
                             unsigned level, unsigned usage,
                             const struct pipe_box *box,
                             struct pipe_transfer **transfer);
static void sz_transfer_unmap(struct pipe_context *pipe,
                              struct pipe_transfer *transfer);

/* Sub-data updates go through the transfer path rather than being special
 * cased: a host-visible resource is written in place, anything else is staged
 * and copied. Leaving them empty is the kind of hole that costs days -- a
 * glBufferSubData silently does nothing, the draw still runs, and the result is
 * a black screen with no error anywhere. */
static void
sz_texture_subdata(struct pipe_context *pipe, struct pipe_resource *res,
                   unsigned level, unsigned usage, const struct pipe_box *box,
                   const void *data, unsigned stride, uintptr_t layer_stride)
{
   struct pipe_transfer *transfer = NULL;
   void *map = sz_transfer_map(pipe, res, level, usage | PIPE_MAP_WRITE, box,
                               &transfer);
   if (!map)
      return;

   const struct util_format_description *desc = util_format_description(res->format);
   const unsigned bw = desc ? desc->block.width : 1;
   const unsigned bh = desc ? desc->block.height : 1;

   util_copy_box((uint8_t *)map, res->format, transfer->stride,
                 transfer->layer_stride, 0, 0, 0,
                 DIV_ROUND_UP(box->width, bw), DIV_ROUND_UP(box->height, bh),
                 box->depth, (const uint8_t *)data, stride, layer_stride,
                 box->x / bw, box->y / bh, box->z);

   sz_transfer_unmap(pipe, transfer);
}

static void
sz_buffer_subdata(struct pipe_context *pipe, struct pipe_resource *res,
                  unsigned usage, unsigned offset, unsigned size,
                  const void *data)
{
   if (size == 0)
      return;

   const struct pipe_box box = {
      .x = offset,
      .width = size,
   };
   struct pipe_transfer *transfer = NULL;
   void *map = sz_transfer_map(pipe, res, 0, usage | PIPE_MAP_WRITE, &box,
                               &transfer);
   if (!map)
      return;

   memcpy(map, data, size);
   sz_transfer_unmap(pipe, transfer);
}

/* Transfers.
 *
 * A resource the CPU can already address is handed out directly; everything else
 * goes through a host-visible staging buffer that is copied on map and back on
 * unmap. The copy is a full submit-and-wait, which is what makes a map a
 * synchronisation point -- correct, and the thing to optimise once profiling
 * says it matters.
 *
 * The staging buffer is per transfer rather than pooled. A pooled one would need
 * a lifetime rule that outlives unmap (the GPU may still be reading it while the
 * next map overwrites it), and getting that wrong shows up as textures that are
 * occasionally one frame stale. */
struct sz_transfer {
   struct pipe_transfer base;

   /* Non-NULL when the resource is not directly addressable, in which case
    * `staging` holds the mapped bytes. */
   struct pipe_resource *staging;

   /* Bytes per row in `staging`. Handed to the caller as base.stride, and used as
    * the copy's bufferRowLength. */
   unsigned row_pitch;
};

static struct pipe_resource *
sz_transfer_staging_create(struct sz_screen *screen, unsigned size)
{
   const struct pipe_resource tmpl = {
      .target = PIPE_BUFFER,
      .format = PIPE_FORMAT_R8_UNORM,
      .bind = 0,
      .usage = PIPE_USAGE_STAGING,
      .width0 = size,
      .height0 = 1,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
   };

   return screen->base.resource_create(&screen->base, &tmpl);
}

/* One recorded transfer, in a shape that can be handed to sz_oneshot(). */
struct sz_copy_job {
   struct sz_resource *res;
   struct sz_resource *staging;
   struct pipe_box box;
   unsigned level;
   bool to_staging;
   unsigned row_pitch;
};

static void
sz_copy_job_record(struct sz_context *ctx, VkCommandBuffer cmd, void *user)
{
   struct sz_copy_job *job = user;
   struct sz_screen *screen = ctx->screen;
   const unsigned bs = util_format_get_blocksize(job->res->base.format);

   if (job->res->image == VK_NULL_HANDLE) {
      /* Buffer to buffer: the region is a plain byte range. */
      sz_resource_access(screen, cmd, job->staging,
                         job->to_staging ? SZ_ACCESS_TRANSFER_DST
                                         : SZ_ACCESS_TRANSFER_SRC);
      sz_resource_access(screen, cmd, job->res,
                         job->to_staging ? SZ_ACCESS_TRANSFER_SRC
                                         : SZ_ACCESS_TRANSFER_DST);

      const VkBufferCopy region = {
         .srcOffset = job->to_staging ? job->box.x : 0,
         .dstOffset = job->to_staging ? 0 : job->box.x,
         .size = job->box.width,
      };

      if (job->to_staging)
         screen->vk.CmdCopyBuffer(cmd, job->res->buffer, job->staging->buffer,
                                  1, &region);
      else
         screen->vk.CmdCopyBuffer(cmd, job->staging->buffer, job->res->buffer,
                                  1, &region);
      return;
   }

   const VkImageLayout layout =
      sz_resource_access(screen, cmd, job->res,
                         job->to_staging ? SZ_ACCESS_TRANSFER_SRC
                                         : SZ_ACCESS_TRANSFER_DST);
   sz_resource_access(screen, cmd, job->staging,
                      job->to_staging ? SZ_ACCESS_TRANSFER_DST
                                      : SZ_ACCESS_TRANSFER_SRC);

   const bool is_3d = job->res->base.target == PIPE_TEXTURE_3D;

   VkBufferImageCopy region = {
      .bufferOffset = 0,
      /* In texels, not bytes; zero would mean "tightly packed", which is not the
       * pitch the caller sees. */
      .bufferRowLength = job->row_pitch / bs,
      .bufferImageHeight = 0,
      .imageSubresource = {
         .aspectMask = sz_format_aspect(job->res->base.format),
         .mipLevel = job->level,
         .baseArrayLayer = is_3d ? 0 : job->box.z,
         .layerCount = is_3d ? 1 : job->box.depth,
      },
      .imageOffset = {
         job->box.x,
         job->box.y,
         is_3d ? job->box.z : 0,
      },
      .imageExtent = {
         job->box.width,
         job->box.height,
         job->box.depth,
      },
   };

   if (job->to_staging)
      screen->vk.CmdCopyImageToBuffer(cmd, job->res->image, layout,
                                      job->staging->buffer, 1, &region);
   else
      screen->vk.CmdCopyBufferToImage(cmd, job->staging->buffer, job->res->image,
                                      layout, 1, &region);
}

/* Runs one recorded job to completion.
 *
 * Whatever the context has recorded so far is submitted first: the job needs its
 * own command buffer, and the resource it touches may already be referenced by
 * the pending one. */
static bool
sz_oneshot(struct sz_context *ctx, struct sz_copy_job *job)
{
   sz_context_submit(ctx);

   struct sz_batch *batch = sz_batch_begin(&ctx->screen->batch_state);
   if (!batch)
      return false;

   sz_copy_job_record(ctx, batch->cmd, job);

   if (!sz_batch_end(batch))
      return false;
   if (!sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
      return false;

   sz_queue_wait_idle(&ctx->screen->batch_state);
   return true;
}

static void *
sz_transfer_map(struct pipe_context *pipe, struct pipe_resource *res,
                unsigned level, unsigned usage, const struct pipe_box *box,
                struct pipe_transfer **transfer)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_screen *screen = ctx->screen;
   struct sz_resource *r = sz_resource(res);

   struct sz_transfer *t = CALLOC_STRUCT(sz_transfer);
   if (!t)
      return NULL;

   t->base.resource = res;
   t->base.level = level;
   t->base.usage = usage;
   t->base.box = *box;

   void *direct = sz_resource_map(r);
   if (direct) {
      t->base.stride = box->width;
      t->base.layer_stride = 0;
      *transfer = &t->base;
      return (char *)direct + box->x;
   }

   const unsigned bs = util_format_get_blocksize(res->format);
   const unsigned row_pitch =
      res->target == PIPE_BUFFER
         ? 0
         : (unsigned)sz_resource_linear_pitch(screen, box->width, bs);
   const uint64_t size =
      res->target == PIPE_BUFFER
         ? box->width
         : (uint64_t)row_pitch * box->height * box->depth;

   t->staging = sz_transfer_staging_create(screen, (unsigned)size);
   if (!t->staging) {
      free(t);
      return NULL;
   }

   t->row_pitch = row_pitch;

   if (usage & PIPE_MAP_READ) {
      struct sz_copy_job job = {
         .res = r,
         .staging = sz_resource(t->staging),
         .box = *box,
         .level = level,
         .to_staging = true,
         .row_pitch = row_pitch,
      };

      if (!sz_oneshot(ctx, &job)) {
         pipe_resource_reference(&t->staging, NULL);
         free(t);
         return NULL;
      }

      sz_bo_invalidate(sz_resource(t->staging)->bo, 0, size);
   }

   t->base.stride = res->target == PIPE_BUFFER ? box->width : row_pitch;
   t->base.layer_stride = res->target == PIPE_BUFFER ? 0 : row_pitch * box->height;

   *transfer = &t->base;
   return sz_resource_map(sz_resource(t->staging));
}

static void
sz_transfer_unmap(struct pipe_context *pipe, struct pipe_transfer *transfer)
{
   struct sz_transfer *t = (struct sz_transfer *)transfer;

   if (t->staging && (t->base.usage & PIPE_MAP_WRITE)) {
      struct sz_context *ctx = sz_context(pipe);
      struct sz_resource *staging = sz_resource(t->staging);
      const uint64_t size = sz_bo_size(staging->bo);

      sz_bo_flush(staging->bo, 0, size);

      struct sz_copy_job job = {
         .res = sz_resource(t->base.resource),
         .staging = staging,
         .box = t->base.box,
         .level = t->base.level,
         .to_staging = false,
         .row_pitch = t->row_pitch,
      };

      sz_oneshot(ctx, &job);
   }

   pipe_resource_reference(&t->staging, NULL);
   free(t);
}

static void *
sz_buffer_map(struct pipe_context *pipe, struct pipe_resource *res,
              unsigned level, unsigned usage, const struct pipe_box *box,
              struct pipe_transfer **transfer)
{
   return sz_transfer_map(pipe, res, level, usage, box, transfer);
}

static void
sz_buffer_unmap(struct pipe_context *pipe, struct pipe_transfer *transfer)
{
   sz_transfer_unmap(pipe, transfer);
}

static void *
sz_texture_map(struct pipe_context *pipe, struct pipe_resource *res,
               unsigned level, unsigned usage, const struct pipe_box *box,
               struct pipe_transfer **transfer)
{
   return sz_transfer_map(pipe, res, level, usage, box, transfer);
}

static void
sz_texture_unmap(struct pipe_context *pipe, struct pipe_transfer *transfer)
{
   sz_transfer_unmap(pipe, transfer);
}

static void
sz_transfer_flush_region(struct pipe_context *pipe,
                         struct pipe_transfer *transfer,
                         const struct pipe_box *box)
{
   /* The whole box is copied at unmap, so there is nothing to flush early. */
}

static void
sz_texture_barrier(struct pipe_context *pipe, unsigned flags)
{
}

static void
sz_resource_release(struct pipe_context *pipe, struct pipe_resource *res)
{
   /* The frontend is done with `res`; the driver owns whatever is left. Dropping
    * the last reference is the entire contract (see u_default_resource_release),
    * and sz_screen_resource_destroy() defers the Vulkan teardown if a submission
    * may still read it. Without this hook st_add_releasebuf() -- which the
    * constant uploader calls whenever it reallocates, once every ~800 frames of
    * glxgears -- jumps through a NULL function pointer. */
   pipe_resource_reference(&res, NULL);
}

static void
sz_context_memory_barrier(struct pipe_context *pipe, unsigned flags)
{
   struct sz_context *ctx = sz_context(pipe);

   /* glMemoryBarrier, and the state tracker's own "shader wrote this, now read
    * it back" paths. There is no finer information to act on than "make every
    * earlier write visible", which is exactly one whole-command-buffer barrier.
    * The heavier cost of getting this wrong (stale pixels) is why it is not a
    * no-op. */
   if (flags == 0)
      return;

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   sz_memory_barrier(ctx->screen, cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

static enum pipe_reset_status
sz_get_device_reset_status(struct pipe_context *pipe)
{
   return PIPE_NO_RESET;
}

static void
sz_set_debug_callback(struct pipe_context *pipe,
                      const struct util_debug_callback *cb)
{
}

static void
sz_context_destroy(struct pipe_context *pipe)
{
   struct sz_context *ctx = sz_context(pipe);

   /* Anything recorded and not yet submitted still names the pipeline and the
    * attachments, so it has to reach the queue before they are destroyed. */
   sz_context_submit(ctx);
   sz_queue_wait_idle(&ctx->screen->batch_state);

   sz_pipeline_finish(ctx);

   /* The constant buffers are the one piece of shadow state the context owns a
    * reference to, so they are the one piece it has to drop. */
   for (unsigned sh = 0; sh < MESA_SHADER_STAGES; sh++) {
      for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++)
         pipe_resource_reference(&ctx->state.constant_buffers[sh][i].buffer,
                                 NULL);
   }

   u_upload_destroy(ctx->base.const_uploader);
   u_upload_destroy(ctx->base.stream_uploader);

   /* Nothing in ctx->state is owned here: every object was created by a create_*
    * entry point and is destroyed by its matching delete_* entry point, which the
    * state tracker always calls. Freeing them again here was a double free --
    * and the sampler arrays are shared with sampler views, so it was also a
    * type-confused free. */
   ctx->screen->context_alive = false;
   free(ctx);
}

/* ------------------------------------------------------------------------- */
/* Self-test                                                                  */
/* ------------------------------------------------------------------------- */

/* Exact comparison is too strict for a rendered texel: the hardware is allowed
 * to round an 8-bit conversion either way, so a correct draw can land one step
 * off. Two steps still catches a wrong colour, a flipped channel or a stale
 * frame. */
static bool
sz_rgba_close(const uint8_t *a, const uint8_t *b, unsigned tolerance)
{
   for (unsigned i = 0; i < 4; i++) {
      const int diff = (int)a[i] - (int)b[i];
      if (diff > (int)tolerance || diff < -(int)tolerance)
         return false;
   }
   return true;
}

unsigned
sz_context_selftest(struct sz_screen *screen)
{
   unsigned failures = 0;
   const unsigned w = 64, h = 64;

   struct pipe_resource *image = NULL;
   struct pipe_resource *readback = NULL;

   const struct pipe_resource img_tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
      .usage = PIPE_USAGE_DEFAULT,
      .width0 = w,
      .height0 = h,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
   };

   image = screen->base.resource_create(&screen->base, &img_tmpl);
   if (!image) {
      mesa_loge("SZ: selftest: could not create the clear target");
      return 1;
   }

   const struct pipe_resource buf_tmpl = {
      .target = PIPE_BUFFER,
      .format = PIPE_FORMAT_R8_UNORM,
      /* STAGING is what buys host-visible memory now that the transfer bind
       * flags are gone. */
      .usage = PIPE_USAGE_STAGING,
      .width0 = w * h * 4,
      .height0 = 1,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
   };

   readback = screen->base.resource_create(&screen->base, &buf_tmpl);
   if (!readback) {
      mesa_loge("SZ: selftest: could not create the readback buffer");
      failures++;
      goto out;
   }

   struct sz_resource *img = sz_resource(image);
   struct sz_resource *rb = sz_resource(readback);

   /* A different value per channel catches channel swaps and endianness
    * mistakes that a uniform colour would hide. */
   const VkClearColorValue clear = {
      .float32 = { 0.25f, 0.5f, 0.75f, 1.0f },
   };

   struct sz_batch *batch = sz_batch_begin(&screen->batch_state);
   if (!batch) {
      failures++;
      goto out;
   }
   const VkCommandBuffer cmd = batch->cmd;

   /* The image has never been written, so the source layout is UNDEFINED rather
    * than whatever the resource happens to default to. */
   sz_resource_discard(img);
   sz_resource_access(screen, cmd, img, SZ_ACCESS_TRANSFER_DST);

   const VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .baseMipLevel = 0,
      .levelCount = 1,
      .baseArrayLayer = 0,
      .layerCount = 1,
   };

   screen->vk.CmdClearColorImage(cmd, img->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
                                 &range);

   sz_resource_access(screen, cmd, img, SZ_ACCESS_TRANSFER_SRC);

   const VkBufferImageCopy region = {
      .bufferOffset = 0,
      .bufferRowLength = 0, /* tightly packed */
      .bufferImageHeight = 0,
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .imageOffset = { 0, 0, 0 },
      .imageExtent = { w, h, 1 },
   };

   screen->vk.CmdCopyImageToBuffer(cmd, img->image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   rb->buffer, 1, &region);

   /* The barrier that makes the copy visible to the host has to be inside the
    * command buffer, so it is recorded before submission, not after. */
   sz_memory_barrier(screen, cmd, VK_PIPELINE_STAGE_2_COPY_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

   if (!sz_batch_end(batch) ||
       !sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)) {
      failures++;
      goto out;
   }

   sz_queue_wait_idle(&screen->batch_state);
   sz_bo_invalidate(rb->bo, 0, sz_bo_size(rb->bo));

   const uint8_t *pixels = sz_resource_map(rb);
   if (!pixels) {
      mesa_loge("SZ: selftest: the readback buffer is not host visible");
      failures++;
      goto out;
   }

   /* 0.25/0.5/0.75 round to 64/128/191 in 8-bit UNORM. One unit of slack,
    * because a driver may round half-to-even instead of truncating. */
   const uint8_t expect[4] = { 64, 128, 191, 255 };

   unsigned bad = 0;
   for (unsigned y = 0; y < h && bad < 8; y++) {
      for (unsigned x = 0; x < w; x++) {
         const uint8_t *texel = pixels + (y * w + x) * 4;
         for (unsigned c = 0; c < 4; c++) {
            const int diff = (int)texel[c] - (int)expect[c];
            if (diff < -1 || diff > 1) {
               if (bad == 0) {
                  mesa_loge("SZ: selftest: texel %u,%u is %u,%u,%u,%u, expected "
                            "%u,%u,%u,%u", x, y, texel[0], texel[1], texel[2],
                            texel[3], expect[0], expect[1], expect[2],
                            expect[3]);
               }
               bad++;
               break;
            }
         }
      }
   }

   if (bad) {
      mesa_loge("SZ: selftest: %u bad texels", bad);
      failures++;
   } else {
      mesa_logi("SZ: selftest: clear/copy/readback %ux%u ok", w, h);
   }

out:
   pipe_resource_reference(&image, NULL);
   pipe_resource_reference(&readback, NULL);
   return failures;
}

/* SZ_DEBUG=selftest: render one triangle through the real Gallium entry points.
 *
 * The 1.C self-test proves recording and submitting work; this one proves a draw
 * reaches pixels. It goes through sz_draw_vbo, sz_renderpass_begin and
 * sz_pipeline_get rather than calling Vulkan directly, because those three are
 * where a bug produces no error at all -- just a black screen.
 *
 * The triangle covers exactly the left half of the target. Checking one texel on
 * each side separates "rasterised in the right place" from "something drew": a
 * full-screen triangle would pass even with the viewport transform inverted. */
unsigned
sz_render_selftest(struct sz_screen *screen)
{
   unsigned failures = 0;
   const unsigned w = 64, h = 64;

   /* Left half in NDC, as six vec4 vertices (two triangles). */
   static const float vertices[6][4] = {
      { -1.0f, -1.0f, 0.0f, 1.0f },
      {  0.0f, -1.0f, 0.0f, 1.0f },
      { -1.0f,  1.0f, 0.0f, 1.0f },
      {  0.0f, -1.0f, 0.0f, 1.0f },
      {  0.0f,  1.0f, 0.0f, 1.0f },
      { -1.0f,  1.0f, 0.0f, 1.0f },
   };

   /* Distinct from the pipeline colour in every channel, so a missing clear and
    * a missing draw cannot both look like a pass. */
   const union pipe_color_union clear_color = {
      .f = { 0.1f, 0.2f, 0.3f, 1.0f },
   };
   const uint8_t expect_clear[4] = { 26, 51, 77, 255 };
   const uint8_t expect_tri[4] = { 64, 128, 191, 255 };

   struct pipe_context *pipe = NULL;
   void *velems = NULL;
   struct pipe_resource *rt = NULL;
   struct pipe_resource *vbuf = NULL;
   struct pipe_resource *readback = NULL;

   const struct pipe_resource rt_tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
      .usage = PIPE_USAGE_DEFAULT,
      .width0 = w,
      .height0 = h,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
   };

   const struct pipe_resource vbuf_tmpl = {
      .target = PIPE_BUFFER,
      .format = PIPE_FORMAT_R8_UNORM,
      .usage = PIPE_USAGE_STAGING,
      .width0 = sizeof(vertices),
      .height0 = 1,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
   };

   const struct pipe_resource rb_tmpl = {
      .target = PIPE_BUFFER,
      .format = PIPE_FORMAT_R8_UNORM,
      .usage = PIPE_USAGE_STAGING,
      .width0 = w * h * 4,
      .height0 = 1,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
   };

   pipe = sz_context_create(screen, 0);
   if (!pipe) {
      mesa_loge("SZ: render selftest: could not create a context");
      return 1;
   }

   rt = screen->base.resource_create(&screen->base, &rt_tmpl);
   vbuf = screen->base.resource_create(&screen->base, &vbuf_tmpl);
   readback = screen->base.resource_create(&screen->base, &rb_tmpl);

   if (!rt || !vbuf || !readback) {
      mesa_loge("SZ: render selftest: could not create the resources");
      failures++;
      goto out;
   }

   memcpy(sz_resource_map(sz_resource(vbuf)), vertices, sizeof(vertices));
   sz_bo_flush(sz_resource(vbuf)->bo, 0, sizeof(vertices));

   const struct pipe_vertex_element elem = {
      .src_offset = 0,
      .vertex_buffer_index = 0,
      .src_format = PIPE_FORMAT_R32G32B32A32_FLOAT,
      .src_stride = sizeof(vertices[0]),
   };

   velems = pipe->create_vertex_elements_state(pipe, 1, &elem);
   if (!velems) {
      mesa_loge("SZ: render selftest: could not create vertex elements");
      failures++;
      goto out;
   }

   struct pipe_framebuffer_state fb = {
      .width = w,
      .height = h,
      .nr_cbufs = 1,
      .cbufs = {
         {
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .texture = rt,
            .level = 0,
            .first_layer = 0,
            .last_layer = 0,
         },
      },
   };

   pipe->set_framebuffer_state(pipe, &fb);
   pipe->bind_vertex_elements_state(pipe, velems);

   const struct pipe_vertex_buffer vb = {
      .is_user_buffer = false,
      .buffer_offset = 0,
      .buffer.resource = vbuf,
   };
   pipe->set_vertex_buffers(pipe, 1, &vb);

   const struct pipe_viewport_state vp = {
      .scale = { w / 2.0f, h / 2.0f, 0.5f },
      .translate = { w / 2.0f, h / 2.0f, 0.5f },
   };
   pipe->set_viewport_states(pipe, 0, 1, &vp);

   const struct pipe_scissor_state sc = {
      .minx = 0, .miny = 0, .maxx = w, .maxy = h,
   };
   pipe->set_scissor_states(pipe, 0, 1, &sc);

   pipe->clear(pipe, PIPE_CLEAR_COLOR, 0, 0, NULL, &clear_color, 0.0, 0);

   const struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLES,
      .instance_count = 1,
   };
   const struct pipe_draw_start_count_bias draw = {
      .start = 0,
      .count = ARRAY_SIZE(vertices),
   };
   pipe->draw_vbo(pipe, &info, 0, NULL, &draw, 1);

   pipe->flush(pipe, NULL, 0);
   sz_queue_wait_idle(&screen->batch_state);

   /* Reading the result back is a transfer on the image, so it is recorded
    * outside the context to keep the test independent of the transfer entry
    * points that Phase 1 has not implemented yet. */
   struct sz_resource *img = sz_resource(rt);
   struct sz_resource *rb = sz_resource(readback);

   struct sz_batch *batch = sz_batch_begin(&screen->batch_state);
   if (!batch) {
      failures++;
      goto out;
   }
   const VkCommandBuffer cmd = batch->cmd;

   sz_resource_access(screen, cmd, img, SZ_ACCESS_TRANSFER_SRC);

   const VkBufferImageCopy region = {
      .imageSubresource = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .imageExtent = { w, h, 1 },
   };

   screen->vk.CmdCopyImageToBuffer(cmd, img->image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   rb->buffer, 1, &region);

   sz_memory_barrier(screen, cmd, VK_PIPELINE_STAGE_2_COPY_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);

   if (!sz_batch_end(batch) ||
       !sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)) {
      failures++;
      goto out;
   }

   sz_queue_wait_idle(&screen->batch_state);
   sz_bo_invalidate(rb->bo, 0, sz_bo_size(rb->bo));

   const uint8_t *pixels = sz_resource_map(rb);
   if (!pixels) {
      mesa_loge("SZ: render selftest: the readback buffer is not host visible");
      failures++;
      goto out;
   }

   /* Inside the triangle's half, and one texel clear of the boundary. */
   const uint8_t *inside = pixels + (h / 2 * w + w / 4) * 4;
   const uint8_t *outside = pixels + (h / 2 * w + 3 * w / 4) * 4;

   if (!sz_rgba_close(inside, expect_tri, 2)) {
      mesa_loge("SZ: render selftest: inside the triangle is %u,%u,%u,%u, "
                "expected %u,%u,%u,%u", inside[0], inside[1], inside[2],
                inside[3], expect_tri[0], expect_tri[1], expect_tri[2],
                expect_tri[3]);
      failures++;
   }

   if (!sz_rgba_close(outside, expect_clear, 2)) {
      mesa_loge("SZ: render selftest: outside the triangle is %u,%u,%u,%u, "
                "expected %u,%u,%u,%u", outside[0], outside[1], outside[2],
                outside[3], expect_clear[0], expect_clear[1], expect_clear[2],
                expect_clear[3]);
      failures++;
   }

   if (failures == 0)
      mesa_logi("SZ: render selftest: draw %ux%u ok", w, h);

out:
   if (velems)
      pipe->delete_vertex_elements_state(pipe, velems);
   pipe_resource_reference(&rt, NULL);
   pipe_resource_reference(&vbuf, NULL);
   pipe_resource_reference(&readback, NULL);
   sz_context_destroy(pipe);
   return failures;
}

/* ------------------------------------------------------------------------- */
/* Entry point                                                                */
/* ------------------------------------------------------------------------- */

struct pipe_context *
sz_context_create(struct sz_screen *screen, unsigned flags)
{
   struct sz_context *ctx = CALLOC_STRUCT(sz_context);
   if (!ctx)
      return NULL;

   struct pipe_context *pipe = &ctx->base;
   pipe->screen = &screen->base;
   ctx->screen = screen;

   pipe->destroy = sz_context_destroy;
   pipe->flush = sz_context_flush;

   /* State objects. */
   pipe->create_blend_state = sz_create_blend_state;
   pipe->bind_blend_state = sz_bind_blend_state;
   pipe->delete_blend_state = sz_delete_state;

   pipe->create_sampler_state = sz_create_sampler_state;
   pipe->bind_sampler_states = sz_bind_sampler_states;
   pipe->delete_sampler_state = sz_delete_state;

   pipe->create_rasterizer_state = sz_create_rasterizer_state;
   pipe->bind_rasterizer_state = sz_bind_rasterizer_state;
   pipe->delete_rasterizer_state = sz_delete_state;

   pipe->create_depth_stencil_alpha_state = sz_create_depth_stencil_alpha_state;
   pipe->bind_depth_stencil_alpha_state = sz_bind_depth_stencil_alpha_state;
   pipe->delete_depth_stencil_alpha_state = sz_delete_state;

   pipe->create_fs_state = sz_create_fs_state;
   pipe->bind_fs_state = sz_bind_fs_state;
   pipe->delete_fs_state = sz_delete_state;

   pipe->create_vs_state = sz_create_vs_state;
   pipe->bind_vs_state = sz_bind_vs_state;
   pipe->delete_vs_state = sz_delete_state;

   pipe->create_gs_state = sz_create_gs_state;
   pipe->bind_gs_state = sz_bind_gs_state;
   pipe->delete_gs_state = sz_delete_state;

   pipe->create_tcs_state = sz_create_tcs_state;
   pipe->bind_tcs_state = sz_bind_tcs_state;
   pipe->delete_tcs_state = sz_delete_state;

   pipe->create_tes_state = sz_create_tes_state;
   pipe->bind_tes_state = sz_bind_tes_state;
   pipe->delete_tes_state = sz_delete_state;

   pipe->create_ts_state = sz_create_ts_state;
   pipe->bind_ts_state = sz_bind_ts_state;
   pipe->delete_ts_state = sz_delete_state;

   pipe->create_ms_state = sz_create_ms_state;
   pipe->bind_ms_state = sz_bind_ms_state;
   pipe->delete_ms_state = sz_delete_state;

   pipe->create_vertex_elements_state = sz_create_vertex_elements_state;
   pipe->bind_vertex_elements_state = sz_bind_vertex_elements_state;
   pipe->delete_vertex_elements_state = sz_delete_state;

   /* Parameter state. */
   pipe->set_blend_color = sz_set_blend_color;
   pipe->set_stencil_ref = sz_set_stencil_ref;
   pipe->set_sample_mask = sz_set_sample_mask;
   pipe->set_framebuffer_state = sz_set_framebuffer_state;
   pipe->set_viewport_states = sz_set_viewport_states;
   pipe->set_scissor_states = sz_set_scissor_states;
   pipe->set_sampler_views = sz_set_sampler_views;
   pipe->set_constant_buffer = sz_set_constant_buffer;
   pipe->set_inlinable_constants = sz_set_inlinable_constants;
   pipe->set_vertex_buffers = sz_set_vertex_buffers;
   pipe->set_shader_buffers = sz_set_shader_buffers;
   pipe->set_shader_images = sz_set_shader_images;
   pipe->set_clip_state = sz_set_clip_state;
   pipe->set_polygon_stipple = sz_set_polygon_stipple;
   pipe->set_tess_state = sz_set_tess_state;
   pipe->set_patch_vertices = sz_set_patch_vertices;
   pipe->set_stream_output_targets = sz_set_stream_output_targets;
   pipe->set_debug_callback = sz_set_debug_callback;

   /* Drawing. */
   pipe->draw_vbo = sz_draw_vbo;
   pipe->draw_vbo_buffers = sz_draw_vbo_buffers;
   pipe->draw_vertex_state = sz_draw_vertex_state;
   pipe->launch_grid = sz_launch_grid;

   /* Transfers, clears and copies. */
   pipe->clear = sz_clear;
   pipe->clear_render_target = sz_clear_render_target;
   pipe->clear_depth_stencil = sz_clear_depth_stencil;
   pipe->clear_texture = sz_clear_texture_entry;
   pipe->clear_buffer = sz_clear_buffer;
   pipe->resource_copy_region = sz_resource_copy_region;
   pipe->blit = sz_blit;
   pipe->flush_resource = sz_flush_resource;
   pipe->texture_subdata = sz_texture_subdata;
   pipe->buffer_subdata = sz_buffer_subdata;
   pipe->buffer_map = sz_buffer_map;
   pipe->buffer_unmap = sz_buffer_unmap;
   pipe->texture_map = sz_texture_map;
   pipe->texture_unmap = sz_texture_unmap;
   pipe->transfer_flush_region = sz_transfer_flush_region;
   pipe->texture_barrier = sz_texture_barrier;
   pipe->memory_barrier = sz_context_memory_barrier;
   pipe->resource_release = sz_resource_release;

   /* Sampler views. */
   pipe->create_sampler_view = sz_create_sampler_view;
   pipe->sampler_view_destroy = sz_sampler_view_destroy;
   pipe->sampler_view_release = sz_sampler_view_release;

   pipe->get_device_reset_status = sz_get_device_reset_status;

   /* Stream and constant uploads. Every driver creates a pair: u_vbuf, u_upload
    * users in st and the clear paths all dereference them unconditionally, and a
    * NULL here shows up as a crash inside upload code rather than as an error. */
   pipe->stream_uploader = u_upload_create_default(pipe);
   if (!pipe->stream_uploader) {
      mesa_loge("SZ: could not create the stream uploader");
      free(ctx);
      return NULL;
   }
   pipe->const_uploader = u_upload_create_default(pipe);
   if (!pipe->const_uploader) {
      mesa_loge("SZ: could not create the constant uploader");
      u_upload_destroy(pipe->stream_uploader);
      free(ctx);
      return NULL;
   }

   (void)flags;

   return pipe;
}
