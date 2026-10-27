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

#include "sz_batch.h"
#include "sz_debug.h"
#include "sz_format.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "util/format/u_format.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"

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

static void
sz_context_submit(struct sz_context *ctx)
{
   if (!ctx->batch)
      return;

   struct sz_batch *batch = ctx->batch;
   ctx->batch = NULL;

   if (!sz_batch_end(batch))
      return;

   sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
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
   if (flags & PIPE_FLUSH_END_OF_FRAME)
      sz_queue_wait_idle(&ctx->screen->batch_state);
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
   sz_state_free(opaque);
}

static void
sz_bind_state(struct pipe_context *pipe, void *opaque)
{
   /* Recording the binding is Phase 2's job: nothing reads it until a pipeline
    * is created. Keeping the entry point rather than leaving it NULL matters
    * because cso_context calls it unconditionally. */
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
   state->u.shader = *templ;

   /* The IR belongs to the state tracker; copying the struct must not make us a
    * co-owner of it. Nothing compiles it yet, so dropping the pointer is
    * correct and avoids a double free later. */
   state->u.shader.ir.nir = NULL;

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

   VkCommandBuffer cmd = sz_context_cmd(ctx);
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

   VkCommandBuffer cmd = sz_context_cmd(ctx);
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

   VkCommandBuffer cmd = sz_context_cmd(ctx);
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

   VkCommandBuffer cmd = sz_context_cmd(ctx);
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
   /* pipe_framebuffer_state holds the surfaces by value in this Mesa version, so
    * a struct copy is the whole update. */
   sz_context(pipe)->state.framebuffer = *fb;
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
}

static void
sz_set_vertex_buffers(struct pipe_context *pipe, unsigned count,
                      const struct pipe_vertex_buffer *buffers)
{
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

static void
sz_draw_vbo(struct pipe_context *pipe, const struct pipe_draw_info *info,
            unsigned drawid_offset,
            const struct pipe_draw_indirect_info *indirect,
            const struct pipe_draw_start_count_bias *draws, unsigned num_draws)
{
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

   VkCommandBuffer cmd = sz_context_cmd(ctx);
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

static void
sz_texture_subdata(struct pipe_context *pipe, struct pipe_resource *res,
                   unsigned level, unsigned usage, const struct pipe_box *box,
                   const void *data, unsigned stride, uintptr_t layer_stride)
{
}

static void
sz_buffer_subdata(struct pipe_context *pipe, struct pipe_resource *res,
                  unsigned usage, unsigned offset, unsigned size,
                  const void *data)
{
}

static void *
sz_buffer_map(struct pipe_context *pipe, struct pipe_resource *res,
              unsigned level, unsigned usage, const struct pipe_box *box,
              struct pipe_transfer **transfer)
{
   return NULL;
}

static void
sz_buffer_unmap(struct pipe_context *pipe, struct pipe_transfer *transfer)
{
}

static void *
sz_texture_map(struct pipe_context *pipe, struct pipe_resource *res,
               unsigned level, unsigned usage, const struct pipe_box *box,
               struct pipe_transfer **transfer)
{
   return NULL;
}

static void
sz_texture_unmap(struct pipe_context *pipe, struct pipe_transfer *transfer)
{
}

static void
sz_transfer_flush_region(struct pipe_context *pipe,
                         struct pipe_transfer *transfer,
                         const struct pipe_box *box)
{
}

static void
sz_texture_barrier(struct pipe_context *pipe, unsigned flags)
{
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

   sz_context_submit(ctx);
   sz_queue_wait_idle(&ctx->screen->batch_state);

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
   pipe->bind_blend_state = sz_bind_state;
   pipe->delete_blend_state = sz_delete_state;

   pipe->create_sampler_state = sz_create_sampler_state;
   pipe->bind_sampler_states = sz_bind_sampler_states;
   pipe->delete_sampler_state = sz_delete_state;

   pipe->create_rasterizer_state = sz_create_rasterizer_state;
   pipe->bind_rasterizer_state = sz_bind_state;
   pipe->delete_rasterizer_state = sz_delete_state;

   pipe->create_depth_stencil_alpha_state = sz_create_depth_stencil_alpha_state;
   pipe->bind_depth_stencil_alpha_state = sz_bind_state;
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
   pipe->bind_vertex_elements_state = sz_bind_state;
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

   /* Sampler views. */
   pipe->create_sampler_view = sz_create_sampler_view;
   pipe->sampler_view_destroy = sz_sampler_view_destroy;
   pipe->sampler_view_release = sz_sampler_view_release;

   pipe->get_device_reset_status = sz_get_device_reset_status;

   (void)flags;

   return pipe;
}
