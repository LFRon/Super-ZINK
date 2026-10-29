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
#include "sz_blit.h"
#include "sz_compiler.h"
#include "sz_debug.h"
#include "sz_format.h"
#include "sz_pipeline.h"
#include "sz_query.h"
#include "sz_renderpass.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "compiler/nir/nir.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/u_atomic.h"
#include "util/u_inlines.h"
#include "util/u_surface.h"
#include "util/u_upload_mgr.h"
#include "vk_format.h"

#include <string.h>

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
VkCommandBuffer
sz_context_cmd(struct sz_context *ctx)
{
   if (!ctx->batch) {
      ctx->batch = sz_batch_begin(&ctx->screen->batch_state);
      if (!ctx->batch)
         return VK_NULL_HANDLE;

      /* A new command buffer inherits none of the previous one's bindings, so
       * every piece of state has to be re-emitted before the next draw. This is
       * the only place where the dirty mask is set wholesale. */
      ctx->dirty = SZ_DIRTY_ALL;
      ctx->pipeline = NULL;
      /* The descriptor buffer belongs to this batch: a slot may only be
       * re-recorded once its submission finished, so the bump pointer can start
       * from zero again at exactly the moment a new command buffer takes it
       * over. Descriptor bindings do not carry into a new command buffer
       * either, which is why sz_descriptor_push() re-binds on every batch. */
      const unsigned slot = ctx->batch->ring_index;
      if (slot < SZ_BATCH_RING_SIZE)
         ctx->screen->db.ring[slot].used = 0;
      /* Descriptor set bindings are per command buffer, so the "already bound"
       * memo has to be cleared with the rest of the inherited state. */
      ctx->screen->bound_sampler_set = VK_NULL_HANDLE;
   }
   return ctx->batch->cmd;
}

/* Common prologue of every state setter: close the draw batch recorded under the
 * old state, mark what has to be re-emitted, and advance the epoch (ADR-005b). */
static void
sz_state_changed(struct sz_context *ctx, unsigned dirty)
{
   sz_draw_flush(ctx);
   ctx->dirty |= dirty;
}

/* For commands Vulkan forbids inside a render pass: clears, copies and
 * pipeline barriers. Ends the pass first, and the next draw starts a fresh one --
 * which costs a begin/end pair per clear/copy interleaved with draws, and is the
 * price of not having to prove that a clear never lands between two draws in
 * Phase 1. */
VkCommandBuffer
sz_context_cmd_no_pass(struct sz_context *ctx)
{
   VkCommandBuffer cmd = sz_context_cmd(ctx);
   if (cmd != VK_NULL_HANDLE && ctx->rendering)
      sz_renderpass_end(ctx, cmd);
   return cmd;
}

void
sz_context_submit(struct sz_context *ctx)
{
   /* Draws are queued, not recorded, so a submission has to drain them even when
    * no command buffer has been acquired yet. */
   sz_draw_flush(ctx);

   if (!ctx->batch)
      return;

   /* A render pass cannot outlive the command buffer's recording. Ending it
    * here rather than in every caller means no path can forget. */
   sz_renderpass_end(ctx, ctx->batch->cmd);

   struct sz_batch *batch = ctx->batch;
   ctx->batch = NULL;

   if (!sz_batch_end(batch))
      return;

   if (sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)) {
      ctx->screen->stats.submits++;
      sz_pipeline_note_submit(ctx);
   }
}

static void
sz_context_flush(struct pipe_context *pipe, struct pipe_fence_handle **fence,
                 unsigned flags)
{
   struct sz_context *ctx = sz_context(pipe);

   sz_trace("flush flags=0x%x fence=%d", flags, fence != NULL);

   sz_context_submit(ctx);

   /* The fence names the value of the submission that carried the work just
    * issued, which is only known once the submit has happened: submitting is
    * what assigns the value.
    *
    * It used to be taken beforehand as next_value + 1. That is wrong whenever
    * there is no batch to submit -- a glFenceSync after the draws have already
    * been submitted, which is exactly what GL_ARB_sync is used for -- because
    * no submission then takes that value, and the fence waits for the next one
    * that does. If the application stops issuing work and waits on the fence,
    * nothing ever signals it and the wait never returns: glClientWaitSync,
    * glFinish and glMapBuffer after a draw all deadlock. */
   if (fence)
      *fence = sz_fence_create(ctx->screen, ctx->screen->batch_state.next_value);

   if (flags & PIPE_FLUSH_HINT_FINISH)
      sz_queue_wait_idle(&ctx->screen->batch_state);

   /* The winsys asks for this before handing a buffer to the display engine.
    * Without it the compositor can sample a surface the GPU is still writing. */
   if (flags & PIPE_FLUSH_END_OF_FRAME) {
      ctx->screen->stats.end_of_frame++;
      sz_queue_wait_idle(&ctx->screen->batch_state);

      /* SZ_DEBUG=stats: a periodic snapshot, because an app killed with
       * SIGTERM never reaches context destroy. */
      if (sz_debug_enabled(SZ_DEBUG_STATS)) {
         static unsigned frames;
         if (++frames % 120 == 0) {
            const struct sz_screen *screen = ctx->screen;
            mesa_logi("SZ: stats: draws=%" PRIu64 " batches=%" PRIu64
                      " (multi %" PRIu64 ") pipeline_binds=%" PRIu64
                      " pipeline_builds=%" PRIu64 " descriptor_pushes=%" PRIu64
                      " vb_binds=%" PRIu64 " state_sets=%" PRIu64
                      " submits=%" PRIu64,
                      screen->stats.draws, screen->stats.batches,
                      screen->stats.multi_batches, screen->stats.pipeline_binds,
                      screen->stats.pipeline_builds,
                      screen->stats.descriptor_pushes,
                      screen->stats.vertex_buffer_binds,
                      screen->stats.state_sets, screen->stats.submits);
            mesa_logi("SZ: stats: blits=%" PRIu64 " (resolves %" PRIu64
                      ", copies %" PRIu64 ") end_of_frame=%" PRIu64,
                      screen->stats.blits, screen->stats.resolves,
                      screen->stats.copies, screen->stats.end_of_frame);
         }
      }

      /* The queue just went idle, so every retired resource whose submission has
       * completed can be torn down. Doing it here rather than only on the next
       * destroy keeps the table from filling up during a long run. */
      sz_resource_reclaim(ctx->screen);
   }
}

/* ------------------------------------------------------------------------- */
/* State objects                                                              */
/* ------------------------------------------------------------------------- */

static uint64_t sz_state_id;

static struct sz_state *
sz_state_alloc(enum sz_state_kind kind)
{
   struct sz_state *state = CALLOC_STRUCT(sz_state);
   if (state) {
      state->kind = kind;
      state->id = p_atomic_inc_return(&sz_state_id);
   }
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
      /* A VkSampler is immutable, so the object owns it outright. Destroying it
       * while a submission still references it is allowed -- samplers are not
       * pipeline state that a command buffer latches -- but the push that
       * recorded it has to have happened, which it did at draw time. */
      if (state->u.sampler.sampler != VK_NULL_HANDLE)
         ctx->screen->vk.DestroySampler(ctx->screen->dev,
                                        state->u.sampler.sampler, NULL);
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
   sz_state_changed(ctx, SZ_DIRTY_PIPELINE);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.blend = NULL;
}

static void
sz_bind_rasterizer_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   /* The scissor *enable* lives in the rasterizer state, and Vulkan has no way
    * to disable a scissor rectangle: the last one stays in effect until it is
    * replaced. Turning GL_SCISSOR_TEST off therefore has to re-emit the full
    * surface scissor. */
   sz_state_changed(ctx, SZ_DIRTY_PIPELINE | SZ_DIRTY_SCISSOR);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.rasterizer = NULL;
}

static void
sz_bind_depth_stencil_alpha_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   sz_state_changed(ctx, SZ_DIRTY_PIPELINE);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.depth_stencil_alpha = NULL;
}

static void
sz_bind_vertex_elements_state(struct pipe_context *pipe, void *opaque)
{
   struct sz_context *ctx = sz_context(pipe);
   sz_state_changed(ctx, SZ_DIRTY_PIPELINE);
   if (opaque)
      sz_bind_typed(ctx, opaque);
   else
      ctx->state.vertex_elements = NULL;
}

/* ------------------------------------------------------------------------- */
/* Samplers                                                                   */
/* ------------------------------------------------------------------------- */

static VkFilter
sz_tex_filter(unsigned filter)
{
   return filter == PIPE_TEX_FILTER_NEAREST ? VK_FILTER_NEAREST
                                            : VK_FILTER_LINEAR;
}

static VkSamplerMipmapMode
sz_mipmap_mode(unsigned filter)
{
   /* PIPE_TEX_MIPFILTER_NONE still needs a mode: it is the minFilter that
    * decides whether mips are used at all, and picking a mode that the sampler
    * never reaches costs nothing. */
   return filter == PIPE_TEX_MIPFILTER_NEAREST
             ? VK_SAMPLER_MIPMAP_MODE_NEAREST
             : VK_SAMPLER_MIPMAP_MODE_LINEAR;
}

static VkSamplerAddressMode
sz_address_mode(enum pipe_tex_wrap wrap)
{
   switch (wrap) {
   case PIPE_TEX_WRAP_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_REPEAT;
   case PIPE_TEX_WRAP_CLAMP:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
      return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
   case PIPE_TEX_WRAP_MIRROR_REPEAT:
      return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
   case PIPE_TEX_WRAP_MIRROR_CLAMP:
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:
      return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER:
      /* Vulkan has no mirrored clamp-to-border; clamped edge is the closest
       * thing that exists and only differs outside the texture. */
      return VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE;
   }
   return VK_SAMPLER_ADDRESS_MODE_REPEAT;
}

/* Vulkan predefines three border colours unless VK_EXT_custom_border_color is
 * enabled. Gallium passes the colour as four floats, so the three legal values
 * are recognised by value and anything else falls back to transparent black with
 * a warning -- a wrong border colour is visible, a silently unsupported one is
 * worse. */
static VkBorderColor
sz_border_color(const union pipe_color_union *color)
{
   const float *c = color->f;
   const bool zero_rgb = c[0] == 0.0f && c[1] == 0.0f && c[2] == 0.0f;
   const bool one_rgb = c[0] == 1.0f && c[1] == 1.0f && c[2] == 1.0f;

   if (zero_rgb && c[3] == 0.0f)
      return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
   if (zero_rgb && c[3] == 1.0f)
      return VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
   if (one_rgb && c[3] == 1.0f)
      return VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;

   sz_warn_once("custom border colours are not supported; using "
                "transparent black");
   return VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
}

static VkSampler
sz_sampler_create(struct sz_screen *screen,
                  const struct pipe_sampler_state *templ)
{
   const bool compare = templ->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE;
   const float max_anisotropy = MAX2(templ->max_anisotropy, 1u);

   /* A rectangle texture addresses in texel units, which GL spells as
    * pipe_sampler_state::unnormalized_coords and Vulkan as
    * VkSamplerCreateInfo::unnormalizedCoordinates. The two are the same feature,
    * so this is a pass-through -- but Vulkan attaches conditions to it that a
    * normalised sampler is free of, and a violation is a validation error rather
    * than a slower path. Rather than reject the sampler, the conditions are
    * imposed: they are exactly what GL requires of a rectangle texture anyway
    * (clamp addressing, no mipmaps), so nothing a conformant application can
    * observe changes.
    *
    * The one case that cannot be expressed is a rect *shadow* sampler: Vulkan
    * forbids compareEnable together with unnormalizedCoordinates. That stays a
    * warning, because silently dropping the comparison would read the wrong
    * channel. */
   const bool unnormalized =
      templ->unnormalized_coords && !compare;

   if (templ->unnormalized_coords && compare)
      sz_warn_once("a shadow sampler with unnormalised coordinates is not "
                   "implementable in Vulkan");

   const VkSamplerCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = sz_tex_filter(templ->mag_img_filter),
      .minFilter = unnormalized ? sz_tex_filter(templ->mag_img_filter)
                                : sz_tex_filter(templ->min_img_filter),
      .mipmapMode = unnormalized ? VK_SAMPLER_MIPMAP_MODE_NEAREST
                                 : sz_mipmap_mode(templ->min_mip_filter),
      .addressModeU = unnormalized ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                                   : sz_address_mode(templ->wrap_s),
      .addressModeV = unnormalized ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                                   : sz_address_mode(templ->wrap_t),
      .addressModeW = unnormalized ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                                   : sz_address_mode(templ->wrap_r),
      .mipLodBias = unnormalized ? 0.0f : templ->lod_bias,
      .anisotropyEnable = !unnormalized && max_anisotropy > 1.0f &&
                          screen->caps.core_features.samplerAnisotropy,
      .maxAnisotropy = max_anisotropy,
      .compareEnable = compare,
      .compareOp = compare ? sz_compare_op(templ->compare_func)
                           : VK_COMPARE_OP_NEVER,
      .minLod = unnormalized ? 0.0f : templ->min_lod,
      .maxLod = unnormalized ? 0.0f : templ->max_lod,
      .borderColor = sz_border_color(&templ->border_color),
      .unnormalizedCoordinates = unnormalized,
   };

   VkSampler sampler = VK_NULL_HANDLE;
   VkResult result = screen->vk.CreateSampler(screen->dev, &ci, NULL, &sampler);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateSampler failed: %d", result);
      return VK_NULL_HANDLE;
   }

   return sampler;
}

static void *
sz_create_sampler_state(struct pipe_context *pipe,
                        const struct pipe_sampler_state *templ)
{
   struct sz_state *state = sz_state_alloc(SZ_STATE_SAMPLER);
   if (!state)
      return NULL;

   state->u.sampler.base = *templ;
   state->u.sampler.sampler =
      sz_sampler_create(sz_context(pipe)->screen, templ);

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
   /* The interpreter of this description lives in the translator: capture
    * placement has to become SPIR-V decorations, so it is needed at shader
    * translation time rather than at draw time. */
   state->u.shader.stream_output = templ->stream_output;

   /* The IR belongs to the state tracker, which frees it independently of us, so
    * the driver needs its own copy: a shader object outlives the one call that
    * created it by design (cso_context keeps the bound object and rebinds it).
    * Pointing at the caller's copy works until the caller frees it, which happens
    * on the very next program change. */
   if (templ->type != PIPE_SHADER_IR_NIR || !templ->ir.nir) {
      sz_warn_once("only NIR shaders are supported");
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

   struct sz_screen *screen = sz_context(pipe)->screen;
   const struct sz_translate_options opts = {
      .alias_1d_shadow = screen->need_2d_zs,
      .clip_distance = screen->have_shader_clip_distance,
      .cull_distance = screen->have_shader_cull_distance,
   };

   char *error = NULL;
   if (!sz_nir_to_spirv(state->u.shader.nir, &state->u.shader.stream_output,
                        &opts, &state->u.shader.spirv,
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
   struct sz_screen *screen = sz_context(pipe)->screen;

   if (sv->view != VK_NULL_HANDLE)
      screen->vk.DestroyImageView(screen->dev, sv->view, NULL);
   if (sv->buffer_view != VK_NULL_HANDLE)
      screen->vk.DestroyBufferView(screen->dev, sv->buffer_view, NULL);

   pipe_resource_reference(&sv->base.texture, NULL);
   free(sv);
}

static VkImageViewType
sz_view_type(enum pipe_texture_target target, unsigned array_size)
{
   switch (target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      /* An aliased 1D depth texture is a 2D image, so its sampler view is 2D;
       * sz_view_type's caller knows the resource, not just the target, so the
       * decision is taken there. */
      return target == PIPE_TEXTURE_1D ? VK_IMAGE_VIEW_TYPE_1D
                                       : VK_IMAGE_VIEW_TYPE_1D_ARRAY;
   case PIPE_TEXTURE_2D:
      return array_size > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                            : VK_IMAGE_VIEW_TYPE_2D;
   case PIPE_TEXTURE_2D_ARRAY:
      return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
   case PIPE_TEXTURE_CUBE:
      return VK_IMAGE_VIEW_TYPE_CUBE;
   case PIPE_TEXTURE_CUBE_ARRAY:
      return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
   case PIPE_TEXTURE_3D:
      return VK_IMAGE_VIEW_TYPE_3D;
   case PIPE_TEXTURE_RECT:
      return VK_IMAGE_VIEW_TYPE_2D;
   default:
      return VK_IMAGE_VIEW_TYPE_2D;
   }
}

static VkComponentSwizzle
sz_component_swizzle(unsigned swizzle)
{
   switch (swizzle) {
   case PIPE_SWIZZLE_X:
      return VK_COMPONENT_SWIZZLE_R;
   case PIPE_SWIZZLE_Y:
      return VK_COMPONENT_SWIZZLE_G;
   case PIPE_SWIZZLE_Z:
      return VK_COMPONENT_SWIZZLE_B;
   case PIPE_SWIZZLE_W:
      return VK_COMPONENT_SWIZZLE_A;
   case PIPE_SWIZZLE_0:
      return VK_COMPONENT_SWIZZLE_ZERO;
   case PIPE_SWIZZLE_1:
      return VK_COMPONENT_SWIZZLE_ONE;
   case PIPE_SWIZZLE_NONE:
   default:
      return VK_COMPONENT_SWIZZLE_IDENTITY;
   }
}

static void
sz_sampler_view_release(struct pipe_context *pipe,
                        struct pipe_sampler_view *view)
{
   /* Drops the reference the frontend held. It is not the same thing as an
    * unbind -- no state is touched -- and a no-op here leaks the view and, with
    * it, the reference it holds on its texture: the validation layer reports
    * that as a VkImage still alive when the device is destroyed. */
   struct pipe_sampler_view *v = view;
   pipe_sampler_view_reference(&v, NULL);
}

static struct pipe_sampler_view *
sz_create_sampler_view(struct pipe_context *pipe, struct pipe_resource *texture,
                       const struct pipe_sampler_view *templ)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_screen *screen = ctx->screen;
   struct sz_resource *res = sz_resource(texture);

   const VkFormat format = sz_vk_format(screen, templ->format);
   if (format == VK_FORMAT_UNDEFINED) {
      /* A view in a format the device cannot read would be rejected by
       * vkCreateImageView with an error that says nothing about which texture
       * asked for it. */
      sz_warn_once("sampler view format %u has no Vulkan equivalent",
                   (unsigned)templ->format);
      return NULL;
   }

   struct sz_sampler_view *sv = CALLOC_STRUCT(sz_sampler_view);
   if (!sv)
      return NULL;

   sv->base = *templ;
   sv->base.context = pipe;
   /* The frontend deliberately does not clear its template -- pipe_sampler_view's
    * texture and context are the driver's own storage, and zeroing them is pure
    * overhead on a hot path -- so both have to be reset here. The copy above
    * leaves stack garbage in them, and referencing that garbage segfaults. */
   sv->base.texture = NULL;
   pipe_resource_reference(&sv->base.texture, texture);
   /* The template's own reference count is whatever the frontend left on its
    * stack: it is a recipe for a view, not a view. Copying it over means the
    * release never reaches zero, so every sampler view leaks -- along with the
    * reference it holds on the texture, which then never gets destroyed and is
    * reported by the validation layer as a VkImage leaked at device teardown. */
   pipe_reference_init(&sv->base.reference, 1);

   /* A buffer texture is a window onto a VkBuffer: a VkBufferView carries the
    * format and the byte range, and a UNIFORM_TEXEL_BUFFER binding takes it
    * directly. GL's glTexBufferRange is exactly the offset+size pair here. */
   if (texture->target == PIPE_BUFFER) {
      const VkBufferViewCreateInfo bv = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
         .buffer = res->buffer,
         .format = format,
         .offset = templ->u.buf.offset,
         .range = templ->u.buf.size ? templ->u.buf.size : VK_WHOLE_SIZE,
      };
      const VkResult r =
         screen->vk.CreateBufferView(screen->dev, &bv, NULL, &sv->buffer_view);
      if (r != VK_SUCCESS) {
         /* A format that cannot be a texel buffer is not a format this driver can
          * offer for a buffer texture; is_format_supported() says so, so reaching
          * here means the application skipped that check. */
         mesa_loge("SZ: vkCreateBufferView(%s) failed: %d",
                   util_format_name(templ->format), r);
         sv->buffer_view = VK_NULL_HANDLE;
      }
      return &sv->base;
   }

   if (res->image == VK_NULL_HANDLE)
      return &sv->base;

   const unsigned first_level = templ->u.tex.first_level;
   const unsigned last_level = templ->u.tex.last_level;
   const unsigned first_layer = templ->u.tex.first_layer;
   const unsigned last_layer = templ->u.tex.last_layer;

   const VkImageViewCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = res->image,
      .viewType = res->aliased_2d
                     ? (templ->target == PIPE_TEXTURE_1D
                           ? VK_IMAGE_VIEW_TYPE_2D
                           : VK_IMAGE_VIEW_TYPE_2D_ARRAY)
                     : sz_view_type(templ->target, texture->array_size),
      .format = format,
      .components = {
         .r = sz_component_swizzle(templ->swizzle_r),
         .g = sz_component_swizzle(templ->swizzle_g),
         .b = sz_component_swizzle(templ->swizzle_b),
         .a = sz_component_swizzle(templ->swizzle_a),
      },
      .subresourceRange = {
         .aspectMask = sz_format_aspect(templ->format),
         .baseMipLevel = first_level,
         .levelCount = last_level > first_level ? last_level - first_level + 1
                                                : 1,
         .baseArrayLayer = first_layer,
         .layerCount = last_layer > first_layer ? last_layer - first_layer + 1
                                                : 1,
      },
   };

   const VkResult result =
      screen->vk.CreateImageView(screen->dev, &ci, NULL, &sv->view);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateImageView(sampler view) failed: %d", result);
      sv->view = VK_NULL_HANDLE;
   }

   return &sv->base;
}

/* ------------------------------------------------------------------------- */
/* Clears                                                                     */
/* ------------------------------------------------------------------------- */

/* Clears a rectangle of one of the current framebuffer's attachments.
 *
 * vkCmdClearColorImage can only clear a whole subresource, so a scissored
 * glClear or a glClearTexSubImage has nowhere to go through the transfer path --
 * they go through vkCmdClearAttachments instead, which is a rectangle inside a
 * render pass. The pass is left open: a clear is normally followed by draws that
 * belong in the same one. */
static bool
sz_clear_rect(struct sz_context *ctx, unsigned attachment, bool depth_stencil,
              VkImageAspectFlags aspects, const VkClearValue *value,
              unsigned x, unsigned y, unsigned width, unsigned height)
{
   /* The clear belongs after everything recorded so far and before everything
    * that follows, so the queued draws have to be emitted first. */
   sz_draw_flush(ctx);

   VkCommandBuffer cmd = sz_context_cmd(ctx);
   if (cmd == VK_NULL_HANDLE)
      return false;

   if (!ctx->rendering && !sz_renderpass_begin(ctx, cmd))
      return false;

   VkClearAttachment clear = {
      .aspectMask = depth_stencil ? aspects : VK_IMAGE_ASPECT_COLOR_BIT,
      .colorAttachment = depth_stencil ? 0 : attachment,
      .clearValue = *value,
   };

   const VkClearRect rect = {
      .rect = {
         .offset = { (int32_t)x, (int32_t)y },
         .extent = { width, height },
      },
      .baseArrayLayer = 0,
      .layerCount = 1,
   };

   ctx->screen->vk.CmdClearAttachments(cmd, 1, &clear, 1, &rect);
   return true;
}

/* True when `box` covers the whole of `level`'s extent, which is the only case
 * vkCmdClearColorImage can express: it clears a whole subresource. Anything
 * smaller goes through sz_clear_rect() instead. */
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
      const VkClearValue value = { .color = {
         .float32 = { color->f[0], color->f[1], color->f[2], color->f[3] } } };
      for (unsigned i = 0; i < ctx->state.framebuffer.nr_cbufs; i++) {
         if (&ctx->state.framebuffer.cbufs[i] != dst)
            continue;
         sz_clear_rect(ctx, i, false, 0, &value, dstx, dsty, width, height);
         return;
      }

      static bool warned;
      if (!warned) {
         mesa_logw("SZ: cannot clear a region of a surface that is not bound; "
                   "ignoring the clear");
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

   const VkClearDepthStencilValue clear = {
      .depth = (float)depth,
      .stencil = stencil,
   };

   if (!sz_box_covers_surface(&res->base, dst->level, dstx, dsty, width, height)) {
      const VkClearValue value = { .depthStencil = clear };
      if (dst != &ctx->state.framebuffer.zsbuf) {
         static bool warned;
         if (!warned) {
            mesa_logw("SZ: cannot clear a region of a surface that is not bound; "
                      "ignoring the clear");
            warned = true;
         }
         return;
      }

      sz_clear_rect(ctx, 0, true, aspect, &value, dstx, dsty, width, height);
      return;
   }

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   sz_resource_access(ctx->screen, cmd, res, SZ_ACCESS_TRANSFER_DST);

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

   /* glClear honours the scissor. Each buffer is cleared through
    * vkCmdClearAttachments, which takes the rectangle directly, so a scissored
    * clear of a small damaged region stays a small clear. */
   if (scissor_state) {
      unsigned x0 = scissor_state->minx;
      unsigned y0 = scissor_state->miny;
      unsigned x1 = MIN2(scissor_state->maxx, fb->width);
      unsigned y1 = MIN2(scissor_state->maxy, fb->height);
      if (x1 <= x0 || y1 <= y0)
         return;

      if (buffers & PIPE_CLEAR_COLOR) {
         const VkClearValue value = { .color = {
            .float32 = { color->f[0], color->f[1], color->f[2], color->f[3] } } };
         for (unsigned i = 0; i < fb->nr_cbufs; i++) {
            if (!fb->cbufs[i].texture)
               continue;
            sz_clear_rect(ctx, i, false, 0, &value, x0, y0, x1 - x0, y1 - y0);
         }
      }

      if ((buffers & (PIPE_CLEAR_DEPTH | PIPE_CLEAR_STENCIL)) &&
          fb->zsbuf.texture) {
         const struct util_format_description *desc =
            util_format_description(fb->zsbuf.format);
         VkImageAspectFlags aspect = 0;
         if ((buffers & PIPE_CLEAR_DEPTH) && util_format_has_depth(desc))
            aspect |= VK_IMAGE_ASPECT_DEPTH_BIT;
         if ((buffers & PIPE_CLEAR_STENCIL) && util_format_has_stencil(desc))
            aspect |= VK_IMAGE_ASPECT_STENCIL_BIT;
         if (aspect != 0) {
            const VkClearValue value = { .depthStencil = {
               .depth = (float)depth, .stencil = stencil } };
            sz_clear_rect(ctx, 0, true, aspect, &value, x0, y0, x1 - x0, y1 - y0);
         }
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

   /* Draws issued before the copy are still queued, not recorded. Recording the
    * copy first would put it in front of them in the command buffer, so those
    * draws would read the destination before the copy wrote it -- the same
    * ordering mistake the upload path had. */
   sz_draw_flush(ctx);

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
      sz_resource_access(ctx->screen, cmd, ssrc, SZ_ACCESS_TRANSFER_SRC);
      sz_resource_access(ctx->screen, cmd, sdst, SZ_ACCESS_TRANSFER_DST);

      /* VkBufferCopy offsets are relative to the VkBuffer, which was already
       * bound at sz_bo_offset() when it was created. Adding the suballocation
       * offset again moves the copy by however far into the shared block the
       * buffer happens to sit -- glCopyBufferSubData then writes someone else's
       * bytes and the destination keeps whatever it had. */
      const VkBufferCopy region = {
         .srcOffset = src_box->x,
         .dstOffset = dstx,
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
   struct sz_context *ctx = sz_context(pipe);
   sz_state_changed(ctx, SZ_DIRTY_BLEND_CONST);
   ctx->state.blend_color = *c;
}

static void
sz_set_stencil_ref(struct pipe_context *pipe, const struct pipe_stencil_ref ref)
{
   struct sz_context *ctx = sz_context(pipe);
   sz_state_changed(ctx, SZ_DIRTY_STENCIL_REF);
   ctx->state.stencil_ref = ref;
}

static void
sz_set_sample_mask(struct pipe_context *pipe, unsigned mask)
{
   struct sz_context *ctx = sz_context(pipe);
   sz_state_changed(ctx, SZ_DIRTY_PIPELINE);
   ctx->state.sample_mask = mask;
}

static void
sz_set_framebuffer_state(struct pipe_context *pipe,
                         const struct pipe_framebuffer_state *fb)
{
   struct sz_context *ctx = sz_context(pipe);
   const struct pipe_framebuffer_state *cur = &ctx->state.framebuffer;

   /* The pending batch belongs to the old attachments. */
   sz_draw_flush(ctx);

   /* A dynamic rendering instance names concrete views, so it cannot be
    * retargeted: changing the attachments mid-pass has to close it. */
   bool changed = cur->nr_cbufs != fb->nr_cbufs || cur->width != fb->width ||
                  cur->height != fb->height ||
                  cur->zsbuf.texture != fb->zsbuf.texture;

   for (unsigned i = 0; !changed && i < fb->nr_cbufs; i++)
      changed = cur->cbufs[i].texture != fb->cbufs[i].texture ||
                cur->cbufs[i].level != fb->cbufs[i].level ||
                cur->cbufs[i].first_layer != fb->cbufs[i].first_layer ||
                cur->cbufs[i].last_layer != fb->cbufs[i].last_layer;

   if (!changed)
      changed = cur->zsbuf.texture != fb->zsbuf.texture ||
                cur->zsbuf.level != fb->zsbuf.level ||
                cur->zsbuf.first_layer != fb->zsbuf.first_layer ||
                cur->zsbuf.last_layer != fb->zsbuf.last_layer;

   if (changed && ctx->rendering) {
      VkCommandBuffer cmd = sz_context_cmd(ctx);
      if (cmd != VK_NULL_HANDLE)
         sz_renderpass_end(ctx, cmd);
   }

   /* The cached attachment views are *not* dropped here. A view a command buffer
    * already recorded keeps being read until that submission retires, and the
    * cache is keyed on the subresource anyway, so switching framebuffers merely
    * looks up a different entry. They are destroyed with the context. */

   /* pipe_framebuffer_state holds the surfaces by value in this Mesa version, so
    * a struct copy is the whole update. Formats are part of the pipeline, so the
    * dirty mask is wholesale. */
   ctx->state.framebuffer = *fb;
   ctx->dirty |= SZ_DIRTY_ALL;
}

static void
sz_set_viewport_states(struct pipe_context *pipe, unsigned start_slot,
                       unsigned num_viewports,
                       const struct pipe_viewport_state *viewports)
{
   struct sz_context *ctx = sz_context(pipe);

   sz_state_changed(ctx, SZ_DIRTY_VIEWPORT);
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

   sz_state_changed(ctx, SZ_DIRTY_SCISSOR);
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

   sz_state_changed(ctx, SZ_DIRTY_DESCRIPTORS);

   /* The views are reference counted and the caller keeps its own references,
    * so the context needs one of its own: storing the pointer alone leaves it
    * dangling the moment the state tracker drops the view, and nothing ever
    * destroys the last reference -- which shows up as a VkImage leaked at device
    * teardown, because a view that is never destroyed never releases its
    * texture. */
   for (unsigned i = 0; i < num_views; i++) {
      const unsigned slot = start_slot + i;
      if (slot >= PIPE_MAX_SAMPLERS)
         break;
      pipe_sampler_view_reference(&ctx->state.sampler_views[shader][slot],
                                  views ? views[i] : NULL);
   }

   for (unsigned i = 0; i < unbind_num_trailing_slots; i++) {
      const unsigned slot = start_slot + num_views + i;
      if (slot >= PIPE_MAX_SAMPLERS)
         break;
      pipe_sampler_view_reference(&ctx->state.sampler_views[shader][slot], NULL);
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

   sz_state_changed(ctx, SZ_DIRTY_DESCRIPTORS);
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
   sz_state_changed(ctx, SZ_DIRTY_PIPELINE);
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
   struct sz_screen *screen = ctx->screen;

   if (shader >= MESA_SHADER_STAGES || index >= SZ_MAX_CONST_BUFFERS) {
      sz_warn_once("ignoring a constant buffer outside the supported range");
      return;
   }

   struct pipe_constant_buffer *dst = &ctx->state.constant_buffers[shader][index];

   if (sz_debug_enabled(SZ_DEBUG_TRACE))
      sz_trace_("constbuf stage=%d index=%u buffer=%p user=%p size=%u off=%u",
                (int)shader, index, (void *)(buf ? buf->buffer : NULL),
                buf ? buf->user_buffer : NULL, buf ? buf->buffer_size : 0,
                buf ? buf->buffer_offset : 0);

   sz_state_changed(ctx, SZ_DIRTY_DESCRIPTORS);

   if (!buf || (!buf->buffer && !buf->user_buffer)) {
      pipe_resource_reference(&dst->buffer, NULL);
      dst->buffer_offset = 0;
      dst->buffer_size = 0;
      dst->user_buffer = NULL;
      return;
   }

   if (buf->buffer) {
      pipe_resource_reference(&dst->buffer, buf->buffer);
      dst->buffer_offset = buf->buffer_offset;
      dst->buffer_size = buf->buffer_size;
      dst->user_buffer = NULL;
      return;
   }

   /* Client memory. st uploads most constants itself, but not all of them: the
    * default uniform block it hands the driver this way carries
    * gl_FbWposYTransform and the other values the frontend derives per draw, and
    * a shader that reads gl_FragCoord reads them. Dropping the pointer leaves
    * the descriptor pointing at whatever the previous draw put there, which is
    * how a texelFetch through gl_FragCoord ends up sampling outside the texture
    * and returning zero.
    *
    * The offset is aligned to minUniformBufferOffsetAlignment because
    * VkDescriptorBufferInfo::offset must be, and the resource the uploader
    * returns is referenced by the caller: it must not be the same buffer the
    * uploader hands out next time. */
   const unsigned alignment =
      MAX2((unsigned)screen->props.limits.minUniformBufferOffsetAlignment, 16u);
   const unsigned size = buf->buffer_size;
   if (size == 0) {
      pipe_resource_reference(&dst->buffer, NULL);
      dst->buffer_offset = 0;
      dst->buffer_size = 0;
      return;
   }

   struct pipe_resource *uploaded = NULL;
   struct pipe_resource *release = NULL;
   unsigned offset = 0;
   u_upload_data(pipe->const_uploader, 0, size, alignment, buf->user_buffer,
                 &offset, &uploaded, &release);
   if (release)
      pipe_resource_reference(&release, NULL);
   u_upload_unmap(pipe->const_uploader);

   if (!uploaded) {
      sz_warn_once("could not upload a constant buffer from client memory");
      return;
   }

   pipe_resource_reference(&dst->buffer, NULL);
   dst->buffer = uploaded;
   dst->buffer_offset = offset;
   dst->buffer_size = size;
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

   /* Records the draws appended under the old bindings before they change. */
   sz_state_changed(ctx, SZ_DIRTY_VERTEX_BUFFERS);

   const unsigned n = MIN2(count, PIPE_MAX_ATTRIBS);
   for (unsigned i = 0; i < n; i++) {
      struct pipe_vertex_buffer *dst = &ctx->state.vertex_buffers[i];

      /* Drop the slot's old reference before the copy overwrites the pointer
       * that names it. */
      pipe_resource_reference(&dst->buffer.resource, NULL);

      *dst = buffers[i];
      /* The caller keeps its own reference and drops it as soon as it likes --
       * cso's vbuf path does so right after the draw -- so a bare pointer copy
       * leaves the state naming a freed resource, and the VkBuffer bound later
       * is whatever the allocator left in its place. */
      dst->buffer.resource = NULL;
      if (!buffers[i].is_user_buffer)
         pipe_resource_reference(&dst->buffer.resource,
                                 buffers[i].buffer.resource);
   }

   /* Slots past `count` are unbound. */
   for (unsigned i = n; i < ctx->state.num_vertex_buffers; i++)
      pipe_resource_reference(&ctx->state.vertex_buffers[i].buffer.resource, NULL);

   ctx->state.num_vertex_buffers = n;
}

static void
sz_set_shader_buffers(struct pipe_context *pipe, mesa_shader_stage shader,
                      unsigned start_slot, unsigned count,
                      const struct pipe_shader_buffer *buffers,
                      unsigned writable_bitmask)
{
}

/* The VkImageView for a bound storage image, and the transition that makes the
 * image readable and writable in a shader. The view is cached on the context
 * like an attachment's, so a draw that rebinds the same image costs a lookup. */
bool
sz_shader_image_view(struct sz_context *ctx, VkCommandBuffer cmd,
                     const struct pipe_image_view *img, VkImageView *out)
{
   if (!img || !img->resource)
      return false;

   struct sz_resource *res = sz_resource(img->resource);
   if (res->image == VK_NULL_HANDLE)
      return false;

   /* GL images are read-write, so the layout is GENERAL rather than a read-only
    * one; that is also the only layout a Vulkan storage image descriptor may
    * name when the shader both loads and stores. */
   sz_resource_access(ctx->screen, cmd, res, SZ_ACCESS_STORAGE);

   const struct pipe_surface surf = {
      .format = img->format,
      .level = img->u.tex.level,
      .first_layer = img->u.tex.first_layer,
      .last_layer = img->u.tex.last_layer,
   };
   *out = sz_surface_view(ctx, res, sz_vk_format(ctx->screen, img->format),
                          &surf);
   return *out != VK_NULL_HANDLE;
}

static void
sz_set_shader_images(struct pipe_context *pipe, mesa_shader_stage shader,
                     unsigned start_slot, unsigned count,
                     unsigned unbind_num_trailing_slots,
                     const struct pipe_image_view *images)
{
   struct sz_context *ctx = sz_context(pipe);

   /* Recorded draws name whatever was bound when they were queued, so the
    * bindings they used have to be submitted before any of them changes. */
   sz_draw_flush(ctx);

   for (unsigned i = 0; i < count; i++) {
      const unsigned slot = start_slot + i;
      if (slot >= SZ_MAX_IMAGES)
         break;

      struct pipe_image_view **dst = &ctx->state.shader_images[shader][slot];
      if (*dst) {
         pipe_resource_reference(&(*dst)->resource, NULL);
         FREE(*dst);
         *dst = NULL;
      }

      if (!images[i].resource)
         continue;

      struct pipe_image_view *copy = CALLOC_STRUCT(pipe_image_view);
      if (!copy)
         continue;
      *copy = images[i];
      pipe_resource_reference(&copy->resource, images[i].resource);
      *dst = copy;
   }

   for (unsigned i = 0; i < unbind_num_trailing_slots; i++) {
      const unsigned slot = start_slot + count + i;
      if (slot >= SZ_MAX_IMAGES)
         break;
      struct pipe_image_view **dst = &ctx->state.shader_images[shader][slot];
      if (*dst) {
         pipe_resource_reference(&(*dst)->resource, NULL);
         FREE(*dst);
         *dst = NULL;
      }
   }
}

static void
sz_render_condition(struct pipe_context *pipe, struct pipe_query *query,
                    bool condition, enum pipe_render_cond_flag mode)
{
   struct sz_context *ctx = sz_context(pipe);

   /* Draws issued before the condition belongs to the previous block. */
   sz_draw_flush(ctx);

   bool passes = true;
   if (query) {
      uint64_t value = 0;
      /* `condition` is glBeginConditionalRender's *inverted* flag, not "draw
       * when true": Mesa passes GL_QUERY_*_INVERTED straight through
       * (condrender.c). So the block runs when the query result is non-zero and
       * the app did not ask for the inverted form.
       *
       * The GL specification allows blocking here (GL_QUERY_WAIT), and the
       * result cannot change afterwards, so the predicate is resolved once
       * instead of being tested per draw. */
      if (sz_query_result_u64(ctx, query, &value, true))
         passes = (value != 0) != condition;
      else {
         sz_warn_once("conditional render with an unavailable query; the "
                      "block is discarded");
         passes = false;
      }
   }

   ctx->condition_query = query;
   ctx->condition_expected = condition;
   ctx->condition_passes = passes;

   (void)mode;
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

/* The target is a bare wrapper: the buffer, the offset and the size all live in
 * the Gallium base structure, and the VkBuffer is reached through the resource. */
struct pipe_stream_output_target *
sz_create_stream_output_target(struct pipe_context *pipe,
                               struct pipe_resource *res, unsigned buffer_offset,
                               unsigned buffer_size)
{
   struct pipe_stream_output_target *target = CALLOC_STRUCT(pipe_stream_output_target);
   if (!target)
      return NULL;

   /* The caller owns one reference. CALLOC leaves the count at zero, which
    * makes the first reference_release free the object while the state tracker
    * still points at it -- the use-after-free shows up in end_transform_feedback
    * long after the release. */
   pipe_reference_init(&target->reference, 1);

   target->context = pipe;
   pipe_resource_reference(&target->buffer, res);
   target->buffer_offset = buffer_offset;
   target->buffer_size = buffer_size;
   return target;
}

void
sz_stream_output_target_destroy(struct pipe_context *pipe,
                                struct pipe_stream_output_target *target)
{
   if (!target)
      return;
   pipe_resource_reference(&target->buffer, NULL);
   free(target);
}

uint32_t
sz_stream_output_target_offset(struct pipe_stream_output_target *target)
{
   /* The number of vertices written is a GL 3.2 (ARB_transform_feedback2) query;
    * returning zero keeps glGetBufferParameteriv from inventing a number. */
   (void)target;
   return 0;
}

/* GL_ARB_texture_multisample's glGetMultisamplefv(GL_SAMPLE_POSITION).
 *
 * Vulkan defines one set of standard sample locations and every implementation
 * uses it, so the answer is not a device query -- there is nothing to ask. The
 * grid below is the standard one from the Vulkan specification's "Standard
 * Sample Locations" table, for the counts GL exposes.
 *
 * The values are in GL's convention: the origin is the lower-left corner of the
 * pixel, which is also Vulkan's for this purpose (Vulkan's sample locations are
 * relative to the upper-left, so y is flipped here to match). */
static void
sz_get_sample_position(struct pipe_context *pipe, unsigned sample_count,
                       unsigned sample_index, float *out_value)
{
   (void)pipe;

   static const float locations_2[][2] = {
      { 0.75f, 0.25f }, { 0.25f, 0.75f },
   };
   static const float locations_4[][2] = {
      { 0.375f, 0.875f }, { 0.875f, 0.375f },
      { 0.125f, 0.625f }, { 0.625f, 0.125f },
   };
   static const float locations_8[][2] = {
      { 0.5625f, 0.3125f }, { 0.4375f, 0.6875f }, { 0.8125f, 0.5625f },
      { 0.3125f, 0.1875f }, { 0.1875f, 0.8125f }, { 0.0625f, 0.4375f },
      { 0.6875f, 0.9375f }, { 0.9375f, 0.0625f },
   };
   static const float locations_16[][2] = {
      { 0.5625f, 0.5625f }, { 0.4375f, 0.3125f }, { 0.3125f, 0.625f },
      { 0.75f, 0.4375f },   { 0.1875f, 0.375f },  { 0.625f, 0.8125f },
      { 0.8125f, 0.6875f }, { 0.6875f, 0.1875f }, { 0.375f, 0.875f },
      { 0.5f, 0.0625f },    { 0.25f, 0.125f },    { 0.125f, 0.75f },
      { 0.0f, 0.5f },       { 0.9375f, 0.25f },   { 0.875f, 0.9375f },
      { 0.0625f, 0.0f },
   };

   const float(*table)[2] = NULL;
   unsigned count = 0;

   switch (sample_count) {
   case 2:
      table = locations_2;
      count = ARRAY_SIZE(locations_2);
      break;
   case 4:
      table = locations_4;
      count = ARRAY_SIZE(locations_4);
      break;
   case 8:
      table = locations_8;
      count = ARRAY_SIZE(locations_8);
      break;
   case 16:
      table = locations_16;
      count = ARRAY_SIZE(locations_16);
      break;
   default:
      break;
   }

   if (!table || sample_index >= count) {
      out_value[0] = 0.5f;
      out_value[1] = 0.5f;
      return;
   }

   out_value[0] = table[sample_index][0];
   out_value[1] = table[sample_index][1];
}

void
sz_set_stream_output_targets(struct pipe_context *pipe, unsigned num_targets,
                             struct pipe_stream_output_target **targets,
                             const unsigned *offsets,
                             enum mesa_prim output_prim)
{
   struct sz_context *ctx = sz_context(pipe);

   /* Draws queued so far belong to the previous capture configuration. */
   sz_draw_flush(ctx);

   const unsigned n = MIN2(num_targets, PIPE_MAX_SO_BUFFERS);
   bool any_reset = false;

   for (unsigned i = 0; i < n; i++) {
      /* `offsets[i] == UINT_MAX` means append, and it must not reach the byte
       * arithmetic: taken as an offset it would put the capture four gigabytes
       * into a buffer that is not that long. */
      const unsigned offset = offsets ? offsets[i] : 0;
      const bool append = offset == UINT_MAX;

      /* A slot that points at a different buffer cannot continue the old
       * counter: it counted bytes into the buffer that was there before. */
      const bool rebound = ctx->so_prev_targets[i] != targets[i];

      pipe_so_target_reference(&ctx->so_targets[i], targets[i]);
      pipe_so_target_reference(&ctx->so_prev_targets[i], targets[i]);
      ctx->so_offsets[i] = append ? UINT_MAX : offset;
      ctx->so_counter_reset[i] = !append || rebound;
      any_reset |= ctx->so_counter_reset[i];
   }
   for (unsigned i = n; i < PIPE_MAX_SO_BUFFERS; i++) {
      pipe_so_target_reference(&ctx->so_targets[i], NULL);
      pipe_so_target_reference(&ctx->so_prev_targets[i], NULL);
   }

   ctx->num_so_targets = n;
   /* Only a capture whose counters actually need zeroing pays for it: it forces
    * the render pass closed, and a capture that spans many batches would
    * otherwise restart at its own beginning every time. */
   ctx->so_counter_dirty = any_reset;
   (void)output_prim;
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
/* The state emitters. Each one is called only when its dirty bit is set, which
 * is what turns a draw from nine unconditional vkCmd calls into one. */

static void
sz_emit_viewport(struct sz_context *ctx, VkCommandBuffer cmd)
{
   const struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   /* GL_ARB_viewport_array: one viewport per slot the application set, in the
    * order it set them. Vulkan takes the same array, and a vertex shader picks
    * with gl_ViewportIndex -- which is the ViewportIndex builtin the translator
    * already declares.
    *
    * All PIPE_MAX_VIEWPORTS slots are written even when the application set one:
    * the pipeline declares every slot dynamic, and Vulkan requires each of them
    * to have been set before a draw (VUID-vkCmdDraw-None-07831). A slot the
    * application never mentioned repeats the last one it did, which is what GL
    * says an unused index does -- the viewport in effect is still viewport 0.
    *
    * Zero viewports means the state tracker has not sent one yet, and the
    * framebuffer's extent is the only sensible answer. */
   const unsigned count = CLAMP(ctx->state.num_viewports, 1, PIPE_MAX_VIEWPORTS);

   VkViewport viewports[PIPE_MAX_VIEWPORTS];

   /* Gallium viewport state is in GL window coordinates: y grows upwards from
    * the bottom-left corner. Vulkan's framebuffer origin is the top-left corner,
    * so `translate - scale` with a positive height maps NDC +1 to row 0 -- which
    * is where GL puts the top of the window. Keeping the height positive is what
    * keeps the rendered image the right way up; a negative height would mirror
    * the frame vertically on screen. */
   /* With no viewport from the state tracker the whole framebuffer is the
    * sensible answer. */
   const struct pipe_viewport_state fallback = {
      .scale = { (float)fb->width / 2.0f, (float)fb->height / 2.0f, 0.5f },
      .translate = { (float)fb->width / 2.0f, (float)fb->height / 2.0f, 0.5f },
   };

   for (unsigned i = 0; i < PIPE_MAX_VIEWPORTS; i++) {
      const unsigned slot = ctx->state.num_viewports == 0
                               ? 0
                               : MIN2(i, count - 1);
      const struct pipe_viewport_state *vp =
         ctx->state.num_viewports == 0 ? &fallback
                                       : &ctx->state.viewports[slot];
      viewports[i] = (VkViewport) {
         .x = vp->translate[0] - vp->scale[0],
         .y = vp->translate[1] - vp->scale[1],
         .width = 2.0f * vp->scale[0],
         .height = 2.0f * vp->scale[1],
         .minDepth = vp->translate[2] - vp->scale[2],
         .maxDepth = vp->translate[2] + vp->scale[2],
      };
   }

   ctx->screen->vk.CmdSetViewport(cmd, 0, PIPE_MAX_VIEWPORTS, viewports);
}

static void
sz_emit_scissor(struct sz_context *ctx, VkCommandBuffer cmd)
{
   const struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   /* The scissor rects are only meaningful while glScissor's test is on, and the
    * state tracker keeps sending them after it is switched off. Applying the
    * last rect regardless clips every later draw to a stale rectangle, which
    * renders as "everything after a scissored draw is missing". */
   const struct sz_state *rast_state = ctx->state.rasterizer;
   const bool scissor_test = rast_state && rast_state->u.rasterizer.scissor;
   const unsigned count =
      CLAMP(ctx->state.num_scissors, 1, PIPE_MAX_VIEWPORTS);

   VkRect2D scissors[PIPE_MAX_VIEWPORTS];

   /* Every slot is written for the same reason the viewports are: the pipeline
    * declares them all dynamic. */
   if (!scissor_test || ctx->state.num_scissors == 0) {
      const VkRect2D full = {
         .offset = { 0, 0 },
         .extent = { fb->width, fb->height },
      };
      for (unsigned i = 0; i < PIPE_MAX_VIEWPORTS; i++)
         scissors[i] = full;
      ctx->screen->vk.CmdSetScissor(cmd, 0, PIPE_MAX_VIEWPORTS, scissors);
      return;
   }

   for (unsigned i = 0; i < PIPE_MAX_VIEWPORTS; i++) {
      const struct pipe_scissor_state *sc =
         &ctx->state.scissors[MIN2(i, count - 1)];
      scissors[i] = (VkRect2D) {
         .offset = { sc->minx, sc->miny },
         .extent = { sc->maxx - sc->minx, sc->maxy - sc->miny },
      };
   }

   ctx->screen->vk.CmdSetScissor(cmd, 0, PIPE_MAX_VIEWPORTS, scissors);
}

static bool
sz_scissor_is_empty(const struct sz_context *ctx)
{
   const struct sz_state *rast_state = ctx->state.rasterizer;
   if (!rast_state || !rast_state->u.rasterizer.scissor ||
       ctx->state.num_scissors == 0)
      return false;

   const struct pipe_scissor_state *sc = &ctx->state.scissors[0];
   return sc->maxx == sc->minx || sc->maxy == sc->miny;
}

static void
sz_emit_blend_const(struct sz_context *ctx, VkCommandBuffer cmd)
{
   /* glBlendColor is dynamic in Vulkan, so it is set when it changes rather than
    * baked into the pipeline. */
   if (!ctx->screen->vk.CmdSetBlendConstants)
      return;

   const struct pipe_blend_color *bc = &ctx->state.blend_color;
   const float constants[4] = { bc->color[0], bc->color[1], bc->color[2],
                                bc->color[3] };
   ctx->screen->vk.CmdSetBlendConstants(cmd, constants);
}

static void
sz_emit_stencil_ref(struct sz_context *ctx, VkCommandBuffer cmd)
{
   /* Core Vulkan can only set one value for both faces; Gallium carries two
    * (GL_STENCIL_BACK_REF from glStencilFuncSeparate) and the driver has no
    * VK_EXT_extended_dynamic_state yet to split them. Two different values are
    * rare enough that saying so once is better than silently using the front
    * one. */
   const unsigned ref = ctx->state.stencil_ref.ref_value[0];
   if (ctx->state.stencil_ref.ref_value[1] != ref)
      sz_warn_once("per-face stencil reference values need "
                   "VK_EXT_extended_dynamic_state; using the front value");

   ctx->screen->vk.CmdSetStencilReference(cmd, VK_STENCIL_FACE_FRONT_AND_BACK,
                                          ref);
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

/* ── Draw stream ─────────────────────────────────────────────────────────── */

static void
sz_emit_draws(struct sz_context *ctx, VkCommandBuffer cmd)
{
   const unsigned n = ctx->pending.count;

   if (n == 0)
      return;

   ctx->screen->stats.draws += n;
   ctx->screen->stats.batches++;

   /* An indirect batch is exactly one record: the counts live in the buffer, so
    * there is no VkMultiDrawInfoEXT for them to travel in. */
   if (ctx->pending.indirect.buffer) {
      const VkBuffer buffer = ctx->pending.indirect.buffer->buffer;
      const VkDeviceSize offset = ctx->pending.indirect.offset;
      const uint32_t stride = ctx->pending.indirect.stride;
      const uint32_t count = ctx->pending.indirect.draw_count;

      /* A count buffer means the record count is itself read by the device --
       * that is GL_PARAMETER_BUFFER. The record count then becomes the
       * *maximum*, which is exactly Vulkan's maxDrawCount. */
      if (ctx->pending.indirect.count_buffer) {
         const VkBuffer count_buffer =
            ctx->pending.indirect.count_buffer->buffer;
         const VkDeviceSize count_offset = ctx->pending.indirect.count_offset;
         if (ctx->pending.indexed)
            ctx->screen->vk.CmdDrawIndexedIndirectCount(
               cmd, buffer, offset, count_buffer, count_offset, count, stride);
         else
            ctx->screen->vk.CmdDrawIndirectCount(
               cmd, buffer, offset, count_buffer, count_offset, count, stride);
      } else if (ctx->pending.indexed)
         ctx->screen->vk.CmdDrawIndexedIndirect(cmd, buffer, offset, count,
                                                stride);
      else
         ctx->screen->vk.CmdDrawIndirect(cmd, buffer, offset, count, stride);
      return;
   }

   /* One call for the whole batch. The data is a plain host array: Vulkan copies
    * it into the command stream while recording, so there is nothing to keep
    * alive afterwards. */
   if (n > 1 && ctx->screen->max_multi_draw_count > 0) {
      ctx->screen->stats.multi_batches++;
      if (ctx->pending.indexed) {
         VkMultiDrawIndexedInfoEXT info[SZ_MAX_BATCHED_DRAWS];
         for (unsigned i = 0; i < n; i++) {
            info[i].firstIndex = ctx->pending.draws[i].first;
            info[i].indexCount = ctx->pending.draws[i].count;
            info[i].vertexOffset = ctx->pending.draws[i].index_bias;
         }
         /* (cmd, drawCount, pIndexInfo, instanceCount, firstInstance, stride,
          * pVertexOffset); a NULL offset means "use vertexOffset from each
          * VkMultiDrawIndexedInfoEXT". */
         ctx->screen->vk.CmdDrawMultiIndexedEXT(cmd, n, info,
                                                ctx->pending.instance_count,
                                                ctx->pending.start_instance,
                                                sizeof(info[0]), NULL);
      } else {
         VkMultiDrawInfoEXT info[SZ_MAX_BATCHED_DRAWS];
         for (unsigned i = 0; i < n; i++) {
            info[i].firstVertex = ctx->pending.draws[i].first;
            info[i].vertexCount = ctx->pending.draws[i].count;
         }
         ctx->screen->vk.CmdDrawMultiEXT(cmd, n, info,
                                         ctx->pending.instance_count,
                                         ctx->pending.start_instance,
                                         sizeof(info[0]));
      }
      return;
   }

   /* No VK_EXT_multi_draw (or a batch of one): one command per draw. */
   for (unsigned i = 0; i < n; i++) {
      const uint32_t count = ctx->pending.draws[i].count;
      const uint32_t first = ctx->pending.draws[i].first;
      const int32_t index_bias = ctx->pending.draws[i].index_bias;

      if (ctx->pending.indexed)
         ctx->screen->vk.CmdDrawIndexed(cmd, count,
                                        ctx->pending.instance_count, first,
                                        index_bias,
                                        ctx->pending.start_instance);
      else
         ctx->screen->vk.CmdDraw(cmd, count, ctx->pending.instance_count,
                                 first, ctx->pending.start_instance);
   }
}

/* The counter buffer exists because Vulkan requires one for a capture to
 * happen at all: vkCmdBeginTransformFeedbackEXT loads the starting dword offset
 * of every buffer from a counter buffer, and RADV (like the other drivers)
 * disables the stream when it is called without counters. */
static bool
sz_so_counter_ensure(struct sz_context *ctx)
{
   if (ctx->so_counter_buffer != VK_NULL_HANDLE)
      return true;

   struct sz_screen *screen = ctx->screen;
   const VkBufferCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = PIPE_MAX_SO_BUFFERS * sizeof(uint32_t),
      .usage = VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buffer = VK_NULL_HANDLE;
   if (screen->vk.CreateBuffer(screen->dev, &ci, NULL, &buffer) != VK_SUCCESS) {
      sz_warn_once("could not create the transform feedback counter buffer");
      return false;
   }

   VkMemoryRequirements req;
   screen->vk.GetBufferMemoryRequirements(screen->dev, buffer, &req);
   struct sz_bo *bo = sz_bo_alloc(screen, &req, 0);
   if (!bo ||
       screen->vk.BindBufferMemory(screen->dev, buffer, sz_bo_memory(bo),
                                   sz_bo_offset(bo)) != VK_SUCCESS) {
      sz_warn_once("could not back the transform feedback counter buffer");
      if (bo)
         sz_bo_free(bo);
      screen->vk.DestroyBuffer(screen->dev, buffer, NULL);
      return false;
   }

   ctx->so_counter_buffer = buffer;
   ctx->so_counter_bo = bo;
   return true;
}

/* Zeroes the counter buffer for the capture about to be recorded, outside the
 * render pass.
 *
 * vkCmdFillBuffer may not be recorded inside a dynamic render pass
 * (VUID-vkCmdFillBuffer-renderpass), and the counters must read zero when
 * vkCmdBeginTransformFeedbackEXT loads the starting offsets from them -- which
 * the driver does at begin time, inside the pass.
 *
 * Zeroing in the command stream rather than on the host matters: a previously
 * submitted capture may still be reading these dwords. */
static bool sz_so_counter_ensure(struct sz_context *ctx);

static void
sz_so_counter_zero(struct sz_context *ctx, VkCommandBuffer cmd)
{
   /* The counter buffer may not exist yet: it is created on first use, and this
    * runs before the capture bracket below would have asked for it. */
   if (ctx->num_so_targets == 0 || !sz_so_counter_ensure(ctx))
      return;

   for (unsigned i = 0; i < ctx->num_so_targets; i++) {
      if (!ctx->so_counter_reset[i])
         continue;
      ctx->screen->vk.CmdFillBuffer(cmd, ctx->so_counter_buffer,
                                    i * sizeof(uint32_t), sizeof(uint32_t), 0);
      ctx->so_counter_reset[i] = false;
   }

   const VkMemoryBarrier2 fill_barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT |
                      VK_PIPELINE_STAGE_2_CLEAR_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFORM_FEEDBACK_BIT_EXT,
      .dstAccessMask = VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_READ_BIT_EXT |
                       VK_ACCESS_2_TRANSFORM_FEEDBACK_COUNTER_WRITE_BIT_EXT,
   };
   const VkDependencyInfo fill_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &fill_barrier,
   };
   ctx->screen->vk.CmdPipelineBarrier2(cmd, &fill_dep);
}

void
sz_draw_flush(struct sz_context *ctx)
{
   if (ctx->pending.count == 0)
      return;

   /* Inside a discarded conditional-render block the draws are dropped, not
    * recorded: GL says the commands have no effect. The state is dropped with
    * them, because a new batch re-emits everything anyway. */
   if (ctx->condition_query && !ctx->condition_passes) {
      ctx->pending.count = 0;
      return;
   }

   VkCommandBuffer cmd = sz_context_cmd(ctx);
   if (cmd == VK_NULL_HANDLE) {
      ctx->pending.count = 0;
      return;
   }

   /* A layout transition is illegal inside a render pass, so a batch whose
    * descriptors changed may have to close it first. */
   if ((ctx->dirty & SZ_DIRTY_DESCRIPTORS) && ctx->rendering &&
       sz_descriptor_sampled_needs_transition(ctx)) {
      ctx->screen->vk.CmdEndRendering(cmd);
      ctx->rendering = false;
   }

   if (ctx->dirty & SZ_DIRTY_DESCRIPTORS)
      sz_descriptor_prepare_sampled(ctx, cmd);

   /* Zeroing the capture counters may not be recorded inside a render pass, so
    * it forces the pass closed -- only when a capture actually starts, not on
    * every batch of one. */
   if (ctx->so_counter_dirty) {
      if (ctx->rendering) {
         ctx->screen->vk.CmdEndRendering(cmd);
         ctx->rendering = false;
      }
      sz_so_counter_zero(ctx, cmd);
      ctx->so_counter_dirty = false;
   }

   if (!ctx->rendering && !sz_renderpass_begin(ctx, cmd)) {
      ctx->pending.count = 0;
      ctx->dirty = 0;
      return;
   }

   if (ctx->dirty & SZ_DIRTY_PIPELINE) {
      struct sz_gfx_pipeline *p = sz_pipeline_get(ctx, ctx->pending.topology);
      if (!p) {
         ctx->pending.count = 0;
         ctx->dirty = 0;
         return;
      }
      if (p != ctx->pipeline) {
         ctx->screen->vk.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                         p->pipeline);
         ctx->pipeline = p;
         ctx->screen->stats.pipeline_binds++;
      }
      ctx->dirty &= ~SZ_DIRTY_PIPELINE;
   }

   if (ctx->dirty & SZ_DIRTY_DESCRIPTORS) {
      sz_descriptor_push(ctx, cmd);
      ctx->screen->stats.descriptor_pushes++;
      ctx->dirty &= ~SZ_DIRTY_DESCRIPTORS;
   }

   /* Barriers are emitted for every batch, not only when a binding changes.
    *
    * The uploaders write new data into a buffer that is already bound -- the
    * binding (resource + offset) can be identical while the contents behind it
    * were rewritten since the last draw. Emitting the write-to-read barrier only
    * on a binding change therefore leaves a window in which the GPU reads
    * vertices that the CPU has already overwritten, which shows up as
    * intermittent garbage geometry. The loop is a handful of pointer checks per
    * batch, not per draw. */
   for (unsigned i = 0; i < ctx->state.num_vertex_buffers; i++) {
      const struct pipe_vertex_buffer *vb = &ctx->state.vertex_buffers[i];
      if (vb->is_user_buffer || !vb->buffer.resource)
         continue;
      struct sz_resource *vres = sz_resource(vb->buffer.resource);
      if (vres->buffer == VK_NULL_HANDLE)
         continue;
      sz_resource_access(ctx->screen, cmd, vres, SZ_ACCESS_VERTEX_BUFFER);
   }

   for (unsigned st = 0; st < MESA_SHADER_STAGES; st++) {
      for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++) {
         const struct pipe_constant_buffer *cb =
            &ctx->state.constant_buffers[st][i];
         if (!cb->buffer)
            continue;
         struct sz_resource *cres = sz_resource(cb->buffer);
         if (cres->buffer == VK_NULL_HANDLE)
            continue;
         sz_resource_access(ctx->screen, cmd, cres, SZ_ACCESS_UNIFORM_BUFFER);
      }
   }

   if (ctx->dirty & SZ_DIRTY_VERTEX_BUFFERS) {
      /* Every slot the frontend bound, not just slot 0: a shader reading two
       * attributes has two vertex buffers, and binding only the first leaves the
       * rest undefined. A slot with no resource behind it is left unbound, which
       * is legal as long as no shader input reads it. */
      for (unsigned i = 0; i < ctx->state.num_vertex_buffers; i++) {
         const struct pipe_vertex_buffer *vb = &ctx->state.vertex_buffers[i];
         if (vb->is_user_buffer || !vb->buffer.resource)
            continue;

         struct sz_resource *vres = sz_resource(vb->buffer.resource);
         if (vres->buffer == VK_NULL_HANDLE)
            continue;

         const VkDeviceSize offset = vb->buffer_offset;
         ctx->screen->vk.CmdBindVertexBuffers(cmd, i, 1, &vres->buffer, &offset);
         ctx->screen->stats.vertex_buffer_binds++;
      }
      ctx->dirty &= ~SZ_DIRTY_VERTEX_BUFFERS;
   }

   if (ctx->pending.indexed && ctx->pending.index_buffer)
      sz_resource_access(ctx->screen, cmd, ctx->pending.index_buffer,
                         SZ_ACCESS_INDEX_BUFFER);

   if (ctx->dirty & SZ_DIRTY_INDEX_BUFFER) {
      if (ctx->pending.indexed && ctx->pending.index_buffer) {
         ctx->screen->vk.CmdBindIndexBuffer(cmd, ctx->pending.index_buffer->buffer,
                                            ctx->pending.index_offset,
                                            ctx->pending.index_type);
      }
      ctx->dirty &= ~SZ_DIRTY_INDEX_BUFFER;
   }

   /* A zero-sized scissor rejects every fragment, and Vulkan forbids the extent
    * itself. The state is still emitted; only the draws are dropped. */
   const bool scissor_empty = sz_scissor_is_empty(ctx);

   if (ctx->dirty & SZ_DIRTY_VIEWPORT) {
      sz_emit_viewport(ctx, cmd);
      ctx->screen->stats.state_sets++;
   }
   if (ctx->dirty & SZ_DIRTY_SCISSOR) {
      sz_emit_scissor(ctx, cmd);
      ctx->screen->stats.state_sets++;
   }
   if (ctx->dirty & SZ_DIRTY_BLEND_CONST) {
      sz_emit_blend_const(ctx, cmd);
      ctx->screen->stats.state_sets++;
   }
   if (ctx->dirty & SZ_DIRTY_STENCIL_REF) {
      sz_emit_stencil_ref(ctx, cmd);
      ctx->screen->stats.state_sets++;
   }

   /* Transform feedback brackets the batch. Beginning and ending per batch is
    * what makes the capture follow the state that was in effect when the draws
    * were queued: set_stream_output_targets() flushes, so a batch never spans
    * two capture configurations. */
   const bool tf = ctx->num_so_targets > 0;

   if (tf) {
      VkBuffer buffers[PIPE_MAX_SO_BUFFERS];
      VkDeviceSize offsets[PIPE_MAX_SO_BUFFERS];
      unsigned n = 0;

      for (unsigned i = 0; i < ctx->num_so_targets; i++) {
         struct pipe_stream_output_target *t = ctx->so_targets[i];
         if (!t || !t->buffer)
            continue;
         struct sz_resource *res = sz_resource(t->buffer);
         if (res->buffer == VK_NULL_HANDLE)
            continue;
         buffers[n] = res->buffer;
         offsets[n] = t->buffer_offset +
                      (ctx->so_offsets[i] == UINT_MAX ? 0 : ctx->so_offsets[i]);
         n++;
      }

      if (n > 0 && ctx->screen->vk.CmdBeginTransformFeedbackEXT &&
          sz_so_counter_ensure(ctx)) {
         VkDeviceSize sizes[PIPE_MAX_SO_BUFFERS];
         VkBuffer counters[PIPE_MAX_SO_BUFFERS];
         VkDeviceSize counter_offsets[PIPE_MAX_SO_BUFFERS];

         for (unsigned i = 0; i < n; i++) {
            const struct pipe_stream_output_target *t = ctx->so_targets[i];
            sizes[i] = t->buffer_size;
            counters[i] = ctx->so_counter_buffer;
            counter_offsets[i] = i * sizeof(uint32_t);
         }

         /* The counters were zeroed before the render pass opened (see
          * sz_so_counter_zero): vkCmdFillBuffer is illegal inside one. */
         ctx->screen->vk.CmdBindTransformFeedbackBuffersEXT(cmd, 0, n, buffers,
                                                            offsets, sizes);
         ctx->screen->vk.CmdBeginTransformFeedbackEXT(cmd, 0, n, counters,
                                                      counter_offsets);
      }
   }

   if (!scissor_empty)
      sz_emit_draws(ctx, cmd);

   if (tf && ctx->screen->vk.CmdEndTransformFeedbackEXT &&
       ctx->so_counter_buffer != VK_NULL_HANDLE) {
      VkBuffer counters[PIPE_MAX_SO_BUFFERS];
      VkDeviceSize counter_offsets[PIPE_MAX_SO_BUFFERS];
      for (unsigned i = 0; i < PIPE_MAX_SO_BUFFERS; i++) {
         counters[i] = ctx->so_counter_buffer;
         counter_offsets[i] = i * sizeof(uint32_t);
      }
      ctx->screen->vk.CmdEndTransformFeedbackEXT(cmd, 0, PIPE_MAX_SO_BUFFERS,
                                                 counters, counter_offsets);
   }

   ctx->dirty = 0;
   ctx->pending.count = 0;
   ctx->pending.indirect.buffer = NULL;
   pipe_resource_reference(&ctx->pending.index_resource, NULL);
   pipe_resource_reference(&ctx->pending.indirect.resource, NULL);
   ctx->pending.indirect.count_buffer = NULL;
   pipe_resource_reference(&ctx->pending.indirect.count_resource, NULL);
}

static bool
sz_pending_compatible(const struct sz_context *ctx, bool indexed,
                      VkPrimitiveTopology topology, int32_t instance_count,
                      uint32_t start_instance, const struct sz_resource *index,
                      VkIndexType type, bool primitive_restart)
{
   if (ctx->pending.count == 0)
      return true;
   if (ctx->pending.indexed != indexed || ctx->pending.topology != topology)
      return false;
   if (ctx->pending.instance_count != instance_count ||
       ctx->pending.start_instance != start_instance)
      return false;
   if (ctx->pending.primitive_restart != primitive_restart)
      return false;
   if (indexed &&
       (ctx->pending.index_buffer != index || ctx->pending.index_type != type))
      return false;
   return true;
}

static void
sz_draw_append(struct sz_context *ctx, const struct pipe_draw_info *info,
               const struct pipe_draw_start_count_bias *draws, unsigned num_draws,
               struct sz_resource *index, VkIndexType type)
{
   const bool indexed = info->index_size != 0;
   const VkPrimitiveTopology topology = sz_topology(info->mode);

   if (!sz_pending_compatible(ctx, indexed, topology,
                              (int32_t)info->instance_count,
                              info->start_instance, index, type,
                              info->primitive_restart))
      sz_draw_flush(ctx);

   if (ctx->pending.count == 0) {
      ctx->pending.indexed = indexed;
      ctx->pending.instance_count = (int32_t)info->instance_count;
      ctx->pending.start_instance = info->start_instance;
      ctx->pending.topology = topology;
      ctx->pending.primitive_restart = info->primitive_restart;
      ctx->pending.index_buffer = index;
      /* The index buffer in pipe_draw_info is only valid for the duration of the
       * call, and this draw is queued rather than recorded, so the context needs
       * a reference of its own until the flush has bound it. */
      pipe_resource_reference(&ctx->pending.index_resource,
                              index ? &index->base : NULL);
      ctx->pending.index_type = type;
      ctx->pending.index_offset = 0;
      /* The index binding is part of the batch, so it is re-emitted whenever a
       * new batch starts. */
      ctx->dirty |= SZ_DIRTY_INDEX_BUFFER | SZ_DIRTY_PIPELINE;
   }

   if (ctx->state.primitive_restart != info->primitive_restart) {
      ctx->state.primitive_restart = info->primitive_restart;
      ctx->dirty |= SZ_DIRTY_PIPELINE;
   }

   for (unsigned i = 0; i < num_draws; i++) {
      if (draws[i].count == 0)
         continue;

      if (ctx->pending.count == ctx->pending.capacity)
         sz_draw_flush(ctx);

      /* After a flush the ring is empty and the batch parameters are already
       * correct: the flush drained exactly this state. */
      ctx->pending.draws[ctx->pending.count].count = draws[i].count;
      ctx->pending.draws[ctx->pending.count].first = draws[i].start;
      ctx->pending.draws[ctx->pending.count].index_bias = draws[i].index_bias;
      ctx->pending.count++;
   }
}

/* One indirect draw, queued the same way a direct one is.
 *
 * Everything the pipeline needs -- topology, index type, primitive restart -- is
 * known from the call; only the counts are not, and Vulkan reads those out of the
 * buffer itself. That is also why the batch is flushed immediately: there is no
 * count for the batcher to merge, and the buffer is only guaranteed to be alive
 * for the duration of this call.
 */
static void
sz_draw_append_indirect(struct sz_context *ctx, const struct pipe_draw_info *info,
                        const struct pipe_draw_indirect_info *indirect,
                        struct sz_resource *index, VkIndexType type)
{
   struct sz_resource *args = indirect->buffer ? sz_resource(indirect->buffer) : NULL;
   if (!args || args->buffer == VK_NULL_HANDLE)
      return;

   sz_draw_flush(ctx);

   const bool indexed = info->index_size != 0;

   ctx->pending.indexed = indexed;
   ctx->pending.topology = sz_topology(info->mode);
   ctx->pending.primitive_restart = info->primitive_restart;
   /* The pipeline key reads the shadow state, not the batch, so primitive
    * restart has to be published there like any other state change. */
   if (ctx->state.primitive_restart != info->primitive_restart) {
      ctx->state.primitive_restart = info->primitive_restart;
      ctx->dirty |= SZ_DIRTY_PIPELINE;
   }
   ctx->pending.index_buffer = index;
   pipe_resource_reference(&ctx->pending.index_resource,
                           index ? &index->base : NULL);
   ctx->pending.index_type = type;
   ctx->pending.index_offset = 0;
   /* Instance and vertex counts come from the buffer, so these are never read;
    * they are kept honest for the pipeline key. */
   ctx->pending.instance_count = 1;
   ctx->pending.start_instance = 0;

   ctx->pending.indirect.buffer = args;
   pipe_resource_reference(&ctx->pending.indirect.resource, &args->base);
   ctx->pending.indirect.offset = indirect->offset;
   /* A stride of zero means one draw command; Vulkan wants a real stride, and any
    * value is legal when the count is one. */
   ctx->pending.indirect.stride = indirect->stride ? indirect->stride
                                                   : sizeof(uint32_t) * 4;
   ctx->pending.indirect.draw_count = indirect->draw_count ? indirect->draw_count : 1;

   /* glMultiDrawArraysIndirectCount: the record count lives in another buffer.
    * draw_count is then the clamp, so it stays as filled above. */
   ctx->pending.indirect.count_buffer = NULL;
   if (indirect->indirect_draw_count) {
      struct sz_resource *count_res =
         sz_resource(indirect->indirect_draw_count);
      if (count_res->buffer != VK_NULL_HANDLE) {
         ctx->pending.indirect.count_buffer = count_res;
         pipe_resource_reference(&ctx->pending.indirect.count_resource,
                                 indirect->indirect_draw_count);
         ctx->pending.indirect.count_offset = indirect->indirect_draw_count_offset;
      }
   }

   ctx->pending.count = 1;
   ctx->pending.draws[0].count = 0;
   ctx->pending.draws[0].first = 0;
   ctx->pending.draws[0].index_bias = 0;

   ctx->dirty |= SZ_DIRTY_INDEX_BUFFER | SZ_DIRTY_PIPELINE;
   sz_draw_flush(ctx);
}

static void
sz_draw_vbo(struct pipe_context *pipe, const struct pipe_draw_info *info,
            unsigned drawid_offset,
            const struct pipe_draw_indirect_info *indirect,
            const struct pipe_draw_start_count_bias *draws, unsigned num_draws)
{
   struct sz_context *ctx = sz_context(pipe);

   if (num_draws == 0 && !indirect)
      return;

   /* A count buffer needs vkCmdDrawIndirectCount, which is core 1.2. Refuse
    * rather than silently drawing the clamp count. */
   if (indirect && indirect->indirect_draw_count &&
       !ctx->screen->vk.CmdDrawIndirectCount) {
      sz_warn_once("indirect draws with a draw count need Vulkan 1.2");
      return;
   }

   /* A draw does not have to read a vertex attribute. A geometry-only program
    * takes its vertices from gl_PrimitiveID, and a vertex shader may derive
    * everything from gl_VertexID; Vulkan accepts an empty vertex input state, so
    * the only draw that cannot be honoured is one whose declared elements have
    * no buffer behind them -- Vulkan requires every binding a pipeline names to
    * be bound. */
   const struct sz_state *ve = ctx->state.vertex_elements;
   const unsigned element_count = ve ? ve->u.ve.count : 0;

   if (element_count && ctx->state.num_vertex_buffers == 0) {
      sz_warn_once("draw without vertex buffers, ignoring");
      return;
   }

   /* A batch accumulates one descriptor-set copy per flush, so a draw stream
    * with many state changes can fill a ring slot. Starting a fresh batch here
    * keeps every offset valid: the descriptors a command buffer references must
    * stay put until that command buffer is submitted, so the only way to reclaim
    * the space is to submit. */
   if (!sz_descriptor_has_room(ctx)) {
      sz_draw_flush(ctx);
      sz_context_submit(ctx);
   }

   struct sz_resource *index = NULL;
   VkIndexType type = VK_INDEX_TYPE_UINT16;

   if (info->index_size) {
      if (info->has_user_indices) {
         sz_warn_once("user index buffers are not implemented, ignoring");
         return;
      }
      index = sz_resource(info->index.resource);
      if (!index || index->buffer == VK_NULL_HANDLE)
         return;
      type = info->index_size == 4 ? VK_INDEX_TYPE_UINT32
                                   : VK_INDEX_TYPE_UINT16;
   }

   if (indirect)
      sz_draw_append_indirect(ctx, info, indirect, index, type);
   else
      sz_draw_append(ctx, info, draws, num_draws, index, type);
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
sz_flush_resource(struct pipe_context *pipe, struct pipe_resource *res)
{
   struct sz_context *ctx = sz_context(pipe);

   /* "Make this resource visible to somebody else." The winsys calls this right
    * before a dma-buf is exported for the compositor, and the compositor samples
    * it without a fence of its own -- so the submission that wrote it has to be
    * complete, not merely queued. Waiting here is cheap (once per presented
    * resource) and removes the window in which a half-rendered frame could be
    * sampled. */
   sz_trace("flush_resource res=%p %ux%u fmt=%s", (void *)res, res->width0,
            res->height0, util_format_name(res->format));
   sz_context_submit(ctx);
   sz_queue_wait_idle(&ctx->screen->batch_state);
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

   /* Neither side carries an offset here. The transfer hands out the box, not
    * the level, so the destination starts at 0,0,0; and the caller's pointer is
    * already the first texel of the sub-image -- st_TexSubImage applies
    * GL_UNPACK_SKIP_PIXELS/ROWS/IMAGE_HEIGHT before calling in. Passing box->x
    * and box->y as source offsets as well walked the read forward by
    * box->y * stride bytes into an array that only holds the sub-image, which is
    * what crashed Minecraft: every sprite of a texture atlas is uploaded at a
    * nonzero y into a client buffer sized for the sprite alone.
    *
    * Width and height stay in texels: util_copy_rect() rounds them up to whole
    * blocks itself, so pre-dividing them would copy a block-compressed box once
    * instead of once per block. */
   util_copy_box((uint8_t *)map, res->format, transfer->stride,
                 transfer->layer_stride, 0, 0, 0,
                 box->width, box->height, box->depth,
                 (const uint8_t *)data, stride, layer_stride, 0, 0, 0);

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
   /* Byte offset inside `staging` the copy reads from. Non-zero for a partial
    * upload: a flush_region copies one range of a transfer that covers more. */
   uint32_t staging_offset;
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
         .srcOffset = job->to_staging ? job->box.x : job->staging_offset,
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

   /* VkBufferImageCopy measures bufferRowLength in *texels*, and for a
    * block-compressed format that is texels of the image, not blocks of the
    * buffer: the value has to be a multiple of the block width
    * (VUID-vkCmdCopyBufferToImage-bufferRowLength-09106). So the row pitch is
    * divided by the bytes of one block and multiplied by how many texels wide
    * that block is. */
   const unsigned block_w = util_format_get_blockwidth(job->res->base.format);
   const unsigned block_h = util_format_get_blockheight(job->res->base.format);

   VkBufferImageCopy region = {
      .bufferOffset = 0,
      /* In texels, not bytes; zero would mean "tightly packed", which is not the
       * pitch the caller sees. */
      .bufferRowLength = (job->row_pitch / bs) * block_w,
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
         /* A copy works in whole blocks, so a partial block at the edge is
          * rounded up -- the buffer holds whole texel blocks. */
         ALIGN_POT(DIV_ROUND_UP(job->box.width, block_w) * block_w, block_w),
         ALIGN_POT(DIV_ROUND_UP(job->box.height, block_h) * block_h, block_h),
         /* For a layered image the box's depth is a layer count and belongs in
          * imageSubresource above; extent.depth must be 1 for every non-3D
          * target (VUID-vkCmdCopyImageToBuffer-srcImage-07980). */
         is_3d ? job->box.depth : 1,
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

   /* A buffer write must not go straight into the resource's memory, however
    * directly the CPU can address it.
    *
    * Draws are recorded into a command buffer and submitted later, so the GPU has
    * not read the previous contents when the next upload overwrites them. An
    * application that uploads vertices, draws, uploads the next batch and draws
    * again -- which is how every renderer builds a frame, and what Minecraft's
    * BufferBuilder does several times per frame -- then has every draw read the
    * last upload, so the frame renders with geometry that belongs to some other
    * draw and nothing lands on screen.
    *
    * Buffers therefore go through the staging path, where the copy is recorded at
    * the position of the upload and is ordered against the draws around it.
    *
    * What is not staged is a map the caller declares it manages itself. The state
    * tracker's uploader keeps one such map open across a whole batch and writes
    * into it between draws, so staging it would put the copies after the draws
    * that need them. Those writes are synchronised instead: the GPU is given the
    * chance to finish reading the buffer before the CPU overwrites it, which is
    * the same stall a read already pays for and happens once per batch rather
    * than once per draw. Textures keep the direct path as well -- they are
    * written once and then only sampled. */
   const bool streaming_write =
      (usage & PIPE_MAP_WRITE) && res->target == PIPE_BUFFER &&
      !(usage & (PIPE_MAP_UNSYNCHRONIZED | PIPE_MAP_FLUSH_EXPLICIT |
                 PIPE_MAP_PERSISTENT));

   void *direct = sz_resource_map(r);
   if (direct && !streaming_write) {
      if (usage & PIPE_MAP_READ) {
         /* GL reads are synchronous: whatever the GPU was asked to write has
          * to be there before the pointer is handed out, and a direct map has
          * no barrier of its own to make that true. */
         sz_context_submit(ctx);
         sz_queue_wait_idle(&screen->batch_state);
         sz_bo_invalidate(r->bo, box->x, box->width);
      }
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

   if (!t->staging) {
      /* A direct mapping wrote straight into device memory. Flushing the range
       * handles non-coherent memory; the flag makes the next access emit the
       * HOST -> read barrier that Vulkan requires even for coherent memory. */
      if (t->base.usage & PIPE_MAP_WRITE) {
         struct sz_resource *r = sz_resource(t->base.resource);
         if (r->bo) {
            sz_bo_flush(r->bo, t->base.box.x, t->base.box.width);
            if (t->base.resource->target == PIPE_BUFFER)
               r->host_written = true;
         }
      }
      free(t);
      return;
   }

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

      /* Uploads are recorded into the command buffer that is already being
       * built, not submitted on their own.
       *
       * They used to go through sz_oneshot(), which ends the command buffer,
       * submits it and waits. That split an occlusion query (or any other
       * begin/end pair) across two command buffers, which Vulkan forbids, and
       * it cost a submit per client-array draw -- a dozen per frame in
       * glxgears. The staging resource stays alive through the existing
       * retirement table, which defers its destruction to the submission this
       * copy is part of. */
      /* The queued draws have to be emitted before this copy, or the two swap
       * places in the command stream and every draw that reads this resource
       * reads the data being uploaded now instead of the data that was there.
       * An application that uploads vertices, draws, uploads the next batch and
       * draws again -- which is how a renderer builds a frame, and what
       * Minecraft's BufferBuilder does several times per frame -- then has every
       * draw read the last upload, so the frame renders with geometry that
       * belongs to some other draw and nothing lands on screen.
       *
       * sz_renderpass_end() flushes too, but only while a pass is open; a batch
       * whose draws are still queued has none yet, which is the common case
       * here. */
      sz_draw_flush(ctx);

      VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
      if (cmd != VK_NULL_HANDLE)
         sz_copy_job_record(ctx, cmd, &job);
      else
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
   /* A mapped write is synchronised at map time (see sz_transfer_map), so the
    * whole box is copied at unmap and there is nothing to flush early. */
}

/* glTextureBarrier (NV_texture_barrier).
 *
 * The promise GL makes is that a draw which samples a texture a *previous* draw
 * rendered into sees that draw's results, and that this needs no explicit
 * synchronisation from the application. Vulkan makes the opposite promise by
 * default -- a read-after-write within one command buffer is a hazard the
 * application has to barrier -- so the barrier is what upholds GL's rule.
 *
 * The draws are already ordered by being recorded in one command buffer, so the
 * only thing missing is the execution and memory dependency between the colour
 * writes and the sampled reads. A full barrier is coarse, and deliberately so:
 * the flags Gallium passes describe which textures the application *promises*
 * it will only read or write, and getting that bookkeeping subtly wrong produces
 * exactly the intermittent corruption the barrier exists to prevent. */
static void
sz_texture_barrier(struct pipe_context *pipe, unsigned flags)
{
   struct sz_context *ctx = sz_context(pipe);

   /* Anything queued has to land first: the barrier has to separate the draws
    * it is between, not the ones already merged into a batch. */
   sz_draw_flush(ctx);

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   sz_memory_barrier(ctx->screen, cmd,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT);

   (void)flags;
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

   pipe_resource_reference(&ctx->sampled_copy, NULL);

   /* Constant buffers and sampler views are the shadow state the context owns a
    * reference to, so they are what it has to drop. */
   for (unsigned i = 0; i < ctx->state.num_vertex_buffers; i++)
      pipe_resource_reference(&ctx->state.vertex_buffers[i].buffer.resource,
                              NULL);
   pipe_resource_reference(&ctx->pending.index_resource, NULL);

   for (unsigned sh = 0; sh < MESA_SHADER_STAGES; sh++) {
      for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++)
         pipe_resource_reference(&ctx->state.constant_buffers[sh][i].buffer,
                                 NULL);
      for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; i++)
         pipe_sampler_view_reference(&ctx->state.sampler_views[sh][i], NULL);
   }

   for (unsigned i = 0; i < PIPE_MAX_SO_BUFFERS; i++) {
      pipe_so_target_reference(&ctx->so_targets[i], NULL);
      pipe_so_target_reference(&ctx->so_prev_targets[i], NULL);
   }

   for (unsigned i = 0; i < ctx->num_surface_views; i++)
      if (ctx->surface_views[i].view != VK_NULL_HANDLE &&
          ctx->surface_views[i].view != ctx->surface_views[i].res->view)
         ctx->screen->vk.DestroyImageView(ctx->screen->dev,
                                          ctx->surface_views[i].view, NULL);
   free(ctx->surface_views);

   if (ctx->so_counter_buffer != VK_NULL_HANDLE) {
      sz_bo_free(ctx->so_counter_bo);
      ctx->screen->vk.DestroyBuffer(ctx->screen->dev, ctx->so_counter_buffer,
                                    NULL);
   }

   u_upload_destroy(ctx->base.const_uploader);
   u_upload_destroy(ctx->base.stream_uploader);

   /* Nothing in ctx->state is owned here: every object was created by a create_*
    * entry point and is destroyed by its matching delete_* entry point, which the
    * state tracker always calls. Freeing them again here was a double free --
    * and the sampler arrays are shared with sampler views, so it was also a
    * type-confused free. */
   if (sz_debug_enabled(SZ_DEBUG_STATS)) {
      const struct sz_screen *screen = ctx->screen;
      mesa_logi("SZ: stats: draws=%" PRIu64 " batches=%" PRIu64
                " (multi %" PRIu64 ") pipeline_binds=%" PRIu64
                " pipeline_builds=%" PRIu64 " descriptor_pushes=%" PRIu64
                " vb_binds=%" PRIu64 " state_sets=%" PRIu64
                " submits=%" PRIu64,
                screen->stats.draws, screen->stats.batches,
                screen->stats.multi_batches, screen->stats.pipeline_binds,
                screen->stats.pipeline_builds, screen->stats.descriptor_pushes,
                screen->stats.vertex_buffer_binds, screen->stats.state_sets,
                screen->stats.submits);
   }

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

   /* The draw stream batches at most this many draws into one call. One means
    * "no VK_EXT_multi_draw", in which case every append flushes immediately and
    * the emit path falls back to one vkCmdDraw* per draw. */
   ctx->pending.capacity =
      MAX2(1u, MIN2((unsigned)SZ_MAX_BATCHED_DRAWS,
                    screen->max_multi_draw_count));

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
   pipe->create_query = sz_create_query;
   pipe->destroy_query = sz_destroy_query;
   pipe->begin_query = sz_begin_query;
   pipe->end_query = sz_end_query;
   pipe->get_query_result = sz_get_query_result;
   pipe->get_query_result_resource = sz_get_query_result_resource;
   pipe->render_condition = sz_render_condition;
   pipe->create_stream_output_target = sz_create_stream_output_target;
   pipe->stream_output_target_destroy = sz_stream_output_target_destroy;
   pipe->stream_output_target_offset = sz_stream_output_target_offset;
   pipe->set_stream_output_targets = sz_set_stream_output_targets;
   pipe->flush_resource = sz_flush_resource;
   pipe->get_sample_position = sz_get_sample_position;
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
