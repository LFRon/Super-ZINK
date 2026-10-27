/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Descriptors. See sz_descriptor.h for the design.
 */

#include "sz_descriptor.h"

#include "sz_compiler.h"
#include "sz_context.h"
#include "sz_format.h"
#include "sz_pipeline.h"
#include "sz_renderpass.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"

#include "vk_format.h"

/* Bindings in the one push set: SZ_MAX_CONST_BUFFERS uniform buffers for each of
 * the stages sz_compiler.h reserves room for. Compute is deliberately absent --
 * the layout cannot hold three stages and still stay inside the 32 descriptors
 * every device is guaranteed (see SZ_DESCRIPTOR_STAGES). */
static const mesa_shader_stage sz_descriptor_stages[SZ_DESCRIPTOR_STAGES] = {
   MESA_SHADER_VERTEX, MESA_SHADER_FRAGMENT,
};

static bool
sz_descriptor_layout_create(struct sz_screen *screen, VkDescriptorSetLayout *out)
{
   VkDescriptorSetLayoutBinding bindings[SZ_DESCRIPTOR_BINDINGS];

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];
      const VkShaderStageFlags flags = sz_descriptor_stage_flags(stage);
      for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++) {
         const unsigned binding = sz_descriptor_binding(stage, i);
         bindings[binding] = (VkDescriptorSetLayoutBinding) {
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = flags,
         };
      }
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         const unsigned binding = sz_sampler_binding(stage, i);
         bindings[binding] = (VkDescriptorSetLayoutBinding) {
            .binding = binding,
            /* A combined image sampler rather than separate image and sampler
             * bindings: GL's sampler uniforms carry both, and one binding per
             * texture unit is half the descriptor budget of two. */
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = flags,
         };
      }
   }

   const VkDescriptorSetLayoutCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      /* ADR-008: nothing is ever allocated from a pool. The flag is what makes
       * vkCmdPushDescriptorSet legal against this layout. */
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
      .bindingCount = SZ_DESCRIPTOR_BINDINGS,
      .pBindings = bindings,
   };

   const VkResult result =
      screen->vk.CreateDescriptorSetLayout(screen->dev, &ci, NULL, out);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateDescriptorSetLayout failed: %d", result);
      *out = VK_NULL_HANDLE;
      return false;
   }

   return true;
}

bool
sz_descriptor_screen_init(struct sz_screen *screen, unsigned max_push_descriptors)
{
   /* VUID 00281: the layout's total descriptor count may not exceed
    * maxPushDescriptors. The layout is fixed, so this is a screen-creation
    * decision rather than something a draw can work around. */
   if (max_push_descriptors != 0 &&
       max_push_descriptors < SZ_DESCRIPTOR_BINDINGS) {
      mesa_loge("SZ: the device pushes only %u descriptors, but the uniform "
                "buffer layout needs %u", max_push_descriptors,
                SZ_DESCRIPTOR_BINDINGS);
      return false;
   }

   return sz_descriptor_layout_create(screen, &screen->descriptor_set_layout);
}

void
sz_descriptor_screen_finish(struct sz_screen *screen)
{
   if (screen->descriptor_set_layout != VK_NULL_HANDLE) {
      screen->vk.DestroyDescriptorSetLayout(screen->dev,
                                            screen->descriptor_set_layout,
                                            NULL);
      screen->descriptor_set_layout = VK_NULL_HANDLE;
   }
}

/* The image behind one combined image sampler slot, or NULL when the slot is not
 * usable as a descriptor: Vulkan rejects a combined image sampler with either
 * half missing, and GL allows a sampler uniform to be sampled with no texture
 * bound, which reads as black rather than as an error. */
static struct sz_resource *
sz_sampled_resource(const struct sz_context *ctx, mesa_shader_stage stage,
                    unsigned slot, const struct sz_state **state_out,
                    const struct sz_sampler_view **view_out)
{
   const struct pipe_sampler_view *view = ctx->state.sampler_views[stage][slot];
   const struct sz_state *state = ctx->state.sampler_states[stage][slot];
   if (!view || !state || state->u.sampler.sampler == VK_NULL_HANDLE)
      return NULL;

   const struct sz_sampler_view *sv = (const struct sz_sampler_view *)view;
   if (sv->view == VK_NULL_HANDLE)
      return NULL;

   struct sz_resource *res = sz_resource(view->texture);
   if (!res || res->image == VK_NULL_HANDLE)
      return NULL;

   if (state_out)
      *state_out = state;
   if (view_out)
      *view_out = sv;
   return res;
}

bool
sz_descriptor_sampled_needs_transition(struct sz_context *ctx)
{
   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         struct sz_resource *res =
            sz_sampled_resource(ctx, sz_descriptor_stages[s], i, NULL, NULL);
         if (!res)
            continue;

         if (res->undefined_contents ||
             res->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            return true;
      }
   }

   return false;
}

/* Whether the resource is one of the attachments of the render pass that is
 * currently open. The layout alone cannot answer this: an image keeps
 * COLOR_ATTACHMENT_OPTIMAL after its pass has ended, and sampling it then is
 * perfectly legal as long as it is transitioned first. Only a texture that is
 * still attached is the undefined case. */
static bool
sz_resource_is_rendering_target(struct sz_context *ctx, struct sz_resource *res)
{
   const struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   for (unsigned i = 0; i < fb->nr_cbufs; i++) {
      if (fb->cbufs[i].texture && sz_resource(fb->cbufs[i].texture) == res)
         return true;
   }

   return fb->zsbuf.texture && sz_resource(fb->zsbuf.texture) == res;
}

/* Copies the image a draw is about to render into so the draw can sample the
 * copy. GL leaves a feedback loop undefined, but every real driver produces the
 * pixels rendered so far, and a browser compositor relies on that: RADV's
 * compressed surfaces make the loop a screenful of noise instead. */
static bool
sz_sampled_feedback_copy(struct sz_context *ctx, VkCommandBuffer cmd,
                         struct sz_resource *res, VkImageView *view_out)
{
   struct sz_screen *screen = ctx->screen;
   const VkFormat format = sz_vk_format(screen, res->base.format);

   if (!ctx->sampled_copy || ctx->sampled_copy_format != format ||
       ctx->sampled_copy_extent[0] != res->base.width0 ||
       ctx->sampled_copy_extent[1] != res->base.height0) {
      pipe_resource_reference(&ctx->sampled_copy, NULL);

      struct pipe_resource tmpl = {
         .target = PIPE_TEXTURE_2D,
         .format = res->base.format,
         .width0 = res->base.width0,
         .height0 = res->base.height0,
         .depth0 = 1,
         .array_size = 1,
         .last_level = 0,
         .nr_samples = 1,
         .usage = PIPE_USAGE_DEFAULT,
         .bind = PIPE_BIND_SAMPLER_VIEW,
      };
      ctx->sampled_copy = screen->base.resource_create(&screen->base, &tmpl);
      if (!ctx->sampled_copy) {
         mesa_loge("SZ: out of memory allocating a feedback copy");
         return false;
      }
      ctx->sampled_copy_format = format;
      ctx->sampled_copy_extent[0] = res->base.width0;
      ctx->sampled_copy_extent[1] = res->base.height0;
   }

   struct sz_resource *copy = sz_resource(ctx->sampled_copy);
   if (copy->view == VK_NULL_HANDLE)
      return false;

   sz_resource_access(screen, cmd, res, SZ_ACCESS_TRANSFER_SRC);
   sz_resource_access(screen, cmd, copy, SZ_ACCESS_TRANSFER_DST);

   const VkImageCopy region = {
      .srcSubresource = { sz_format_aspect(res->base.format), 0, 0, 1 },
      .dstSubresource = { sz_format_aspect(res->base.format), 0, 0, 1 },
      .extent = { res->base.width0, res->base.height0, 1 },
   };
   screen->vk.CmdCopyImage(cmd, res->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           copy->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);

   /* Back to an attachment: the draw path re-opens the pass on this image right
    * after this returns. The copy itself stays readable. */
   sz_resource_access(screen, cmd, res, SZ_ACCESS_COLOR_ATTACH);
   sz_resource_access(screen, cmd, copy, SZ_ACCESS_SAMPLED);

   *view_out = copy->view;
   return true;
}

void
sz_descriptor_prepare_sampled(struct sz_context *ctx, VkCommandBuffer cmd)
{
   memset(ctx->sampled_override.view, 0, sizeof(ctx->sampled_override.view));

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         struct sz_resource *res =
            sz_sampled_resource(ctx, sz_descriptor_stages[s], i, NULL, NULL);
         if (!res)
            continue;

         /* Sampling an image that is also one of this framebuffer's attachments
          * cannot work: the transition alone would make the render pass render
          * into a read-only image, and sampling the tiled attachment without it
          * reads noise. A copy of what has been rendered so far is what the GL
          * caller means. */
         if (sz_resource_is_rendering_target(ctx, res)) {
            if (ctx->rendering)
               sz_renderpass_end(ctx, cmd);
            if (sz_sampled_feedback_copy(ctx, cmd, res,
                                         &ctx->sampled_override.view[s][i]))
               continue;
         }

         sz_resource_access(ctx->screen, cmd, res, SZ_ACCESS_SAMPLED);
      }
   }
}

void
sz_descriptor_push(struct sz_context *ctx, VkCommandBuffer cmd)
{
   struct sz_screen *screen = ctx->screen;
   /* One layout for the whole screen (ADR-005c), so the push does not depend on
    * which pipeline is bound -- only on the descriptor set layout, which never
    * changes. */
   const VkPipelineLayout layout = screen->pipeline_layout;
   if (layout == VK_NULL_HANDLE)
      return;

   /* One call per stage, all into the same set. A push updates only the bindings
    * it names, so the second call never disturbs what the first one wrote, and
    * each call stays far below maxPushDescriptors. */
   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];

      VkDescriptorBufferInfo infos[SZ_MAX_CONST_BUFFERS];
      VkWriteDescriptorSet writes[SZ_MAX_CONST_BUFFERS];
      unsigned count = 0;

      for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++) {
         const struct pipe_constant_buffer *cb =
            &ctx->state.constant_buffers[stage][i];
         if (!cb->buffer)
            continue;

         struct sz_resource *res = sz_resource(cb->buffer);
         if (res->buffer == VK_NULL_HANDLE)
            continue;

         sz_resource_access(screen, cmd, res, SZ_ACCESS_UNIFORM_BUFFER);

         /* A zero range means "the rest of the buffer", which Vulkan spells
          * VK_WHOLE_SIZE; passing zero through is a validation error. */
         const uint64_t offset = cb->buffer_offset;
         uint64_t range = cb->buffer_size ? cb->buffer_size
                                          : VK_WHOLE_SIZE;
         if (range != VK_WHOLE_SIZE)
            range = MIN2(range, (uint64_t)cb->buffer->width0 - offset);

         infos[count] = (VkDescriptorBufferInfo) {
            .buffer = res->buffer,
            .offset = offset,
            .range = range,
         };

         writes[count] = (VkWriteDescriptorSet) {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            /* dstSet is ignored by a push; the write goes to the layout the
             * pipeline was created with, at dstBinding. */
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = sz_descriptor_binding(stage, i),
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &infos[count],
         };

         count++;
      }

      /* Textures follow the constant buffers in the same push. A unit is
       * written only when it has both a view and a sampler: Vulkan rejects a
       * combined image sampler with either half missing, and GL allows a
       * sampler uniform to be sampled with no texture bound, which reads as
       * black rather than as an error. */
      VkDescriptorImageInfo image_infos[SZ_MAX_SAMPLERS];
      VkWriteDescriptorSet image_writes[SZ_MAX_SAMPLERS];
      unsigned image_count = 0;

      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         const struct sz_state *state = NULL;
         const struct sz_sampler_view *sv = NULL;
         struct sz_resource *res =
            sz_sampled_resource(ctx, stage, i, &state, &sv);
         if (!res)
            continue;

         image_infos[image_count] = (VkDescriptorImageInfo) {
            .sampler = state->u.sampler.sampler,
            .imageView = ctx->sampled_override.view[s][i]
                            ? ctx->sampled_override.view[s][i]
                            : sv->view,
            /* sz_descriptor_prepare_sampled() put the image in this layout
             * before the render pass opened; a descriptor that claims otherwise
             * makes the sampled reads undefined. */
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         };

         image_writes[image_count] = (VkWriteDescriptorSet) {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = sz_sampler_binding(stage, i),
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_infos[image_count],
         };

         image_count++;
      }

      if (count == 0 && image_count == 0)
         continue;

      /* Buffers and images cannot share one call -- each write names its own
       * descriptor type and pNext-free info array -- but they can share the
       * set, and a push only touches the bindings it names. */
      if (count) {
         screen->vk.CmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            layout, SZ_DESCRIPTOR_SET, count,
                                            writes);
      }
      if (image_count) {
         screen->vk.CmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            layout, SZ_DESCRIPTOR_SET,
                                            image_count, image_writes);
      }
   }
}
