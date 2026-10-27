/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Descriptors. See sz_descriptor.h for the design.
 */

#include "sz_descriptor.h"

#include "sz_compiler.h"
#include "sz_context.h"
#include "sz_pipeline.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"

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

void
sz_descriptor_push(struct sz_context *ctx, VkCommandBuffer cmd)
{
   struct sz_screen *screen = ctx->screen;
   const VkPipelineLayout layout = ctx->pipeline ? ctx->pipeline->layout
                                                 : VK_NULL_HANDLE;
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

      if (count == 0)
         continue;

      screen->vk.CmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                         layout, SZ_DESCRIPTOR_SET, count,
                                         writes);
   }
}
