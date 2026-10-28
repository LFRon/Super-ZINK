/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_sync.h"

#include "sz_format.h"

#include "util/format/u_format.h"
#include "util/log.h"
#include "util/macros.h"

/* The purpose maps onto (stage, access, layout). The three always travel
 * together: a layout that does not match the access mask is a validation error
 * on strict drivers and a silent performance cliff on all of them.
 *
 * `depth_stencil` asks for the combined depth/stencil layout, which is what a
 * format carrying a stencil aspect has to be in whenever the stencil aspect is
 * used as an attachment (VUID-vkCmdBeginRendering-pStencilAttachment-06098).
 * Using the depth-only layout there is invalid, not merely slower. */
static struct sz_sync_state
sz_sync_state_for(enum sz_access access, bool depth_stencil)
{
   switch (access) {
   case SZ_ACCESS_COLOR_ATTACH:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
         .access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                   VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT,
         .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      };

   case SZ_ACCESS_DEPTH_ATTACH:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
         .access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
         .layout = depth_stencil ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                                 : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      };

   case SZ_ACCESS_SAMPLED:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
         .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      };

   case SZ_ACCESS_STORAGE:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .access = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                   VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
         .layout = VK_IMAGE_LAYOUT_GENERAL,
      };

   case SZ_ACCESS_TRANSFER_SRC:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_COPY_BIT |
                   VK_PIPELINE_STAGE_2_BLIT_BIT |
                   VK_PIPELINE_STAGE_2_RESOLVE_BIT,
         .access = VK_ACCESS_2_TRANSFER_READ_BIT,
         .layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      };

   case SZ_ACCESS_TRANSFER_DST:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_COPY_BIT |
                   VK_PIPELINE_STAGE_2_BLIT_BIT |
                   VK_PIPELINE_STAGE_2_CLEAR_BIT |
                   VK_PIPELINE_STAGE_2_RESOLVE_BIT,
         .access = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      };

   case SZ_ACCESS_VERTEX_BUFFER:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT,
         .access = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT,
         .layout = VK_IMAGE_LAYOUT_UNDEFINED,
      };

   case SZ_ACCESS_INDEX_BUFFER:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_INDEX_INPUT_BIT,
         .access = VK_ACCESS_2_INDEX_READ_BIT,
         .layout = VK_IMAGE_LAYOUT_UNDEFINED,
      };

   case SZ_ACCESS_UNIFORM_BUFFER:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .access = VK_ACCESS_2_UNIFORM_READ_BIT,
         .layout = VK_IMAGE_LAYOUT_UNDEFINED,
      };

   case SZ_ACCESS_TEXEL_BUFFER:
      /* A buffer texture is read through the same uniform-read path as a UBO:
       * SPIR-V sees it as OpImageFetch on a DimBuffer image, and Vulkan counts
       * that as a read access, not as a sampled-image read. */
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .access = VK_ACCESS_2_UNIFORM_READ_BIT,
         .layout = VK_IMAGE_LAYOUT_UNDEFINED,
      };

   case SZ_ACCESS_PRESENT:
      return (struct sz_sync_state) {
         .stages = VK_PIPELINE_STAGE_2_NONE,
         .access = 0,
         .layout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      };
   }

   UNREACHABLE("unknown sz_access");
}

/* Recovers the producer stage/access from the layout recorded on the resource.
 * Returns false for an untracked layout, in which case the contents must be
 * treated as disposable. */
static bool
sz_sync_state_from_layout(VkImageLayout layout, struct sz_sync_state *out)
{
   *out = (struct sz_sync_state) { .layout = layout, .stages = 0, .access = 0 };

   switch (layout) {
   case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
      out->stages = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      out->access = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
      return true;
   case VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL:
   case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
      out->stages = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
      out->access = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
      return true;
   case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
      out->stages = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      out->access = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
      return true;
   case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
      out->stages = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
      out->access = VK_ACCESS_2_TRANSFER_READ_BIT;
      return true;
   case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
      out->stages = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT |
                    VK_PIPELINE_STAGE_2_CLEAR_BIT;
      out->access = VK_ACCESS_2_TRANSFER_WRITE_BIT;
      return true;
   case VK_IMAGE_LAYOUT_GENERAL:
      out->stages = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
      out->access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
      return true;
   case VK_IMAGE_LAYOUT_PRESENT_SRC_KHR:
      /* The presentation engine already consumed it; nothing to flush. */
      out->stages = VK_PIPELINE_STAGE_2_NONE;
      out->access = 0;
      return true;
   default:
      return false;
   }
}

/* Whether an access mask contains any write. A barrier only has to be recorded
 * for a reader when the previous use actually wrote. */
static bool
sz_access_has_write(VkAccessFlags2 access)
{
   return (access & (VK_ACCESS_2_SHADER_WRITE_BIT |
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                     VK_ACCESS_2_TRANSFER_WRITE_BIT |
                     VK_ACCESS_2_HOST_WRITE_BIT |
                     VK_ACCESS_2_MEMORY_WRITE_BIT |
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)) != 0;
}

/* The stage/access a resource is *already* visible through.
 *
 * VK_IMAGE_LAYOUT_UNDEFINED is only legal when the old contents may be thrown
 * away. Using it when they must be preserved silently discards them on tilers,
 * which is why the source state is derived from the resource and not assumed. */
static void
sz_sync_source(const struct sz_resource *res, VkImageLayout *out_layout,
               VkPipelineStageFlags2 *out_stage, VkAccessFlags2 *out_access)
{
   struct sz_sync_state current;

   if (res->undefined_contents ||
       !sz_sync_state_from_layout(res->layout, &current)) {
      /* Untracked layout: treat the contents as disposable rather than
       * claiming they are valid. */
      *out_layout = VK_IMAGE_LAYOUT_UNDEFINED;
      *out_stage = VK_PIPELINE_STAGE_2_NONE;
      *out_access = 0;
      return;
   }

   *out_layout = current.layout;
   *out_stage = current.stages;
   *out_access = current.access;
}

static uint32_t
sz_image_layer_count(const struct sz_resource *res)
{
   switch (res->base.target) {
   case PIPE_TEXTURE_3D:
      /* 3D images are not layered; VK_REMAINING_ARRAY_LAYERS is the only legal
       * value. */
      return VK_REMAINING_ARRAY_LAYERS;
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      /* Gallium counts a cube's six faces in array_size, so no multiplication
       * happens here -- doing it once too often made every barrier on a cube map
       * describe six times the layers the image has. */
      return MAX2(res->base.array_size, 1);
   default:
      return MAX2(res->base.array_size, 1);
   }
}

void
sz_resource_discard(struct sz_resource *res)
{
   if (!res || res->buffer != VK_NULL_HANDLE)
      return;
   res->undefined_contents = true;
}


/* The layout one mip level is in. `layout` is the answer unless that level was
 * transitioned on its own and recorded in `levels`. */
static VkImageLayout
sz_level_layout(const struct sz_resource *res, unsigned level)
{
   if (res->levels && level <= res->base.last_level)
      return res->levels[level];
   return res->layout;
}

static void
sz_level_set_layout(struct sz_resource *res, unsigned level, VkImageLayout layout)
{
   /* Only the "no overrides yet and none needed" case may return early. Once
    * the array exists, a level returning to the default layout still has to be
    * written: leaving the old value there makes the next transition claim a
    * starting layout the image is not in, which is
    * VUID-VkImageMemoryBarrier2-oldLayout-01197 and, worse, a barrier that
    * does not actually wait for what it says it does. */
   if (!res->levels && layout == res->layout)
      return;

   if (!res->levels) {
      res->levels = calloc(res->base.last_level + 1, sizeof(*res->levels));
      if (!res->levels)
         return;
      for (unsigned i = 0; i <= res->base.last_level; i++)
         res->levels[i] = res->layout;
   }

   res->levels[level] = layout;
}

/* Whether every level is back in `res->layout`, which is when the array can go. */
static void
sz_levels_drop_if_uniform(struct sz_resource *res)
{
   if (!res->levels)
      return;

   for (unsigned i = 0; i <= res->base.last_level; i++)
      if (res->levels[i] != res->layout)
         return;

   free(res->levels);
   res->levels = NULL;
}

VkImageLayout
sz_resource_access(struct sz_screen *screen, VkCommandBuffer cmd,
                   struct sz_resource *res, enum sz_access access)
{
   return sz_resource_access_levels(screen, cmd, res, access, 0,
                                    VK_REMAINING_MIP_LEVELS);
}

VkImageLayout
sz_resource_access_levels(struct sz_screen *screen, VkCommandBuffer cmd,
                          struct sz_resource *res, enum sz_access access,
                          unsigned base_level, unsigned level_count)
{
   if (!res)
      return VK_IMAGE_LAYOUT_UNDEFINED;

   /* A combined depth/stencil image has to be in DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    * for the render pass to attach its stencil aspect. Depth-only formats keep
    * the narrower layout, which lets tilers compress it as depth alone. */
   const bool depth_stencil =
      access == SZ_ACCESS_DEPTH_ATTACH && res->base.format &&
      util_format_has_stencil(util_format_description(res->base.format));

   const struct sz_sync_state dst = sz_sync_state_for(access, depth_stencil);

   /* Buffers have no layout, so the only thing to synchronise is a pending
    * write. A buffer that was never written needs nothing, which is the common
    * case for a static vertex buffer drawn every frame. */
   if (res->buffer != VK_NULL_HANDLE) {
      /* Host writes through a mapped pointer are only visible to the device
       * after a barrier whose source is HOST/HOST_WRITE. The uploaders write
       * vertices and uniforms that way, so this is the common case for a
       * buffer, not an exotic one. */
      VkPipelineStageFlags2 src_stages = res->pending_write_stages;
      VkAccessFlags2 src_access = res->pending_write_access;
      if (res->host_written) {
         src_stages |= VK_PIPELINE_STAGE_2_HOST_BIT;
         src_access |= VK_ACCESS_2_HOST_WRITE_BIT;
      }

      if (src_stages != 0) {
         const VkBufferMemoryBarrier2 barrier = {
            .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
            .srcStageMask = src_stages,
            .srcAccessMask = src_access,
            .dstStageMask = dst.stages,
            .dstAccessMask = dst.access,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .buffer = res->buffer,
            .offset = 0,
            .size = VK_WHOLE_SIZE,
         };

         const VkDependencyInfo dep = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .bufferMemoryBarrierCount = 1,
            .pBufferMemoryBarriers = &barrier,
         };

         screen->vk.CmdPipelineBarrier2(cmd, &dep);
         res->pending_write_stages = 0;
         res->pending_write_access = 0;
         res->host_written = false;
      }

      /* Marked before the write is recorded, not after: the caller issues the
       * copy or fill immediately after this returns, so in command order the
       * write precedes any later read's barrier either way. */
      if (sz_access_has_write(dst.access)) {
         res->pending_write_stages = dst.stages;
         res->pending_write_access = dst.access;
      }

      return VK_IMAGE_LAYOUT_UNDEFINED;
   }

   /* Nothing to do when the layout already matches and the contents are still
    * valid. Skipping is not merely an optimisation: a barrier whose source mask
    * is empty against a non-UNDEFINED old layout stalls the pipeline on some
    * drivers for no benefit. */
   if (!res->undefined_contents && !res->levels && res->layout == dst.layout)
      return res->layout;

   VkImageLayout old_layout;
   VkPipelineStageFlags2 src_stage;
   VkAccessFlags2 src_access;
   sz_sync_source(res, &old_layout, &src_stage, &src_access);

   /* One barrier per run of consecutive levels that share a layout. A mipmap
    * chain normally has exactly one run -- every level is in the same layout --
    * but generating level N reads N-1 and writes N, and a single barrier
    * covering the whole chain then tells the driver the wrong starting layout
    * for one of them. That is not a validation error; it is a wrong picture, or
    * a hang on a driver that trusts the claim. */
   const VkImageAspectFlags aspect = sz_format_aspect(res->base.format);

   const unsigned last_level =
      level_count == VK_REMAINING_MIP_LEVELS
         ? res->base.last_level
         : MIN2(base_level + level_count - 1, res->base.last_level);

   for (unsigned level = base_level; level <= last_level; level++) {
      if (sz_level_layout(res, level) == dst.layout)
         continue;

      /* A run is levels that are in the *same starting layout*, not merely
       * levels that all need changing: the barrier states one oldLayout, and
       * claiming TRANSFER_SRC for a level that is TRANSFER_DST is
       * VUID-VkImageMemoryBarrier2-oldLayout-01197 and a barrier that does not
       * wait for the write it was supposed to. */
      uint32_t old = res->undefined_contents ? VK_IMAGE_LAYOUT_UNDEFINED
                                             : sz_level_layout(res, level);
      unsigned run_end = level;
      while (run_end + 1 <= last_level &&
             (res->undefined_contents
                 ? VK_IMAGE_LAYOUT_UNDEFINED
                 : sz_level_layout(res, run_end + 1)) == old &&
             sz_level_layout(res, run_end + 1) != dst.layout)
         run_end++;

      const VkImageMemoryBarrier2 barrier = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = src_stage,
         .srcAccessMask = src_access,
         .dstStageMask = dst.stages,
         .dstAccessMask = dst.access,
         .oldLayout = old,
         .newLayout = dst.layout,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = res->image,
         .subresourceRange = {
            .aspectMask = aspect,
            .baseMipLevel = level,
            .levelCount = run_end - level + 1,
            .baseArrayLayer = 0,
            .layerCount = sz_image_layer_count(res),
         },
      };

      const VkDependencyInfo dep = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 1,
         .pImageMemoryBarriers = &barrier,
      };
      screen->vk.CmdPipelineBarrier2(cmd, &dep);

      for (unsigned i = level; i <= run_end; i++)
         sz_level_set_layout(res, i, dst.layout);
      level = run_end;
   }

   /* A ranged transition says nothing about the levels it did not touch, so the
    * whole-image answer only becomes `dst.layout` when every level is in it. */
   if (base_level == 0 && last_level == res->base.last_level)
      res->layout = dst.layout;
   res->undefined_contents = false;
   sz_levels_drop_if_uniform(res);

   return sz_level_layout(res, base_level);
}

void
sz_memory_barrier(struct sz_screen *screen, VkCommandBuffer cmd,
                  VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                  VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
   const VkMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = dst_stage,
      .dstAccessMask = dst_access,
   };

   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &barrier,
   };
   screen->vk.CmdPipelineBarrier2(cmd, &dep);
}
