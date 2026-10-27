/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * pipe_context::blit.
 *
 * GL needs three different things from one entry point: a straight copy, a
 * scaled/filtered blit, and an MSAA resolve. Vulkan spells them with two
 * commands -- vkCmdBlitImage2 does the first two, vkCmdResolveImage2 the third,
 * and neither can do the other's job:
 *
 *  - a resolve requires identical formats and extents, and a source with more
 *    samples than the destination;
 *  - a blit cannot resolve, and cannot touch the stencil aspect at all.
 *
 * So the cases are dispatched explicitly rather than being papered over by one
 * "copy" path. Anything outside the implemented set is refused with a warning:
 * a blit that silently does nothing produces a picture that is wrong in a way
 * nobody can trace back to the driver.
 */

#include "sz_blit.h"

#include "sz_context.h"
#include "sz_format.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "util/format/u_formats.h"
#include "util/log.h"
#include "util/macros.h"

#include <string.h>

/* Gallium boxes are signed and may describe a flip for the source. Vulkan's
 * blit offsets are just two corners, so a flipped box is expressed by handing
 * the corners over in the reverse order. */
static void
sz_blit_offsets(const struct pipe_box *box, bool is_3d, VkOffset3D out[2])
{
   const int x0 = box->x;
   const int y0 = box->y;
   const int x1 = box->x + box->width;
   const int y1 = box->y + box->height;

   if (is_3d) {
      out[0] = (VkOffset3D) { x0, y0, box->z };
      out[1] = (VkOffset3D) { x1, y1, box->z + box->depth };
   } else {
      out[0] = (VkOffset3D) { x0, y0, 0 };
      out[1] = (VkOffset3D) { x1, y1, 1 };
   }
}

static void
sz_blit_subresource(const struct pipe_resource *res, unsigned level,
                    const struct pipe_box *box, VkImageAspectFlags aspect,
                    VkImageSubresourceLayers *out)
{
   *out = (VkImageSubresourceLayers) {
      .aspectMask = aspect,
      .mipLevel = level,
      .baseArrayLayer = res->target == PIPE_TEXTURE_3D ? 0 : box->z,
      .layerCount = res->target == PIPE_TEXTURE_3D ? 1 : MAX2(box->depth, 1),
   };
}

/* Positive extent of a box, which is what Vulkan wants for a resolve. */
static VkExtent3D
sz_box_extent(const struct pipe_box *box)
{
   return (VkExtent3D) {
      .width = abs(box->width),
      .height = abs(box->height),
      .depth = abs(box->depth),
   };
}

static bool
sz_box_equals(const struct pipe_box *a, const struct pipe_box *b)
{
   const VkExtent3D ea = sz_box_extent(a);
   const VkExtent3D eb = sz_box_extent(b);
   return ea.width == eb.width && ea.height == eb.height && ea.depth == eb.depth;
}

static void
sz_blit_resolve(struct sz_context *ctx, VkCommandBuffer cmd,
                const struct pipe_blit_info *info, struct sz_resource *src,
                struct sz_resource *dst, VkImageAspectFlags aspect)
{
   const bool src_3d = info->src.resource->target == PIPE_TEXTURE_3D;
   const bool dst_3d = info->dst.resource->target == PIPE_TEXTURE_3D;

   VkOffset3D src_off[2], dst_off[2];
   sz_blit_offsets(&info->src.box, src_3d, src_off);
   sz_blit_offsets(&info->dst.box, dst_3d, dst_off);

   VkImageResolve2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2,
      .srcOffset = src_off[0],
      .dstOffset = dst_off[0],
      .extent = sz_box_extent(&info->src.box),
   };
   sz_blit_subresource(info->src.resource, info->src.level, &info->src.box,
                       aspect, &region.srcSubresource);
   sz_blit_subresource(info->dst.resource, info->dst.level, &info->dst.box,
                       aspect, &region.dstSubresource);

   ctx->screen->stats.resolves++;

   const VkResolveImageInfo2 resolve = {
      .sType = VK_STRUCTURE_TYPE_RESOLVE_IMAGE_INFO_2,
      .srcImage = src->image,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage = dst->image,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &region,
   };

   ctx->screen->vk.CmdResolveImage2(cmd, &resolve);
}

static void
sz_blit_image(struct sz_context *ctx, VkCommandBuffer cmd,
              const struct pipe_blit_info *info, struct sz_resource *src,
              struct sz_resource *dst, VkImageAspectFlags aspect)
{
   const bool src_3d = info->src.resource->target == PIPE_TEXTURE_3D;
   const bool dst_3d = info->dst.resource->target == PIPE_TEXTURE_3D;

   VkOffset3D src_off[2], dst_off[2];
   sz_blit_offsets(&info->src.box, src_3d, src_off);
   sz_blit_offsets(&info->dst.box, dst_3d, dst_off);

   VkImageBlit2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2,
   };
   memcpy(region.srcOffsets, src_off, sizeof(src_off));
   memcpy(region.dstOffsets, dst_off, sizeof(dst_off));
   sz_blit_subresource(info->src.resource, info->src.level, &info->src.box,
                       aspect, &region.srcSubresource);
   sz_blit_subresource(info->dst.resource, info->dst.level, &info->dst.box,
                       aspect, &region.dstSubresource);

   const VkBlitImageInfo2 blit = {
      .sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2,
      .srcImage = src->image,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage = dst->image,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &region,
      .filter = info->filter == PIPE_TEX_FILTER_LINEAR ? VK_FILTER_LINEAR
                                                       : VK_FILTER_NEAREST,
   };

   ctx->screen->vk.CmdBlitImage2(cmd, &blit);
}

/* A same-size, same-format copy. Used for depth, where a blit may be
 * unsupported, and for stencil, which has no blit path at all. */
static void
sz_blit_copy(struct sz_context *ctx, VkCommandBuffer cmd,
             const struct pipe_blit_info *info, struct sz_resource *src,
             struct sz_resource *dst, VkImageAspectFlags aspect)
{
   VkImageCopy2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .extent = sz_box_extent(&info->src.box),
      .srcOffset = { info->src.box.x, info->src.box.y, 0 },
      .dstOffset = { info->dst.box.x, info->dst.box.y, 0 },
   };
   sz_blit_subresource(info->src.resource, info->src.level, &info->src.box,
                       aspect, &region.srcSubresource);
   sz_blit_subresource(info->dst.resource, info->dst.level, &info->dst.box,
                       aspect, &region.dstSubresource);

   ctx->screen->stats.copies++;

   const VkCopyImageInfo2 copy = {
      .sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2,
      .srcImage = src->image,
      .srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .dstImage = dst->image,
      .dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .regionCount = 1,
      .pRegions = &region,
   };

   ctx->screen->vk.CmdCopyImage2(cmd, &copy);
}

/* Stencil has no blit path in Vulkan at all, and a resolve never covers it.
 * A same-format, same-size copy is the only correct answer, and it is also what
 * glBlitFramebuffer means for the stencil aspect in practice. */
static void
sz_blit_stencil(struct sz_context *ctx, VkCommandBuffer cmd,
                const struct pipe_blit_info *info, struct sz_resource *src,
                struct sz_resource *dst)
{
   if (!sz_box_equals(&info->src.box, &info->dst.box)) {
      sz_warn_once("scaled stencil blit is not implemented");
      return;
   }

   sz_blit_copy(ctx, cmd, info, src, dst, VK_IMAGE_ASPECT_STENCIL_BIT);
}


static void
sz_blit_aspect(struct sz_context *ctx, VkCommandBuffer cmd,
               const struct pipe_blit_info *info, struct sz_resource *src,
               struct sz_resource *dst, VkImageAspectFlags aspect,
               VkFormat src_format, VkFormat dst_format, unsigned src_samples,
               unsigned dst_samples)
{
   if (src_format != dst_format) {
      sz_warn_once("blit between different Vulkan formats is not implemented "
                   "(%s vs %s)",
                   util_format_name(info->src.resource->format),
                   util_format_name(info->dst.resource->format));
      return;
   }

   sz_resource_access(ctx->screen, cmd, src, SZ_ACCESS_TRANSFER_SRC);
   sz_resource_access(ctx->screen, cmd, dst, SZ_ACCESS_TRANSFER_DST);

   /* A resolve needs the same extent, a multisampled source and a
    * single-sampled destination. Everything else is a blit, which handles
    * scaling and filtering.
    *
    * Depth is the exception: Vulkan only allows a depth blit when the format
    * advertises VK_FORMAT_FEATURE_BLIT_SRC/DST, which several drivers (RADV
    * among them) do not for depth formats. A same-size, same-format copy is
    * always allowed and is also exactly what glBlitFramebuffer means when
    * nothing is scaled, so that is the path taken first. */
   const bool same_size = sz_box_equals(&info->src.box, &info->dst.box);
   const bool resolve = src_samples > 1 && dst_samples == 1 && same_size;
   const bool depth_copy =
      aspect == VK_IMAGE_ASPECT_DEPTH_BIT && same_size && src_samples == 1 &&
      dst_samples == 1 && info->src.box.width >= 0 && info->src.box.height >= 0;

   if (resolve)
      sz_blit_resolve(ctx, cmd, info, src, dst, aspect);
   else if (depth_copy)
      sz_blit_copy(ctx, cmd, info, src, dst, aspect);
   else
      sz_blit_image(ctx, cmd, info, src, dst, aspect);
}

void
sz_blit(struct pipe_context *pipe, const struct pipe_blit_info *info)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_screen *screen = ctx->screen;

   if (!info->src.resource || !info->dst.resource)
      return;

   struct sz_resource *src = sz_resource(info->src.resource);
   struct sz_resource *dst = sz_resource(info->dst.resource);
   if (src->image == VK_NULL_HANDLE || dst->image == VK_NULL_HANDLE)
      return;

   /* Window rectangles, render conditions, blending and swizzling each need a
    * different mechanism (scissored blits, conditional rendering, a shader).
    * Refusing is honest; doing half of it is a wrong picture. */
   if (info->num_window_rectangles || info->render_condition_enable ||
       info->alpha_blend || info->swizzle_enable) {
      sz_warn_once("blit with window rectangles, a render condition, alpha "
                   "blending or swizzle is not implemented");
      return;
   }

   const VkFormat src_format = sz_vk_format(screen, info->src.resource->format);
   const VkFormat dst_format = sz_vk_format(screen, info->dst.resource->format);
   const unsigned src_samples = MAX2(info->src.resource->nr_samples, 1);
   const unsigned dst_samples = MAX2(info->dst.resource->nr_samples, 1);

   ctx->screen->stats.blits++;

   /* Both commands are illegal inside a render pass, and anything already
    * queued must land before the blit does. */
   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return;

   if (info->mask & PIPE_MASK_RGBA)
      sz_blit_aspect(ctx, cmd, info, src, dst, VK_IMAGE_ASPECT_COLOR_BIT,
                     src_format, dst_format, src_samples, dst_samples);

   if (info->mask & PIPE_MASK_Z)
      sz_blit_aspect(ctx, cmd, info, src, dst, VK_IMAGE_ASPECT_DEPTH_BIT,
                     src_format, dst_format, src_samples, dst_samples);

   if (info->mask & PIPE_MASK_S) {
      if (src_samples > 1 || dst_samples > 1) {
         sz_warn_once("multisampled stencil blit is not implemented");
      } else if (src_format != dst_format) {
         sz_warn_once("stencil blit between different formats is not "
                      "implemented");
      } else {
         sz_resource_access(screen, cmd, src, SZ_ACCESS_TRANSFER_SRC);
         sz_resource_access(screen, cmd, dst, SZ_ACCESS_TRANSFER_DST);
         sz_blit_stencil(ctx, cmd, info, src, dst);
      }
   }
}
