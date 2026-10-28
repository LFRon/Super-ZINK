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

#include "util/format/u_format.h"
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

/* Positive extent of a box, which is what Vulkan wants for a resolve.
 *
 * For a layered target the box's depth is a layer count, and Vulkan keeps that in
 * imageSubresourceLayers: for a 1D or 2D image extent.depth has to be 1
 * (VUID-vkCmdBlitImage2-pRegions-00217). Only a 3D image has depth here. */
static VkExtent3D
sz_box_extent(const struct pipe_box *box, bool is_3d)
{
   return (VkExtent3D) {
      .width = abs(box->width),
      .height = abs(box->height),
      .depth = is_3d ? abs(box->depth) : 1,
   };
}

static bool
sz_box_equals(const struct pipe_box *a, const struct pipe_box *b)
{
   const VkExtent3D ea = sz_box_extent(a, true);
   const VkExtent3D eb = sz_box_extent(b, true);
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
      .extent = sz_box_extent(&info->src.box, src_3d),
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
   const bool src_3d = info->src.resource->target == PIPE_TEXTURE_3D;

   VkImageCopy2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .extent = sz_box_extent(&info->src.box, src_3d),
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


/* Blits between two formats that Vulkan will not blit between.
 *
 * vkCmdBlitImage only accepts "compatible" formats -- same size and family --
 * so the format conversions GL allows (RGBA8 to RGBA32F, sRGB to linear, UINT to
 * float) have no vkCmd* equivalent at all. The state tracker hits this on the
 * ordinary glBlitFramebuffer path, not on some exotic one.
 *
 * The conversion therefore happens where Mesa already has a complete and tested
 * implementation for it: util_format_translate. Both ends go through the same
 * transfer path the driver already uses for texture upload and readback, which
 * means staging allocations, barriers and waiting are all already handled -- and
 * the cost is two round trips, which is the right trade for a blit that changes
 * format. A shader-based blit would be faster and is the natural upgrade once
 * there is an internal pipeline to hang it on.
 *
 * Scaling is not supported here: util_format_translate converts a rectangle, it
 * does not resample. A cross-format blit that also scales is refused rather than
 * silently done without filtering.
 */
static bool
sz_blit_cross_format(struct pipe_context *pipe,
                     const struct pipe_blit_info *info)
{
   struct sz_context *ctx = sz_context(pipe);
   const enum pipe_format src_format = info->src.resource->format;
   const enum pipe_format dst_format = info->dst.resource->format;

   if (!sz_box_equals(&info->src.box, &info->dst.box)) {
      sz_warn_once("a blit that changes format and scales is not implemented");
      return true;
   }

   const struct util_format_description *src_desc =
      util_format_description(src_format);
   const struct util_format_description *dst_desc =
      util_format_description(dst_format);
   if (!src_desc || !dst_desc ||
       util_format_get_blockwidth(src_format) != 1 ||
       util_format_get_blockheight(src_format) != 1 ||
       util_format_get_blockwidth(dst_format) != 1 ||
       util_format_get_blockheight(dst_format) != 1) {
      sz_warn_once("cross-format blit between compressed formats is not "
                   "implemented");
      return true;
   }

   struct pipe_transfer *src_transfer = NULL, *dst_transfer = NULL;
   struct pipe_box box = info->src.box;
   if (box.width <= 0 || box.height <= 0 || box.depth <= 0)
      return true;

   /* The destination box may be at a different origin, so it is described
    * separately even though the extents match. */
   const struct pipe_box dst_box = info->dst.box;

   const void *src = pipe->texture_map(pipe, info->src.resource, info->src.level,
                                        PIPE_MAP_READ, &box, &src_transfer);
   if (!src) {
      sz_warn_once("could not map the source of a cross-format blit");
      return true;
   }

   void *dst = pipe->texture_map(pipe, info->dst.resource, info->dst.level,
                                  PIPE_MAP_WRITE, &dst_box, &dst_transfer);
   if (!dst) {
      pipe->texture_unmap(pipe, src_transfer);
      sz_warn_once("could not map the destination of a cross-format blit");
      return true;
   }

   /* Level by level: util_format_translate converts one 2D slice, so a blit of a
    * 3D resource (or an array, whose layers are separate slices) walks the
    * slices here. */
   const unsigned slices = MAX2(box.depth, 1);
   for (unsigned z = 0; z < slices; z++) {
      util_format_translate(dst_format, dst, dst_transfer->stride, dst_box.x,
                            dst_box.y, src_format, src,
                            src_transfer->stride, box.x, box.y, box.width,
                            box.height);
      dst = (uint8_t *)dst + dst_transfer->layer_stride;
      src = (const uint8_t *)src + src_transfer->layer_stride;
   }

   pipe->texture_unmap(pipe, dst_transfer);
   pipe->texture_unmap(pipe, src_transfer);

   ctx->screen->stats.copies++;
   return true;
}

static void
sz_blit_aspect(struct sz_context *ctx, VkCommandBuffer cmd,
               const struct pipe_blit_info *info, struct sz_resource *src,
               struct sz_resource *dst, VkImageAspectFlags aspect,
               VkFormat src_format, VkFormat dst_format, unsigned src_samples,
               unsigned dst_samples)
{
   if (src_format != dst_format) {
      /* Only the colour aspect has a generic conversion; a depth or stencil
       * blit that also changes format has no equivalent anywhere in Vulkan or
       * in util_format. */
      if (aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
         sz_warn_once("a depth or stencil blit that changes format is not "
                      "implemented");
         return;
      }
      sz_blit_cross_format(&ctx->base, info);
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

   /* Window rectangles, conditional rendering, blending and swizzling each need
    * a different mechanism (scissored blits, predication, a shader). Refusing is
    * honest; doing half of it is a wrong picture.
    *
    * The condition is taken from our own state rather than from
    * pipe_blit_info::render_condition_enable: the state tracker only fills that
    * field in its glDrawPixels path, so on the glBlitFramebuffer path it is
    * uninitialised stack garbage (observed as 1 with no condition set). */
   if (info->num_window_rectangles || ctx->condition_query ||
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
