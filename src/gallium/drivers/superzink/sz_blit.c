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
#include "sz_renderpass.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/log.h"
#include "util/u_debug.h"
#include "util/u_blitter.h"
#include "util/macros.h"
#include "util/u_math.h"

#include <stdlib.h>
#include <stdlib.h>
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

/* Whether vkCmdBlitImage2 can read and write this format.
 *
 * Depth is the reason this is asked: a copy is the obvious way to move depth
 * between two single-sampled images, but on RADV a depth image written by
 * vkCmdCopyImage2 and then attached as a depth buffer reads back as the
 * fast-clear value in scattered tiles -- HTILE is only brought up to date by the
 * render pass paths, not by a transfer. A blit of a depth format goes through
 * the driver's own depth blit instead, which does maintain it. */
static bool
sz_format_can_blit(struct sz_screen *screen, VkFormat format)
{
   VkFormatProperties props;
   screen->vk.GetPhysicalDeviceFormatProperties(screen->pdev, format, &props);
   const VkFormatFeatureFlags needed =
      VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
   return (props.optimalTilingFeatures & needed) == needed;
}

/* The blits no single Vulkan command can express, handed to Mesa's blitter.
 *
 * Three of them are worth spelling out, because each is a legal GL request with
 * no Vulkan counterpart at all:
 *
 *  - a multisampled blit whose source and destination have the *same* sample
 *    count (glBlitFramebuffer on a multisampled framebuffer, or glCopyPixels
 *    with one). Vulkan forbids blits, copies and resolves between multisampled
 *    images; the blitter draws each sample instead.
 *  - a multisampled blit with different extents, which GL defines as "resolve,
 *    then scale" and Vulkan cannot do in one step.
 *  - a depth or stencil resolve whose two corners do not line up, which the
 *    render pass resolve cannot express because a render pass reads and writes
 *    the same coordinates.
 *
 * The blitter builds its shaders in TGSI and draws them through the ordinary
 * pipeline, so this costs the driver no second renderer. */
static bool
sz_blit_blitter(struct sz_context *ctx, const struct pipe_blit_info *info)
{
   struct blitter_context *blitter = ctx->blitter;
   if (!blitter || !util_blitter_is_blit_supported(blitter, info))
      return false;

   util_blitter_save_framebuffer(blitter, &ctx->state.framebuffer);
   util_blitter_save_vertex_buffers(blitter, ctx->state.vertex_buffers,
                                    ctx->state.num_vertex_buffers);
   util_blitter_save_vertex_elements(blitter, ctx->state.vertex_elements);
   util_blitter_save_viewport(blitter, ctx->state.viewports);
   util_blitter_save_scissor(blitter, ctx->state.scissors);
   util_blitter_save_rasterizer(blitter, ctx->state.rasterizer);
   util_blitter_save_depth_stencil_alpha(blitter,
                                         ctx->state.depth_stencil_alpha);
   util_blitter_save_blend(blitter, ctx->state.blend);
   util_blitter_save_stencil_ref(blitter, &ctx->state.stencil_ref);
   util_blitter_save_sample_mask(blitter, ctx->state.sample_mask, 1);
   util_blitter_save_vertex_shader(blitter,
                                   ctx->state.shaders[MESA_SHADER_VERTEX]);
   util_blitter_save_tessctrl_shader(blitter,
                                     ctx->state.shaders[MESA_SHADER_TESS_CTRL]);
   util_blitter_save_tesseval_shader(blitter,
                                     ctx->state.shaders[MESA_SHADER_TESS_EVAL]);
   util_blitter_save_geometry_shader(blitter,
                                     ctx->state.shaders[MESA_SHADER_GEOMETRY]);
   util_blitter_save_fragment_shader(blitter,
                                     ctx->state.shaders[MESA_SHADER_FRAGMENT]);
   util_blitter_save_so_targets(blitter, ctx->num_so_targets, ctx->so_targets,
                               MESA_PRIM_UNKNOWN);
   util_blitter_save_fragment_sampler_states(
      blitter, PIPE_MAX_SAMPLERS,
      ctx->state.sampler_states[MESA_SHADER_FRAGMENT]);
   util_blitter_save_fragment_sampler_views(
      blitter, PIPE_MAX_SAMPLERS,
      ctx->state.sampler_views[MESA_SHADER_FRAGMENT]);
   util_blitter_save_fragment_constant_buffer_slot(
      blitter, &ctx->state.constant_buffers[MESA_SHADER_FRAGMENT][0]);

   util_blitter_blit(blitter, info, NULL);
   return true;
}

/* A depth or stencil resolve.
 *
 * vkCmdResolveImage2 is not available here: it may only touch those aspects
 * with the maintenance10 feature, which RADV does not enable (and without the
 * VkResolveImageModeInfoKHR this driver has no way to pass, its depth path
 * dereferences NULL and crashes outright).
 *
 * Vulkan's other resolve mechanism is the render pass: VkRenderingAttachmentInfo
 * names a resolve image, and the resolve runs over the render area when the pass
 * ends. No draw is recorded at all -- the attachment is loaded, resolved, and
 * stored with DONT_CARE, so the multisampled image is left untouched. Sample
 * zero is GL's rule for both aspects: unlike colour, neither is averaged. */
static void
sz_blit_resolve_ds(struct sz_context *ctx, VkCommandBuffer cmd,
                   const struct pipe_blit_info *info, struct sz_resource *src,
                   struct sz_resource *dst, VkImageAspectFlags aspect)
{
   struct sz_screen *screen = ctx->screen;

   VkOffset3D src_off[2], dst_off[2];
   sz_blit_offsets(&info->src.box, false, src_off);
   sz_blit_offsets(&info->dst.box, false, dst_off);

   /* The render area is measured in the destination image, and the attachment
    * is read from the same coordinates, so the two corners have to agree. */
   if (src_off[0].x != dst_off[0].x || src_off[0].y != dst_off[0].y) {
      sz_warn_once("a depth or stencil resolve from an offset");
      return;
   }

   const unsigned first_layer = MAX2(info->dst.box.z, 0);
   const unsigned layer_count = MAX2(info->dst.box.depth, 1);
   const VkFormat format = sz_vk_format(screen, info->dst.format);
   const VkImageAspectFlags src_aspects =
      aspect == VK_IMAGE_ASPECT_DEPTH_BIT
         ? VK_IMAGE_ASPECT_DEPTH_BIT
         : VK_IMAGE_ASPECT_STENCIL_BIT;

   /* Both images are read and written as attachments for the duration of the
    * resolve, so this is the layout pair the render pass expects. */
   sz_resource_access_levels(screen, cmd, src, SZ_ACCESS_DEPTH_ATTACH,
                             info->src.level, 1);
   sz_resource_access_levels(screen, cmd, dst, SZ_ACCESS_DEPTH_ATTACH,
                             info->dst.level, 1);

   const VkImageView src_view =
      sz_resource_view(ctx, src, format, src_aspects, info->src.level, 1,
                       first_layer, layer_count);
   const VkImageView dst_view =
      sz_resource_view(ctx, dst, format, src_aspects, info->dst.level, 1,
                       first_layer, layer_count);
   if (src_view == VK_NULL_HANDLE || dst_view == VK_NULL_HANDLE) {
      sz_warn_once("a depth or stencil resolve without a view");
      return;
   }

   const VkRenderingAttachmentInfo attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = src_view,
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .resolveMode = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .resolveImageView = dst_view,
      .resolveImageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
   };

   const VkExtent3D extent = sz_box_extent(&info->dst.box, false);
   const VkRenderingInfo resolve = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {
         .offset = { dst_off[0].x, dst_off[0].y },
         .extent = { extent.width, extent.height },
      },
      .layerCount = 1,
      .pDepthAttachment =
         aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? &attachment : NULL,
      .pStencilAttachment =
         aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? &attachment : NULL,
   };

   ctx->screen->stats.resolves++;

   screen->vk.CmdBeginRendering(cmd, &resolve);
   screen->vk.CmdEndRendering(cmd);
}

static void sz_blit_copy(struct sz_context *ctx, VkCommandBuffer cmd,
                         const struct pipe_blit_info *info,
                         struct sz_resource *src, struct sz_resource *dst,
                         VkImageAspectFlags aspect);

/* A depth or stencil resolve whose two corners do not line up.
 *
 * The render pass resolve reads and writes the same coordinates, so it can only
 * express "resolve this rectangle onto itself". glBlitFramebuffer asks for
 * exactly that in the common case -- but glBlitFramebuffer between two regions
 * of one multisampled framebuffer, and glCopyPixels, both move the result. The
 * resolve therefore lands in a temporary single-sampled image at the source's
 * own coordinates, and a copy moves it to the destination.
 *
 * Both halves are operations this file already performs, so no new Vulkan
 * feature is involved. The temporary is released through the screen's retire
 * list, which keeps it alive until the submission that names it has completed. */
static bool
sz_blit_resolve_offset(struct sz_context *ctx, VkCommandBuffer cmd,
                       const struct pipe_blit_info *info,
                       struct sz_resource *src, struct sz_resource *dst,
                       VkImageAspectFlags aspect)
{
   struct sz_screen *screen = ctx->screen;

   /* A copy cannot mirror, and neither can a render pass: a flipped source has
    * no path here and is left to the blitter. */
   if (info->src.box.width < 0 || info->src.box.height < 0 ||
       info->dst.box.width < 0 || info->dst.box.height < 0)
      return false;

   const unsigned width = u_minify(info->src.resource->width0, info->src.level);
   const unsigned height = u_minify(info->src.resource->height0, info->src.level);

   const struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = info->src.resource->format,
      .bind = PIPE_BIND_DEPTH_STENCIL,
      .usage = PIPE_USAGE_DEFAULT,
      .width0 = width,
      .height0 = height,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 1,
      .nr_storage_samples = 1,
   };

   struct pipe_resource *temp =
      screen->base.resource_create(&screen->base, &templ);
   if (!temp) {
      sz_warn_once("no temporary for a depth or stencil resolve");
      return false;
   }
   struct sz_resource *tmp = sz_resource(temp);

   if (tmp->image == VK_NULL_HANDLE) {
      screen->base.resource_destroy(&screen->base, temp);
      return false;
   }

   const struct pipe_box box = {
      .x = info->src.box.x,
      .y = info->src.box.y,
      .z = 0,
      .width = info->src.box.width,
      .height = info->src.box.height,
      .depth = 1,
   };

   struct pipe_blit_info resolve = *info;
   resolve.dst.resource = temp;
   resolve.dst.level = 0;
   resolve.dst.format = info->src.resource->format;
   resolve.dst.box = box;
   resolve.mask = aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? PIPE_MASK_Z
                                                       : PIPE_MASK_S;
   resolve.filter = PIPE_TEX_FILTER_NEAREST;
   resolve.swizzle_enable = false;
   sz_blit_resolve_ds(ctx, cmd, &resolve, src, tmp, aspect);

   struct pipe_blit_info copy = *info;
   copy.src.resource = temp;
   copy.src.level = 0;
   copy.src.format = info->src.resource->format;
   copy.src.box = box;
   sz_blit_copy(ctx, cmd, &copy, tmp, dst, aspect);

   screen->base.resource_destroy(&screen->base, temp);
   return true;
}

static void
sz_blit_resolve(struct sz_context *ctx, VkCommandBuffer cmd,
                const struct pipe_blit_info *info, struct sz_resource *src,
                struct sz_resource *dst, VkImageAspectFlags aspect)
{
   if (aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
      sz_blit_resolve_ds(ctx, cmd, info, src, dst, aspect);
      return;
   }

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
      /* Vulkan requires NEAREST for a depth or stencil format, and GL ignores
       * the filter for them too: neither value has anything to interpolate. */
      .filter = aspect == VK_IMAGE_ASPECT_COLOR_BIT &&
                info->filter == PIPE_TEX_FILTER_LINEAR ? VK_FILTER_LINEAR
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
 * float, and depth's float-to-unorm) have no vkCmd* equivalent at all. The state
 * tracker hits this on the ordinary glBlitFramebuffer path, not on some exotic
 * one.
 *
 * The conversion therefore happens where Mesa already has a complete and tested
 * implementation for it: util_format_translate. Both ends go through the same
 * transfer path the driver already uses for texture upload and readback, which
 * means staging allocations, barriers and waiting are all already handled -- and
 * the cost is two round trips, which is the right trade for a blit that changes
 * format. A shader-based blit would be faster and is the natural upgrade once
 * there is an internal pipeline to hang it on.
 *
 * `aspect` says which half of the texel moves. A depth or stencil blit maps only
 * that aspect, so the other one is neither read nor written, and the formats
 * handed to the translator describe the aspect's bytes rather than the
 * resource's -- a depth aspect is stored in the device's depth format, which may
 * be wider than the pipe format names, and a stencil aspect is always one byte
 * per texel.
 *
 * Scaling is not supported here: util_format_translate converts a rectangle, it
 * does not resample. A cross-format blit that also scales is refused rather than
 * silently done without filtering.
 */
static bool
sz_blit_cross_format(struct pipe_context *pipe,
                     const struct pipe_blit_info *info,
                     VkImageAspectFlags aspect)
{
   struct sz_context *ctx = sz_context(pipe);
   const enum pipe_format src_res_format = info->src.resource->format;
   const enum pipe_format dst_res_format = info->dst.resource->format;
   enum pipe_format src_format = src_res_format;
   enum pipe_format dst_format = dst_res_format;
   unsigned map_flags = 0;

   if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      map_flags = PIPE_MAP_DEPTH_ONLY;
      src_format = sz_vk_depth_pipe_format(ctx->screen, src_res_format);
      dst_format = sz_vk_depth_pipe_format(ctx->screen, dst_res_format);
      if (src_format == PIPE_FORMAT_NONE || dst_format == PIPE_FORMAT_NONE)
         return true;
   } else if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
      map_flags = PIPE_MAP_STENCIL_ONLY;
      src_format = PIPE_FORMAT_S8_UINT;
      dst_format = PIPE_FORMAT_S8_UINT;
   }

   /* The source box may be flipped, which sz_box_equals() reads as a different
    * extent, so the sizes are compared directly. */
   if (info->src.box.width != info->dst.box.width ||
       abs(info->src.box.height) != abs(info->dst.box.height) ||
       info->src.box.depth != info->dst.box.depth) {
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
   if (box.width <= 0 || box.height == 0 || box.depth <= 0)
      return true;

   /* A negative height is GL's flipped source: the rectangle runs upwards from
    * box.y, which is the normal case for glBlitFramebuffer because its origin is
    * the bottom-left. Dropping it left the destination holding whatever it had
    * before -- for a window that is the previous frame, or the clear colour,
    * forever. It is normalised here so the mapping addresses a positive
    * rectangle, and the flip is applied when the rows are written. */
   const bool flip = box.height < 0;
   if (flip) {
      /* A flipped box spans from y - 1 down to y + height, so the lowest row it
       * names is y + height. */
      box.y = MAX2(box.y + box.height, 0);
      box.height = -box.height;
   }
   const unsigned rows = (unsigned)box.height;

   /* The destination box may be at a different origin, so it is described
    * separately even though the extents match. */
   const struct pipe_box dst_box = info->dst.box;

   const void *src = pipe->texture_map(pipe, info->src.resource, info->src.level,
                                        PIPE_MAP_READ | map_flags, &box,
                                        &src_transfer);
   if (!src) {
      sz_warn_once("could not map the source of a cross-format blit");
      return true;
   }

   void *dst = pipe->texture_map(pipe, info->dst.resource, info->dst.level,
                                  PIPE_MAP_WRITE | map_flags, &dst_box,
                                  &dst_transfer);
   if (!dst) {
      pipe->texture_unmap(pipe, src_transfer);
      sz_warn_once("could not map the destination of a cross-format blit");
      return true;
   }

   /* Level by level: util_format_translate converts one 2D slice, so a blit of a
    * 3D resource (or an array, whose layers are separate slices) walks the
    * slices here.
    *
    * The mapped pointers address the first texel of their own box, so every
    * offset handed to the translator is relative to it. */
   const unsigned slices = MAX2(box.depth, 1);
   for (unsigned z = 0; z < slices; z++) {
      if (!flip) {
         util_format_translate(dst_format, dst, dst_transfer->stride, 0, 0,
                               src_format, src, src_transfer->stride, 0, 0,
                               (unsigned)box.width, rows);
      } else {
         for (unsigned r = 0; r < rows; r++)
            util_format_translate(dst_format, dst, dst_transfer->stride, 0,
                                  rows - 1 - r, src_format, src,
                                  src_transfer->stride, 0, r,
                                  (unsigned)box.width, 1);
      }
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
   /* Two different VkFormats is not by itself a conversion Vulkan cannot do:
    * formats in one compatibility class blit into each other by channel, and
    * RGBA8 to BGRA8 -- every glBlitFramebuffer to the window of a little-endian
    * display -- is exactly that. Anything else is a numeric conversion GL spells
    * out (a depth of 0.25 stays 0.25 whatever the format, float becomes unorm,
    * 32 bits become 16) and has no vkCmd* equivalent at all.
    *
    * A multisampled source is the one case that cannot go through the CPU path
    * even then: mapping it names one sample's texels, not the samples, so the
    * transfer cannot see what it is supposed to copy. A draw can, and the
    * blitter's shader reads and writes both ends as floats. */
   const bool natively_blittable =
      src_format == dst_format ||
      (sz_format_blit_compatible(info->src.format, info->dst.format) &&
       sz_format_can_blit(ctx->screen, src_format) &&
       sz_format_can_blit(ctx->screen, dst_format));

   if (!natively_blittable) {
      if (src_samples > 1 &&
          (aspect == VK_IMAGE_ASPECT_DEPTH_BIT ||
           aspect == VK_IMAGE_ASPECT_STENCIL_BIT)) {
         struct pipe_blit_info sub = *info;
         sub.mask = aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? PIPE_MASK_Z
                                                        : PIPE_MASK_S;
         if (!sz_blit_blitter(ctx, &sub))
            sz_warn_once("a multisampled depth or stencil blit that changes "
                         "format");
         return;
      }
      sz_blit_cross_format(&ctx->base, info, aspect);
      return;
   }

   /* Level-ranged, because a mipmap generation reads level N-1 and writes level
    * N of the *same* image: two whole-image transitions would leave it in
    * whichever layout was asked for second, and the blit would then name the
    * wrong one for the half it reads (VUID-VkBlitImageInfo2-srcImageLayout-00221).
    * The same applies to the shared-image case where src and dst differ only in
    * level. */
   sz_resource_access_levels(ctx->screen, cmd, src, SZ_ACCESS_TRANSFER_SRC,
                             info->src.level, 1);
   sz_resource_access_levels(ctx->screen, cmd, dst, SZ_ACCESS_TRANSFER_DST,
                             info->dst.level, 1);

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

   if (resolve) {
      /* The resolve names its own layouts: a depth or stencil one runs through
       * a render pass, where the images are attachments rather than transfer
       * sources. vkCmdResolveImage2 moves the region itself, a render pass does
       * not, so a depth or stencil resolve with two different corners goes
       * through a temporary image. */
      const bool aligned = info->src.box.x == info->dst.box.x &&
                           info->src.box.y == info->dst.box.y;

      if (aspect == VK_IMAGE_ASPECT_COLOR_BIT || aligned) {
         sz_blit_resolve(ctx, cmd, info, src, dst, aspect);
         return;
      }

      if (sz_blit_resolve_offset(ctx, cmd, info, src, dst, aspect))
         return;

      struct pipe_blit_info sub = *info;
      sub.mask = aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? PIPE_MASK_Z
                                                     : PIPE_MASK_S;
      if (!sz_blit_blitter(ctx, &sub))
         sz_warn_once("a depth or stencil resolve from an offset");
      return;
   }

   sz_resource_access_levels(ctx->screen, cmd, src, SZ_ACCESS_TRANSFER_SRC,
                             info->src.level, 1);
   sz_resource_access_levels(ctx->screen, cmd, dst, SZ_ACCESS_TRANSFER_DST,
                             info->dst.level, 1);

   const bool depth_copy =
      aspect == VK_IMAGE_ASPECT_DEPTH_BIT && same_size && src_samples == 1 &&
      dst_samples == 1 && info->src.box.width >= 0 && info->src.box.height >= 0;

   if (src_samples > 1 || dst_samples > 1) {
      /* vkCmdBlitImage cannot read or write a multisampled image at all, and a
       * resolve needs a single-sampled destination: a blit between two images
       * of the *same* sample count, or one that also scales, is a draw. */
      struct pipe_blit_info sub = *info;
      sub.mask = aspect == VK_IMAGE_ASPECT_DEPTH_BIT
                    ? PIPE_MASK_Z
                    : aspect == VK_IMAGE_ASPECT_STENCIL_BIT ? PIPE_MASK_S
                                                            : PIPE_MASK_RGBA;
      if (!sz_blit_blitter(ctx, &sub))
         sz_warn_once("a multisampled blit that is not a same-size resolve");
      return;
   } else if (depth_copy && sz_format_can_blit(ctx->screen,
                                              sz_vk_format(ctx->screen,
                                                           info->dst.format))) {
      /* A native blit is the cheapest and the most precise thing for a depth
       * copy, and the only path RADV gets right on every format. */
      sz_blit_image(ctx, cmd, info, src, dst, aspect);
   } else if (depth_copy && src != dst) {
      /* No blit for this depth format, and the two images differ, so the copy is
       * not the only option -- and it is the wrong one. A depth image written by
       * vkCmdCopyImage2 and then sampled or attached reads back as the
       * fast-clear value in whole 8-row HTILE tiles on RADV, which is what made
       * ext_framebuffer_multisample@accuracy fail only at samples = 0: that
       * sub-case is a plain depth blit between two single-sampled images.
       * Drawing the copy instead keeps the image in the state the renderer
       * expects, at the cost of a pipeline and a quad. */
      struct pipe_blit_info sub = *info;
      sub.mask = PIPE_MASK_Z;
      sz_blit_blitter(ctx, &sub);
   } else if (depth_copy) {
      /* Source and destination are the same image, which the blitter cannot
       * read and write in one draw: GL's own copies within one attachment keep
       * the transfer path. */
      sz_blit_copy(ctx, cmd, info, src, dst, aspect);
   } else {
      sz_blit_image(ctx, cmd, info, src, dst, aspect);
   }
}

/* One axis of a blit, clipped to what the destination can hold.
 *
 * Vulkan requires every blit region to lie inside the image
 * (VUID-VkBlitImageInfo2-dstOffset-00248), and GL's own rule is that a blit is
 * clipped to the framebuffer and to the scissor box. The state tracker clips
 * too, but it clips against the framebuffer's reported bounds -- which for the
 * winsys framebuffer are not the attachment's, so a glBlitFramebuffer whose
 * destination is INT_MAX wide arrives here with its extent intact. RADV
 * silently drops such a blit, which is why fbo-blit-check-limits saw nothing
 * happen.
 *
 * Both boxes describe the same mapping, so clipping the destination and
 * carrying the fraction over to the source keeps the scale. Rounding is
 * outwards on the source side: a destination pixel has to read the source texel
 * it lands in, not the one next to it. */
struct sz_blit_axis {
   int dst_pos, dst_len;
   int src_pos, src_len;
};

static bool
sz_blit_clip_axis(struct sz_blit_axis *a, int dst_lo, int dst_hi, int src_limit)
{
   /* A zero-length axis has nothing to copy. */
   if (a->dst_len == 0 || a->src_len == 0)
      return false;

   /* Signed boxes mean a flip: the span runs from pos+len up to pos. Both ends
    * are made positive here and the sign is restored on the way out. */
   const int dsign = a->dst_len < 0 ? -1 : 1;
   const int ssign = a->src_len < 0 ? -1 : 1;
   int dlo = a->dst_pos, dhi = a->dst_pos + a->dst_len;
   int slo = a->src_pos, shi = a->src_pos + a->src_len;
   if (dlo > dhi) { int t = dlo; dlo = dhi; dhi = t; }
   if (slo > shi) { int t = slo; slo = shi; shi = t; }

   const int new_dlo = MAX2(dlo, dst_lo);
   const int new_dhi = MIN2(dhi, dst_hi);
   if (new_dlo >= new_dhi)
      return false;

   /* The source range the clipped destination range covers. The scale is
    * (shi - slo) / (dhi - dlo), and it is applied to both ends. */
   const int64_t dspan = dhi - dlo;
   const int64_t sspan = shi - slo;
   int new_slo = slo;
   int new_shi = shi;
   if (new_dlo != dlo || new_dhi != dhi) {
      /* Where the kept destination endpoints land in the source. A flipped
       * source runs the other way, so the destination's low end lands at its
       * high end -- cropping the wrong edge there is what put the texture's top
       * rows at the bottom of a window blit. The source end is rounded outwards:
       * a destination pixel has to read the whole texel it lands in. */
      const int64_t lo_at = (int64_t)(new_dlo - dlo) * sspan;
      const int64_t hi_at = (int64_t)(new_dhi - dlo) * sspan;
      if (ssign > 0) {
         new_slo = slo + (int)(lo_at / dspan);
         new_shi = slo + (int)((hi_at + dspan - 1) / dspan);
      } else {
         new_slo = shi - (int)((hi_at + dspan - 1) / dspan);
         new_shi = shi - (int)(lo_at / dspan);
      }
      new_slo = CLAMP(new_slo, 0, src_limit);
      new_shi = CLAMP(new_shi, 0, src_limit);
      if (new_dlo == dlo)
         ssign > 0 ? (new_slo = slo) : (new_shi = shi);
      if (new_dhi == dhi)
         ssign > 0 ? (new_shi = shi) : (new_slo = slo);
      if (new_slo >= new_shi)
         new_shi = MIN2(new_slo + 1, src_limit);
      if (new_slo >= new_shi)
         return false;
   }

   /* A negative length means a flip: Gallium stores such a box with `pos` at
    * the far end, covering [pos + len, pos). */
   a->dst_pos = dsign < 0 ? new_dhi : new_dlo;
   a->dst_len = (new_dhi - new_dlo) * dsign;
   a->src_pos = ssign < 0 ? new_shi : new_slo;
   a->src_len = (new_shi - new_slo) * ssign;
   return true;
}

/* The blit restricted to what the destination image and the scissor box can
 * hold. Returns false when nothing is left. */
static bool
sz_blit_clip(const struct pipe_blit_info *info, unsigned dst_w, unsigned dst_h,
             unsigned src_w, unsigned src_h, struct pipe_blit_info *out)
{
   *out = *info;
   out->scissor_enable = false;

   int dst_xlo = 0, dst_ylo = 0;
   int dst_xhi = (int)dst_w, dst_yhi = (int)dst_h;
   if (info->scissor_enable) {
      dst_xlo = MAX2(dst_xlo, (int)info->scissor.minx);
      dst_ylo = MAX2(dst_ylo, (int)info->scissor.miny);
      dst_xhi = MIN2(dst_xhi, (int)info->scissor.maxx);
      dst_yhi = MIN2(dst_yhi, (int)info->scissor.maxy);
   }

   struct sz_blit_axis x = { info->dst.box.x, info->dst.box.width,
                             info->src.box.x, info->src.box.width };
   struct sz_blit_axis y = { info->dst.box.y, info->dst.box.height,
                             info->src.box.y, info->src.box.height };

   if (!sz_blit_clip_axis(&x, dst_xlo, dst_xhi, (int)src_w) ||
       !sz_blit_clip_axis(&y, dst_ylo, dst_yhi, (int)src_h))
      return false;

   out->dst.box.x = x.dst_pos;
   out->dst.box.width = x.dst_len;
   out->src.box.x = x.src_pos;
   out->src.box.width = x.src_len;
   out->dst.box.y = y.dst_pos;
   out->dst.box.height = y.dst_len;
   out->src.box.y = y.src_pos;
   out->src.box.height = y.src_len;
   return true;
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

   /* A blit that the destination cannot hold, or that a scissor restricts, goes
    * to the blitter: it draws the rectangle, so the rasteriser clips and scissors
    * it exactly the way GL says. Vulkan's transfer commands have neither, and
    * clipping them here means recomputing the source rectangle from the fraction
    * that survived -- arithmetic that has to agree with what the application's
    * own shader does for the same blit, which is what a scaled-blit test compares
    * against.
    *
    * The clipped native path below is the fallback for when the blitter declines
    * (a format it cannot sample, for instance). Vulkan requires the regions to be
    * inside the images, so the boxes still have to be brought into range. */
   const unsigned src_w = u_minify(info->src.resource->width0, info->src.level);
   const unsigned src_h = u_minify(info->src.resource->height0, info->src.level);
   const unsigned dst_w = u_minify(info->dst.resource->width0, info->dst.level);
   const unsigned dst_h = u_minify(info->dst.resource->height0, info->dst.level);
   const struct pipe_box *db = &info->dst.box;
   const bool dst_out_of_range =
      db->x < 0 || db->y < 0 ||
      db->x + abs(db->width) > (int)dst_w ||
      db->y + abs(db->height) > (int)dst_h;

   struct pipe_blit_info clipped;
   if (info->scissor_enable || dst_out_of_range) {
      if (sz_blit_blitter(ctx, info))
         return;
      if (!sz_blit_clip(info, dst_w, dst_h, src_w, src_h, &clipped))
         return;
      info = &clipped;
   }

   /* Window rectangles and conditional rendering need a mechanism of their own
    * (scissored blits, predication). Refusing is honest; doing half of it is a
    * wrong picture.
    *
    * The condition is taken from our own state rather than from
    * pipe_blit_info::render_condition_enable: the state tracker only fills that
    * field in its glDrawPixels path, so on the glBlitFramebuffer path it is
    * uninitialised stack garbage (observed as 1 with no condition set). */
   if (info->num_window_rectangles || ctx->condition_query) {
      sz_warn_once("blit with window rectangles or a render condition is not "
                   "implemented");
      return;
   }

   const VkFormat src_format = sz_vk_format(screen, info->src.resource->format);
   const VkFormat dst_format = sz_vk_format(screen, info->dst.resource->format);
   const unsigned src_samples = MAX2(info->src.resource->nr_samples, 1);
   const unsigned dst_samples = MAX2(info->dst.resource->nr_samples, 1);

   if (debug_get_bool_option("SZ_BLIT_LOG", false))
      fprintf(stderr,
              "SZBLIT mask=%#x src=%s dst=%s srcbox=%d,%d %dx%d dstbox=%d,%d "
              "%dx%d scissor=%d swz=%d ab=%d\n",
              info->mask, util_format_name(info->src.resource->format),
              util_format_name(info->dst.resource->format), info->src.box.x,
              info->src.box.y, info->src.box.width, info->src.box.height,
              info->dst.box.x, info->dst.box.y, info->dst.box.width,
              info->dst.box.height, info->scissor_enable,
              info->swizzle_enable, info->alpha_blend);

   ctx->screen->stats.blits++;



   /* Both commands are illegal inside a render pass, and anything already
    * queued must land before the blit does.
    *
    * The flush has to be explicit. sz_context_cmd_no_pass() ends a render pass
    * only when one is open, and a batch whose draws are still queued has none
    * yet -- so without this the blit is recorded ahead of the very draws it is
    * supposed to copy. An application that renders into a framebuffer and then
    * blits it to the window (which is how Minecraft and every other modern
    * renderer presents) then shows the previous frame's content, or the clear
    * colour, forever. */
   sz_draw_flush(ctx);

   /* A per-channel swizzle or alpha blending is a draw and nothing else:
    * vkCmdBlitImage, vkCmdCopyImage2 and vkCmdResolveImage2 all move texels the
    * way they are, and Vulkan has no render-target swizzle to apply on the way
    * out. util_blitter does both (the swizzle on the sampler view, the blending
    * through a blend state), so the whole mask goes there.
    *
    * The state tracker asks for this whenever the source and destination base
    * formats differ -- a GL_RGBA framebuffer blitted to a BGRX window, which is
    * every glBlitFramebuffer to the screen on a little-endian display server. */
   if (info->alpha_blend) {
      ctx->screen->stats.blits++;
      if (!sz_blit_blitter(ctx, info))
         sz_warn_once("a blit with a swizzle or blending that the blitter "
                      "cannot draw");
      return;
   }

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
      if (src_format != dst_format && src_samples == 1 && dst_samples == 1) {
         /* Two images whose stencil aspects are both S8_UINT still differ in
          * format, and a copy cannot cross formats. The CPU path copies the
          * stencil bytes one at a time, which for S8 to S8 is the same byte. */
         sz_blit_cross_format(&ctx->base, info, VK_IMAGE_ASPECT_STENCIL_BIT);
      } else if (src_format != dst_format || src_samples > 1) {
         /* Everything else that touches stencil is a draw: the blitter writes
          * the stencil export, which is the only way to move a stencil sample
          * in Vulkan -- vkCmdCopyImage2 moves a single-sample stencil plane and
          * nothing moves a multisampled one. */
         sz_blit_aspect(ctx, cmd, info, src, dst, VK_IMAGE_ASPECT_STENCIL_BIT,
                        src_format, dst_format, src_samples, dst_samples);
      } else {
         sz_resource_access(screen, cmd, src, SZ_ACCESS_TRANSFER_SRC);
         sz_resource_access(screen, cmd, dst, SZ_ACCESS_TRANSFER_DST);
         sz_blit_stencil(ctx, cmd, info, src, dst);
      }
   }
}
