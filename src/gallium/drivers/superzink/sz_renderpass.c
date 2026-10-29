/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_renderpass.h"

#include "sz_debug.h"
#include "sz_format.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "vk_format.h"

#include "util/format/u_format.h"
#include "util/log.h"
#include "util/macros.h"

#include <string.h>

bool
sz_renderpass_is_valid(struct sz_context *ctx)
{
   const struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   if (fb->width == 0 || fb->height == 0)
      return false;
   if (fb->nr_cbufs == 0 && !fb->zsbuf.texture)
      return false;
   if (fb->nr_cbufs > PIPE_MAX_COLOR_BUFS)
      return false;

   return true;
}


/* A cache bound rather than a hard limit: past this, attachments use the
 * whole-image view. */
#define SZ_MAX_SURFACE_VIEWS 64

/* An image view for exactly the subresource a framebuffer attachment names.
 *
 * `res->view` addresses the whole image, which is right for a 2D texture but
 * wrong for anything layered: attaching one cube face or one array layer of a
 * layered render target and rendering through the whole-image view writes to
 * *all* of them. GL's glFramebufferTextureLayer and every cube map's per-face
 * FBO selection land here.
 *
 * The views are cached on the context and dropped whenever the framebuffer
 * changes, because that is exactly when the set is no longer needed; the
 * objects themselves live until the image does. */
VkImageView
sz_surface_view(struct sz_context *ctx, struct sz_resource *res, VkFormat format,
                const struct pipe_surface *surf)
{
   const unsigned first_level = surf->level;
   const unsigned level_count = 1;
   const unsigned first_layer = surf->first_layer;
   const unsigned layer_count = MAX2(surf->last_layer - surf->first_layer + 1, 1);

   /* A view is only needed when it addresses less than the whole image, or when
    * it reinterprets the texels. The whole-image view carries the resource's own
    * format, and a storage image unit names the format it wants to read and
    * write the texels with -- GL lets an application bind an r8_snorm image over
    * an r8 texture, and reading through the resource's view then returns the
    * texel as unorm. */
   const bool whole_levels = first_level == 0 && level_count > res->base.last_level;
   const bool whole_layers =
      first_layer == 0 && layer_count >= res->base.array_size;
   const bool same_format =
      format == sz_vk_format(ctx->screen, res->base.format);
   if (whole_levels && whole_layers && same_format)
      return res->view;

   /* A single layer of a cube or cube array is a 2D view of that face; a
    * multi-layer cube range is a cube (array) view. Vulkan has no layered 1D
    * view that is also arrayed, so a 1D array collapses to 2D exactly as the
    * whole-image view does. */
   VkImageViewType type;
   switch (res->base.target) {
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      type = layer_count > 6 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
             : layer_count == 6
                ? (res->base.target == PIPE_TEXTURE_CUBE
                      ? VK_IMAGE_VIEW_TYPE_CUBE
                      : VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)
                : VK_IMAGE_VIEW_TYPE_2D;
      break;
   case PIPE_TEXTURE_3D:
      type = VK_IMAGE_VIEW_TYPE_3D;
      break;
   default:
      if (res->base.target == PIPE_TEXTURE_1D)
         type = VK_IMAGE_VIEW_TYPE_1D;
      else if (res->base.target == PIPE_TEXTURE_1D_ARRAY)
         type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      else if (layer_count > 1 || res->base.array_size > 1)
         type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      else
         type = VK_IMAGE_VIEW_TYPE_2D;
      break;
   }

   for (unsigned i = 0; i < ctx->num_surface_views; i++) {
      const struct sz_surface_view *v = &ctx->surface_views[i];
      if (v->res == res && v->format == format && v->type == type &&
          v->first_level == first_level && v->level_count == level_count &&
          v->first_layer == first_layer && v->layer_count == layer_count)
         return v->view;
   }

   /* Reaching the cap means an application attached more distinct
    * subresources than this driver bothers to cache; rendering goes through the
    * whole-image view then, which is correct for an unlayered attachment and is
    * what happened for every attachment before this cache existed. Growing
    * without a bound would leak a view per FBO rebind. */
   if (ctx->num_surface_views == SZ_MAX_SURFACE_VIEWS)
      return res->view;

   if (ctx->num_surface_views == ctx->surface_views_capacity) {
      const unsigned capacity = MIN2(MAX2(ctx->surface_views_capacity * 2, 8),
                                     SZ_MAX_SURFACE_VIEWS);
      void *grown = realloc(ctx->surface_views, capacity * sizeof(*ctx->surface_views));
      if (!grown)
         return res->view;
      ctx->surface_views = grown;
      ctx->surface_views_capacity = capacity;
   }

   const VkImageViewCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = res->image,
      .viewType = type,
      .format = format,
      .subresourceRange = {
         .aspectMask = sz_format_aspect(res->base.format),
         .baseMipLevel = first_level,
         .levelCount = level_count,
         .baseArrayLayer = first_layer,
         .layerCount = layer_count,
      },
   };

   VkImageView view = VK_NULL_HANDLE;
   if (ctx->screen->vk.CreateImageView(ctx->screen->dev, &ci, NULL, &view) !=
       VK_SUCCESS)
      return res->view;

   struct sz_surface_view *v = &ctx->surface_views[ctx->num_surface_views++];
   *v = (struct sz_surface_view) {
      .view = view,
      .res = res,
      .format = format,
      .first_level = first_level,
      .level_count = level_count,
      .type = type,
      .first_layer = first_layer,
      .layer_count = layer_count,
   };
   return view;
}

bool
sz_renderpass_begin(struct sz_context *ctx, VkCommandBuffer cmd)
{
   struct sz_screen *screen = ctx->screen;
   struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   if (!sz_renderpass_is_valid(ctx)) {
      sz_warn("draw without a complete framebuffer, ignoring (%ux%u, %u colour "
              "buffer%s, %s depth)",
              fb->width, fb->height, fb->nr_cbufs,
              fb->nr_cbufs == 1 ? "" : "s",
              fb->zsbuf.texture ? "with" : "no");
      return false;
   }

   VkRenderingAttachmentInfo color[PIPE_MAX_COLOR_BUFS] = { 0 };
   VkFormat color_formats[PIPE_MAX_COLOR_BUFS] = { 0 };
   unsigned color_count = 0;

   for (unsigned i = 0; i < fb->nr_cbufs; i++) {
      if (!fb->cbufs[i].texture)
         break;

      struct sz_resource *res = sz_resource(fb->cbufs[i].texture);
      color_formats[color_count] = sz_vk_format(screen, fb->cbufs[i].format);

      sz_resource_access(screen, cmd, res, SZ_ACCESS_COLOR_ATTACH);

      color[color_count] = (VkRenderingAttachmentInfo) {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = sz_surface_view(ctx, res, color_formats[color_count],
                                      &fb->cbufs[i]),
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      color_count++;
   }

   VkRenderingAttachmentInfo depth = { 0 };
   bool has_depth = fb->zsbuf.texture != NULL;
   /* A depth-only format is not a stencil attachment: naming it as one is a
    * validation error, and the format has to agree with what the pipeline
    * declared. */
   const bool has_stencil =
      has_depth &&
      util_format_has_stencil(util_format_description(fb->zsbuf.format));

   if (has_depth) {
      struct sz_resource *res = sz_resource(fb->zsbuf.texture);

      sz_resource_access(screen, cmd, res,
                         SZ_ACCESS_DEPTH_ATTACH);

      depth = (VkRenderingAttachmentInfo) {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = sz_surface_view(ctx, res,
                                      sz_vk_format(screen, fb->zsbuf.format),
                                      &fb->zsbuf),
         /* The stencil attachment shares this struct, so the layout has to be
          * legal for both aspects: DEPTH_ATTACHMENT_OPTIMAL is not
          * (VUID-vkCmdBeginRendering-pStencilAttachment-06098). */
         .imageLayout = has_stencil
                           ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                           : VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };

   }

   const VkRenderingInfo info = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {
         .offset = { 0, 0 },
         .extent = { fb->width, fb->height },
      },
      .layerCount = 1,
      .colorAttachmentCount = color_count,
      .pColorAttachments = color_count ? color : NULL,
      .pDepthAttachment = has_depth ? &depth : NULL,
      .pStencilAttachment = has_stencil ? &depth : NULL,
   };

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      static unsigned last_color = ~0u, last_depth = ~0u;
      const unsigned color_samples =
         color_count && fb->cbufs[0].texture
            ? MAX2(fb->cbufs[0].texture->nr_samples, 1)
            : 0;
      const unsigned depth_samples =
         has_depth ? MAX2(fb->zsbuf.texture->nr_samples, 1) : 0;
      if (color_samples != last_color || depth_samples != last_depth) {
         mesa_logi("SZ: framebuffer %ux%u: %u colour buffer(s) at %ux, depth "
                   "at %ux", fb->width, fb->height, color_count,
                   MAX2(color_samples, 1), MAX2(depth_samples, 1));
         last_color = color_samples;
         last_depth = depth_samples;
      }
   }

   screen->vk.CmdBeginRendering(cmd, &info);

   ctx->rendering = true;
   ctx->render_extent[0] = fb->width;
   ctx->render_extent[1] = fb->height;

   memcpy(ctx->render_color_formats, color_formats,
          color_count * sizeof(color_formats[0]));
   ctx->render_color_count = color_count;

   return true;
}

void
sz_renderpass_end(struct sz_context *ctx, VkCommandBuffer cmd)
{
   /* A render pass cannot be ended with draws still queued for it. */
   sz_draw_flush(ctx);

   if (!ctx->rendering)
      return;

   ctx->screen->vk.CmdEndRendering(cmd);
   ctx->rendering = false;
}
