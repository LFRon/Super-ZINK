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
         .imageView = res->view,
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
         .imageView = res->view,
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
   if (!ctx->rendering)
      return;

   ctx->screen->vk.CmdEndRendering(cmd);
   ctx->rendering = false;
}
