/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * The render pass.
 *
 * There is no VkRenderPass. VK_KHR_dynamic_rendering is core in Vulkan 1.3 and
 * already on the required capability list, so a render pass is described at
 * record time from the current framebuffer state. That removes the combinatorial
 * explosion of VkRenderPass/VkFramebuffer objects that GL's "any attachment
 * combination at any time" usage would otherwise cause -- which is the single
 * biggest structural advantage this driver has over wrapping Vulkan 1.0.
 */

#ifndef SZ_RENDERPASS_H
#define SZ_RENDERPASS_H

#include "sz_context.h"
#include "sz_types.h"

/* True when the current framebuffer state can be described at all. */
/* The image view for one subresource range of `res`, cached on the context.
 * Shared with the storage-image descriptors, which address the same kind of
 * range. */
VkImageView sz_surface_view(struct sz_context *ctx, struct sz_resource *res,
                            VkFormat format, const struct pipe_surface *surf);

bool sz_renderpass_is_valid(struct sz_context *ctx);

/* Transitions the attachments and starts dynamic rendering.
 *
 * Load op is always LOAD: clears are issued as explicit vkCmdClear*Image by
 * sz_clear(), so by the time a draw happens the contents must be preserved.
 * Switching the first draw's attachment to LOAD_OP_CLEAR would skip the separate
 * clear, but only once "clear then draw" is known to be adjacent -- that is a
 * Phase 3 measurement, and guessing it now risks discarding a clear that a
 * glReadPixels depends on.
 *
 * Returns false if nothing can be rendered. */
bool sz_renderpass_begin(struct sz_context *ctx, VkCommandBuffer cmd);

void sz_renderpass_end(struct sz_context *ctx, VkCommandBuffer cmd);

#endif /* SZ_RENDERPASS_H */
