/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Barriers and layout transitions.
 *
 * Gallium has no barrier concept: the state tracker issues draws, copies and
 * clears and expects the driver to work out whatever synchronisation Vulkan
 * needs. So the driver has to *infer* hazards instead of being told about them,
 * which is why every resource tracks its current VkImageLayout.
 *
 * All barriers go through VK_KHR_synchronization2 (`*2` entry points). That
 * extension is on the required capability list, and the `2` forms express stage
 * masks as stage *bits* rather than pipeline-stage flags, which removes the
 * "which of these two mask types am I looking at" bug class entirely.
 */

#ifndef SZ_SYNC_H
#define SZ_SYNC_H

#include "sz_resource.h"
#include "sz_types.h"

/* What a resource is about to be used for.
 *
 * The purpose decides the layout *and* the stage/access masks together, because
 * a layout that disagrees with the access mask either fails validation or
 * silently under-synchronises -- and the latter only shows up as intermittent
 * corruption on some drivers. Callers therefore never spell out raw flags. */
enum sz_access {
   SZ_ACCESS_COLOR_ATTACH,  /* rendering into it as a colour target */
   SZ_ACCESS_DEPTH_ATTACH,  /* rendering into it as depth/stencil */
   SZ_ACCESS_SAMPLED,       /* reading it in a shader */
   SZ_ACCESS_STORAGE,       /* shader image load/store */
   SZ_ACCESS_TRANSFER_SRC,  /* copying out of it */
   SZ_ACCESS_TRANSFER_DST,  /* copying into it, or clearing it */
   SZ_ACCESS_PRESENT,       /* handing it to the winsys */
};

struct sz_sync_state {
   VkPipelineStageFlags2 stages;
   VkAccessFlags2 access;
   VkImageLayout layout;
};

/* Transitions `res` into the layout implied by `access`, recording a
 * vkCmdPipelineBarrier2 only when something actually changes. A redundant
 * barrier is not merely wasted work: on some drivers a source mask of zero
 * against a non-UNDEFINED old layout stalls the whole pipeline.
 *
 * Buffers and NULL resources are accepted and return VK_IMAGE_LAYOUT_UNDEFINED,
 * so callers do not need to special-case them.
 *
 * Returns the layout the resource is in afterwards. */
VkImageLayout sz_resource_access(struct sz_screen *screen, VkCommandBuffer cmd,
                                 struct sz_resource *res, enum sz_access access);

/* Marks the current contents as discardable. The next transition will use
 * VK_IMAGE_LAYOUT_UNDEFINED as its source layout, which lets tilers skip reading
 * back the old contents. Correct whenever the next use fully overwrites the
 * image: a full-surface clear, or a render pass with loadOp=CLEAR. */
void sz_resource_discard(struct sz_resource *res);

/* A whole-command-buffer memory barrier, for the cases where the hazard is
 * between two buffer accesses rather than image layouts. */
void sz_memory_barrier(struct sz_screen *screen, VkCommandBuffer cmd,
                       VkPipelineStageFlags2 src_stage,
                       VkAccessFlags2 src_access,
                       VkPipelineStageFlags2 dst_stage,
                       VkAccessFlags2 dst_access);

#endif /* SZ_SYNC_H */
