/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Device memory allocations.
 *
 * Gallium expects cheap suballocations (zink gets ~2000 of them for a single
 * glxgears frame), while Vulkan hands out whole VkDeviceMemory objects. The
 * mismatch is resolved the same way every serious Vulkan renderer resolves it:
 * one large VkDeviceMemory per memory type, carved up by a range allocator.
 *
 * Two allocation shapes exist:
 *
 *   pooled      - carved out of a shared block. Buffers always use this, and so
 *                 do images the driver does not insist on owning exclusively.
 *   dedicated   - a private VkDeviceMemory of exactly the required size, for
 *                 every image that reports VkMemoryDedicatedRequirements::
 *                 requiresDedicatedAllocation, and for every image that reports
 *                 prefersDedicatedAllocation (some drivers alias memory across
 *                 images and lose compression otherwise).
 *
 * A sz_bo is a *view*: { memory, offset, size }. It does not own a VkBuffer or
 * VkImage; sz_resource does, and binds its sz_bo to it.
 */

#ifndef SZ_BO_H
#define SZ_BO_H

#include <stdbool.h>
#include <stdint.h>

#include <vulkan/vulkan_core.h>

#include "sz_range.h"

struct sz_screen;

void sz_bo_init(struct sz_screen *screen);
void sz_bo_finish(struct sz_screen *screen);

/* Allocates `size` bytes that satisfy `req` and carry `props`.
 *
 * Returns NULL when no memory type satisfies both the requirement mask and the
 * property flags, or when the driver refuses the allocation. */
struct sz_bo *sz_bo_alloc(struct sz_screen *screen,
                          const VkMemoryRequirements *req,
                          VkMemoryPropertyFlags props);

/* Same, but on a private VkDeviceMemory that is never shared. */
struct sz_bo *sz_bo_alloc_dedicated(struct sz_screen *screen,
                                    const VkMemoryRequirements *req,
                                    VkMemoryPropertyFlags props);

void sz_bo_free(struct sz_bo *bo);

/* Persistent CPU pointer for host-visible memory, NULL otherwise. Blocks are
 * mapped once at creation, so this is a pointer load, not a map call. */
void *sz_bo_map(struct sz_bo *bo);

/* Explicit cache maintenance. Both are no-ops on coherent memory. */
void sz_bo_flush(struct sz_bo *bo, uint64_t offset, uint64_t size);
void sz_bo_invalidate(struct sz_bo *bo, uint64_t offset, uint64_t size);

/* Read-only accessors; the struct itself stays private to sz_bo.c. */
uint64_t sz_bo_size(const struct sz_bo *bo);
uint64_t sz_bo_offset(const struct sz_bo *bo);
VkDeviceMemory sz_bo_memory(const struct sz_bo *bo);

struct sz_bo_stats {
   uint64_t allocated;       /* bytes handed out to live allocations */
   uint64_t committed;       /* bytes of VkDeviceMemory we hold */
   uint64_t peak_allocated;
   unsigned live;            /* live allocation count */
   unsigned blocks;          /* VkDeviceMemory objects created so far */
};

void sz_bo_get_stats(const struct sz_screen *screen, struct sz_bo_stats *out);
void sz_bo_dump_stats(const struct sz_screen *screen);

#endif /* SZ_BO_H */
