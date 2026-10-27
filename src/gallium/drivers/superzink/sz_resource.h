/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * pipe_resource backed by a VkImage or a VkBuffer.
 *
 * Every resource owns exactly one sz_bo. That is a deliberate simplification
 * over zink, which keeps a separate surface/level table: the state tracker
 * creates a resource per GL object and destroys it with the object, so the
 * lifetime matches and the extra indirection buys nothing here.
 *
 * Image layout tracking lives here too (sz_resource::layout). Gallium does not
 * tell us about barriers -- it has no such concept -- so the driver infers
 * required transitions from usage. Keeping the layout on the resource is what
 * makes that inference possible without a separate state machine keyed by
 * something else.
 */

#ifndef SZ_RESOURCE_H
#define SZ_RESOURCE_H

#include "sz_bo.h"
#include "sz_types.h"

struct sz_resource {
   struct pipe_resource base;

   struct sz_screen *screen;
   struct sz_bo *bo;

   /* Exactly one of these is non-NULL. */
   VkImage image;
   VkBuffer buffer;

   /* A whole-resource view of `image`, created alongside it.
    *
    * Vulkan cannot bind an image to a framebuffer or a descriptor without a
    * view, and the two differ only in how much of the image they address. The
    * cost is one extra object per texture, which is far less than creating and
    * destroying a view per binding -- and GL rebinds constantly. Views that
    * reinterpret the format or address a sub-range need their own object and
    * arrive with the sampler-view work. */
   VkImageView view;

   /* Current layout, or VK_IMAGE_LAYOUT_UNDEFINED for buffers. */
   VkImageLayout layout;

   /* Buffers only, because images express the same information through their
    * layout. The last stage and access that wrote this buffer, used to emit a
    * barrier only when a write is actually pending. Without it a buffer read as
    * vertex data after a transfer write is a data race that Vulkan will not
    * detect and that shows up as stale vertices.
    *
    * There is no per-range tracking: one stage/access pair covers the whole
    * buffer, so a write to one end forces a barrier before reading the other.
    * Coarse but never wrong. */
   VkPipelineStageFlags2 pending_write_stages;
   VkAccessFlags2 pending_write_access;

   /* Set when the resource was created with linear tiling, which forbids
    * VK_IMAGE_TILING_OPTIMAL-only usage and needs explicit cache flushes. */
   bool linear;

   /* Set when the contents are known to be undefined, so the next write may use
    * loadOp=DONT_CARE / discard instead of preserving garbage. */
   bool undefined_contents;
};

static inline struct sz_resource *
sz_resource(struct pipe_resource *res)
{
   return (struct sz_resource *)res;
}

/* Small accessors, so callers do not need the sz_bo layout. */
uint64_t sz_bo_size(const struct sz_bo *bo);
uint64_t sz_bo_offset(const struct sz_bo *bo);
VkDeviceMemory sz_bo_memory(const struct sz_bo *bo);

/* Aligns a row pitch to the device's optimalBufferCopyRowPitchAlignment, which
 * is what a linear image must use to be copyable. */
uint64_t sz_resource_linear_pitch(struct sz_screen *screen, uint64_t width,
                                  uint64_t bytes_per_pixel);

/* Convenience for the transfer paths: one memcpy-able pointer for a buffer or a
 * linear image. Returns NULL for tiled images, which cannot be mapped. */
void *sz_resource_map(struct sz_resource *res);

/* Installs the pipe_screen resource entry points. */
void sz_resource_screen_init(struct sz_screen *screen);

/* SZ_DEBUG=selftest: round-trips representative buffers and images through the
 * real device and the allocator. Returns the number of failed checks, so a
 * caller can turn a failure into a hard error.
 *
 * This exists because resources are the first subsystem that can be verified
 * without a pipe_context: everything here is a create-and-bind round trip. It
 * catches the two mistakes that are otherwise silent until much later --
 * binding a VkBuffer to the wrong offset, and handing the same bytes to two
 * resources. */
unsigned sz_resource_selftest(struct sz_screen *screen);

#endif /* SZ_RESOURCE_H */
