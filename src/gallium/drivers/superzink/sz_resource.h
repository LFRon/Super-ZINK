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

/* One view of a subresource range of an image, remembered so a framebuffer that
 * names the same layer twice does not create a second object.
 *
 * These live on the resource, not on the context, because their lifetime is the
 * image's: Vulkan requires every view of an image to be destroyed before the
 * image, and a cache that outlived the resource handed out handles that had
 * already been destroyed whenever a new resource reused the old address. */
struct sz_subresource_view {
   VkImageView view;
   VkFormat format;
   VkImageViewType type;
   VkImageAspectFlags aspects;
   unsigned first_level, level_count;
   unsigned first_layer, layer_count;
};

/* A cache bound rather than a hard limit: past this, callers fall back to the
 * whole-image view for unlayered attachments. */
#define SZ_MAX_SUBRESOURCE_VIEWS 32

struct sz_resource {
   struct pipe_resource base;

   struct sz_screen *screen;
   struct sz_bo *bo;

   /* Exactly one of these is non-NULL. */
   VkImage image;
   VkBuffer buffer;
   /* Device address of `buffer`, valid when it is not VK_NULL_HANDLE. A
    * descriptor names a buffer by address in the descriptor-buffer path, and
    * asking once at creation beats asking on every draw. */
   uint64_t address;
   /* A 1D depth/stencil resource created as a 2D image one texel high, because
    * the device has no 1D depth. Views and shader image dimensions follow it. */
   bool aliased_2d;

   /* A whole-resource view of `image`, created alongside it.
    *
    * Vulkan cannot bind an image to a framebuffer or a descriptor without a
    * view, and the two differ only in how much of the image they address. The
    * cost is one extra object per texture, which is far less than creating and
    * destroying a view per binding -- and GL rebinds constantly. Views that
    * reinterpret the format or address a sub-range need their own object and
    * arrive with the sampler-view work. */
   VkImageView view;

   /* Views of the subresource ranges of `image` that a framebuffer attachment
    * or a storage image unit named: one cube face, one array layer, or a
    * format reinterpretation of the whole thing. Destroyed with the image. */
   struct sz_subresource_view *subresource_views;
   unsigned num_subresource_views;
   unsigned subresource_views_capacity;

   /* Current layout, or VK_IMAGE_LAYOUT_UNDEFINED for buffers.
    *
    * This is the layout of every mip level, and it is what almost every
    * transition uses. A mipmap chain is the exception: generating level N reads
    * level N-1 and writes level N, and the two are in different layouts at that
    * moment. `levels` records the levels that differ from `layout` -- NULL until
    * one does, so the common case costs no allocation and no lookups. */
   VkImageLayout layout;
   VkImageLayout *levels;

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

   /* A buffer whose contents the CPU wrote through a direct mapping and which
    * the GPU has not yet been told about. Vulkan makes host writes visible to
    * the device only after a barrier from HOST/HOST_WRITE, and the uploaders
    * write vertices and uniforms this way, so the flag is what turns "the
    * buffer is already bound" into "there is nothing to synchronise" -- which
    * was wrong and showed up as intermittent garbage geometry.
    *
    * Cleared once a barrier has covered the writes, but only while no write
    * mapping is still open: see write_maps. */
   bool host_written;

   /* How many direct write mappings the CPU currently holds on this buffer.
    *
    * Non-zero means the CPU may write at any time, so the flag above cannot be
    * cleared -- there is no later event that would mark the next write. This is
    * not an edge case: u_upload_mgr maps its streaming buffer once and keeps it
    * mapped, so every vertex and index upload Minecraft makes lands in a buffer
    * whose mapping never closes, and the barrier that the flag exists to emit
    * was therefore emitted exactly never. The visible symptom is one vertex of
    * a quad reading another draw's data, which turns the logo into a bow tie. */
   unsigned write_maps;

   /* The timeline value of the last submission that named this buffer.
    *
    * Only used by SZ_DEBUG=race: a CPU write into a buffer whose last reader is
    * still on the device is a write-after-read the application is supposed to
    * have prevented with a fence, so seeing one means the fence the application
    * trusted reported completion too early. */
   uint64_t last_use_value;

   /* Set when the resource was created with linear tiling, which forbids
    * VK_IMAGE_TILING_OPTIMAL-only usage and needs explicit cache flushes. */
   bool linear;

   /* Set when the tiling is VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT rather than
    * plain linear. Such an image is still described as linear pitch-wise, but
    * Vulkan refuses to describe its subresources with a colour/depth aspect: for
    * vkGetImageSubresourceLayout the aspect has to be
    * VK_IMAGE_ASPECT_MEMORY_PLANE_i_BIT_EXT (VUID 09433). */
   bool drm_modifier;

   /* Set when the contents are known to be undefined, so the next write may use
    * loadOp=DONT_CARE / discard instead of preserving garbage. */
   bool undefined_contents;

   /* Set when the VkImage was created with an external memory handle type, so
    * vkGetMemoryFdKHR may be called on its memory. Vulkan refuses the export
    * otherwise, and the refusal is a validation error rather than a VkResult. */
   bool exportable;

   /* Set when the memory came from a dma-buf instead of sz_bo. The fd itself is
    * owned by Vulkan after the import, so the only thing that has to be freed
    * here is the VkDeviceMemory. */
   VkDeviceMemory imported_memory;
};

/* The VkDeviceMemory behind a resource, whichever way it was obtained.
 * VK_NULL_HANDLE for a resource that owns no memory yet. */
VkDeviceMemory sz_resource_memory(const struct sz_resource *res);

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

/* Destroys every resource still waiting for a submission to finish. Called from
 * screen teardown, after the queue has gone idle and before the device is
 * destroyed. */
void sz_resource_reclaim_all(struct sz_screen *screen);

/* Destroys the retired resources whose submission has completed. Cheap when the
 * table is empty, so callers on hot paths may call it unconditionally. */
void sz_resource_reclaim(struct sz_screen *screen);

/* Hands `view`/`buffer_view` over to the screen, which destroys them once the
 * submission that may still name them has completed. */
void sz_screen_retire_view(struct sz_screen *screen, VkImageView view,
                           VkBufferView buffer_view);

/* Wraps a VkImage whose memory was imported from a dma-buf into a
 * pipe_resource. Used by sz_dmabuf's resource_from_handle, which owns the
 * VkImage creation because only it knows the plane layout. */
struct pipe_resource *sz_resource_wrap_imported(struct sz_screen *screen,
                                                const struct pipe_resource *tmpl,
                                                VkImage image,
                                                VkDeviceMemory memory,
                                                bool drm_modifier);

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
