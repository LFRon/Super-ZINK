/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * pipe_resource <-> VkImage / VkBuffer.
 *
 * Design notes that are not obvious from the code:
 *
 *  - Buffers never use dedicated memory. A GL buffer object is small, numerous
 *    and short-lived, which is exactly the shape a pooled allocator is for.
 *
 *  - Images ask the driver first. VkMemoryDedicatedRequirements decides whether
 *    the allocation must be private, and the answer differs per driver and per
 *    format (RADV wants dedicated memory for every compressed image).
 *
 *  - Tiled images are never host-visible. Mapping one would need a staging copy
 *    and a queue submission, which requires a VkCommandBuffer, which requires a
 *    pipe_context. Until Phase 1.C exists, sz_resource_map() returns NULL for
 *    them and the transfer path is simply unavailable -- better than silently
 *    handing back a pointer into memory the GPU is reading.
 *
 *  - Linear images exist so the winsys can hand out dmabufs. Their row pitch
 *    must be the driver's optimalBufferCopyRowPitchAlignment, not a guess.
 */

#include "sz_resource.h"

#include "sz_format.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

#include "vk_format.h"

#include <inttypes.h>
#include <string.h>

uint64_t
sz_resource_linear_pitch(struct sz_screen *screen, uint64_t width,
                         uint64_t bytes_per_pixel)
{
   const uint64_t align =
      MAX2(screen->props.limits.optimalBufferCopyRowPitchAlignment, 1ull);
   const uint64_t row = MAX2(width * bytes_per_pixel, 1ull);
   return (row + align - 1) & ~(align - 1);
}

static VkImageUsageFlags
sz_buffer_usage_from_bindings(unsigned bindings)
{
   VkBufferUsageFlags usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT;

   if (bindings & PIPE_BIND_VERTEX_BUFFER)
      usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
   if (bindings & PIPE_BIND_INDEX_BUFFER)
      usage |= VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
   if (bindings & PIPE_BIND_CONSTANT_BUFFER)
      usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
   if (bindings & PIPE_BIND_SHADER_BUFFER)
      usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
   if (bindings & PIPE_BIND_SAMPLER_VIEW)
      usage |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
   if (bindings & PIPE_BIND_SHADER_IMAGE)
      usage |= VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT;
   if (bindings & PIPE_BIND_COMMAND_ARGS_BUFFER)
      usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
   if (bindings & PIPE_BIND_QUERY_BUFFER)
      usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;

   /* bufferDeviceAddress is a required capability, so every buffer can carry a
    * device address. Declaring the usage up front avoids a second VkBuffer when
    * a shader starts using pointers. */
   usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

   return usage;
}

static VkMemoryPropertyFlags
sz_buffer_memory_properties(unsigned bindings, enum pipe_resource_usage usage)
{
   /* STAGING means the CPU is the main user, so the buffer has to be mappable
    * whatever else it is bound as. Mesa removed the old PIPE_BIND_TRANSFER_*
    * flags, so usage is now the only signal for this. */
   if (usage == PIPE_USAGE_STAGING) {
      return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   }

   /* Any buffer the CPU may write goes to host-visible memory and is flushed
    * lazily. Buffers that are only ever read by the GPU (index/vertex data
    * uploaded once, indirect args) prefer device-local. */
   if (bindings & (PIPE_BIND_CONSTANT_BUFFER | PIPE_BIND_SHADER_BUFFER |
                   PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE |
                   PIPE_BIND_QUERY_BUFFER | PIPE_BIND_COMMAND_ARGS_BUFFER))
      return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

   if (bindings & (PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_INDEX_BUFFER))
      return VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

   return VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
}

static struct sz_resource *
sz_resource_alloc(struct sz_screen *screen, const struct pipe_resource *tmpl)
{
   struct sz_resource *res = CALLOC_STRUCT(sz_resource);
   if (!res)
      return NULL;

   /* Shallow copy: base.width0/format/bind/... must survive, and pipe_resource
    * has no own allocator that a deep copy would need to duplicate. */
   res->base = *tmpl;
   res->base.next = NULL;
   /* Gallium reads base.screen in a lot of places -- the DRI frontend asks the
    * resource's own screen to export it, for one -- and a NULL there is a
    * segfault deep inside Mesa rather than anywhere near the resource code. */
   res->base.screen = &screen->base;
   /* The caller's template has a zeroed refcount, so the new resource has to
    * start at one. Without this the count goes 0 -> 1 on the first reference and
    * 1 -> 0 on the first release, and it can be freed twice: once here and once
    * by whoever took a second reference later. The failure shows up as heap
    * corruption in the frontend, nowhere near the resource code. */
   pipe_reference_init(&res->base.reference, 1);
   res->screen = screen;
   res->layout = VK_IMAGE_LAYOUT_UNDEFINED;
   res->undefined_contents = true;

   return res;
}

/* ── buffers ─────────────────────────────────────────────────────────────── */

static struct pipe_resource *
sz_buffer_create(struct sz_screen *screen, const struct pipe_resource *tmpl)
{
   if (tmpl->width0 == 0) {
      sz_warn("SZ: refusing to create a zero-sized buffer");
      return NULL;
   }

   VkBufferCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = tmpl->width0,
      .usage = sz_buffer_usage_from_bindings(tmpl->bind),
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };

   struct sz_resource *res = sz_resource_alloc(screen, tmpl);
   if (!res)
      return NULL;

   VkResult vkres = screen->vk.CreateBuffer(screen->dev, &ci, NULL, &res->buffer);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateBuffer(%" PRIu64 " bytes, usage 0x%x) failed "
                "(VkResult %d)", (uint64_t)tmpl->width0, ci.usage, vkres);
      FREE(res);
      return NULL;
   }

   VkMemoryRequirements req;
   screen->vk.GetBufferMemoryRequirements(screen->dev, res->buffer, &req);

   res->bo = sz_bo_alloc(screen, &req, sz_buffer_memory_properties(tmpl->bind, tmpl->usage));
   if (!res->bo) {
      screen->vk.DestroyBuffer(screen->dev, res->buffer, NULL);
      FREE(res);
      return NULL;
   }

   vkres = screen->vk.BindBufferMemory(screen->dev, res->buffer,
                                       sz_bo_memory(res->bo),
                                       sz_bo_offset(res->bo));
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkBindBufferMemory failed (VkResult %d)", vkres);
      sz_bo_free(res->bo);
      screen->vk.DestroyBuffer(screen->dev, res->buffer, NULL);
      FREE(res);
      return NULL;
   }

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: buffer %u bytes, bind 0x%x -> %" PRIu64 " bytes @ offset %"
                PRIu64,
                tmpl->width0, tmpl->bind, sz_bo_size(res->bo),
                sz_bo_offset(res->bo));
   }

   return &res->base;
}

/* ── images ──────────────────────────────────────────────────────────────── */

static VkImageCreateInfo
sz_image_create_info(struct sz_screen *screen, const struct pipe_resource *tmpl)
{
   const VkFormat format = vk_format_from_pipe_format(tmpl->format);
   const VkImageType type = sz_image_type(tmpl->target);
   const bool linear = sz_bindings_require_linear(tmpl->bind);

   VkImageUsageFlags usage = sz_image_usage_from_bindings(tmpl->bind, tmpl->format);
   VkImageCreateFlags flags = type == VK_IMAGE_TYPE_2D
                                 ? sz_image_create_flags(tmpl->target)
                                 : 0;

   /* 3D images are never attachments in Gallium's model. */
   if (type == VK_IMAGE_TYPE_3D) {
      usage &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      if (usage == 0)
         usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   }

   const unsigned levels = MAX2(tmpl->last_level + 1, 1);
   const unsigned layers = MAX2(tmpl->array_size, 1);

   /* A cube is six layers; PIPE_TEXTURE_CUBE reports array_size 1. */
   const bool cube = tmpl->target == PIPE_TEXTURE_CUBE ||
                     tmpl->target == PIPE_TEXTURE_CUBE_ARRAY;
   const unsigned depth = type == VK_IMAGE_TYPE_3D ? MAX2(tmpl->depth0, 1)
                          : cube                      ? 6
                                                      : layers;

   return (VkImageCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags,
      .imageType = type,
      .format = format,
      .extent = {
         .width = tmpl->width0,
         .height = type == VK_IMAGE_TYPE_1D ? 1 : MAX2(tmpl->height0, 1),
         .depth = depth,
      },
      .mipLevels = levels,
      .arrayLayers = layers,
      .samples = sz_vk_sample_count(tmpl->nr_samples),
      .tiling = linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
}

static struct pipe_resource *
sz_image_create(struct sz_screen *screen, const struct pipe_resource *tmpl)
{
   const VkFormat format = vk_format_from_pipe_format(tmpl->format);
   if (format == VK_FORMAT_UNDEFINED) {
      mesa_loge("SZ: no VkFormat for pipe format %s",
                util_format_name(tmpl->format));
      return NULL;
   }

   if (tmpl->width0 == 0 || tmpl->height0 == 0) {
      sz_warn("SZ: refusing to create a %ux%u image", tmpl->width0, tmpl->height0);
      return NULL;
   }

   VkImageCreateInfo ci = sz_image_create_info(screen, tmpl);

   struct sz_resource *res = sz_resource_alloc(screen, tmpl);
   if (!res)
      return NULL;
   res->linear = ci.tiling == VK_IMAGE_TILING_LINEAR;

   VkResult vkres = screen->vk.CreateImage(screen->dev, &ci, NULL, &res->image);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateImage(%ux%ux%u %s, %u level%s, %u samples, "
                "usage 0x%x, %s) failed (VkResult %d)",
                ci.extent.width, ci.extent.height, ci.extent.depth,
                util_format_name(tmpl->format), ci.mipLevels,
                ci.mipLevels == 1 ? "" : "s", ci.samples, ci.usage,
                res->linear ? "linear" : "optimal", vkres);
      FREE(res);
      return NULL;
   }

   /* Ask before allocating: whether an image must own its memory is a property
    * of the driver, not something to be guessed from the format. */
   VkMemoryDedicatedRequirements dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
   };
   VkMemoryRequirements2 req2 = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
      .pNext = &dedicated,
   };
   const VkImageMemoryRequirementsInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .image = res->image,
   };

   if (screen->vk.GetImageMemoryRequirements2)
      screen->vk.GetImageMemoryRequirements2(screen->dev, &info, &req2);
   else
      screen->vk.GetImageMemoryRequirements(screen->dev, res->image, &req2.memoryRequirements);

   const bool dedicated_alloc =
      dedicated.requiresDedicatedAllocation || dedicated.prefersDedicatedAllocation;

   /* Linear images must be host-visible: that is the entire point of asking for
    * one, since the winsys reads the pixels directly. Tiled images never are. */
   const VkMemoryPropertyFlags props =
      res->linear ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                  : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

   res->bo = dedicated_alloc
                ? sz_bo_alloc_dedicated(screen, &req2.memoryRequirements, props)
                : sz_bo_alloc(screen, &req2.memoryRequirements, props);
   if (!res->bo) {
      screen->vk.DestroyImage(screen->dev, res->image, NULL);
      FREE(res);
      return NULL;
   }

   vkres = screen->vk.BindImageMemory(screen->dev, res->image,
                                      sz_bo_memory(res->bo),
                                      sz_bo_offset(res->bo));
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkBindImageMemory failed (VkResult %d)", vkres);
      sz_bo_free(res->bo);
      screen->vk.DestroyImage(screen->dev, res->image, NULL);
      FREE(res);
      return NULL;
   }

   /* A whole-resource view, which is what a framebuffer attachment needs. Cube
    * maps get a cube view; everything else becomes 2D or 3D view of the same
    * shape, so the view never reinterprets anything. */
   VkImageViewType view_type;
   switch (tmpl->target) {
   case PIPE_TEXTURE_1D:
      view_type = VK_IMAGE_VIEW_TYPE_1D;
      break;
   case PIPE_TEXTURE_1D_ARRAY:
      view_type = VK_IMAGE_VIEW_TYPE_1D_ARRAY;
      break;
   case PIPE_TEXTURE_2D_ARRAY:
      view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      break;
   case PIPE_TEXTURE_3D:
      view_type = VK_IMAGE_VIEW_TYPE_3D;
      break;
   case PIPE_TEXTURE_CUBE:
      view_type = VK_IMAGE_VIEW_TYPE_CUBE;
      break;
   case PIPE_TEXTURE_CUBE_ARRAY:
      view_type = VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
      break;
   default:
      view_type = VK_IMAGE_VIEW_TYPE_2D;
      break;
   }

   const VkImageViewCreateInfo view_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = res->image,
      .viewType = view_type,
      .format = format,
      .subresourceRange = {
         .aspectMask = sz_format_aspect(tmpl->format),
         .baseMipLevel = 0,
         .levelCount = VK_REMAINING_MIP_LEVELS,
         .baseArrayLayer = 0,
         .layerCount = VK_REMAINING_ARRAY_LAYERS,
      },
   };

   vkres = screen->vk.CreateImageView(screen->dev, &view_ci, NULL, &res->view);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateImageView failed (VkResult %d)", vkres);
      sz_bo_free(res->bo);
      screen->vk.DestroyImage(screen->dev, res->image, NULL);
      FREE(res);
      return NULL;
   }

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: image %ux%ux%u %s, %u level%s, %u sample%s, %s%s -> "
                "%" PRIu64 " bytes",
                ci.extent.width, ci.extent.height, ci.extent.depth,
                util_format_name(tmpl->format), ci.mipLevels,
                ci.mipLevels == 1 ? "" : "s", ci.samples,
                ci.samples == 1 ? "" : "s",
                res->linear ? "linear" : "optimal",
                dedicated_alloc ? " dedicated" : "", sz_bo_size(res->bo));
   }

   return &res->base;
}

/* ── pipe_screen entry points ────────────────────────────────────────────── */

static bool
sz_screen_can_create_resource(struct pipe_screen *pscreen,
                              const struct pipe_resource *tmpl)
{
   if (tmpl->target == PIPE_BUFFER)
      return tmpl->width0 > 0 && tmpl->width0 <=
                                   sz_screen(pscreen)->props.limits.maxStorageBufferRange;

   return tmpl->width0 > 0 && tmpl->width0 <=
                                 sz_screen(pscreen)->props.limits.maxImageDimension2D;
}

static struct pipe_resource *
sz_screen_resource_create(struct pipe_screen *pscreen,
                          const struct pipe_resource *tmpl)
{
   struct sz_screen *screen = sz_screen(pscreen);

   if (!tmpl)
      return NULL;

   if (!screen->dev) {
      mesa_loge("SZ: resource_create before the VkDevice exists");
      return NULL;
   }

   return tmpl->target == PIPE_BUFFER ? sz_buffer_create(screen, tmpl)
                                      : sz_image_create(screen, tmpl);
}

static void
sz_screen_resource_destroy(struct pipe_screen *pscreen,
                           struct pipe_resource *pres)
{
   if (!pres)
      return;

   struct sz_screen *screen = sz_screen(pscreen);
   struct sz_resource *res = sz_resource(pres);

   /* Nothing defers the destroy yet, which is only safe because a resource is
    * never destroyed while a batch that touches it is still in flight: the state
    * tracker destroys resources at GL object deletion, and the batch that last
    * used one has been submitted (not necessarily completed). That is a real
    * hazard and becomes a deferred free keyed on the timeline value in Phase 3. */
   if (res->buffer != VK_NULL_HANDLE)
      screen->vk.DestroyBuffer(screen->dev, res->buffer, NULL);
   if (res->view != VK_NULL_HANDLE)
      screen->vk.DestroyImageView(screen->dev, res->view, NULL);
   if (res->image != VK_NULL_HANDLE)
      screen->vk.DestroyImage(screen->dev, res->image, NULL);

   sz_bo_free(res->bo);

   FREE(res);
}

void *
sz_resource_map(struct sz_resource *res)
{
   if (!res || !res->bo)
      return NULL;

   return sz_bo_map(res->bo);
}

/* Exporting a buffer to the winsys is what presentation needs: DRI3 hands the
 * compositor a dma-buf, and without it there is no swapchain. That is the next
 * piece of work, but the entry point has to exist now: the DRI frontend calls it
 * unconditionally, and a NULL pointer there is a segfault inside Mesa rather
 * than a message anyone can act on.
 *
 * Returning false is the honest answer -- it makes the DRI3 buffer allocation
 * fail, which is exactly the situation. */
static bool
sz_screen_resource_get_handle(struct pipe_screen *pscreen,
                              struct pipe_context *pipe,
                              struct pipe_resource *res,
                              struct winsys_handle *handle,
                              unsigned usage)
{
   static bool warned;
   if (!warned) {
      mesa_logw("SZ: dma-buf export is not implemented yet, so nothing can be "
                "presented. Presentation arrives with the WSI phase.");
      warned = true;
   }
   return false;
}

void
sz_resource_screen_init(struct sz_screen *screen)
{
   screen->base.can_create_resource = sz_screen_can_create_resource;
   screen->base.resource_create = sz_screen_resource_create;
   screen->base.resource_destroy = sz_screen_resource_destroy;
   screen->base.resource_get_handle = sz_screen_resource_get_handle;
}

/* ── self-test ───────────────────────────────────────────────────────────── */

#define SZ_SELFTEST_BUFFERS 64
#define SZ_SELFTEST_ALIGN 256

static void
sz_selftest_tmpl(struct pipe_resource *tmpl, enum pipe_texture_target target,
                 enum pipe_format format, unsigned width, unsigned height,
                 unsigned bind)
{
   *tmpl = (struct pipe_resource) {
      .target = target,
      .format = format,
      .width0 = width,
      .height0 = height,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 1,
      .bind = bind,
      .usage = PIPE_USAGE_DEFAULT,
   };
}

static unsigned
sz_selftest_buffer_roundtrip(struct sz_screen *screen)
{
   unsigned failures = 0;
   struct pipe_resource *bufs[SZ_SELFTEST_BUFFERS] = { 0 };
   struct sz_bo_stats before, mid, after;

   sz_bo_get_stats(screen, &before);

   /* 1. Many small buffers must share blocks: 64 x 4 KiB is 256 KiB, so a
    *    pooling allocator commits one block, not 64 allocations. */
   for (unsigned i = 0; i < SZ_SELFTEST_BUFFERS; i++) {
      struct pipe_resource tmpl;
      sz_selftest_tmpl(&tmpl, PIPE_BUFFER, PIPE_FORMAT_R8_UNORM, 4096, 1,
                       PIPE_BIND_VERTEX_BUFFER);
      bufs[i] = screen->base.resource_create(&screen->base, &tmpl);
      if (!bufs[i]) {
         mesa_loge("SZ: selftest: buffer %u of %u could not be created", i,
                   SZ_SELFTEST_BUFFERS);
         failures++;
         break;
      }
   }

   sz_bo_get_stats(screen, &mid);
   /* Pooling means one new VkDeviceMemory for the whole batch, not one per
    * buffer. Counting blocks is the direct measurement; comparing bytes would
    * only work if we knew the driver's allocation granularity. */
   const unsigned new_blocks = mid.blocks - before.blocks;
   if (new_blocks != 1) {
      mesa_loge("SZ: selftest: %u small buffers needed %u VkDeviceMemory "
                "objects, i.e. no pooling happened", SZ_SELFTEST_BUFFERS,
                new_blocks);
      failures++;
   }
   if (mid.allocated < (uint64_t)SZ_SELFTEST_BUFFERS * 4096) {
      mesa_loge("SZ: selftest: %u live allocations account for only %" PRIu64
                " bytes", mid.live, mid.allocated);
      failures++;
   }

   /* 2. Distinct buffers must not alias: their offsets within a block differ,
    *    and their mapped pointers differ. */
   for (unsigned i = 0; i < SZ_SELFTEST_BUFFERS; i++) {
      if (!bufs[i])
         continue;
      struct sz_resource *ri = sz_resource(bufs[i]);
      if (!ri->bo || sz_bo_size(ri->bo) < 4096) {
         mesa_loge("SZ: selftest: buffer %u has no usable allocation", i);
         failures++;
      }
      for (unsigned j = i + 1; j < SZ_SELFTEST_BUFFERS; j++) {
         if (!bufs[j])
            continue;
         struct sz_resource *rj = sz_resource(bufs[j]);
         const uint64_t a0 = sz_bo_offset(ri->bo);
         const uint64_t a1 = a0 + sz_bo_size(ri->bo);
         const uint64_t b0 = sz_bo_offset(rj->bo);
         const uint64_t b1 = b0 + sz_bo_size(rj->bo);
         if (sz_bo_memory(ri->bo) == sz_bo_memory(rj->bo) && a0 < b1 && b0 < a1) {
            mesa_loge("SZ: selftest: buffers %u and %u overlap in the same "
                      "VkDeviceMemory", i, j);
            failures++;
         }
      }
   }

   /* 3. Host-visible buffers must be writable and read back. */
   if (bufs[0]) {
      uint8_t *p = sz_resource_map(sz_resource(bufs[0]));
      if (!p) {
         mesa_loge("SZ: selftest: a vertex buffer is not host-visible");
         failures++;
      } else {
         for (unsigned i = 0; i < 4096; i++)
            p[i] = (uint8_t)(i * 7u);
         for (unsigned i = 0; i < 4096; i++) {
            if (p[i] != (uint8_t)(i * 7u)) {
               mesa_loge("SZ: selftest: host write/read mismatch at %u", i);
               failures++;
               break;
            }
         }
      }
   }

   /* 4. Free every other buffer, then re-allocate the same count. Freed ranges
    *    must be reused instead of growing the committed total. */
   for (unsigned i = 0; i < SZ_SELFTEST_BUFFERS; i += 2) {
      screen->base.resource_destroy(&screen->base, bufs[i]);
      bufs[i] = NULL;
   }

   for (unsigned i = 0; i < SZ_SELFTEST_BUFFERS; i += 2) {
      struct pipe_resource tmpl;
      sz_selftest_tmpl(&tmpl, PIPE_BUFFER, PIPE_FORMAT_R8_UNORM, 4096, 1,
                       PIPE_BIND_INDEX_BUFFER);
      bufs[i] = screen->base.resource_create(&screen->base, &tmpl);
      if (!bufs[i]) {
         mesa_loge("SZ: selftest: could not reallocate buffer %u", i);
         failures++;
      }
   }

   sz_bo_get_stats(screen, &after);
   if (after.committed > mid.committed + 4096) {
      mesa_loge("SZ: selftest: reuse did not happen -- committed grew from %"
                PRIu64 " to %" PRIu64 " bytes for the same working set",
                mid.committed, after.committed);
      failures++;
   }

   for (unsigned i = 0; i < SZ_SELFTEST_BUFFERS; i++) {
      if (bufs[i])
         screen->base.resource_destroy(&screen->base, bufs[i]);
   }

   return failures;
}

static unsigned
sz_selftest_image_roundtrip(struct sz_screen *screen)
{
   static const struct {
      const char *name;
      enum pipe_texture_target target;
      enum pipe_format format;
      unsigned width, height, samples;
      unsigned bind;
   } cases[] = {
      { "rgba8 render target", PIPE_TEXTURE_2D, PIPE_FORMAT_R8G8B8A8_UNORM,
        256, 256, 1, PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW },
      { "bgra8 scanout", PIPE_TEXTURE_2D, PIPE_FORMAT_B8G8R8A8_UNORM,
        1920, 1080, 1, PIPE_BIND_RENDER_TARGET | PIPE_BIND_DISPLAY_TARGET |
                       PIPE_BIND_SCANOUT },
      { "d24s8 depth", PIPE_TEXTURE_2D, PIPE_FORMAT_Z24_UNORM_S8_UINT,
        256, 256, 1, PIPE_BIND_DEPTH_STENCIL | PIPE_BIND_SAMPLER_VIEW },
      { "rgba8 4x msaa", PIPE_TEXTURE_2D, PIPE_FORMAT_R8G8B8A8_UNORM,
        256, 256, 4, PIPE_BIND_RENDER_TARGET },
      { "srgb8 2d array", PIPE_TEXTURE_2D_ARRAY, PIPE_FORMAT_R8G8B8A8_SRGB,
        64, 64, 1, PIPE_BIND_SAMPLER_VIEW },
      { "rgba8 cube", PIPE_TEXTURE_CUBE, PIPE_FORMAT_R8G8B8A8_UNORM,
        64, 64, 1, PIPE_BIND_SAMPLER_VIEW },
      { "r32f 3d", PIPE_TEXTURE_3D, PIPE_FORMAT_R32_FLOAT,
        32, 32, 1, PIPE_BIND_SAMPLER_VIEW },
      { "rgba8 mipped", PIPE_TEXTURE_2D, PIPE_FORMAT_R8G8B8A8_UNORM,
        128, 128, 1, PIPE_BIND_SAMPLER_VIEW },
   };

   unsigned failures = 0;

   for (unsigned i = 0; i < ARRAY_SIZE(cases); i++) {
      struct pipe_resource tmpl = {
         .target = cases[i].target,
         .format = cases[i].format,
         .width0 = cases[i].width,
         .height0 = cases[i].height,
         .depth0 = cases[i].target == PIPE_TEXTURE_3D ? 8 : 1,
         .array_size = cases[i].target == PIPE_TEXTURE_2D_ARRAY ? 4 : 1,
         .last_level = i == ARRAY_SIZE(cases) - 1 ? 3 : 0,
         .nr_samples = cases[i].samples,
         .bind = cases[i].bind,
         .usage = PIPE_USAGE_DEFAULT,
      };

      struct pipe_resource *res = screen->base.resource_create(&screen->base, &tmpl);
      if (!res) {
         /* Not every device can make every one of these; a 4x MSAA or 3D image
          * refusal is a legitimate answer, so only report it as information. */
         mesa_logi("SZ: selftest: %-20s not creatable on this device",
                   cases[i].name);
         continue;
      }

      struct sz_resource *r = sz_resource(res);
      if (r->image == VK_NULL_HANDLE || !r->bo) {
         mesa_loge("SZ: selftest: %s created without an image or allocation",
                   cases[i].name);
         failures++;
      }

      /* A tiled image must not be mappable: there is no way to write it from
       * the CPU without a staging copy, and pretending otherwise corrupts. */
      if (!r->linear && sz_resource_map(r) != NULL) {
         mesa_loge("SZ: selftest: %s is tiled but reported a host pointer",
                   cases[i].name);
         failures++;
      }

      if (sz_debug_enabled(SZ_DEBUG_INFO)) {
         mesa_logi("SZ: selftest: %-20s ok, %" PRIu64 " bytes%s",
                   cases[i].name, sz_bo_size(r->bo),
                   r->linear ? " (linear)" : "");
      }

      screen->base.resource_destroy(&screen->base, res);
   }

   return failures;
}

unsigned
sz_resource_selftest(struct sz_screen *screen)
{
   struct sz_bo_stats start, end;
   sz_bo_get_stats(screen, &start);

   unsigned failures = sz_selftest_buffer_roundtrip(screen);
   failures += sz_selftest_image_roundtrip(screen);

   sz_bo_get_stats(screen, &end);

   /* Everything created above was destroyed again, so the live count must be
    * back where it started. A non-zero difference is a leak in destroy. */
   if (end.live != start.live || end.allocated != start.allocated) {
      mesa_loge("SZ: selftest: leaked %u allocation%s (%" PRIu64 " bytes)",
                end.live - start.live, end.live - start.live == 1 ? "" : "s",
                end.allocated - start.allocated);
      failures++;
   }

   if (failures)
      mesa_loge("SZ: selftest: %u check%s FAILED", failures,
                failures == 1 ? "" : "s");
   else
      mesa_logi("SZ: selftest: passed (%u blocks, %" PRIu64 " KiB committed, "
                "peak %" PRIu64 " KiB)", end.blocks, end.committed / 1024,
                end.peak_allocated / 1024);

   return failures;
}
