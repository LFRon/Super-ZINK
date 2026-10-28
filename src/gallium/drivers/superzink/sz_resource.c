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

#include "sz_batch.h"
#include "sz_dmabuf.h"
#include "sz_format.h"
#include "sz_sync.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

#include "drm-uapi/drm_fourcc.h"
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
   if (bindings & PIPE_BIND_STREAM_OUTPUT)
      usage |= VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT;

   /* Gallium lets a buffer be bound for a purpose its creation bind flags never
    * mentioned -- the state tracker uploads client vertex arrays into a buffer
    * created for something else, then binds it with vkCmdBindVertexBuffers,
    * which requires VK_BUFFER_USAGE_VERTEX_BUFFER_BIT (VUID 00627). Vulkan has
    * no way to widen a VkBuffer's usage after creation, and buffers are untyped
    * linear memory, so the flags are declared as a superset instead: the cost is
    * a few bits, the cost of the alternative is a validation error now and a
    * driver-side rejection later. */
   usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

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
                   PIPE_BIND_QUERY_BUFFER | PIPE_BIND_COMMAND_ARGS_BUFFER |
                   PIPE_BIND_STREAM_OUTPUT))
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
   res->imported_memory = VK_NULL_HANDLE;

   return res;
}

VkDeviceMemory
sz_resource_memory(const struct sz_resource *res)
{
   if (res->imported_memory != VK_NULL_HANDLE)
      return res->imported_memory;
   return res->bo ? sz_bo_memory(res->bo) : VK_NULL_HANDLE;
}

/* ── buffers ─────────────────────────────────────────────────────────────── */

static struct pipe_resource *
sz_buffer_create(struct sz_screen *screen, const struct pipe_resource *tmpl)
{
   if (tmpl->width0 == 0) {
      sz_warn("refusing to create a zero-sized buffer");
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

   /* A descriptor names a buffer by device address in the descriptor-buffer
    * path, so the address is taken once here rather than per draw. */
   const VkBufferDeviceAddressInfo address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = res->buffer,
   };
   res->address = screen->vk.GetBufferDeviceAddress(screen->dev, &address_info);

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: buffer %u bytes, bind 0x%x -> %" PRIu64 " bytes @ offset %"
                PRIu64 " address 0x%" PRIx64,
                tmpl->width0, tmpl->bind, sz_bo_size(res->bo),
                sz_bo_offset(res->bo), res->address);
   }

   return &res->base;
}

/* ── images ──────────────────────────────────────────────────────────────── */

static VkImageCreateInfo
sz_image_create_info(struct sz_screen *screen, const struct pipe_resource *tmpl)
{
   const VkFormat format = sz_vk_format(screen, tmpl->format);
   const VkImageType type = sz_image_vk_type(screen, tmpl->target, tmpl->format);
   const bool linear = sz_bindings_require_linear(tmpl->bind);

   VkImageUsageFlags usage = sz_image_usage_from_bindings(tmpl->bind, tmpl->format);
   VkImageCreateFlags flags = type == VK_IMAGE_TYPE_2D
                                 ? sz_image_create_flags(tmpl->target)
                                 : 0;

   /* 3D images are never attachments in Gallium's model, and a 1D image cannot
    * have a depth/stencil attachment in Vulkan at all -- but GL has 1D depth
    * textures, so the sampled use has to survive while the attachment use does
    * not. Creating the image with the attachment bit is what would fail. */
   if (type == VK_IMAGE_TYPE_3D) {
      usage &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      if (usage == 0)
         usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   } else if (tmpl->target == PIPE_TEXTURE_1D ||
              tmpl->target == PIPE_TEXTURE_1D_ARRAY) {
      usage &= ~(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      if (usage == 0)
         usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   }

   const unsigned levels = MAX2(tmpl->last_level + 1, 1);
   const unsigned layers = MAX2(tmpl->array_size, 1);

   /* Only a 3D image has depth; for every 1D, 2D, array and cube target Vulkan
    * requires extent.depth to be 1 and puts the layers in arrayLayers instead
    * (VUID-VkImageCreateInfo-imageType-00957). Gallium already counts a cube as
    * six layers in array_size, so the two agree without any special case. */
   const unsigned depth = type == VK_IMAGE_TYPE_3D ? MAX2(tmpl->depth0, 1) : 1;

   return (VkImageCreateInfo) {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags,
      .imageType = type,
      .format = format,
      .extent = {
         .width = tmpl->width0,
         .height = type == VK_IMAGE_TYPE_1D ||
                     tmpl->target == PIPE_TEXTURE_1D ||
                     tmpl->target == PIPE_TEXTURE_1D_ARRAY
                      ? 1
                      : MAX2(tmpl->height0, 1),
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

/* A whole-resource view, which is what a framebuffer attachment needs. Cube maps
 * get a cube view; everything else becomes a 1D/2D/3D view of the same shape, so
 * the view never reinterprets anything.
 *
 * Shared with the dma-buf import path, which creates its VkImage outside this
 * file and would otherwise carry a second copy of this switch. */
static bool
sz_resource_create_view(struct sz_screen *screen, struct sz_resource *res,
                        VkFormat format)
{
   VkImageViewType view_type;
   /* A 1D depth/stencil texture is aliased to a 2D image one texel high on a
    * device without 1D depth, so its view has to be 2D as well -- a 1D view of a
    * 2D image is rejected. */
   const bool aliased_2d = screen->need_2d_zs &&
                           util_format_is_depth_or_stencil(res->base.format) &&
                           (res->base.target == PIPE_TEXTURE_1D ||
                            res->base.target == PIPE_TEXTURE_1D_ARRAY);

   switch (res->base.target) {
   case PIPE_TEXTURE_1D:
      view_type = aliased_2d ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_1D;
      break;
   case PIPE_TEXTURE_1D_ARRAY:
      view_type = aliased_2d ? VK_IMAGE_VIEW_TYPE_2D_ARRAY
                             : VK_IMAGE_VIEW_TYPE_1D_ARRAY;
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
         .aspectMask = sz_format_aspect(res->base.format),
         .baseMipLevel = 0,
         .levelCount = VK_REMAINING_MIP_LEVELS,
         .baseArrayLayer = 0,
         .layerCount = VK_REMAINING_ARRAY_LAYERS,
      },
   };

   VkResult vkres = screen->vk.CreateImageView(screen->dev, &view_ci, NULL,
                                               &res->view);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateImageView failed (VkResult %d)", vkres);
      return false;
   }
   return true;
}

/* Defines the contents of a scanout image before anybody else can look at it.
 *
 * An exported image goes to the winsys as a dma-buf file descriptor, and the X
 * server starts sampling the imported pixmap as soon as the window is mapped --
 * on a DRI3 server that happens *before* the client renders its first frame.
 * The image lives in DEVICE_LOCAL memory, so nothing else has defined its
 * contents: the compositor would show whatever the allocator handed back, which
 * for a linear image is the previous tenant's pixel data (the familiar flash of
 * garbage in the window corner on startup).
 *
 * Clearing costs one submission per back buffer at allocation time. GL makes the
 * contents of a fresh image undefined, so there is no data to preserve -- the
 * only thing being chosen here is *which* undefined contents, and zeros are the
 * only ones that are safe to show. */
static void
sz_resource_zero_exported(struct sz_screen *screen, struct sz_resource *res)
{
   /* Depth/stencil scanout exists but is rare, and the clear needs a
    * depth-stencil layout rather than a colour one. Leave it alone rather than
    * clear the wrong aspect. */
   const struct util_format_description *desc =
      util_format_description(res->base.format);
   if (util_format_has_depth(desc) || util_format_has_stencil(desc))
      return;

   struct sz_batch *batch = sz_batch_begin(&screen->batch_state);
   if (!batch)
      return;

   sz_resource_access(screen, batch->cmd, res, SZ_ACCESS_TRANSFER_DST);

   const VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1,
   };
   const VkClearColorValue zero = { .float32 = { 0.0f, 0.0f, 0.0f, 0.0f } };

   screen->vk.CmdClearColorImage(batch->cmd, res->image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1,
                                 &range);

   /* The image stays in TRANSFER_DST_OPTIMAL: the first draw or copy out of it
    * transitions it again, and the layout tracker now knows what it holds. */

   if (!sz_batch_end(batch) ||
       !sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT))
      return;

   /* The export that follows has no fence of its own, so the zeros must be in
    * memory -- not merely queued -- before the file descriptor exists. */
   sz_queue_wait_idle(&screen->batch_state);
}

static struct pipe_resource *
sz_image_create(struct sz_screen *screen, const struct pipe_resource *tmpl)
{
   const VkFormat format = sz_vk_format(screen, tmpl->format);
   if (format == VK_FORMAT_UNDEFINED) {
      mesa_loge("SZ: no VkFormat for pipe format %s (%ux%u target %d bind 0x%x "
                "usage 0x%x)", util_format_name(tmpl->format), tmpl->width0,
                tmpl->height0, (int)tmpl->target, tmpl->bind, tmpl->usage);
      return NULL;
   }

   if (tmpl->width0 == 0 || tmpl->height0 == 0) {
      sz_warn("refusing to create a %ux%u image", tmpl->width0, tmpl->height0);
      return NULL;
   }

   /* A texture the state tracker wants to hand to somebody else has to be in a
    * layout that other process can describe. DRM_FORMAT_MOD_LINEAR is the one
    * layout every consumer supports, so that is the only one this phase claims;
    * declaring it through VK_EXT_image_drm_format_modifier is what makes Vulkan
    * honour it instead of picking a driver-internal layout and telling us the
    * row pitch afterwards. */
   const bool exportable = sz_bindings_require_export(tmpl->bind) &&
                           screen->caps.external_memory_dma_buf &&
                           tmpl->target == PIPE_TEXTURE_2D &&
                           tmpl->last_level == 0 && MAX2(tmpl->array_size, 1) == 1;

   VkImageCreateInfo ci = sz_image_create_info(screen, tmpl);

   VkExternalMemoryImageCreateInfo ext_ci = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   const uint64_t linear_modifier = DRM_FORMAT_MOD_LINEAR;
   VkImageDrmFormatModifierListCreateInfoEXT mod_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
      .drmFormatModifierCount = 1,
      .pDrmFormatModifiers = &linear_modifier,
   };

   if (exportable) {
      ci.tiling = screen->caps.drm_format_modifier
                     ? VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
                     : VK_IMAGE_TILING_LINEAR;
      /* A list of one leaves the driver no choice, which is the point: the
       * consumer will be told LINEAR and cannot be told otherwise. */
      ext_ci.pNext = screen->caps.drm_format_modifier ? &mod_ci : NULL;
      ci.pNext = &ext_ci;
      ci.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   }

   struct sz_resource *res = sz_resource_alloc(screen, tmpl);
   if (!res)
      return NULL;
   res->aliased_2d = screen->need_2d_zs &&
                     util_format_is_depth_or_stencil(tmpl->format) &&
                     (tmpl->target == PIPE_TEXTURE_1D ||
                      tmpl->target == PIPE_TEXTURE_1D_ARRAY);
   res->linear = ci.tiling != VK_IMAGE_TILING_OPTIMAL;
   res->drm_modifier = ci.tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   res->exportable = exportable;

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

   /* An exported image lives in VRAM even though it is linear: the consumer
    * samples it through its own GPU, never through the CPU, and a back buffer in
    * host-visible memory crosses PCIe on every presented frame. CPU access via
    * PIPE_BIND_LINEAR still works -- sz_transfer_map() falls back to a staging
    * buffer when the resource has no persistent mapping. */
   const VkMemoryPropertyFlags props =
      exportable ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
      : res->linear ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

   res->bo = exportable
                ? sz_bo_alloc_export(screen, &req2.memoryRequirements, props,
                                     res->image)
                : dedicated_alloc
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

   if (!sz_resource_create_view(screen, res, format)) {
      sz_bo_free(res->bo);
      screen->vk.DestroyImage(screen->dev, res->image, NULL);
      FREE(res);
      return NULL;
   }

   if (exportable)
      sz_resource_zero_exported(screen, res);

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: image %ux%ux%u %s, %u level%s, %u sample%s, %s%s%s -> "
                "%" PRIu64 " bytes, bind 0x%x, VkImage %p",
                ci.extent.width, ci.extent.height, ci.extent.depth,
                util_format_name(tmpl->format), ci.mipLevels,
                ci.mipLevels == 1 ? "" : "s", ci.samples,
                ci.samples == 1 ? "" : "s",
                res->linear ? "linear" : "optimal",
                dedicated_alloc ? " dedicated" : "", exportable ? " export" : "",
                sz_bo_size(res->bo), res->base.bind, (void *)res->image);
   }

   return &res->base;
}

/* Wraps a VkImage whose memory came from a dma-buf. The caller has already
 * created the image and imported the memory; everything that is common with a
 * locally allocated image -- the resource header, the view, the refcount -- is
 * done here so the two paths cannot drift. */
struct pipe_resource *
sz_resource_wrap_imported(struct sz_screen *screen,
                          const struct pipe_resource *tmpl,
                          VkImage image, VkDeviceMemory memory,
                          bool drm_modifier)
{
   const VkFormat format = sz_vk_format(screen, tmpl->format);

   struct sz_resource *res = sz_resource_alloc(screen, tmpl);
   if (!res)
      return NULL;

   res->image = image;
   res->imported_memory = memory;
   /* Linear by definition: it is a dma-buf, and this phase only accepts
    * DRM_FORMAT_MOD_LINEAR. */
   res->linear = true;
   res->drm_modifier = drm_modifier;
   /* The contents belong to whoever made the buffer, so the first write may not
    * discard them. */
   res->undefined_contents = false;

   if (!sz_resource_create_view(screen, res, format)) {
      FREE(res);
      return NULL;
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
sz_resource_destroy_now(struct sz_screen *screen, struct pipe_resource *pres)
{
   if (!pres)
      return;

   struct sz_resource *res = sz_resource(pres);

   if (res->buffer != VK_NULL_HANDLE)
      screen->vk.DestroyBuffer(screen->dev, res->buffer, NULL);
   if (res->view != VK_NULL_HANDLE)
      screen->vk.DestroyImageView(screen->dev, res->view, NULL);
   if (res->image != VK_NULL_HANDLE)
      screen->vk.DestroyImage(screen->dev, res->image, NULL);

   /* sz_bo_free refuses a NULL bo, so an imported resource needs nothing more
    * than the memory handle it owns. */
   if (res->imported_memory != VK_NULL_HANDLE)
      screen->vk.FreeMemory(screen->dev, res->imported_memory, NULL);

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      if (res->image != VK_NULL_HANDLE)
         mesa_logi("SZ: destroy image %p", (void *)res->image);
      else if (res->buffer != VK_NULL_HANDLE)
         mesa_logi("SZ: destroy buffer %p", (void *)res->buffer);
   }

   sz_bo_free(res->bo);
   free(res->levels);

   FREE(res);
}

void
sz_resource_reclaim(struct sz_screen *screen)
{
   const uint64_t completed = screen->batch_state.completed_value;

   for (unsigned i = 0; i < screen->num_retired_resources; ) {
      if (screen->retired_resources[i].value > completed) {
         i++;
         continue;
      }

      struct pipe_resource *pres = screen->retired_resources[i].res;
      screen->num_retired_resources--;
      screen->retired_resources[i] =
         screen->retired_resources[screen->num_retired_resources];
      sz_resource_destroy_now(screen, pres);
   }
}

static void
sz_screen_resource_destroy(struct pipe_screen *pscreen,
                           struct pipe_resource *pres)
{
   if (!pres)
      return;

   struct sz_screen *screen = sz_screen(pscreen);

   /* The last reference just went away, which says nothing about the GPU: a
    * draw recorded into the current command buffer, or a batch already
    * submitted, may still name this VkBuffer. Destroying it there is a
    * use-after-free the validation layer reports as a device loss, so the
    * teardown waits for the submission that will make it safe.
    *
    * next_value + 1 rather than next_value: the batch being recorded has not
    * signalled yet and will take the next value on submission. Waiting for one
    * value more than strictly needed costs nothing -- the value only has to be
    * *at least* as large as the submission that references the resource. */
   const uint64_t retire_value = screen->batch_state.next_value + 1;

   sz_resource_reclaim(screen);

   if (screen->num_retired_resources == screen->retired_resources_capacity) {
      const unsigned capacity = MAX2(screen->retired_resources_capacity * 2, 64);
      struct sz_retired_resource *grown =
         realloc(screen->retired_resources, capacity * sizeof(*grown));
      if (grown) {
         screen->retired_resources = grown;
         screen->retired_resources_capacity = capacity;
      }
   }

   if (screen->num_retired_resources < screen->retired_resources_capacity) {
      screen->retired_resources[screen->num_retired_resources].res = pres;
      screen->retired_resources[screen->num_retired_resources].value =
         retire_value;
      screen->num_retired_resources++;
      return;
   }

   /* Only reachable when the table could not grow. Waiting is the one thing that
    * makes progress; the resource is then destroyed by the reclaim below rather
    * than here, because the recording command buffer may still name it. */
   sz_queue_wait_idle(&screen->batch_state);
   sz_resource_reclaim(screen);
   screen->retired_resources[screen->num_retired_resources].res = pres;
   screen->retired_resources[screen->num_retired_resources].value = retire_value;
   screen->num_retired_resources++;
}

void
sz_resource_reclaim_all(struct sz_screen *screen)
{
   for (unsigned i = 0; i < screen->num_retired_resources; i++)
      sz_resource_destroy_now(screen, screen->retired_resources[i].res);
   screen->num_retired_resources = 0;
   free(screen->retired_resources);
   screen->retired_resources = NULL;
   screen->retired_resources_capacity = 0;
}

void *
sz_resource_map(struct sz_resource *res)
{
   if (!res || !res->bo)
      return NULL;

   return sz_bo_map(res->bo);
}

void
sz_resource_screen_init(struct sz_screen *screen)
{
   screen->base.can_create_resource = sz_screen_can_create_resource;
   screen->base.resource_create = sz_screen_resource_create;
   screen->base.resource_destroy = sz_screen_resource_destroy;

   /* resource_get_handle / resource_from_handle and the modifier queries all
    * belong to the dma-buf path, which needs the external-memory extensions. */
   sz_dmabuf_screen_init(screen);
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

   /* resource_destroy() defers the teardown until the submission that may still
    * name the resource has retired, so the deferred table has to be drained
    * before the counters say anything: nothing else runs during screen
    * creation, which makes waiting here free. */
   sz_queue_wait_idle(&screen->batch_state);
   sz_resource_reclaim_all(screen);

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
