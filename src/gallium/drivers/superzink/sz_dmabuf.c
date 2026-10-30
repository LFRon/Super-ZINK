/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * dma-buf export and import. See sz_dmabuf.h for the design.
 *
 * The three moves that are easy to get wrong, all of them enforced by Vulkan
 * rather than by the compiler:
 *
 *  - The VkImage has to be created with VkExternalMemoryImageCreateInfo. An
 *    image created without it cannot export its memory at all, and the failure
 *    is a validation error on vkGetMemoryFdKHR, not a VkResult.
 *
 *  - Exportable memory must be a whole VkDeviceMemory allocated with
 *    VkExportMemoryAllocateInfo; it cannot be carved out of sz_bo's pool.
 *
 *  - An imported fd is owned by Vulkan after vkAllocateMemory, while Gallium's
 *    contract says the caller keeps its fd. Those two cannot both be true of the
 *    same descriptor, so the fd is dup'd here and the original stays the
 *    caller's.
 */

#include "sz_dmabuf.h"

#include "sz_batch.h"
#include "sz_context.h"
#include "sz_debug.h"
#include "sz_device.h"
#include "sz_format.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"

#include "drm-uapi/drm_fourcc.h"
#include "vk_format.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define SZ_DMABUF_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT

static bool
sz_dmabuf_available(const struct sz_screen *screen)
{
   return screen->caps.external_memory_dma_buf &&
          screen->vk.GetMemoryFdKHR && screen->vk.GetMemoryFdPropertiesKHR;
}

/* One warning per kind, not one per call: a missing feature is reported by every
 * allocation attempt, and a log that repeats a thousand times per second is how
 * a real error gets missed. */
static void
sz_dmabuf_unsupported_once(struct sz_screen *screen, const char *what)
{
   if (screen->dmabuf_warned)
      return;
   screen->dmabuf_warned = true;
   mesa_logw("SZ: %s needs VK_EXT_external_memory_dma_buf, which this device "
             "does not offer; nothing can be presented", what);
}

/* ── queries ─────────────────────────────────────────────────────────────── */

/* Whether one modifier/format pair is usable for a texture that leaves the
 * process. Reuses the validated sz_format_supported() rather than querying the
 * driver again, so a format this driver refuses to render to cannot come back
 * here as "supported for dma-buf". */
static bool
sz_dmabuf_modifier_ok(struct sz_screen *screen, uint64_t modifier,
                      enum pipe_format format, bool *external_only)
{
   const unsigned bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW |
                         PIPE_BIND_SHARED;

   if (modifier != DRM_FORMAT_MOD_LINEAR && modifier != DRM_FORMAT_MOD_INVALID)
      return false;

   if (!sz_format_supported(screen, format, PIPE_TEXTURE_2D, 0, 0, bind))
      return false;

   if (external_only)
      *external_only = false;
   return true;
}

static void
sz_screen_query_dmabuf_modifiers(struct pipe_screen *pscreen,
                                 enum pipe_format format, int max,
                                 uint64_t *modifiers,
                                 unsigned *external_only, int *count)
{
   struct sz_screen *screen = sz_screen(pscreen);

   if (!sz_dmabuf_available(screen) || !count) {
      if (count)
         *count = 0;
      return;
   }

   if (!sz_dmabuf_modifier_ok(screen, DRM_FORMAT_MOD_LINEAR, format, NULL)) {
      *count = 0;
      return;
   }

   /* max == 0 is the "how many do you have" form of the call; the array
    * arguments are not written. */
   if (max > 0 && modifiers) {
      modifiers[0] = DRM_FORMAT_MOD_LINEAR;
      if (external_only)
         external_only[0] = 0;
   }
   *count = 1;
}

static bool
sz_screen_is_dmabuf_modifier_supported(struct pipe_screen *pscreen,
                                       uint64_t modifier, enum pipe_format format,
                                       bool *external_only)
{
   struct sz_screen *screen = sz_screen(pscreen);

   if (!sz_dmabuf_available(screen))
      return false;

   return sz_dmabuf_modifier_ok(screen, modifier, format, external_only);
}

/* ── layout queries ──────────────────────────────────────────────────────── */

/* A DRM-modifier image has no colour/depth aspect to report a layout for: its
 * subresources are planes, and Vulkan rejects anything else from
 * vkGetImageSubresourceLayout (VUID 09433). Everything we create or import uses
 * LINEAR, i.e. one plane, so plane 0 is the answer. */
static VkImageAspectFlags
sz_dmabuf_layout_aspect(const struct sz_resource *res)
{
   if (res->drm_modifier)
      return VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT;

   return sz_format_aspect(res->base.format);
}

/* The DRI frontend asks for a resource's stride, offset and modifier before it
 * asks for a handle, because a dma-buf it cannot describe is not usable. Those
 * queries arrive as `WINSYS_HANDLE_TYPE_KMS` on the older code path and as
 * PIPE_RESOURCE_PARAM_* on this one; both have to be answered, and answering
 * here keeps the KMS form out of the export path entirely.
 *
 * Vulkan reports the layout through the image rather than the memory, and only
 * for a linear or DRM-modifier image -- which is exactly what is exportable. */
static bool
sz_screen_resource_get_param(struct pipe_screen *pscreen,
                             struct pipe_context *context,
                             struct pipe_resource *pres, unsigned plane,
                             unsigned layer, unsigned level,
                             enum pipe_resource_param param,
                             unsigned handle_usage, uint64_t *value)
{
   struct sz_screen *screen = sz_screen(pscreen);
   struct sz_resource *res = sz_resource(pres);

   if (!res || !value)
      return false;

   if (param == PIPE_RESOURCE_PARAM_NPLANES) {
      *value = 1;
      return true;
   }

   /* Linear images answer this whether they were made here or imported: the
    * pitch is whatever the caller that made the dma-buf chose, and Vulkan knows
    * it either way. */
   if (res->image == VK_NULL_HANDLE || !res->linear)
      return false;

   if (param != PIPE_RESOURCE_PARAM_STRIDE && param != PIPE_RESOURCE_PARAM_OFFSET &&
       param != PIPE_RESOURCE_PARAM_MODIFIER && param != PIPE_RESOURCE_PARAM_LAYER_STRIDE)
      return false;

   const VkImageSubresource subresource = {
      .aspectMask = sz_dmabuf_layout_aspect(res),
      .mipLevel = level,
      .arrayLayer = layer,
   };
   VkSubresourceLayout layout;
   screen->vk.GetImageSubresourceLayout(screen->dev, res->image, &subresource,
                                        &layout);

   switch (param) {
   case PIPE_RESOURCE_PARAM_STRIDE:
      *value = layout.rowPitch;
      break;
   case PIPE_RESOURCE_PARAM_LAYER_STRIDE:
      *value = layout.arrayPitch;
      break;
   case PIPE_RESOURCE_PARAM_OFFSET:
      *value = layout.offset;
      break;
   case PIPE_RESOURCE_PARAM_MODIFIER:
      *value = DRM_FORMAT_MOD_LINEAR;
      break;
   default:
      return false;
   }
   return true;
}

/* ── frame dump ──────────────────────────────────────────────────────────── */

/* SZ_DEBUG=dump_frame writes the image the driver is about to hand to the
 * display to a PPM. This exists because "the window is wrong" and "the driver
 * drew the wrong thing" are the same picture on screen, and only the second one
 * is a driver bug: dumping the presented image separates them without depending
 * on the window being mapped, focused or visible at all.
 *
 * One dump happens when /tmp/sz-dump-now exists (the file is removed), and one
 * every SZ_DUMP_EVERY presented frames otherwise. Files go to SZ_DUMP_DIR,
 * which defaults to /tmp/sz-frames. */
static void
sz_dump_write_ppm(const char *path, const uint8_t *pixels, unsigned width,
                  unsigned height, unsigned row_pitch, VkFormat format)
{
   FILE *f = fopen(path, "wb");
   if (!f)
      return;

   fprintf(f, "P6\n%u %u\n255\n", width, height);
   for (unsigned y = 0; y < height; y++) {
      const uint8_t *row = pixels + (size_t)y * row_pitch;
      for (unsigned x = 0; x < width; x++) {
         const uint8_t *p = row + (size_t)x * 4;
         /* What a window is allocated in, versus what some configurations end up
          * with. Every other 32-bit format is reported as blue-green-red too, so
          * a wrong guess shows up as a colour swap rather than as nothing. */
         const bool bgr = format == VK_FORMAT_B8G8R8A8_UNORM ||
                          format == VK_FORMAT_B8G8R8A8_SRGB;
         fputc(bgr ? p[2] : p[0], f);
         fputc(p[1], f);
         fputc(bgr ? p[0] : p[2], f);
      }
   }
   fclose(f);
}

static void
sz_dump_frame(struct sz_screen *screen, struct sz_resource *res)
{
   const unsigned width = res->base.width0;
   const unsigned height = res->base.height0;
   const VkFormat format =
      res->image != VK_NULL_HANDLE ? sz_vk_format(screen, res->base.format)
                                   : VK_FORMAT_UNDEFINED;

   if (res->image == VK_NULL_HANDLE || !width || !height ||
       vk_format_get_blocksize(format) != 4) {
      mesa_logw("SZ: dump_frame: cannot dump %ux%u %s", width, height,
                util_format_name(res->base.format));
      return;
   }

   const VkImageSubresource subresource = {
      .aspectMask = sz_dmabuf_layout_aspect(res),
      .mipLevel = 0,
      .arrayLayer = 0,
   };
   VkSubresourceLayout image_layout;
   screen->vk.GetImageSubresourceLayout(screen->dev, res->image, &subresource,
                                        &image_layout);

   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = image_layout.rowPitch * height,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer buffer = VK_NULL_HANDLE;
   if (screen->vk.CreateBuffer(screen->dev, &buffer_info, NULL, &buffer) !=
       VK_SUCCESS) {
      mesa_loge("SZ: dump_frame: CreateBuffer failed");
      return;
   }

   VkMemoryRequirements req;
   screen->vk.GetBufferMemoryRequirements(screen->dev, buffer, &req);
   struct sz_bo *bo = sz_bo_alloc(screen, &req,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                  VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   if (!bo) {
      screen->vk.DestroyBuffer(screen->dev, buffer, NULL);
      mesa_loge("SZ: dump_frame: no host-visible memory");
      return;
   }
   screen->vk.BindBufferMemory(screen->dev, buffer, sz_bo_memory(bo),
                               sz_bo_offset(bo));

   /* The image is in whatever layout the frame left it in. Going to
    * TRANSFER_SRC and back keeps the resource's tracked layout truthful, which
    * is what everything after this dump relies on. */
   struct sz_batch *batch = sz_batch_begin(&screen->batch_state);
   if (!batch) {
      sz_bo_free(bo);
      screen->vk.DestroyBuffer(screen->dev, buffer, NULL);
      mesa_loge("SZ: dump_frame: no command buffer");
      return;
   }
   const VkCommandBuffer cmd = batch->cmd;

   const VkImageMemoryBarrier2 to_src = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
      .oldLayout = res->levels ? VK_IMAGE_LAYOUT_GENERAL : res->layout,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = res->image,
      .subresourceRange = {
         .aspectMask = subresource.aspectMask,
         .baseMipLevel = 0, .levelCount = 1,
         .baseArrayLayer = 0, .layerCount = 1,
      },
   };
   const VkDependencyInfo to_src_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &to_src,
   };
   screen->vk.CmdPipelineBarrier2(cmd, &to_src_dep);

   const VkBufferImageCopy region = {
      .bufferRowLength = image_layout.rowPitch / 4,
      .bufferImageHeight = height,
      .imageSubresource = {
         .aspectMask = subresource.aspectMask,
         .mipLevel = 0,
         .baseArrayLayer = 0,
         .layerCount = 1,
      },
      .imageExtent = { width, height, 1 },
   };
   screen->vk.CmdCopyImageToBuffer(cmd, res->image,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer,
                                   1, &region);

   VkImageMemoryBarrier2 back = to_src;
   back.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
   back.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
   back.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
   back.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
   back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   back.newLayout = to_src.oldLayout;
   const VkDependencyInfo back_dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &back,
   };
   screen->vk.CmdPipelineBarrier2(cmd, &back_dep);

   if (!sz_batch_end(batch) ||
       !sz_batch_submit(batch, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)) {
      sz_bo_free(bo);
      screen->vk.DestroyBuffer(screen->dev, buffer, NULL);
      mesa_loge("SZ: dump_frame: submit failed");
      return;
   }
   sz_queue_wait_idle(&screen->batch_state);

   static unsigned seq;
   const char *dir = getenv("SZ_DUMP_DIR");
   if (!dir || !*dir)
      dir = "/tmp/sz-frames";
   char path[512];
   snprintf(path, sizeof(path), "%s/frame-%05u.ppm", dir, seq++);

   const char *pixels = sz_bo_map(bo);
   if (pixels) {
      sz_dump_write_ppm(path, (const uint8_t *)pixels, width, height,
                        image_layout.rowPitch, format);
      mesa_logi("SZ: dumped %s (%ux%u %s, bind 0x%x, exportable %d, layout %u)",
                path, width, height, util_format_name(res->base.format),
                res->base.bind, res->exportable, (unsigned)res->layout);
   }

   sz_bo_free(bo);
   screen->vk.DestroyBuffer(screen->dev, buffer, NULL);
}

/* Returns true when this presented frame should be dumped. A touch of
 * /tmp/sz-dump-now asks for exactly one frame, which is what makes it possible
 * to capture "the frame where it is stuck" instead of a stream of them. */
static bool
sz_dump_frame_wanted(void)
{
   if (access("/tmp/sz-dump-now", F_OK) == 0) {
      unlink("/tmp/sz-dump-now");
      return true;
   }

   static unsigned every;
   if (!every) {
      const char *env = getenv("SZ_DUMP_EVERY");
      every = env && atoi(env) > 0 ? (unsigned)atoi(env) : 0;
   }

   static unsigned presented;
   return every && ++presented % every == 0;
}

/* Dumps the presented image when a dump was asked for. Called once per frame
 * from pipe_context::flush, which is the only place that knows a frame is
 * being handed to the display. */
void
sz_dump_frame_if_wanted(struct sz_screen *screen, struct sz_resource *res)
{
   if (!sz_debug_enabled(SZ_DEBUG_DUMP_FRAME) || !res || !sz_dump_frame_wanted())
      return;

   sz_dump_frame(screen, res);
}

/* ── export ──────────────────────────────────────────────────────────────── */

static bool
sz_screen_resource_get_handle(struct pipe_screen *pscreen,
                              struct pipe_context *pipe,
                              struct pipe_resource *pres,
                              struct winsys_handle *handle, unsigned usage)
{
   struct sz_screen *screen = sz_screen(pscreen);
   struct sz_resource *res = sz_resource(pres);

   if (!handle || !res)
      return false;

   sz_trace("get_handle res=%p %ux%u fmt=%s type=%u", (void *)pres, pres->width0,
            pres->height0, util_format_name(pres->format), handle->type);

   if (handle->type != WINSYS_HANDLE_TYPE_FD) {
      /* Flink and KMS handles name a GEM object; a Vulkan allocation has none. */
      if (!screen->dmabuf_warned) {
         screen->dmabuf_warned = true;
         mesa_logw("SZ: only dma-buf(FD) handles can be exported, not type %u",
                   handle->type);
      }
      return false;
   }

   if (!sz_dmabuf_available(screen)) {
      sz_dmabuf_unsupported_once(screen, "presentation");
      return false;
   }

   if (res->image == VK_NULL_HANDLE || !res->exportable) {
      if (!screen->dmabuf_warned) {
         screen->dmabuf_warned = true;
         mesa_logw("SZ: resource is not exportable (image=%d, exportable=%d)",
                   res->image != VK_NULL_HANDLE, res->exportable);
      }
      return false;
   }

   const VkDeviceMemory memory = sz_resource_memory(res);
   if (memory == VK_NULL_HANDLE)
      return false;

   /* Whoever receives this fd will sample the buffer with their own GPU. Until
    * fences exist, "the frame was submitted" is not "the frame was rendered", so
    * the only correct thing to hand out is a buffer nobody is writing. This is
    * the FIFO part of the phase: it costs one stall per presented frame and
    * disappears when pipe_fence_handle gets a timeline value.
    *
    * This is also the whole of the present_wait fallback. That extension would
    * let the driver ask the presentation engine when a frame is on screen
    * instead of stalling; VK_KHR_present_wait is not core in any version, so a
    * 1.3 (or a 1.4) device may not have it, and the wait below is then the
    * correct behaviour rather than a degraded one. See SZ_CAP_OPTIONAL_LIST. */
   const int64_t export_wait = sz_time_now();
   sz_queue_wait_idle(&screen->batch_state);
   sz_time_slow("queue wait (export to display)", export_wait);

   const VkImageSubresource subresource = {
      .aspectMask = sz_dmabuf_layout_aspect(res),
      .mipLevel = 0,
      .arrayLayer = handle->layer,
   };
   VkSubresourceLayout layout;
   screen->vk.GetImageSubresourceLayout(screen->dev, res->image, &subresource,
                                        &layout);

   const VkMemoryGetFdInfoKHR fd_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = memory,
      .handleType = SZ_DMABUF_HANDLE_TYPE,
   };

   int fd = -1;
   VkResult vkres = screen->vk.GetMemoryFdKHR(screen->dev, &fd_info, &fd);
   if (vkres != VK_SUCCESS || fd < 0) {
      mesa_loge("SZ: vkGetMemoryFdKHR failed (VkResult %d)", vkres);
      return false;
   }

   handle->handle = fd;
   handle->stride = layout.rowPitch;
   handle->offset = layout.offset;
   handle->modifier = DRM_FORMAT_MOD_LINEAR;
   handle->size = res->bo ? sz_bo_size(res->bo) : 0;

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: exported %ux%u %s as fd %d, stride %u, offset %" PRIu64,
                res->base.width0, res->base.height0,
                util_format_name(res->base.format), fd, handle->stride,
                (uint64_t)layout.offset);
   }

   return true;
}

/* ── import ──────────────────────────────────────────────────────────────── */

static struct pipe_resource *
sz_screen_resource_from_handle(struct pipe_screen *pscreen,
                               const struct pipe_resource *tmpl,
                               struct winsys_handle *handle, unsigned usage)
{
   struct sz_screen *screen = sz_screen(pscreen);

   if (!tmpl || !handle || handle->type != WINSYS_HANDLE_TYPE_FD)
      return NULL;

   if (!sz_dmabuf_available(screen)) {
      sz_dmabuf_unsupported_once(screen, "importing a window buffer");
      return NULL;
   }

   if (tmpl->target != PIPE_TEXTURE_2D) {
      mesa_loge("SZ: cannot import a dma-buf of target %u; only 2D is supported",
                (unsigned)tmpl->target);
      return NULL;
   }

   const uint64_t modifier = handle->modifier;
   if (modifier != DRM_FORMAT_MOD_LINEAR && modifier != DRM_FORMAT_MOD_INVALID) {
      mesa_loge("SZ: dma-buf with modifier 0x%" PRIx64 " cannot be imported; "
                "only DRM_FORMAT_MOD_LINEAR is supported", modifier);
      return NULL;
   }

   const VkFormat format = sz_vk_format(screen, tmpl->format);
   if (format == VK_FORMAT_UNDEFINED) {
      mesa_loge("SZ: no VkFormat for imported pipe format %s",
                util_format_name(tmpl->format));
      return NULL;
   }

   const bool use_modifier = screen->caps.drm_format_modifier;

   VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = { tmpl->width0, tmpl->height0, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = use_modifier ? VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT
                             : VK_IMAGE_TILING_LINEAR,
      .usage = sz_image_usage_from_bindings(screen, tmpl->bind,
                                            tmpl->format) |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };

   VkExternalMemoryImageCreateInfo ext_ci = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = SZ_DMABUF_HANDLE_TYPE,
   };

   /* Explicit plane layout is the only way to make Vulkan use the row pitch the
    * buffer already has. Without it the driver picks its own linear pitch and the
    * image comes out sheared with no error anywhere. */
   VkSubresourceLayout plane = {
      .offset = handle->offset,
      .rowPitch = handle->stride,
   };
   VkImageDrmFormatModifierExplicitCreateInfoEXT mod_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
      .drmFormatModifierPlaneCount = 1,
      .pPlaneLayouts = &plane,
   };

   if (use_modifier) {
      mod_ci.pNext = &ext_ci;
      ci.pNext = &mod_ci;
   } else {
      ci.pNext = &ext_ci;
   }

   VkImage image = VK_NULL_HANDLE;
   VkResult vkres = screen->vk.CreateImage(screen->dev, &ci, NULL, &image);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateImage for an imported %ux%u %s buffer failed "
                "(VkResult %d)", ci.extent.width, ci.extent.height,
                util_format_name(tmpl->format), vkres);
      return NULL;
   }

   VkMemoryRequirements req;
   screen->vk.GetImageMemoryRequirements(screen->dev, image, &req);

   VkMemoryFdPropertiesKHR fd_props = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR,
   };
   vkres = screen->vk.GetMemoryFdPropertiesKHR(screen->dev, SZ_DMABUF_HANDLE_TYPE,
                                               handle->handle, &fd_props);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkGetMemoryFdPropertiesKHR failed (VkResult %d)", vkres);
      screen->vk.DestroyImage(screen->dev, image, NULL);
      return NULL;
   }

   const uint32_t usable = fd_props.memoryTypeBits & req.memoryTypeBits;
   if (usable == 0) {
      mesa_loge("SZ: the dma-buf's memory types (0x%x) do not overlap what the "
                "image requires (0x%x)", fd_props.memoryTypeBits,
                req.memoryTypeBits);
      screen->vk.DestroyImage(screen->dev, image, NULL);
      return NULL;
   }

   /* Vulkan takes ownership of an imported fd, Gallium's contract says the
    * caller keeps theirs, so hand Vulkan a duplicate. */
   const int owned_fd = dup(handle->handle);
   if (owned_fd < 0) {
      mesa_loge("SZ: dup() of the dma-buf fd failed");
      screen->vk.DestroyImage(screen->dev, image, NULL);
      return NULL;
   }

   VkMemoryDedicatedAllocateInfo dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .image = image,
   };
   VkImportMemoryFdInfoKHR import = {
      .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = &dedicated,
      .handleType = SZ_DMABUF_HANDLE_TYPE,
      .fd = owned_fd,
   };
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import,
      .allocationSize = req.size,
      .memoryTypeIndex = 0,
   };

   /* Prefer device-local: the buffer is rendered into and scanned out by GPUs,
    * and host-visible imported memory makes every render target write cross the
    * bus. */
   ai.memoryTypeIndex =
      sz_device_find_memory_type(screen, usable, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (ai.memoryTypeIndex == VK_MAX_MEMORY_TYPES)
      ai.memoryTypeIndex = sz_device_find_memory_type(screen, usable, 0);
   if (ai.memoryTypeIndex == VK_MAX_MEMORY_TYPES) {
      mesa_loge("SZ: no memory type for an imported dma-buf (mask 0x%x)", usable);
      close(owned_fd);
      screen->vk.DestroyImage(screen->dev, image, NULL);
      return NULL;
   }

   VkDeviceMemory memory = VK_NULL_HANDLE;
   vkres = screen->vk.AllocateMemory(screen->dev, &ai, NULL, &memory);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkAllocateMemory for an imported dma-buf failed "
                "(VkResult %d)", vkres);
      /* vkAllocateMemory took the fd on success only. */
      close(owned_fd);
      screen->vk.DestroyImage(screen->dev, image, NULL);
      return NULL;
   }

   /* The plane offset lives in the create info, so the binding itself is at zero;
    * adding it twice would shift the image by its own offset. */
   vkres = screen->vk.BindImageMemory(screen->dev, image, memory, 0);
   if (vkres != VK_SUCCESS) {
      mesa_loge("SZ: vkBindImageMemory for an imported dma-buf failed "
                "(VkResult %d)", vkres);
      screen->vk.FreeMemory(screen->dev, memory, NULL);
      screen->vk.DestroyImage(screen->dev, image, NULL);
      return NULL;
   }

   struct pipe_resource *pres =
      sz_resource_wrap_imported(screen, tmpl, image, memory, use_modifier);
   if (!pres) {
      screen->vk.FreeMemory(screen->dev, memory, NULL);
      screen->vk.DestroyImage(screen->dev, image, NULL);
      return NULL;
   }

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: imported %ux%u %s dma-buf, stride %u, offset %u",
                tmpl->width0, tmpl->height0, util_format_name(tmpl->format),
                handle->stride, handle->offset);
   }

   return pres;
}

/* ------------------------------------------------------------------------- */
/* Self test                                                                  */
/* ------------------------------------------------------------------------- */

/* Round trips a rendered image through a dma-buf.
 *
 * Getting an fd back is only a syscall, so that is not the thing worth testing.
 * What is worth testing is that the memory behind the fd is the memory we
 * rendered into, described with a row pitch the importer ends up agreeing with:
 * a wrong pitch fails nowhere, it shears the image, and on screen that reads as
 * "the driver is slow" rather than as a bug. */
unsigned
sz_dmabuf_selftest(struct sz_screen *screen)
{
   unsigned failures = 0;
   const unsigned w = 64, h = 64;
   bool have_fd = false;

   const union pipe_color_union clear_color = {
      .f = { 0.4f, 0.5f, 0.6f, 1.0f },
   };

   struct pipe_context *pipe = NULL;
   struct pipe_resource *rt = NULL;
   struct pipe_resource *imported = NULL;
   struct winsys_handle wh = { .type = WINSYS_HANDLE_TYPE_FD };

   const struct pipe_resource tmpl = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      /* SHARED is what makes the image exportable rather than the texture being
       * a render target, so both binds are needed to test the presentation
       * case. */
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW |
              PIPE_BIND_SHARED,
      .usage = PIPE_USAGE_DEFAULT,
      .width0 = w,
      .height0 = h,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
   };

   pipe = sz_context_create(screen, 0);
   if (!pipe) {
      mesa_loge("SZ: dma-buf selftest: could not create a context");
      return 1;
   }

   rt = screen->base.resource_create(&screen->base, &tmpl);
   if (!rt) {
      mesa_loge("SZ: dma-buf selftest: could not create the exportable image");
      failures++;
      goto out;
   }

   if (!sz_resource(rt)->exportable) {
      mesa_loge("SZ: dma-buf selftest: the image was created without external "
                "memory");
      failures++;
      goto out;
   }

   /* Something has to be in the buffer, or the test cannot tell shared memory
    * from a fresh zeroed allocation. */
   const struct pipe_framebuffer_state fb = {
      .width = w,
      .height = h,
      .samples = 1,
      .nr_cbufs = 1,
      .cbufs = {
         {
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .texture = rt,
            .level = 0,
            .first_layer = 0,
            .last_layer = 0,
         },
      },
   };
   pipe->set_framebuffer_state(pipe, &fb);
   pipe->clear(pipe, PIPE_CLEAR_COLOR, 0, 0, NULL, &clear_color, 0.0, 0);
   pipe->flush(pipe, NULL, 0);
   sz_queue_wait_idle(&screen->batch_state);

   if (!screen->base.resource_get_handle(&screen->base, pipe, rt, &wh,
                                         PIPE_HANDLE_USAGE_FRAMEBUFFER_WRITE)) {
      mesa_loge("SZ: dma-buf selftest: exporting the render target failed");
      failures++;
      goto out;
   }
   have_fd = true;

   if (wh.handle == 0) {
      mesa_loge("SZ: dma-buf selftest: export returned fd 0");
      failures++;
   }

   if (wh.modifier != DRM_FORMAT_MOD_LINEAR) {
      mesa_loge("SZ: dma-buf selftest: exported modifier 0x%" PRIx64 ", "
                "expected DRM_FORMAT_MOD_LINEAR", (uint64_t)wh.modifier);
      failures++;
   }

   if (wh.stride < w * 4) {
      mesa_loge("SZ: dma-buf selftest: exported stride %u is narrower than a "
                "row of %u RGBA8 pixels", wh.stride, w);
      failures++;
   }

   if (wh.size < (uint64_t)wh.stride * h) {
      mesa_loge("SZ: dma-buf selftest: exported size %" PRIu64 " is smaller "
                "than %u rows of %u bytes", (uint64_t)wh.size, h, wh.stride);
      failures++;
   }

   /* The param path is what DRI3 uses for the modifier and the pitch, and it has
    * to agree with the handle path down to the byte. */
   uint64_t param_stride = 0, param_modifier = 0;
   if (!screen->base.resource_get_param(&screen->base, pipe, rt, 0, 0, 0,
                                        PIPE_RESOURCE_PARAM_STRIDE, 0,
                                        &param_stride) ||
       !screen->base.resource_get_param(&screen->base, pipe, rt, 0, 0, 0,
                                        PIPE_RESOURCE_PARAM_MODIFIER, 0,
                                        &param_modifier)) {
      mesa_loge("SZ: dma-buf selftest: resource_get_param refused the exportable "
                "image");
      failures++;
   } else if (param_stride != wh.stride || param_modifier != wh.modifier) {
      mesa_loge("SZ: dma-buf selftest: get_param says stride %" PRIu64
                " modifier 0x%" PRIx64 ", get_handle says %u and 0x%" PRIx64,
                param_stride, param_modifier, wh.stride, (uint64_t)wh.modifier);
      failures++;
   }

   imported = screen->base.resource_from_handle(&screen->base, &tmpl, &wh,
                                               PIPE_HANDLE_USAGE_FRAMEBUFFER_WRITE);
   if (!imported) {
      mesa_loge("SZ: dma-buf selftest: importing our own dma-buf failed");
      failures++;
      goto out;
   }

   /* This is the assertion the sheared-window bug cannot survive: Vulkan lays
    * the imported image out again from the plane layout we gave it, and the
    * pitch it reports back has to be the pitch we exported. */
   uint64_t imported_stride = 0;
   if (!screen->base.resource_get_param(&screen->base, NULL, imported, 0, 0, 0,
                                        PIPE_RESOURCE_PARAM_STRIDE, 0,
                                        &imported_stride)) {
      mesa_loge("SZ: dma-buf selftest: the imported image has no stride");
      failures++;
   } else if (imported_stride != wh.stride) {
      mesa_loge("SZ: dma-buf selftest: the imported image has stride %" PRIu64
                ", but the dma-buf has stride %u", imported_stride, wh.stride);
      failures++;
   }

   /* And it has to be usable as a framebuffer, which is the only thing the
    * swapchain will ever ask of it. */
   const struct pipe_framebuffer_state imported_fb = {
      .width = w,
      .height = h,
      .samples = 1,
      .nr_cbufs = 1,
      .cbufs = {
         {
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .texture = imported,
            .level = 0,
            .first_layer = 0,
            .last_layer = 0,
         },
      },
   };
   pipe->set_framebuffer_state(pipe, &imported_fb);
   pipe->clear(pipe, PIPE_CLEAR_COLOR, 0, 0, NULL, &clear_color, 0.0, 0);
   pipe->flush(pipe, NULL, 0);
   sz_queue_wait_idle(&screen->batch_state);

   if (failures == 0) {
      mesa_logi("SZ: dma-buf selftest: %ux%u export/import round trip ok "
                "(stride %u)", w, h, wh.stride);
   }

out:
   if (have_fd)
      close(wh.handle);
   pipe_resource_reference(&imported, NULL);
   pipe_resource_reference(&rt, NULL);
   if (pipe)
      pipe->destroy(pipe);
   return failures;
}

void
sz_dmabuf_screen_init(struct sz_screen *screen)
{
   screen->base.resource_get_handle = sz_screen_resource_get_handle;
   screen->base.resource_from_handle = sz_screen_resource_from_handle;
   screen->base.resource_get_param = sz_screen_resource_get_param;
   screen->base.query_dmabuf_modifiers = sz_screen_query_dmabuf_modifiers;
   screen->base.is_dmabuf_modifier_supported =
      sz_screen_is_dmabuf_modifier_supported;
}
