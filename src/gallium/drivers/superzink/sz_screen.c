/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Phase 0: Vulkan bootstrap and capability enumeration.
 *
 * Scope of this file at Phase 0:
 *   - load the Vulkan loader
 *   - create a VkInstance
 *   - pick a physical device and *hard require* VkPhysicalDeviceProperties
 *     ::apiVersion >= 1.4
 *   - publish a pipe_screen whose context_create() is not implemented yet
 *
 * Anything that needs a working VkDevice (queues, memory, descriptors) is
 * Phase 1 work.
 */

#include "sz_caps.h"
#include "sz_device.h"
#include "sz_format.h"
#include "sz_resource.h"
#include "sz_types.h"
#include "sz_debug.h"

#include "util/u_debug.h"
#include "util/u_dl.h"
#include "util/u_memory.h"
#include "util/u_screen.h"
#include "util/os_file.h"

#include "util/log.h"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#if DETECT_OS_WINDOWS
#define SZ_VK_LIBNAME "vulkan-1.dll"
#elif DETECT_OS_APPLE
#define SZ_VK_LIBNAME "@rpath/libvulkan.1.dylib"
#elif DETECT_OS_ANDROID
#define SZ_VK_LIBNAME "libvulkan.so"
#else
#define SZ_VK_LIBNAME "libvulkan.so.1"
#endif

/* ADR-001: 1.3 is the floor; the real gate is the required capability list in
 * sz_caps.h. A 1.4 device is not waved through -- 1.4 promotes only 16
 * extensions and gates none of the ones we depend on most. */
#define SZ_MIN_API_VERSION SZ_CAPS_MIN_API_VERSION

#define SZ_INSTANCE_EXTENSIONS \
   VK_EXT_DEBUG_UTILS_EXTENSION_NAME

/* ── Vulkan loader bootstrap ─────────────────────────────────────────────── */

static bool
sz_load_loader(struct sz_screen *screen)
{
   screen->loader_lib = util_dl_open(SZ_VK_LIBNAME);
   if (!screen->loader_lib) {
      mesa_loge("SZ: failed to load " SZ_VK_LIBNAME);
      return false;
   }

   screen->vk_GetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)util_dl_get_proc_address(screen->loader_lib,
                                                          "vkGetInstanceProcAddr");
   screen->vk_GetDeviceProcAddr =
      (PFN_vkGetDeviceProcAddr)util_dl_get_proc_address(screen->loader_lib,
                                                        "vkGetDeviceProcAddr");
   if (!screen->vk_GetInstanceProcAddr || !screen->vk_GetDeviceProcAddr) {
      mesa_loge("SZ: " SZ_VK_LIBNAME " is missing vkGetInstanceProcAddr/vkGetDeviceProcAddr");
      return false;
   }

   return true;
}

#define SZ_LOAD_PROC_IMPL(e, field, proc, inst)                                 \
   do {                                                                         \
      (e)->field = (PFN_##proc)screen->vk_GetInstanceProcAddr((inst), #proc);   \
      if (!(e)->field) {                                                        \
         mesa_loge("SZ: vkGetInstanceProcAddr(\"%s\") failed", #proc);          \
         return false;                                                          \
      }                                                                         \
   } while (0)

/* Resolved with a NULL instance: only truly global entry points. */
#define SZ_LOAD_GLOBAL_PROC(e, field, proc) \
   SZ_LOAD_PROC_IMPL(e, field, proc, VK_NULL_HANDLE)

/* Resolved against a live instance. */
#define SZ_LOAD_INSTANCE_PROC(e, field, proc) \
   SZ_LOAD_PROC_IMPL(e, field, proc, screen->instance)

static bool
sz_load_global_entrypoints(struct sz_screen *screen)
{
   SZ_LOAD_GLOBAL_PROC(&screen->vk, CreateInstance, vkCreateInstance);
   /* Present since 1.1; absent only on a 1.0 loader, which we reject anyway. */
   screen->vk.EnumerateInstanceVersion =
      (PFN_vkEnumerateInstanceVersion)
         screen->vk_GetInstanceProcAddr(VK_NULL_HANDLE,
                                        "vkEnumerateInstanceVersion");
   return true;
}

static bool
sz_load_instance_entrypoints(struct sz_screen *screen)
{
   SZ_LOAD_INSTANCE_PROC(&screen->vk, DestroyInstance, vkDestroyInstance);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, EnumeratePhysicalDevices, vkEnumeratePhysicalDevices);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, GetPhysicalDeviceProperties,
                         vkGetPhysicalDeviceProperties);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, GetPhysicalDeviceMemoryProperties,
                         vkGetPhysicalDeviceMemoryProperties);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, GetPhysicalDeviceFormatProperties,
                         vkGetPhysicalDeviceFormatProperties);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, GetPhysicalDeviceImageFormatProperties2,
                         vkGetPhysicalDeviceImageFormatProperties2);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, EnumerateDeviceExtensionProperties,
                         vkEnumerateDeviceExtensionProperties);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, GetPhysicalDeviceFeatures2,
                         vkGetPhysicalDeviceFeatures2);
   SZ_LOAD_INSTANCE_PROC(&screen->vk, GetPhysicalDeviceProperties2,
                         vkGetPhysicalDeviceProperties2);
   /* Available since 1.0, but only reachable through the instance. */
   SZ_LOAD_INSTANCE_PROC(&screen->vk, CreateDevice, vkCreateDevice);
   return true;
}

#undef SZ_LOAD_INSTANCE_PROC
#undef SZ_LOAD_GLOBAL_PROC
#undef SZ_LOAD_PROC_IMPL

static VkResult
sz_create_instance(struct sz_screen *screen)
{
   const char *exts[] = { SZ_INSTANCE_EXTENSIONS };

   /* Ask for the highest version the loader offers, capped by our headers.
    * Requesting only the floor would pin every 1.4 driver to the extension
    * path, because a device never reports an apiVersion above its instance. */
   uint32_t api = VK_API_VERSION_1_0;
   if (screen->vk.EnumerateInstanceVersion) {
      VkResult res = screen->vk.EnumerateInstanceVersion(&api);
      if (res != VK_SUCCESS)
         api = VK_API_VERSION_1_0;
   }
   api = MIN2(api, VK_HEADER_VERSION_COMPLETE);

   if (api < SZ_MIN_API_VERSION) {
      mesa_loge("SZ: the Vulkan loader supports %u.%u, but Super-ZINK needs "
                "%u.%u or newer.",
                VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api),
                VK_API_VERSION_MAJOR(SZ_MIN_API_VERSION),
                VK_API_VERSION_MINOR(SZ_MIN_API_VERSION));
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }

   screen->instance_api_version = api;

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "Super-ZINK",
      .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
      .pEngineName = "Super-ZINK",
      .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
      .apiVersion = api,
   };

   VkInstanceCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
      .enabledExtensionCount = ARRAY_SIZE(exts),
      .ppEnabledExtensionNames = exts,
   };

   return screen->vk.CreateInstance(&ci, NULL, &screen->instance);
}

/* ── Physical device selection ───────────────────────────────────────────── */

static bool
sz_select_physical_device(struct sz_screen *screen)
{
   uint32_t count = 0;
   VkResult res = screen->vk.EnumeratePhysicalDevices(screen->instance, &count, NULL);
   if (res != VK_SUCCESS || count == 0) {
      mesa_loge("SZ: no Vulkan physical devices (VkResult %d)", res);
      return false;
   }

   VkPhysicalDevice *devs = CALLOC(count, sizeof(*devs));
   if (!devs)
      return false;

   res = screen->vk.EnumeratePhysicalDevices(screen->instance, &count, devs);
   if (res != VK_SUCCESS) {
      mesa_loge("SZ: vkEnumeratePhysicalDevices failed (VkResult %d)", res);
      FREE(devs);
      return false;
   }

   /* ADR-001 is a filter, not a post-check: a device that fails it must not win
    * on deviceType and then sink the whole screen. Ranking therefore only
    * considers devices that pass the version floor *and* the required
    * capability list. Probing costs one features query plus one extension
    * enumeration per device, which is negligible next to ~100 format queries
    * the screen does later. */
   int best_rank = -1;
   VkPhysicalDevice best = VK_NULL_HANDLE;
   struct sz_caps best_caps = { 0 };

   char reject_why[512] = "";
   const char *reject_name = NULL;

   for (uint32_t i = 0; i < count; i++) {
      VkPhysicalDeviceProperties props;
      screen->vk.GetPhysicalDeviceProperties(devs[i], &props);

      if (sz_debug_enabled(SZ_DEBUG_INFO)) {
         mesa_logi("SZ: device[%u] \"%s\" type=%d apiVersion=%u.%u.%u driver=0x%x",
                   i, props.deviceName, (int)props.deviceType,
                   VK_API_VERSION_MAJOR(props.apiVersion),
                   VK_API_VERSION_MINOR(props.apiVersion),
                   VK_API_VERSION_PATCH(props.apiVersion),
                   props.driverVersion);
      }

      if (props.apiVersion < SZ_MIN_API_VERSION) {
         if (!reject_name) reject_name = props.deviceName;
         snprintf(reject_why, sizeof(reject_why),
                  "reports Vulkan %u.%u, below the %u.%u floor",
                  VK_API_VERSION_MAJOR(props.apiVersion),
                  VK_API_VERSION_MINOR(props.apiVersion),
                  VK_API_VERSION_MAJOR(SZ_MIN_API_VERSION),
                  VK_API_VERSION_MINOR(SZ_MIN_API_VERSION));
         continue;
      }

      struct sz_caps caps;
      sz_caps_init(&caps, devs[i], screen->instance_api_version,
                   screen->vk.EnumerateDeviceExtensionProperties,
                   screen->vk.GetPhysicalDeviceFeatures2,
                   screen->vk.GetPhysicalDeviceProperties2);

      const unsigned missing = sz_caps_missing_required_count(&caps);
      if (missing > 0) {
         if (!reject_name) reject_name = props.deviceName;
         char list[384];
         sz_caps_missing_required_str(&caps, list, sizeof(list));
         snprintf(reject_why, sizeof(reject_why),
                  "is missing %u required capabilit%s: %s",
                  missing, missing == 1 ? "y" : "ies", list);
         continue;
      }

      int rank;
      switch (props.deviceType) {
      case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   rank = 0; break;
      case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: rank = 1; break;
      case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    rank = 2; break;
      default:                                     rank = 3; break;
      }

      if (rank < best_rank || best_rank < 0) {
         best_rank = rank;
         best = devs[i];
         screen->props = props;
         best_caps = caps;
      }
   }

   FREE(devs);

   if (best == VK_NULL_HANDLE) {
      if (reject_name) {
         mesa_loge("SZ: no usable Vulkan device. \"%s\" %s. "
                   "No other device qualified, and there is no fallback path "
                   "by design. Run with SZ_DEBUG=info to see the full "
                   "capability table.",
                   reject_name, reject_why);
      } else {
         mesa_loge("SZ: no usable Vulkan device found.");
      }
      return false;
   }

   screen->caps = best_caps;
   screen->pdev = best;
   screen->api_version = screen->props.apiVersion;
   screen->is_cpu = screen->props.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;

   screen->vk.GetPhysicalDeviceMemoryProperties(best, &screen->mem_props);

   return true;
}

/* ── pipe_screen callbacks ───────────────────────────────────────────────── */

static const char *
sz_screen_get_name(struct pipe_screen *pscreen)
{
   return sz_screen(pscreen)->renderer_string;
}

static const char *
sz_screen_get_vendor(struct pipe_screen *pscreen)
{
   return "Super-ZINK";
}

static const char *
sz_screen_get_device_vendor(struct pipe_screen *pscreen)
{
   struct sz_screen *screen = sz_screen(pscreen);
   return screen->props.vendorID == 0x1002 ? "AMD" :
          screen->props.vendorID == 0x8086 ? "Intel" :
          screen->props.vendorID == 0x10de ? "NVIDIA" :
          screen->props.vendorID == 0x5143 ? "Qualcomm" :
          screen->props.vendorID == 0x13b5 ? "ARM" :
                                             "Unknown";
}

static bool
sz_screen_is_format_supported(struct pipe_screen *pscreen,
                              enum pipe_format format,
                              enum pipe_texture_target target,
                              unsigned sample_count,
                              unsigned storage_sample_count,
                              unsigned bindings)
{
   return sz_format_supported(sz_screen(pscreen), format, target, sample_count,
                              storage_sample_count, bindings);
}

static void
sz_screen_query_memory_info(struct pipe_screen *pscreen,
                            struct pipe_memory_info *info)
{
   struct sz_screen *screen = sz_screen(pscreen);

   memset(info, 0, sizeof(*info));

   for (uint32_t i = 0; i < screen->mem_props.memoryHeapCount; i++) {
      if (screen->mem_props.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
         info->total_device_memory += screen->mem_props.memoryHeaps[i].size;
      else
         info->total_staging_memory += screen->mem_props.memoryHeaps[i].size;
   }
}

static struct pipe_context *
sz_screen_context_create(struct pipe_screen *pscreen, void *priv, unsigned flags)
{
   mesa_loge("SZ: context creation is not implemented yet (Phase 1: the "
             "VkDevice exists, the pipe_context does not).");
   return NULL;
}

static void
sz_screen_destroy(struct pipe_screen *pscreen)
{
   struct sz_screen *screen = sz_screen(pscreen);

   sz_bo_finish(screen);
   sz_device_finish(screen);

   /* The instance is destroyed via the entrypoint resolved while it was alive. */
   if (screen->instance != VK_NULL_HANDLE && screen->vk_GetInstanceProcAddr) {
      PFN_vkDestroyInstance DestroyInstance =
         (PFN_vkDestroyInstance)screen->vk_GetInstanceProcAddr(screen->instance,
                                                              "vkDestroyInstance");
      if (DestroyInstance)
         DestroyInstance(screen->instance, NULL);
   }

   if (screen->loader_lib)
      util_dl_close(screen->loader_lib);

   if (screen->drm_fd >= 0)
      close(screen->drm_fd);

   ralloc_free(screen);
}

/* ── Screen creation ─────────────────────────────────────────────────────── */

static struct pipe_screen *
sz_screen_create(struct sw_winsys *winsys, int fd,
                 const struct pipe_screen_config *config)
{
   struct sz_screen *screen = rzalloc(NULL, struct sz_screen);
   if (!screen)
      return NULL;

   sz_debug_init();

   screen->drm_fd = fd >= 0 ? os_dupfd_cloexec(fd) : -1;
   screen->winsys = winsys;
   screen->driver_name_is_inferred = config && config->driver_name_is_inferred;
   screen->instance = VK_NULL_HANDLE;
   screen->dev = VK_NULL_HANDLE;
   screen->api_version = 0;

   if (!sz_load_loader(screen))
      goto fail;

   if (!sz_load_global_entrypoints(screen))
      goto fail;

   VkResult res = sz_create_instance(screen);
   if (res != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateInstance failed (VkResult %d). "
                "A Vulkan 1.4 capable driver is required.", res);
      goto fail;
   }

   if (!sz_load_instance_entrypoints(screen))
      goto fail;

   if (!sz_select_physical_device(screen))
      goto fail;

   if (!sz_device_init(screen))
      goto fail;

   sz_bo_init(screen);
   sz_resource_screen_init(screen);

   /* pipe_screen base */
   screen->base.destroy = sz_screen_destroy;
   screen->base.get_name = sz_screen_get_name;
   screen->base.get_vendor = sz_screen_get_vendor;
   screen->base.get_device_vendor = sz_screen_get_device_vendor;
   screen->base.is_format_supported = sz_screen_is_format_supported;
   screen->base.query_memory_info = sz_screen_query_memory_info;
   screen->base.context_create = sz_screen_context_create;

   u_init_pipe_screen_caps(&screen->base, !screen->is_cpu);

   /* Capabilities are deliberately minimal at Phase 0: the screen exists so
    * the loader can see a device, but no context can be made yet. */
   {
      struct pipe_caps *caps = (struct pipe_caps *)&screen->base.caps;
      caps->glsl_feature_level = 120;
      caps->glsl_feature_level_compatibility = 120;
      caps->max_texture_2d_size = 16384;
      caps->max_texture_cube_levels = 15;
      caps->max_texture_3d_levels = 12;
      caps->max_texture_array_layers = 2048;
      caps->max_render_targets = 8;
      caps->max_dual_source_render_targets = 1;
   }

   snprintf(screen->renderer_string, sizeof(screen->renderer_string),
            "Super-ZINK (Vulkan %u.%u.%u, %s)",
            VK_API_VERSION_MAJOR(screen->api_version),
            VK_API_VERSION_MINOR(screen->api_version),
            VK_API_VERSION_PATCH(screen->api_version),
            screen->props.deviceName);

   sz_caps_dump(&screen->caps);

   if (sz_debug_enabled(SZ_DEBUG_SELFTEST)) {
      if (sz_resource_selftest(screen) != 0) {
         mesa_loge("SZ: SZ_DEBUG=selftest failed; refusing to publish a screen "
                   "whose resource layer does not round-trip");
         goto fail;
      }
   }

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: screen created: %s", screen->renderer_string);
      mesa_logi("SZ: deviceType=%d vendorID=0x%04x deviceID=0x%04x",
                (int)screen->props.deviceType,
                screen->props.vendorID, screen->props.deviceID);
      mesa_logi("SZ: queues: graphics=family %u, compute=family %u, transfer=family %u",
                screen->queues[SZ_QUEUE_GRAPHICS].family,
                screen->queues[SZ_QUEUE_COMPUTE].family,
                screen->queues[SZ_QUEUE_TRANSFER].family);
      mesa_logi("SZ: memory types=%u heaps=%u",
                screen->mem_props.memoryTypeCount,
                screen->mem_props.memoryHeapCount);
      for (uint32_t i = 0; i < screen->mem_props.memoryHeapCount; i++) {
         mesa_logi("SZ:   heap[%u] size=%" PRIu64 " MiB flags=0x%x", i,
                   screen->mem_props.memoryHeaps[i].size / (1024 * 1024),
                   screen->mem_props.memoryHeaps[i].flags);
      }
   }

   return &screen->base;

fail:
   sz_bo_finish(screen);
   sz_device_finish(screen);
   if (screen->instance != VK_NULL_HANDLE && screen->vk_GetInstanceProcAddr) {
      PFN_vkDestroyInstance DestroyInstance =
         (PFN_vkDestroyInstance)screen->vk_GetInstanceProcAddr(screen->instance,
                                                              "vkDestroyInstance");
      if (DestroyInstance)
         DestroyInstance(screen->instance, NULL);
   }
   if (screen->loader_lib)
      util_dl_close(screen->loader_lib);
   if (screen->drm_fd >= 0)
      close(screen->drm_fd);
   ralloc_free(screen);
   return NULL;
}

/* ── Public entry points ─────────────────────────────────────────────────── */

struct pipe_screen *
superzink_create_screen(struct sw_winsys *winsys,
                        const struct pipe_screen_config *config)
{
   return sz_screen_create(winsys, -1, config);
}

#ifndef _WIN32
struct pipe_screen *
superzink_drm_create_screen(int fd,
                            const struct pipe_screen_config *config,
                            struct renderonly *ro)
{
   if (ro) {
      mesa_loge("SZ: renderonly GPUs are not supported yet");
      return NULL;
   }
   return sz_screen_create(NULL, fd, config);
}
#endif
