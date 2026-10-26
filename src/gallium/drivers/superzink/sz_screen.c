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

#include "sz_format.h"
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

/* Vulkan 1.4 is a hard requirement (ADR-001). There is no fallback path. */
#define SZ_MIN_API_VERSION VK_API_VERSION_1_4

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
   return true;
}

#undef SZ_LOAD_INSTANCE_PROC
#undef SZ_LOAD_GLOBAL_PROC
#undef SZ_LOAD_PROC_IMPL

static VkResult
sz_create_instance(struct sz_screen *screen)
{
   const char *exts[] = { SZ_INSTANCE_EXTENSIONS };

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "Super-ZINK",
      .applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
      .pEngineName = "Super-ZINK",
      .engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0),
      .apiVersion = SZ_MIN_API_VERSION,
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

   /* ADR-001 is a filter, not a post-check: a device that cannot meet the floor
    * must not win on deviceType and then sink the whole screen. Rank only among
    * devices that clear 1.4. */
   int best_rank = -1;
   VkPhysicalDevice best = VK_NULL_HANDLE;
   const char *rejected_name = NULL;
   uint32_t rejected_version = 0;

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
         rejected_name = props.deviceName;
         rejected_version = props.apiVersion;
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
      }
   }

   FREE(devs);

   if (best == VK_NULL_HANDLE) {
      if (rejected_name) {
         mesa_loge("SZ: no usable Vulkan device. \"%s\" reports Vulkan %u.%u but "
                   "Super-ZINK requires Vulkan %u.%u or newer, and no other "
                   "device qualifies. There is no fallback path by design.",
                   rejected_name,
                   VK_API_VERSION_MAJOR(rejected_version),
                   VK_API_VERSION_MINOR(rejected_version),
                   VK_API_VERSION_MAJOR(SZ_MIN_API_VERSION),
                   VK_API_VERSION_MINOR(SZ_MIN_API_VERSION));
      } else {
         mesa_loge("SZ: no usable Vulkan device found.");
      }
      return false;
   }

   if (screen->props.apiVersion < SZ_MIN_API_VERSION) {
      mesa_loge("SZ: invalid apiVersion 0x%x", screen->props.apiVersion);
      return false;
   }

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
   mesa_loge("SZ: context creation is not implemented yet (Phase 1). "
             "The screen enumerates devices only.");
   return NULL;
}

static void
sz_screen_destroy(struct pipe_screen *pscreen)
{
   struct sz_screen *screen = sz_screen(pscreen);

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

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: screen created: %s", screen->renderer_string);
      mesa_logi("SZ: deviceType=%d vendorID=0x%04x deviceID=0x%04x",
                (int)screen->props.deviceType,
                screen->props.vendorID, screen->props.deviceID);
      mesa_logi("SZ: memory heaps=%u", screen->mem_props.memoryHeapCount);
      for (uint32_t i = 0; i < screen->mem_props.memoryHeapCount; i++) {
         mesa_logi("SZ:   heap[%u] size=%" PRIu64 " MiB flags=0x%x", i,
                   screen->mem_props.memoryHeaps[i].size / (1024 * 1024),
                   screen->mem_props.memoryHeaps[i].flags);
      }
   }

   return &screen->base;

fail:
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
