/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef SZ_TYPES_H
#define SZ_TYPES_H

#include <vulkan/vulkan_core.h>

#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"

#include "util/simple_mtx.h"
#include "util/u_dl.h"

#include "sz_caps.h"
#include "sz_debug.h"
#include "sz_public.h"

/* Instance/device entry points resolved through the loader. Kept on the screen
 * so every later module shares one resolution pass. */
struct sz_vk_entrypoints {
   PFN_vkCreateInstance CreateInstance;
   PFN_vkEnumerateInstanceVersion EnumerateInstanceVersion;
   PFN_vkDestroyInstance DestroyInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
   PFN_vkGetPhysicalDeviceImageFormatProperties2
      GetPhysicalDeviceImageFormatProperties2;
   PFN_vkEnumerateDeviceExtensionProperties EnumerateDeviceExtensionProperties;
   PFN_vkGetPhysicalDeviceFeatures2 GetPhysicalDeviceFeatures2;
   PFN_vkGetPhysicalDeviceProperties2 GetPhysicalDeviceProperties2;
   PFN_vkGetPhysicalDeviceQueueFamilyProperties GetPhysicalDeviceQueueFamilyProperties;
   PFN_vkCreateDevice CreateDevice;

   /* Device-level entry points, resolved through vkGetDeviceProcAddr once the
    * VkDevice exists. */
   PFN_vkDestroyDevice DestroyDevice;
   PFN_vkDeviceWaitIdle DeviceWaitIdle;
   PFN_vkGetDeviceQueue2 GetDeviceQueue2;

   /* Memory and resources (sz_bo / sz_resource). */
   PFN_vkAllocateMemory AllocateMemory;
   PFN_vkFreeMemory FreeMemory;
   PFN_vkMapMemory MapMemory;
   PFN_vkUnmapMemory UnmapMemory;
   PFN_vkFlushMappedMemoryRanges FlushMappedMemoryRanges;
   PFN_vkInvalidateMappedMemoryRanges InvalidateMappedMemoryRanges;
   PFN_vkCreateBuffer CreateBuffer;
   PFN_vkDestroyBuffer DestroyBuffer;
   PFN_vkGetBufferMemoryRequirements GetBufferMemoryRequirements;
   PFN_vkGetBufferMemoryRequirements2 GetBufferMemoryRequirements2;
   PFN_vkBindBufferMemory BindBufferMemory;
   PFN_vkCreateImage CreateImage;
   PFN_vkDestroyImage DestroyImage;
   PFN_vkGetImageMemoryRequirements GetImageMemoryRequirements;
   PFN_vkGetImageMemoryRequirements2 GetImageMemoryRequirements2;
   PFN_vkBindImageMemory BindImageMemory;
};

/* Declared before struct sz_screen because sz_screen embeds them by value. */
enum sz_queue_role {
   SZ_QUEUE_GRAPHICS,
   SZ_QUEUE_COMPUTE,
   SZ_QUEUE_TRANSFER,
   SZ_QUEUE_COUNT,
};

struct sz_queue {
   VkQueue queue;
   uint32_t family;
   uint32_t index;
};

/* Forward declarations for the per-screen subsystems, so sz_types.h stays the
 * single place that defines the screen without pulling in every module header. */
struct sz_bo_state;
struct sz_bo;
struct sz_resource;

struct sz_screen {
   struct pipe_screen base;
   struct pipe_screen *wrapped; /* debug_screen_wrap() wrapper, freed on destroy */

   struct sw_winsys *winsys;

   /* Vulkan loader */
   void *loader_lib;
   PFN_vkGetInstanceProcAddr vk_GetInstanceProcAddr;
   PFN_vkGetDeviceProcAddr vk_GetDeviceProcAddr;

   struct sz_vk_entrypoints vk;

   VkInstance instance;
   uint32_t instance_api_version;
   VkDevice dev;
   VkPhysicalDevice pdev;
   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceMemoryProperties mem_props;
   uint32_t api_version;

   /* Built once here and treated as read-only afterwards (ADR-001). */
   struct sz_caps caps;

   struct sz_queue queues[SZ_QUEUE_COUNT];

   /* Device memory suballocator (sz_bo). */
   struct sz_bo_state *bo;

   int drm_fd;

   /* Built once during screen creation: get_name() must be stable and
    * thread-safe, so it cannot format into a function-local static buffer. */
   char renderer_string[384];

   bool is_cpu;
   bool driver_name_is_inferred;
};

static inline struct sz_screen *
sz_screen(struct pipe_screen *screen)
{
   return (struct sz_screen *)screen;
}

#endif /* SZ_TYPES_H */
