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

#include "sz_debug.h"
#include "sz_public.h"

/* Instance/device entry points resolved through the loader. Kept on the screen
 * so every later module shares one resolution pass. */
struct sz_vk_entrypoints {
   PFN_vkCreateInstance CreateInstance;
   PFN_vkDestroyInstance DestroyInstance;
   PFN_vkEnumeratePhysicalDevices EnumeratePhysicalDevices;
   PFN_vkGetPhysicalDeviceProperties GetPhysicalDeviceProperties;
   PFN_vkGetPhysicalDeviceMemoryProperties GetPhysicalDeviceMemoryProperties;
   PFN_vkGetPhysicalDeviceFormatProperties GetPhysicalDeviceFormatProperties;
   PFN_vkGetPhysicalDeviceImageFormatProperties2
      GetPhysicalDeviceImageFormatProperties2;
};

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
   VkPhysicalDevice pdev;
   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceMemoryProperties mem_props;
   uint32_t api_version;

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
