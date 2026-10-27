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

#include "sz_batch.h"
#include "sz_caps.h"
#include "sz_debug.h"
#include "sz_public.h"

/* Instance/device entry points resolved through the loader. Kept on the screen
 * so every later module shares one resolution pass. */
struct sz_vk_entrypoints {
   PFN_vkCreateInstance CreateInstance;
   PFN_vkEnumerateInstanceVersion EnumerateInstanceVersion;
   PFN_vkEnumerateInstanceExtensionProperties EnumerateInstanceExtensionProperties;
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
   PFN_vkCreateSampler CreateSampler;
   PFN_vkDestroySampler DestroySampler;
   PFN_vkCreateImageView CreateImageView;
   PFN_vkDestroyImageView DestroyImageView;
   PFN_vkGetImageSubresourceLayout GetImageSubresourceLayout;

   /* External memory (sz_dmabuf). All three are extension entry points, so they
    * stay NULL until the extensions are enabled and are only called after
    * sz_caps says they are there. */
   PFN_vkGetMemoryFdKHR GetMemoryFdKHR;
   PFN_vkGetMemoryFdPropertiesKHR GetMemoryFdPropertiesKHR;
   PFN_vkGetImageDrmFormatModifierPropertiesEXT GetImageDrmFormatModifierPropertiesEXT;

   /* Descriptors (sz_descriptor). ADR-008: push descriptors only, so this is
    * the whole descriptor surface: a layout to push against and the push. */
   PFN_vkCreateDescriptorSetLayout CreateDescriptorSetLayout;
   PFN_vkDestroyDescriptorSetLayout DestroyDescriptorSetLayout;
   PFN_vkCmdPushDescriptorSetKHR CmdPushDescriptorSetKHR;

   /* Command submission (sz_batch / sz_sync). */
   PFN_vkCreateCommandPool CreateCommandPool;
   PFN_vkDestroyCommandPool DestroyCommandPool;
   PFN_vkAllocateCommandBuffers AllocateCommandBuffers;
   PFN_vkFreeCommandBuffers FreeCommandBuffers;
   PFN_vkBeginCommandBuffer BeginCommandBuffer;
   PFN_vkEndCommandBuffer EndCommandBuffer;
   PFN_vkResetCommandBuffer ResetCommandBuffer;
   PFN_vkQueueSubmit2 QueueSubmit2;
   PFN_vkQueueWaitIdle QueueWaitIdle;
   PFN_vkCreateSemaphore CreateSemaphore;
   PFN_vkDestroySemaphore DestroySemaphore;
   PFN_vkWaitSemaphores WaitSemaphores;
   PFN_vkGetSemaphoreCounterValue GetSemaphoreCounterValue;
   PFN_vkCmdPipelineBarrier2 CmdPipelineBarrier2;
   PFN_vkCmdClearColorImage CmdClearColorImage;
   PFN_vkCmdClearDepthStencilImage CmdClearDepthStencilImage;
   PFN_vkCmdCopyImageToBuffer CmdCopyImageToBuffer;
   PFN_vkCmdCopyBufferToImage CmdCopyBufferToImage;
   PFN_vkCmdCopyImage CmdCopyImage;
   PFN_vkCmdCopyImage2 CmdCopyImage2;
   PFN_vkCmdBlitImage2 CmdBlitImage2;
   PFN_vkCmdResolveImage2 CmdResolveImage2;

   /* Queries (sz_query). */
   PFN_vkCreateQueryPool CreateQueryPool;
   PFN_vkDestroyQueryPool DestroyQueryPool;
   PFN_vkResetQueryPool ResetQueryPool;
   PFN_vkCmdBeginQuery CmdBeginQuery;
   PFN_vkCmdEndQuery CmdEndQuery;
   PFN_vkGetQueryPoolResults GetQueryPoolResults;
   PFN_vkCmdCopyBuffer CmdCopyBuffer;
   PFN_vkCmdFillBuffer CmdFillBuffer;
   PFN_vkCmdUpdateBuffer CmdUpdateBuffer;
   PFN_vkCmdBeginRendering CmdBeginRendering;
   PFN_vkCmdEndRendering CmdEndRendering;
   PFN_vkCmdSetViewport CmdSetViewport;
   PFN_vkCmdSetScissor CmdSetScissor;
   PFN_vkCmdSetBlendConstants CmdSetBlendConstants;
   PFN_vkCmdSetStencilReference CmdSetStencilReference;
   PFN_vkCmdBindPipeline CmdBindPipeline;
   PFN_vkCmdBindVertexBuffers CmdBindVertexBuffers;
   PFN_vkCmdBindIndexBuffer CmdBindIndexBuffer;
   PFN_vkCmdDraw CmdDraw;
   PFN_vkCmdDrawIndexed CmdDrawIndexed;
   PFN_vkCmdClearAttachments CmdClearAttachments;
   /* VK_EXT_multi_draw. NULL when the extension is not enabled, which is the
    * only thing the draw stream checks before falling back to a loop of
    * vkCmdDraw/vkCmdDrawIndexed. */
   PFN_vkCmdDrawMultiEXT CmdDrawMultiEXT;
   PFN_vkCmdDrawMultiIndexedEXT CmdDrawMultiIndexedEXT;

   /* Pipelines and shaders (sz_pipeline). */
   PFN_vkCreateShaderModule CreateShaderModule;
   PFN_vkDestroyShaderModule DestroyShaderModule;
   PFN_vkCreatePipelineLayout CreatePipelineLayout;
   PFN_vkDestroyPipelineLayout DestroyPipelineLayout;
   PFN_vkCreateGraphicsPipelines CreateGraphicsPipelines;
   PFN_vkDestroyPipeline DestroyPipeline;
   PFN_vkCreatePipelineCache CreatePipelineCache;
   PFN_vkDestroyPipelineCache DestroyPipelineCache;
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

/* How many destroyed resources may wait for a submission to finish before a
 * destroy has to stall the queue to make room. Destroying a resource is common
 * (the constant-buffer uploader churns one per frame), so this is not a rare
 * path. */
#define SZ_MAX_RETIRED_RESOURCES 64

/* A pipe_resource the frontend has released while a submission that may still
 * read its VkBuffer/VkImage is in flight. `value` is the timeline value that
 * submission signals; once completed_value reaches it the resource is free to
 * be torn down. */
struct sz_retired_resource {
   struct pipe_resource *res;
   uint64_t value;
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
   uint32_t instance_api_version;
   VkDevice dev;
   VkPhysicalDevice pdev;
   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceMemoryProperties mem_props;
   uint32_t api_version;

   /* Built once here and treated as read-only afterwards (ADR-001). */
   struct sz_caps caps;

   struct sz_queue queues[SZ_QUEUE_COUNT];

   /* Command submission state for the graphics family. Phase 1 has one context
    * at a time, so the pool lives on the screen rather than being duplicated per
    * context. */
   struct sz_queue_state batch_state;

   /* Resources awaiting the timeline value in `value`. See
    * sz_screen_resource_destroy(). */
   struct sz_retired_resource retired_resources[SZ_MAX_RETIRED_RESOURCES];
   unsigned num_retired_resources;

   /* pipe_context::destroy has no way to reach the screen's flag, so the screen
    * tracks it here. See sz_screen_context_create(). */
   bool context_alive;

   /* Device memory suballocator (sz_bo). */
   struct sz_bo_state *bo;

   /* One descriptor set layout per stage, indexed by descriptor set. */
   VkDescriptorSetLayout descriptor_set_layout;

   /* Screen-owned objects every pipeline shares. The layout is immutable and
    * depends only on the descriptor set layout, so a per-pipeline copy would be
    * pure overhead -- and creating one per rebuild was most of the cost of a
    * state change. */
   VkPipelineLayout pipeline_layout;
   VkPipelineCache pipeline_cache;

   /* Shaders used when a draw has no program bound (the driver's own self
    * tests). Created on first use, destroyed with the device. */
   VkShaderModule bootstrap_vs;
   VkShaderModule bootstrap_fs;

   /* VK_EXT_multi_draw: how many draws one vkCmdDrawMultiEXT may carry. Zero
    * means the extension is unavailable and the draw stream emits one command
    * per draw. */
   unsigned max_multi_draw_count;

   int drm_fd;

   /* Built once during screen creation: get_name() must be stable and
    * thread-safe, so it cannot format into a function-local static buffer. */
   char renderer_string[384];

   bool is_cpu;
   bool driver_name_is_inferred;

   /* Depth/stencil format availability, decided once at screen creation.
    *
    * Vulkan devices are free not to support D24_UNORM_S8_UINT at all -- RADV on
    * AMD reports no features for it -- while GL's DEPTH24_STENCIL8 still has to
    * be backed by something. The closest supported format is used instead, which
    * is what every other Gallium-on-Vulkan driver does. The probe happens once
    * because the answer is a device property and sz_vk_format() is called from
    * hot paths (the pipeline key comparison among them). */
   bool have_d24_unorm_s8_uint;
   bool have_x8_d24_unorm_pack32;
   bool have_d32_sfloat;
   bool have_d32_sfloat_s8_uint;

   /* Set by the first dma-buf failure, so a missing capability is reported once
    * instead of on every allocation attempt. */
   bool dmabuf_warned;

   /* SZ_DEBUG=stats counters. Cheap enough to leave enabled in release builds:
    * each one is a single increment on a path that already does real work. */
   struct {
      uint64_t draws;            /* draw commands the app issued */
      uint64_t batches;          /* draw batches that reached the command buffer */
      uint64_t multi_batches;    /* batches that used one vkCmdDrawMulti* call */
      uint64_t pipeline_binds;
      uint64_t pipeline_builds;
      uint64_t descriptor_pushes;
      uint64_t vertex_buffer_binds;
      uint64_t state_sets;       /* viewport/scissor/blend/stencil emissions */
      uint64_t blits;            /* pipe_context::blit calls */
      uint64_t resolves;         /* of those, MSAA resolves */
      uint64_t copies;           /* of those, same-size image copies */
      uint64_t end_of_frame;     /* flushes that carried PIPE_FLUSH_END_OF_FRAME */
      uint64_t submits;
   } stats;
};

static inline struct sz_screen *
sz_screen(struct pipe_screen *screen)
{
   return (struct sz_screen *)screen;
}

#endif /* SZ_TYPES_H */
