/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Phase 1: VkDevice, queues and memory types.
 *
 * The extension and feature lists are derived from the capability snapshot the
 * screen already built, never re-derived from scratch. ADR-001 makes
 * screen->caps the single gate: if it says a capability is unavailable, no
 * extension name and no feature bit for it is ever passed to vkCreateDevice.
 *
 * Two rules make this less trivial than it looks:
 *
 *   1. An extension whose dependency is not enabled fails vkCreateDevice. The
 *      capability table names a feature, not its dependency chain, so chains
 *      such as present_wait -> swapchain + present_id are expanded here.
 *   2. A promoted extension is only needed below its core version. On a 1.4
 *      device the 1.4 feature struct carries the bit and naming the extension
 *      is redundant; below it, the extension name is the only way in.
 */

#include "sz_device.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_memory.h"

#include <string.h>

#define SZ_MAX_DEVICE_EXTENSIONS 32

/* Capabilities the GL 4.6 feature set needs, gated on what the device reports.
 * Anything not listed here stays disabled even when the hardware offers it:
 * enabling a feature the driver does not need only widens the surface for
 * driver-specific slow paths. */
#define SZ_CORE_FEATURES(X)                                                     \
   X(independentBlend)                                                          \
   X(dualSrcBlend)                                                              \
   X(geometryShader)                                                            \
   X(tessellationShader)                                                        \
   X(sampleRateShading)                                                         \
   X(depthClamp)                                                                \
   X(fillModeNonSolid)                                                          \
   X(wideLines)                                                                 \
   X(largePoints)                                                               \
   X(alphaToOne)                                                                \
   X(logicOp)                                                                   \
   X(depthBiasClamp)                                                            \
   X(drawIndirectFirstInstance)                                                 \
   X(multiDrawIndirect)                                                         \
   X(imageCubeArray)                                                            \
   X(fragmentStoresAndAtomics)                                                  \
   X(shaderImageGatherExtended)                                                 \
   X(shaderStorageImageExtendedFormats)                                         \
   X(shaderStorageImageReadWithoutFormat)                                       \
   X(shaderStorageImageWriteWithoutFormat)                                      \
   X(shaderClipDistance)                                                        \
   X(shaderCullDistance)                                                        \
   X(shaderTessellationAndGeometryPointSize)                                    \
   X(occlusionQueryPrecise)                                                     \
   X(pipelineStatisticsQuery)                                                   \
   X(vertexPipelineStoresAndAtomics)

/* Extension dependency expansion. Every driver that advertises the left-hand
 * extension also advertises its dependencies, but vkCreateDevice validates the
 * list we pass, so they have to be spelled out. */
struct sz_ext_dep {
   const char *extension;
   const char *depends[3];
};

static const struct sz_ext_dep sz_ext_deps[] = {
   { VK_KHR_PRESENT_WAIT_EXTENSION_NAME,
     { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_PRESENT_ID_EXTENSION_NAME, NULL } },
   { VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME,
     { VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, NULL, NULL } },
   /* VK_KHR_external_memory is core in 1.1, but a driver may still want the
    * extension spelled out, and vkCreateDevice validates the chain we pass. */
   { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
     { VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, NULL, NULL } },
   { VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
     { VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, NULL, NULL } },
   { VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
     { VK_KHR_IMAGE_FORMAT_LIST_EXTENSION_NAME,
       VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME, NULL } },
};

struct sz_ext_list {
   const char *names[SZ_MAX_DEVICE_EXTENSIONS];
   unsigned count;
   unsigned overflow;
};

static bool
sz_ext_list_has(const struct sz_ext_list *list, const char *name)
{
   for (unsigned i = 0; i < list->count; i++) {
      if (strcmp(list->names[i], name) == 0)
         return true;
   }
   return false;
}

static void
sz_ext_list_add(struct sz_ext_list *list, const char *name)
{
   if (sz_ext_list_has(list, name))
      return;
   if (list->count == SZ_MAX_DEVICE_EXTENSIONS) {
      list->overflow++;
      return;
   }
   list->names[list->count++] = name;
}

static void
sz_ext_list_remove(struct sz_ext_list *list, const char *name)
{
   for (unsigned i = 0; i < list->count; i++) {
      if (strcmp(list->names[i], name) != 0)
         continue;
      memmove(&list->names[i], &list->names[i + 1],
              (list->count - i - 1) * sizeof(list->names[0]));
      list->count--;
      return;
   }
}

static bool
sz_device_extension_advertised(const VkExtensionProperties *exts, uint32_t count,
                               const char *name)
{
   for (uint32_t i = 0; i < count; i++) {
      if (strcmp(exts[i].extensionName, name) == 0)
         return true;
   }
   return false;
}

/* The extension list is exactly "the capabilities caps says we have, minus the
 * ones we get from core on this device". No speculative enables: asking for an
 * extension we do not use costs a driver-side branch on every call that could
 * take the core path. */
static void
sz_build_extension_list(const struct sz_screen *screen,
                        const VkExtensionProperties *advertised,
                        uint32_t advertised_count, struct sz_ext_list *list)
{
   const struct sz_caps *caps = &screen->caps;

   for (const struct sz_cap_row *r = sz_caps_table(); r->name; r++) {
      const bool have = *(const bool *)((const char *)caps + r->offset);
      if (!have)
         continue;

      /* Core on this device: the version alone unlocks it. */
      if (r->core_version != SZ_CORE_NONE &&
          screen->api_version >= r->core_version)
         continue;

      if (!sz_device_extension_advertised(advertised, advertised_count,
                                          r->extension))
         continue;

      sz_ext_list_add(list, r->extension);
   }

   for (const struct sz_cap_ext_row *r = sz_caps_ext_table(); r->name; r++) {
      const bool have = *(const bool *)((const char *)caps + r->offset);
      if (!have)
         continue;
      if (!sz_device_extension_advertised(advertised, advertised_count,
                                          r->extension))
         continue;
      sz_ext_list_add(list, r->extension);
   }

   /* Expand dependencies, dropping anything whose chain is incomplete rather
    * than letting vkCreateDevice fail with a generic error later. */
   const unsigned snapshot = list->count;
   for (unsigned i = 0; i < snapshot; i++) {
      const char *name = list->names[i];
      for (unsigned d = 0; d < ARRAY_SIZE(sz_ext_deps); d++) {
         if (strcmp(sz_ext_deps[d].extension, name) != 0)
            continue;

         bool ok = true;
         const char *missing = NULL;
         for (unsigned k = 0; k < ARRAY_SIZE(sz_ext_deps[d].depends); k++) {
            const char *dep = sz_ext_deps[d].depends[k];
            if (!dep)
               break;
            if (!sz_device_extension_advertised(advertised, advertised_count, dep)) {
               ok = false;
               missing = dep;
               break;
            }
            sz_ext_list_add(list, dep);
         }
         if (!ok) {
            sz_warn("extension %s needs %s, which the device does not "
                    "advertise; dropping it", name, missing);
            sz_ext_list_remove(list, name);
            i--;
         }
         break;
      }
   }

   if (list->overflow) {
      sz_warn("%u device extensions did not fit in the list; they stay "
              "disabled", list->overflow);
   }
}

static struct sz_vk_feature_selection
sz_feature_selection(const struct sz_screen *screen)
{
   const struct sz_caps *caps = &screen->caps;

   return (struct sz_vk_feature_selection) {
      .local_read = caps->dynamic_rendering_local_read,
      .host_image_copy = caps->host_image_copy,
      .descriptor_buffer = caps->descriptor_buffer,
      .present_wait = caps->present_wait,
      .graphics_pipeline_library = caps->graphics_pipeline_library,
      .msrtss = caps->msrtss,
      .shader_object = caps->shader_object,
      .fragment_shading_rate = caps->fragment_shading_rate,
      .shader_image_int64 = caps->shader_image_int64_atomics,
      .multi_draw = caps->multi_draw,
      .list_restart = caps->list_restart,
      .transform_feedback = caps->transform_feedback,
   };
}

/* Asks the driver what it supports (before vkCreateDevice) or what it enabled
 * (after). The query is tagged with the magic value the spec prescribes, which
 * is also what makes the post-create call a legal way to read the enabled set. */
static void
sz_query_features(struct sz_screen *screen, struct sz_vk_features *features,
                  VkPhysicalDeviceFeatures2 *chained)
{
   const struct sz_vk_feature_selection sel = sz_feature_selection(screen);

   sz_vk_features_prepare(features, screen->api_version);

   VkPhysicalDeviceFeatures2 query = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
   };
   sz_vk_features_link(features, screen->api_version, &sel, &query);
   if (query.pNext)
      screen->vk.GetPhysicalDeviceFeatures2(screen->pdev, &query);
   features->core = query.features;

   if (chained) {
      sz_vk_features_link(features, screen->api_version, &sel, chained);
      chained->features = features->core;
   }
}

/* Clears every bit the capability snapshot did not confirm, so a driver that
 * answers differently on the second query cannot slip a feature past the
 * gate. The snapshot stays authoritative (ADR-001). */
static void
sz_gate_features(const struct sz_screen *screen, struct sz_vk_features *features)
{
   const struct sz_caps *caps = &screen->caps;

#define SZ_GATE_CORE(field) features->core.field &= caps->core_features.field;
   SZ_CORE_FEATURES(SZ_GATE_CORE)
#undef SZ_GATE_CORE

   features->f12.timelineSemaphore &= caps->timeline_semaphore;
   features->f12.uniformBufferStandardLayout &=
      caps->uniform_buffer_standard_layout;
   features->f12.samplerMirrorClampToEdge &=
      caps->sampler_mirror_clamp_to_edge;
   features->f12.bufferDeviceAddress &= caps->buffer_device_address;
   features->f13.dynamicRendering &= caps->dynamic_rendering;
   features->f13.synchronization2 &= caps->synchronization2;
   features->f13.maintenance4 &= caps->maintenance4;

   if (screen->api_version >= VK_API_VERSION_1_4) {
      features->f14.dynamicRenderingLocalRead &= caps->dynamic_rendering_local_read;
      features->f14.hostImageCopy &= caps->host_image_copy;
      features->f14.pushDescriptor &= caps->push_descriptor;

   } else {
      features->local_read.dynamicRenderingLocalRead &=
         caps->dynamic_rendering_local_read;
      features->host_image_copy.hostImageCopy &= caps->host_image_copy;
      /* pushDescriptor lives in VkPhysicalDeviceVulkan14Features only. Below
       * 1.4 there is no bit to set: the enabled extension is the whole
       * contract, and maxPushDescriptors comes from its properties. */
   }

   features->descriptor_buffer.descriptorBuffer &= caps->descriptor_buffer;
   features->present_wait.presentWait &= caps->present_wait;
   features->gpl.graphicsPipelineLibrary &= caps->graphics_pipeline_library;
   features->msrtss.multisampledRenderToSingleSampled &= caps->msrtss;
   features->shader_object.shaderObject &= caps->shader_object;

   /* VK_KHR_fragment_shading_rate exposes three independent bits; GL has no
    * use for the primitive-rate one, so only the two that map onto GL's
    * shading-rate state are ever enabled. */
   features->fsr.pipelineFragmentShadingRate &= caps->fragment_shading_rate;
   features->fsr.primitiveFragmentShadingRate = false;
   features->fsr.attachmentFragmentShadingRate &= caps->fragment_shading_rate;

   features->image_int64.shaderImageInt64Atomics &=
      caps->shader_image_int64_atomics;
   features->multi_draw.multiDraw &= caps->multi_draw;
   features->list_restart.primitiveTopologyListRestart &= caps->list_restart;
   features->transform_feedback.transformFeedback &= caps->transform_feedback;
}

/* Post-create self-check. A feature that we asked for but the device did not
 * enable produces a silent miscompile far from here, so it is verified once at
 * startup instead of being discovered by the first draw that needs it. */
static void
sz_verify_enabled_features(struct sz_screen *screen,
                           const struct sz_vk_features *requested)
{
   struct sz_vk_features actual;
   sz_query_features(screen, &actual, NULL);

   unsigned missing = 0;
   char names[256] = "";
   size_t used = 0;

#define SZ_CHECK(field, label)                                                 \
   do {                                                                        \
      if (requested->field && !actual.field) {                                 \
         missing++;                                                            \
         int n = snprintf(names + used, sizeof(names) - used, "%s%s",          \
                          used ? "," : "", label);                             \
         if (n > 0 && (size_t)n < sizeof(names) - used)                        \
            used += n;                                                         \
      }                                                                        \
   } while (0)

   SZ_CHECK(f12.timelineSemaphore, "timelineSemaphore");
   SZ_CHECK(f12.bufferDeviceAddress, "bufferDeviceAddress");
   SZ_CHECK(f13.dynamicRendering, "dynamicRendering");
   SZ_CHECK(f13.synchronization2, "synchronization2");
   SZ_CHECK(f13.maintenance4, "maintenance4");
   SZ_CHECK(descriptor_buffer.descriptorBuffer, "descriptorBuffer");
   SZ_CHECK(present_wait.presentWait, "presentWait");
   SZ_CHECK(gpl.graphicsPipelineLibrary, "graphicsPipelineLibrary");
   SZ_CHECK(msrtss.multisampledRenderToSingleSampled, "msrtss");
   SZ_CHECK(shader_object.shaderObject, "shaderObject");
   SZ_CHECK(image_int64.shaderImageInt64Atomics, "shaderImageInt64Atomics");
   if (screen->api_version >= VK_API_VERSION_1_4) {
      SZ_CHECK(f14.dynamicRenderingLocalRead, "dynamicRenderingLocalRead");
      SZ_CHECK(f14.hostImageCopy, "hostImageCopy");
      SZ_CHECK(f14.pushDescriptor, "pushDescriptor");
   } else {
      SZ_CHECK(local_read.dynamicRenderingLocalRead, "dynamicRenderingLocalRead");
      SZ_CHECK(host_image_copy.hostImageCopy, "hostImageCopy");
   }
#undef SZ_CHECK

   if (missing) {
      sz_warn("the device accepted the feature chain but reports %u of its "
              "bits as disabled: %s", missing, names);
   } else if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ:   enabled features verified");
   }
}

static const char *
sz_queue_role_name(enum sz_queue_role role)
{
   switch (role) {
   case SZ_QUEUE_GRAPHICS: return "graphics";
   case SZ_QUEUE_COMPUTE:  return "compute";
   case SZ_QUEUE_TRANSFER: return "transfer";
   default:                return "?";
   }
}

/* Picks one family per role. Graphics and compute are allowed to share a
 * family: splitting them would cost a second submit per frame for no gain,
 * because Super-ZINK never runs asynchronous compute. The transfer role
 * prefers a dedicated family only when it is not also a graphics family. */
static bool
sz_pick_queue_families(struct sz_screen *screen, uint32_t *graphics,
                       uint32_t *compute, uint32_t *transfer)
{
   uint32_t count = 0;
   screen->vk.GetPhysicalDeviceQueueFamilyProperties(screen->pdev, &count, NULL);
   if (count == 0) {
      mesa_loge("SZ: the device exposes no queue families");
      return false;
   }

   VkQueueFamilyProperties *families = CALLOC(count, sizeof(*families));
   if (!families)
      return false;
   screen->vk.GetPhysicalDeviceQueueFamilyProperties(screen->pdev, &count, families);

   int g = -1, c = -1, t = -1;
   for (uint32_t i = 0; i < count; i++) {
      const VkQueueFlags f = families[i].queueFlags;
      const bool has_graphics = f & VK_QUEUE_GRAPHICS_BIT;
      const bool has_compute = f & VK_QUEUE_COMPUTE_BIT;
      const bool has_transfer = f & VK_QUEUE_TRANSFER_BIT;

      if (sz_debug_enabled(SZ_DEBUG_INFO)) {
         mesa_logi("SZ: queue family[%u] flags=0x%x count=%u%s", i, f,
                   families[i].queueCount,
                   families[i].timestampValidBits ? "" : " (no timestamps)");
      }

      if (has_graphics && g < 0) {
         g = i;
         /* Timestamps are a property of the queue, not of the device: a family
          * without valid bits silently writes nothing. The graphics family is
          * the one every submission goes through. */
         screen->have_timestamps = families[i].timestampValidBits > 0;
      }
      if (has_compute && c < 0)
         c = i;
      /* A transfer-only family is the only reason to use a third queue;
       * otherwise the compute family already covers the role. */
      if (has_transfer && !(f & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) &&
          t < 0)
         t = i;
   }

   FREE(families);

   if (g < 0) {
      mesa_loge("SZ: no graphics queue family; this device cannot run a GL "
                "context");
      return false;
   }

   *graphics = (uint32_t)g;
   *compute = c >= 0 ? (uint32_t)c : (uint32_t)g;
   *transfer = t >= 0 ? (uint32_t)t : *compute;

   return true;
}

static void
sz_load_device_entrypoints(struct sz_screen *screen)
{
#define SZ_DEV_PROC(field, proc)                                                \
   screen->vk.field = (PFN_##proc)screen->vk_GetDeviceProcAddr(screen->dev, #proc)

   SZ_DEV_PROC(DestroyDevice, vkDestroyDevice);
   SZ_DEV_PROC(DeviceWaitIdle, vkDeviceWaitIdle);
   SZ_DEV_PROC(GetDeviceQueue2, vkGetDeviceQueue2);

   SZ_DEV_PROC(AllocateMemory, vkAllocateMemory);
   SZ_DEV_PROC(FreeMemory, vkFreeMemory);
   SZ_DEV_PROC(MapMemory, vkMapMemory);
   SZ_DEV_PROC(UnmapMemory, vkUnmapMemory);
   SZ_DEV_PROC(FlushMappedMemoryRanges, vkFlushMappedMemoryRanges);
   SZ_DEV_PROC(InvalidateMappedMemoryRanges, vkInvalidateMappedMemoryRanges);
   SZ_DEV_PROC(CreateBuffer, vkCreateBuffer);
   SZ_DEV_PROC(DestroyBuffer, vkDestroyBuffer);
   SZ_DEV_PROC(GetBufferMemoryRequirements, vkGetBufferMemoryRequirements);
   SZ_DEV_PROC(GetBufferMemoryRequirements2, vkGetBufferMemoryRequirements2);
   SZ_DEV_PROC(BindBufferMemory, vkBindBufferMemory);
   SZ_DEV_PROC(CreateImage, vkCreateImage);
   SZ_DEV_PROC(DestroyImage, vkDestroyImage);
   SZ_DEV_PROC(GetImageMemoryRequirements, vkGetImageMemoryRequirements);
   SZ_DEV_PROC(GetImageMemoryRequirements2, vkGetImageMemoryRequirements2);
   SZ_DEV_PROC(BindImageMemory, vkBindImageMemory);
   SZ_DEV_PROC(CreateSampler, vkCreateSampler);
   SZ_DEV_PROC(DestroySampler, vkDestroySampler);
   SZ_DEV_PROC(CreateImageView, vkCreateImageView);
   SZ_DEV_PROC(DestroyImageView, vkDestroyImageView);
   SZ_DEV_PROC(GetImageSubresourceLayout, vkGetImageSubresourceLayout);
   /* Null unless the external-memory extensions were enabled; every caller
    * checks screen->caps before reaching for them. */
   SZ_DEV_PROC(GetMemoryFdKHR, vkGetMemoryFdKHR);
   SZ_DEV_PROC(GetMemoryFdPropertiesKHR, vkGetMemoryFdPropertiesKHR);
   SZ_DEV_PROC(GetImageDrmFormatModifierPropertiesEXT,
               vkGetImageDrmFormatModifierPropertiesEXT);

   SZ_DEV_PROC(CreateDescriptorSetLayout, vkCreateDescriptorSetLayout);
   SZ_DEV_PROC(DestroyDescriptorSetLayout, vkDestroyDescriptorSetLayout);
   /* The sampler set is an ordinary descriptor set: it is not a push set (a
    * pipeline layout may hold only one of those), so it is allocated from a pool
    * and bound with vkCmdBindDescriptorSets. */
   /* bufferDeviceAddress is a required capability, so this always resolves. */
   SZ_DEV_PROC(GetBufferDeviceAddress, vkGetBufferDeviceAddress);

   if (screen->caps.descriptor_buffer) {
      SZ_DEV_PROC(GetDescriptorSetLayoutSizeEXT, vkGetDescriptorSetLayoutSizeEXT);
      SZ_DEV_PROC(GetDescriptorSetLayoutBindingOffsetEXT,
                  vkGetDescriptorSetLayoutBindingOffsetEXT);
      SZ_DEV_PROC(GetDescriptorEXT, vkGetDescriptorEXT);
      SZ_DEV_PROC(CmdBindDescriptorBuffersEXT, vkCmdBindDescriptorBuffersEXT);
      SZ_DEV_PROC(CmdSetDescriptorBufferOffsetsEXT,
                  vkCmdSetDescriptorBufferOffsetsEXT);
   }

   SZ_DEV_PROC(CreateDescriptorPool, vkCreateDescriptorPool);
   SZ_DEV_PROC(DestroyDescriptorPool, vkDestroyDescriptorPool);
   SZ_DEV_PROC(AllocateDescriptorSets, vkAllocateDescriptorSets);
   SZ_DEV_PROC(UpdateDescriptorSets, vkUpdateDescriptorSets);
   SZ_DEV_PROC(CmdBindDescriptorSets, vkCmdBindDescriptorSets);
   /* vkCmdPushDescriptorSetKHR only resolves if VK_KHR_push_descriptor was
    * enabled, and a 1.4 device needs no extension at all: there the command is
    * core as vkCmdPushDescriptorSet, which kept the same signature (only
    * vkCmdPushDescriptorSet2 takes the VkPushDescriptorSetInfo struct), so the
    * one function pointer type covers both. Below 1.4 the extension is in the
    * required capability list and therefore always enabled. */
   SZ_DEV_PROC(CmdPushDescriptorSetKHR, vkCmdPushDescriptorSetKHR);
   if (screen->vk.CmdPushDescriptorSetKHR == NULL)
      SZ_DEV_PROC(CmdPushDescriptorSetKHR, vkCmdPushDescriptorSet);

   SZ_DEV_PROC(CreateCommandPool, vkCreateCommandPool);
   SZ_DEV_PROC(DestroyCommandPool, vkDestroyCommandPool);
   SZ_DEV_PROC(AllocateCommandBuffers, vkAllocateCommandBuffers);
   SZ_DEV_PROC(FreeCommandBuffers, vkFreeCommandBuffers);
   SZ_DEV_PROC(BeginCommandBuffer, vkBeginCommandBuffer);
   SZ_DEV_PROC(EndCommandBuffer, vkEndCommandBuffer);
   SZ_DEV_PROC(ResetCommandBuffer, vkResetCommandBuffer);
   SZ_DEV_PROC(QueueSubmit2, vkQueueSubmit2);
   SZ_DEV_PROC(QueueWaitIdle, vkQueueWaitIdle);
   SZ_DEV_PROC(CreateSemaphore, vkCreateSemaphore);
   SZ_DEV_PROC(DestroySemaphore, vkDestroySemaphore);
   SZ_DEV_PROC(WaitSemaphores, vkWaitSemaphores);
   SZ_DEV_PROC(GetSemaphoreCounterValue, vkGetSemaphoreCounterValue);
   SZ_DEV_PROC(CmdPipelineBarrier2, vkCmdPipelineBarrier2);
   SZ_DEV_PROC(CmdClearColorImage, vkCmdClearColorImage);
   SZ_DEV_PROC(CmdClearDepthStencilImage, vkCmdClearDepthStencilImage);
   SZ_DEV_PROC(CmdCopyImageToBuffer, vkCmdCopyImageToBuffer);
   SZ_DEV_PROC(CmdCopyBufferToImage, vkCmdCopyBufferToImage);
   SZ_DEV_PROC(CmdCopyImage, vkCmdCopyImage);
   SZ_DEV_PROC(CmdCopyImage2, vkCmdCopyImage2);
   SZ_DEV_PROC(CmdBlitImage2, vkCmdBlitImage2);
   SZ_DEV_PROC(CmdResolveImage2, vkCmdResolveImage2);
   SZ_DEV_PROC(CreateQueryPool, vkCreateQueryPool);
   SZ_DEV_PROC(DestroyQueryPool, vkDestroyQueryPool);
   SZ_DEV_PROC(ResetQueryPool, vkResetQueryPool);
   SZ_DEV_PROC(CmdBeginQuery, vkCmdBeginQuery);
   SZ_DEV_PROC(CmdWriteTimestamp2, vkCmdWriteTimestamp2);
   SZ_DEV_PROC(CmdEndQuery, vkCmdEndQuery);
   SZ_DEV_PROC(GetQueryPoolResults, vkGetQueryPoolResults);
   SZ_DEV_PROC(CmdBindTransformFeedbackBuffersEXT, vkCmdBindTransformFeedbackBuffersEXT);
   SZ_DEV_PROC(CmdBeginTransformFeedbackEXT, vkCmdBeginTransformFeedbackEXT);
   SZ_DEV_PROC(CmdEndTransformFeedbackEXT, vkCmdEndTransformFeedbackEXT);
   SZ_DEV_PROC(CmdCopyBuffer, vkCmdCopyBuffer);
   SZ_DEV_PROC(CmdFillBuffer, vkCmdFillBuffer);
   SZ_DEV_PROC(CmdUpdateBuffer, vkCmdUpdateBuffer);
   SZ_DEV_PROC(CmdBeginRendering, vkCmdBeginRendering);
   SZ_DEV_PROC(CmdEndRendering, vkCmdEndRendering);
   SZ_DEV_PROC(CmdSetViewport, vkCmdSetViewport);
   SZ_DEV_PROC(CmdSetScissor, vkCmdSetScissor);
   SZ_DEV_PROC(CmdSetBlendConstants, vkCmdSetBlendConstants);
   SZ_DEV_PROC(CmdSetStencilReference, vkCmdSetStencilReference);
   SZ_DEV_PROC(CmdBindPipeline, vkCmdBindPipeline);
   SZ_DEV_PROC(CmdBindVertexBuffers, vkCmdBindVertexBuffers);
   SZ_DEV_PROC(CmdBindIndexBuffer, vkCmdBindIndexBuffer);
   SZ_DEV_PROC(CmdDraw, vkCmdDraw);
   SZ_DEV_PROC(CmdDrawIndexed, vkCmdDrawIndexed);
   SZ_DEV_PROC(CmdDrawIndirect, vkCmdDrawIndirect);
   SZ_DEV_PROC(CmdDrawIndexedIndirect, vkCmdDrawIndexedIndirect);
   SZ_DEV_PROC(CmdClearAttachments, vkCmdClearAttachments);
   /* NULL unless VK_EXT_multi_draw was enabled; the draw stream checks. */
   SZ_DEV_PROC(CmdDrawMultiEXT, vkCmdDrawMultiEXT);
   SZ_DEV_PROC(CmdDrawMultiIndexedEXT, vkCmdDrawMultiIndexedEXT);
   SZ_DEV_PROC(CreatePipelineCache, vkCreatePipelineCache);
   SZ_DEV_PROC(DestroyPipelineCache, vkDestroyPipelineCache);

   SZ_DEV_PROC(CreateShaderModule, vkCreateShaderModule);
   SZ_DEV_PROC(DestroyShaderModule, vkDestroyShaderModule);
   SZ_DEV_PROC(CreatePipelineLayout, vkCreatePipelineLayout);
   SZ_DEV_PROC(DestroyPipelineLayout, vkDestroyPipelineLayout);
   SZ_DEV_PROC(CreateGraphicsPipelines, vkCreateGraphicsPipelines);
   SZ_DEV_PROC(DestroyPipeline, vkDestroyPipeline);

#undef SZ_DEV_PROC
}

void
sz_device_finish(struct sz_screen *screen)
{
   if (screen->dev != VK_NULL_HANDLE && screen->vk.DestroyDevice) {
      screen->vk.DestroyDevice(screen->dev, NULL);
      screen->dev = VK_NULL_HANDLE;
   }
}

bool
sz_device_init(struct sz_screen *screen)
{
   screen->vk.GetPhysicalDeviceQueueFamilyProperties =
      (PFN_vkGetPhysicalDeviceQueueFamilyProperties)
         screen->vk_GetInstanceProcAddr(screen->instance,
                                        "vkGetPhysicalDeviceQueueFamilyProperties");
   if (!screen->vk.GetPhysicalDeviceQueueFamilyProperties) {
      mesa_loge("SZ: vkGetPhysicalDeviceQueueFamilyProperties is unavailable");
      return false;
   }

   /* Enabled extensions are validated against a fresh enumeration: the caps
    * table records which features exist, not which names the driver exposes,
    * and only the driver's own list is authoritative for vkCreateDevice. */
   uint32_t advertised_count = 0;
   screen->vk.EnumerateDeviceExtensionProperties(screen->pdev, NULL,
                                                 &advertised_count, NULL);
   VkExtensionProperties *advertised = NULL;
   if (advertised_count) {
      advertised = CALLOC(advertised_count, sizeof(*advertised));
      if (!advertised)
         return false;
      VkResult res = screen->vk.EnumerateDeviceExtensionProperties(
         screen->pdev, NULL, &advertised_count, advertised);
      if (res != VK_SUCCESS && res != VK_INCOMPLETE) {
         FREE(advertised);
         advertised = NULL;
         advertised_count = 0;
      }
   }

   struct sz_ext_list exts = { 0 };
   sz_build_extension_list(screen, advertised, advertised_count, &exts);

   struct sz_vk_features features;
   VkPhysicalDeviceFeatures2 chained = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
   };
   sz_query_features(screen, &features, &chained);
   sz_gate_features(screen, &features);
   screen->have_shader_clip_distance = features.core.shaderClipDistance;
   screen->have_shader_cull_distance = features.core.shaderCullDistance;
   /* `chained.pNext` now points at members of `features`, which is still in
    * scope, so it can be handed straight to vkCreateDevice. */
   chained.features = features.core;

   uint32_t graphics, compute, transfer;
   if (!sz_pick_queue_families(screen, &graphics, &compute, &transfer)) {
      FREE(advertised);
      return false;
   }

   /* One queue per distinct family: Phase 1 only submits on graphics, and the
    * other roles exist so the families are already validated. */
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci[3] = { 0 };
   unsigned qci_count = 0;
   uint32_t families[3] = { graphics, compute, transfer };
   unsigned family_count = 1;
   if (compute != graphics) {
      families[family_count++] = compute;
   }
   if (transfer != graphics && transfer != compute) {
      families[family_count++] = transfer;
   }
   for (unsigned i = 0; i < family_count; i++) {
      qci[qci_count++] = (VkDeviceQueueCreateInfo) {
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueFamilyIndex = families[i],
         .queueCount = 1,
         .pQueuePriorities = &prio,
      };
   }

   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &chained,
      .queueCreateInfoCount = qci_count,
      .pQueueCreateInfos = qci,
      .enabledExtensionCount = exts.count,
      .ppEnabledExtensionNames = exts.names,
   };

   if (sz_debug_enabled(SZ_DEBUG_INFO)) {
      mesa_logi("SZ: creating VkDevice with %u extension%s, queues g=%u c=%u t=%u",
                exts.count, exts.count == 1 ? "" : "s", graphics, compute, transfer);
      for (unsigned i = 0; i < exts.count; i++)
         mesa_logi("SZ:   device ext: %s", exts.names[i]);

      static const char *const core_names[] = {
#define SZ_CORE_NAME(field) #field,
         SZ_CORE_FEATURES(SZ_CORE_NAME)
#undef SZ_CORE_NAME
      };
      /* VkPhysicalDeviceFeatures is an array of VkBool32, not of bool, so it
       * cannot be walked by pointer arithmetic. Copy the fields out instead. */
      static const VkBool32 dummy = 0;
      const VkBool32 core_vals[] = {
#define SZ_CORE_VAL(field) features.core.field,
         SZ_CORE_FEATURES(SZ_CORE_VAL)
#undef SZ_CORE_VAL
      };
      (void)dummy;

      char off[320] = "";
      size_t used = 0;
      unsigned on_count = 0;
      for (unsigned i = 0; i < ARRAY_SIZE(core_names); i++) {
         if (core_vals[i]) {
            on_count++;
            continue;
         }
         int n = snprintf(off + used, sizeof(off) - used, "%s%s",
                          used ? "," : "", core_names[i]);
         if (n > 0 && (size_t)n < sizeof(off) - used)
            used += n;
      }
      mesa_logi("SZ:   core features: %u/%zu enabled%s%s", on_count,
                ARRAY_SIZE(core_names), off[0] ? ", unavailable: " : "",
                off);
   }

   VkResult res = screen->vk.CreateDevice(screen->pdev, &dci, NULL, &screen->dev);
   FREE(advertised);

   if (res != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateDevice failed (VkResult %d)", res);
      return false;
   }

   sz_verify_enabled_features(screen, &features);

   sz_load_device_entrypoints(screen);
   if (!screen->vk.DestroyDevice || !screen->vk.GetDeviceQueue2 ||
       !screen->vk.AllocateMemory || !screen->vk.CreateBuffer ||
       !screen->vk.CreateImage || !screen->vk.BindBufferMemory) {
      mesa_loge("SZ: the device is missing required entry points");
      sz_device_finish(screen);
      return false;
   }

   /* How many draws one vkCmdDrawMulti* call may carry. Zero means the
    * extension is unavailable and the draw stream emits one command per draw --
    * the only difference between the two paths. */
   screen->max_multi_draw_count =
      (screen->caps.multi_draw && screen->vk.CmdDrawMultiEXT &&
       screen->vk.CmdDrawMultiIndexedEXT)
         ? screen->caps.max_multi_draw_count
         : 0;

   static const VkQueueFlags role_flags[SZ_QUEUE_COUNT] = {
      [SZ_QUEUE_GRAPHICS] = VK_QUEUE_GRAPHICS_BIT,
      [SZ_QUEUE_COMPUTE]  = VK_QUEUE_COMPUTE_BIT,
      [SZ_QUEUE_TRANSFER] = VK_QUEUE_TRANSFER_BIT,
   };

   for (unsigned role = 0; role < SZ_QUEUE_COUNT; role++) {
      struct sz_queue *q = &screen->queues[role];
      q->family = role == SZ_QUEUE_GRAPHICS ? graphics :
                  role == SZ_QUEUE_COMPUTE  ? compute  : transfer;
      q->index = 0;

      VkDeviceQueueInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_INFO_2,
         .queueFamilyIndex = q->family,
         .queueIndex = 0,
      };
      screen->vk.GetDeviceQueue2(screen->dev, &info, &q->queue);

      if (sz_debug_enabled(SZ_DEBUG_INFO)) {
         mesa_logi("SZ: %s queue: family %u index %u flags 0x%x",
                   sz_queue_role_name(role), q->family, q->index,
                   role_flags[role]);
      }
   }

   return true;
}

uint32_t
sz_device_find_memory_type(const struct sz_screen *screen, uint32_t type_bits,
                           VkMemoryPropertyFlags props)
{
   for (uint32_t i = 0; i < screen->mem_props.memoryTypeCount; i++) {
      if (!(type_bits & (1u << i)))
         continue;
      if ((screen->mem_props.memoryTypes[i].propertyFlags & props) != props)
         continue;
      return i;
   }
   return VK_MAX_MEMORY_TYPES;
}

uint32_t
sz_device_pick_local_memory(const struct sz_screen *screen)
{
   for (uint32_t i = 0; i < screen->mem_props.memoryTypeCount; i++) {
      const uint32_t heap = screen->mem_props.memoryTypes[i].heapIndex;
      if (screen->mem_props.memoryHeaps[heap].flags &
          VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
         return i;
   }
   return 0;
}
