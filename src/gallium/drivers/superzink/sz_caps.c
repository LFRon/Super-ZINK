/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>

#include "util/macros.h"
#include "util/u_memory.h"

#include "sz_caps.h"
#include "sz_debug.h"

static bool
sz_api_at_least(uint32_t api, uint32_t required)
{
   if (required == SZ_CORE_NONE)
      return false;
   return api >= required;
}

static bool
sz_ext_advertised(const VkExtensionProperties *exts, uint32_t count,
                  const char *name)
{
   for (uint32_t i = 0; i < count; i++) {
      if (strcmp(exts[i].extensionName, name) == 0)
         return true;
   }
   return false;
}

/* Appends `s` to the pNext chain rooted at *head, linking through *tail.
 *
 * The link must go through the struct's own pNext field. Writing to the struct
 * address instead would overwrite sType, because sType sits at offset 0. */
static void
sz_chain(VkBaseOutStructure **head, VkBaseOutStructure **tail, void *s)
{
   VkBaseOutStructure *b = (VkBaseOutStructure *)s;
   b->pNext = NULL;

   if (*tail)
      (*tail)->pNext = b;
   else
      *head = b;

   *tail = b;
}

void
sz_vk_features_prepare(struct sz_vk_features *f, uint32_t api_version)
{
   memset(f, 0, sizeof(*f));

   f->f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
   f->f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
   f->f14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;

   /* Alias structure types: identical to their core counterparts, so linking
    * them on a 1.4 device answers the same question the core struct would. */
   f->local_read.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES;
   f->host_image_copy.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
   f->descriptor_buffer.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
   f->present_wait.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR;
   f->gpl.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
   f->msrtss.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT;
   f->shader_object.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
   f->fsr.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
   f->image_int64.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT;
   f->list_restart.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRIMITIVE_TOPOLOGY_LIST_RESTART_FEATURES_EXT;
   f->multi_draw.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_FEATURES_EXT;
   f->transform_feedback.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT;

   (void)api_version;
}

void
sz_vk_features_link(struct sz_vk_features *f, uint32_t api_version,
                    const struct sz_vk_feature_selection *sel,
                    VkPhysicalDeviceFeatures2 *features2)
{
   features2->pNext = NULL;

   VkBaseOutStructure *head = NULL;
   VkBaseOutStructure *tail = NULL;

   /* Core structs only exist on devices that know them; chaining a 1.4 struct
    * into a 1.3 driver makes vkGetPhysicalDeviceFeatures2 return
    * VK_ERROR_FEATURE_NOT_PRESENT on some drivers. */
   if (sz_api_at_least(api_version, VK_API_VERSION_1_2))
      sz_chain(&head, &tail, &f->f12);
   if (sz_api_at_least(api_version, VK_API_VERSION_1_3))
      sz_chain(&head, &tail, &f->f13);
   if (sz_api_at_least(api_version, VK_API_VERSION_1_4))
      sz_chain(&head, &tail, &f->f14);

   /* Promoted-but-not-mandatory extensions: ask through their own structure so
    * the answer is available on 1.3 as well as 1.4. On 1.4 the core struct
    * already carries the feature bit. */
   if (!sz_api_at_least(api_version, VK_API_VERSION_1_4) && sel->local_read)
      sz_chain(&head, &tail, &f->local_read);
   if (!sz_api_at_least(api_version, VK_API_VERSION_1_4) && sel->host_image_copy)
      sz_chain(&head, &tail, &f->host_image_copy);

   /* Extensions that 1.4 did not promote. */
   if (sel->descriptor_buffer)
      sz_chain(&head, &tail, &f->descriptor_buffer);
   if (sel->present_wait)
      sz_chain(&head, &tail, &f->present_wait);
   if (sel->graphics_pipeline_library)
      sz_chain(&head, &tail, &f->gpl);
   if (sel->msrtss)
      sz_chain(&head, &tail, &f->msrtss);
   if (sel->shader_object)
      sz_chain(&head, &tail, &f->shader_object);
   if (sel->fragment_shading_rate)
      sz_chain(&head, &tail, &f->fsr);
   if (sel->shader_image_int64)
      sz_chain(&head, &tail, &f->image_int64);
   if (sel->multi_draw)
      sz_chain(&head, &tail, &f->multi_draw);
   if (sel->list_restart)
      sz_chain(&head, &tail, &f->list_restart);
   if (sel->transform_feedback)
      sz_chain(&head, &tail, &f->transform_feedback);

   features2->pNext = head;
}

void
sz_caps_init(struct sz_caps *caps,
             VkPhysicalDevice pdev,
             uint32_t instance_api_version,
             PFN_vkEnumerateDeviceExtensionProperties enum_ext,
             PFN_vkGetPhysicalDeviceFeatures2 get_features2,
             PFN_vkGetPhysicalDeviceProperties2 get_props2)
{
   memset(caps, 0, sizeof(*caps));

   VkPhysicalDeviceProperties props;
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
   };

   /* The caller only resolved a subset of entry points; recover apiVersion
    * without requiring GetPhysicalDeviceProperties. */
   uint32_t api = instance_api_version;
   if (get_props2) {
      get_props2(pdev, &props2);
      props = props2.properties;
      api = props.apiVersion;
   }
   caps->instance_api_version = instance_api_version;
   caps->device_api_version = api;

   /* ── extension list ────────────────────────────────────────────────────── */
   VkExtensionProperties *exts = NULL;
   uint32_t ext_count = 0;

   if (enum_ext) {
      uint32_t n = 0;
      if (enum_ext(pdev, NULL, &n, NULL) == VK_SUCCESS && n > 0) {
         exts = CALLOC(n, sizeof(*exts));
         if (exts) {
            VkResult res = enum_ext(pdev, NULL, &n, exts);
            if (res == VK_SUCCESS || res == VK_INCOMPLETE)
               ext_count = n;
         }
      }
   }

   /* ── one features query, chained ───────────────────────────────────────── */
   struct sz_vk_feature_selection sel = {
      .local_read = sz_ext_advertised(exts, ext_count,
                                      VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME),
      .host_image_copy = sz_ext_advertised(exts, ext_count,
                                           VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME),
      .descriptor_buffer = sz_ext_advertised(exts, ext_count,
                                             VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME),
      .present_wait = sz_ext_advertised(exts, ext_count,
                                        VK_KHR_PRESENT_WAIT_EXTENSION_NAME),
      .graphics_pipeline_library =
         sz_ext_advertised(exts, ext_count,
                           VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME),
      .msrtss = sz_ext_advertised(exts, ext_count,
                                  VK_EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXTENSION_NAME),
      .shader_object = sz_ext_advertised(exts, ext_count,
                                         VK_EXT_SHADER_OBJECT_EXTENSION_NAME),
      .fragment_shading_rate = sz_ext_advertised(exts, ext_count,
                                                 VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME),
      .shader_image_int64 = sz_ext_advertised(exts, ext_count,
                                              VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME),
      .list_restart = sz_ext_advertised(exts, ext_count,
         VK_EXT_PRIMITIVE_TOPOLOGY_LIST_RESTART_EXTENSION_NAME),
      .multi_draw = sz_ext_advertised(exts, ext_count,
                                      VK_EXT_MULTI_DRAW_EXTENSION_NAME),
      .transform_feedback = sz_ext_advertised(exts, ext_count,
                                              VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME),
   };

   struct sz_vk_features q;
   sz_vk_features_prepare(&q, api);

   VkPhysicalDeviceFeatures2 features2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
   };
   sz_vk_features_link(&q, api, &sel, &features2);
   if (get_features2) {
      get_features2(pdev, &features2);
      q.core = features2.features;
   }

   /* ── promoted capabilities: pick core or extension ─────────────────────── */
   const bool vk14 = sz_api_at_least(api, VK_API_VERSION_1_4);

   const bool cap_dynamic_rendering_local_read =
      vk14 ? q.f14.dynamicRenderingLocalRead : q.local_read.dynamicRenderingLocalRead;
   const bool cap_host_image_copy =
      vk14 ? q.f14.hostImageCopy : q.host_image_copy.hostImageCopy;
   /* VK_KHR_push_descriptor has no feature struct of its own ("nofeatures" in
    * vk.xml): advertising the extension is the whole answer. */
   const bool cap_push_descriptor =
      vk14 ? q.f14.pushDescriptor : true;
   const bool cap_local_read_depth_stencil =
      vk14 && q.props14.dynamicRenderingLocalReadDepthStencilAttachments;
   const bool cap_local_read_multisampled =
      vk14 && q.props14.dynamicRenderingLocalReadMultisampledAttachments;

   /* maxPushDescriptors is a property rather than a feature, so it comes from its
    * own chained query. It bounds how many uniform buffers one push can carry:
    * sz_descriptor_screen_init() refuses to start when it is too small. */
   if (get_props2) {
      VkPhysicalDevicePushDescriptorPropertiesKHR push = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR,
      };
      VkPhysicalDeviceProperties2 query = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
         .pNext = &push,
      };
      VkPhysicalDeviceMultiDrawPropertiesEXT multi = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTI_DRAW_PROPERTIES_EXT,
      };
      VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
      };
      VkPhysicalDeviceDescriptorBufferPropertiesEXT desc = {
         .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT,
      };
      push.pNext = &multi;
      multi.pNext = &xfb;
      xfb.pNext = &desc;
      get_props2(pdev, &query);
      caps->max_push_descriptors = push.maxPushDescriptors;
      caps->max_multi_draw_count = multi.maxMultiDrawCount;
      caps->max_transform_feedback_buffers = xfb.maxTransformFeedbackBuffers;
      caps->descriptor_buffer_offset_alignment =
         desc.descriptorBufferOffsetAlignment;
      caps->descriptor_size_uniform_buffer = desc.uniformBufferDescriptorSize;
      caps->descriptor_size_combined_image_sampler =
         desc.combinedImageSamplerDescriptorSize;
   }

   /* ── resolve every listed capability ───────────────────────────────────── */
#define SZ_CAP_RESOLVE(field, name, ext, core, expr)                           \
   caps->field = (expr) &&                                                     \
      (sz_api_at_least(api, core) || sz_ext_advertised(exts, ext_count, ext));
   SZ_CAP_REQUIRED_LIST(SZ_CAP_RESOLVE)
   SZ_CAP_OPTIONAL_LIST(SZ_CAP_RESOLVE)
#undef SZ_CAP_RESOLVE

   caps->core_features = q.core;
   caps->uses_vk14_core = vk14;

   /* Extension-only capabilities. The dependency is folded in rather than left
    * to the caller: a dma-buf we cannot turn into a file descriptor, or a
    * modifier we cannot describe, is not a capability. */
   caps->external_memory_fd =
      sz_ext_advertised(exts, ext_count,
                        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
   caps->external_memory_dma_buf =
      caps->external_memory_fd &&
      sz_ext_advertised(exts, ext_count,
                        VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
   caps->drm_format_modifier =
      caps->external_memory_dma_buf &&
      sz_ext_advertised(exts, ext_count,
                        VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME);

   FREE(exts);
}

const struct sz_cap_row *
sz_caps_table(void)
{
   static const struct sz_cap_row rows[] = {
#define SZ_CAP_ROW(field, name, ext, core, expr)                               \
      { name, ext, core, offsetof(struct sz_caps, field), true },
      SZ_CAP_REQUIRED_LIST(SZ_CAP_ROW)
#undef SZ_CAP_ROW
#define SZ_CAP_ROW(field, name, ext, core, expr)                               \
      { name, ext, core, offsetof(struct sz_caps, field), false },
      SZ_CAP_OPTIONAL_LIST(SZ_CAP_ROW)
#undef SZ_CAP_ROW
      { NULL, NULL, 0, 0, false },
   };
   return rows;
}

const struct sz_cap_ext_row *
sz_caps_ext_table(void)
{
   static const struct sz_cap_ext_row rows[] = {
#define SZ_CAP_EXT_ROW(field, ext)                                             \
      { #field, ext, offsetof(struct sz_caps, field) },
      SZ_CAP_EXT_ONLY_LIST(SZ_CAP_EXT_ROW)
#undef SZ_CAP_EXT_ROW
      { NULL, NULL, 0 },
   };
   return rows;
}

#define SZ_CAP_BOOL(caps, row)                                                 \
   (*(const bool *)((const char *)(caps) + (row)->offset))

unsigned
sz_caps_missing_required_count(const struct sz_caps *caps)
{
   unsigned n = 0;
   for (const struct sz_cap_row *r = sz_caps_table(); r->name; r++) {
      if (r->required && !SZ_CAP_BOOL(caps, r))
         n++;
   }
   return n;
}

const char *
sz_caps_missing_required_str(const struct sz_caps *caps, char *buf, size_t buflen)
{
   size_t used = 0;
   buf[0] = '\0';

   for (const struct sz_cap_row *r = sz_caps_table(); r->name; r++) {
      if (!r->required || SZ_CAP_BOOL(caps, r))
         continue;
      int n = snprintf(buf + used, buflen - used, "%s%s",
                       used ? ", " : "", r->name);
      if (n < 0 || (size_t)n >= buflen - used)
         break;
      used += n;
   }
   return buf;
}

void
sz_caps_dump(const struct sz_caps *caps)
{
   if (!sz_debug_enabled(SZ_DEBUG_INFO))
      return;

   mesa_logi("SZ: caps: device apiVersion=%u.%u.%u instance=%u.%u.%u%s",
             VK_API_VERSION_MAJOR(caps->device_api_version),
             VK_API_VERSION_MINOR(caps->device_api_version),
             VK_API_VERSION_PATCH(caps->device_api_version),
             VK_API_VERSION_MAJOR(caps->instance_api_version),
             VK_API_VERSION_MINOR(caps->instance_api_version),
             VK_API_VERSION_PATCH(caps->instance_api_version),
             caps->uses_vk14_core ? " (1.4 core path)" : " (extension path)");

   for (const struct sz_cap_row *r = sz_caps_table(); r->name; r++) {
      const bool have = SZ_CAP_BOOL(caps, r);
      const char *source = "missing";
      if (have)
         source = sz_api_at_least(caps->device_api_version, r->core_version) ? "core" : "ext";

      mesa_logi("SZ:   %-34s %-8s %-4s %s",
                r->name, source, r->required ? "req" : "opt",
                have ? "ok" : (r->required ? "FAIL" : "off"));
   }

   for (const struct sz_cap_ext_row *r = sz_caps_ext_table(); r->name; r++) {
      const bool have = SZ_CAP_BOOL(caps, r);
      mesa_logi("SZ:   %-34s %-8s %-4s %s", r->name, "ext", "opt",
                have ? "ok" : "off");
   }
}
