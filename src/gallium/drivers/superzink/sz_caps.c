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

/* Everything we ask the driver about, chained through VkPhysicalDeviceFeatures2
 * in one call. Members are read by the feature expressions in sz_caps.h. */
struct sz_caps_query {
   VkPhysicalDeviceVulkan12Features f12;
   VkPhysicalDeviceVulkan13Features f13;
   VkPhysicalDeviceVulkan14Features f14;

   /* Promoted extensions keep their original structure type as an alias of the
    * core one (e.g. VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT
    * == ..._FEATURES), so these are valid to chain on both 1.3 and 1.4. */
   VkPhysicalDeviceDynamicRenderingLocalReadFeatures local_read;
   VkPhysicalDeviceHostImageCopyFeaturesEXT host_image_copy;

   VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer;
   VkPhysicalDevicePresentWaitFeaturesKHR present_wait;
   VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
   VkPhysicalDeviceMultisampledRenderToSingleSampledFeaturesEXT msrtss;
   VkPhysicalDeviceShaderObjectFeaturesEXT shader_object;
   VkPhysicalDeviceFragmentShadingRateFeaturesKHR fsr;
   VkPhysicalDeviceShaderImageAtomicInt64FeaturesEXT image_int64;

   VkPhysicalDeviceVulkan14Properties props14;
};

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
   struct sz_caps_query q;
   memset(&q, 0, sizeof(q));

   VkBaseOutStructure *head = NULL;
   VkBaseOutStructure *tail = NULL;

   /* Core feature structs are only chained on devices that know them. */
   if (sz_api_at_least(api, VK_API_VERSION_1_2)) {
      q.f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
      sz_chain(&head, &tail, &q.f12);
   }
   if (sz_api_at_least(api, VK_API_VERSION_1_3)) {
      q.f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
      sz_chain(&head, &tail, &q.f13);
   }
   if (sz_api_at_least(api, VK_API_VERSION_1_4)) {
      q.f14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
      sz_chain(&head, &tail, &q.f14);
   }

   /* Promoted-but-not-mandatory extensions: ask through their own structure so
    * the answer is available on 1.3 as well as 1.4. */
   if (!sz_api_at_least(api, VK_API_VERSION_1_4) &&
       sz_ext_advertised(exts, ext_count, VK_KHR_DYNAMIC_RENDERING_LOCAL_READ_EXTENSION_NAME)) {
      q.local_read.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES;
      sz_chain(&head, &tail, &q.local_read);
   }
   if (!sz_api_at_least(api, VK_API_VERSION_1_4) &&
       sz_ext_advertised(exts, ext_count, VK_EXT_HOST_IMAGE_COPY_EXTENSION_NAME)) {
      q.host_image_copy.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_FEATURES_EXT;
      sz_chain(&head, &tail, &q.host_image_copy);
   }

   /* Extensions that 1.4 did not promote. */
   if (sz_ext_advertised(exts, ext_count, VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
      q.descriptor_buffer.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
      sz_chain(&head, &tail, &q.descriptor_buffer);
   }
   if (sz_ext_advertised(exts, ext_count, VK_KHR_PRESENT_WAIT_EXTENSION_NAME)) {
      q.present_wait.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_FEATURES_KHR;
      sz_chain(&head, &tail, &q.present_wait);
   }
   if (sz_ext_advertised(exts, ext_count, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME)) {
      q.gpl.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
      sz_chain(&head, &tail, &q.gpl);
   }
   if (sz_ext_advertised(exts, ext_count,
                         VK_EXT_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_EXTENSION_NAME)) {
      q.msrtss.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MULTISAMPLED_RENDER_TO_SINGLE_SAMPLED_FEATURES_EXT;
      sz_chain(&head, &tail, &q.msrtss);
   }
   if (sz_ext_advertised(exts, ext_count, VK_EXT_SHADER_OBJECT_EXTENSION_NAME)) {
      q.shader_object.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
      sz_chain(&head, &tail, &q.shader_object);
   }
   if (sz_ext_advertised(exts, ext_count, VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME)) {
      q.fsr.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR;
      sz_chain(&head, &tail, &q.fsr);
   }
   if (sz_ext_advertised(exts, ext_count,
                         VK_EXT_SHADER_IMAGE_ATOMIC_INT64_EXTENSION_NAME)) {
      q.image_int64.sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_IMAGE_ATOMIC_INT64_FEATURES_EXT;
      sz_chain(&head, &tail, &q.image_int64);
   }

   if (get_features2 && head) {
      VkPhysicalDeviceFeatures2 features2 = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
         .pNext = head,
      };
      get_features2(pdev, &features2);
   }

   /* The local-read nesting depth/stencil and multisampled bits live in
    * VkPhysicalDeviceVulkan14Properties, which has no extension equivalent
    * anywhere in vk.xml. On 1.3 they are simply unknowable, so both stay false
    * and the depth/MSAA local-read paths stay off. */
   if (get_props2 && sz_api_at_least(api, VK_API_VERSION_1_4)) {
      q.props14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
      props2.pNext = &q.props14;
      get_props2(pdev, &props2);
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

   /* ── resolve every listed capability ───────────────────────────────────── */
#define SZ_CAP_RESOLVE(field, name, ext, core, expr)                           \
   caps->field = (expr) &&                                                     \
      (sz_api_at_least(api, core) || sz_ext_advertised(exts, ext_count, ext));
   SZ_CAP_REQUIRED_LIST(SZ_CAP_RESOLVE)
   SZ_CAP_OPTIONAL_LIST(SZ_CAP_RESOLVE)
#undef SZ_CAP_RESOLVE

   caps->uses_vk14_core = vk14;

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
}
