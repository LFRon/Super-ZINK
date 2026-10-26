/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Capability snapshot.
 *
 * ADR-001: the Vulkan version only decides whether a feature can be taken from
 * core or must come from an extension. It never decides on its own that the
 * feature exists, because a high declared version does not imply the presence
 * of extensions that version did not promote:
 *
 *   lavapipe  claims 1.4 but has no VK_KHR_present_wait at all.
 *   RADV      claims 1.4 but gates present_wait on has_timeline_syncobj,
 *             host_image_copy on gfx_level >= GFX10, graphics_pipeline_library
 *             on !use_llvm, and msrtss on DETECT_OS_ANDROID || experimental.
 *
 * So every capability is resolved as
 *
 *     feature bit set  &&  (apiVersion >= core version || extension advertised)
 *
 * The feature bit is always consulted, even for core features: promotion to
 * core makes a feature reachable without enabling an extension, it does not
 * make it mandatory. What the version check saves is only the extension-name
 * lookup.
 *
 * The snapshot is built once, in sz_caps_init(). Modules read the booleans and
 * must not query the driver again: no lookups in per-context or per-draw code.
 */

#ifndef SZ_CAPS_H
#define SZ_CAPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <vulkan/vulkan_core.h>

/* Lowest apiVersion whose core set we require unconditionally. Anything older
 * is rejected before extension enumeration. */
#define SZ_CAPS_MIN_API_VERSION VK_API_VERSION_1_3

/* Sentinel for "this capability was never promoted to core". */
#define SZ_CORE_NONE 0

/* Capabilities without which Super-ZINK refuses to run.
 *
 * X(field, printed name, primary extension, core version, feature expression)
 *
 * The feature expression is evaluated inside sz_caps_init() and may reference
 * the query locals documented there. */
#define SZ_CAP_REQUIRED_LIST(X)                                                \
   X(dynamic_rendering,            "dynamicRendering",                          \
     "VK_KHR_dynamic_rendering",            VK_API_VERSION_1_3,                \
     q.f13.dynamicRendering)                                                    \
   X(synchronization2,             "synchronization2",                          \
     "VK_KHR_synchronization2",             VK_API_VERSION_1_3,                \
     q.f13.synchronization2)                                                     \
   X(timeline_semaphore,           "timelineSemaphore",                         \
     "VK_KHR_timeline_semaphore",           VK_API_VERSION_1_2,                \
     q.f12.timelineSemaphore)                                                    \
   X(buffer_device_address,        "bufferDeviceAddress",                       \
     "VK_KHR_buffer_device_address",        VK_API_VERSION_1_2,                \
     q.f12.bufferDeviceAddress)                                                   \
   X(format_feature_flags2,        "formatFeatureFlags2",                       \
     "VK_KHR_format_feature_flags2",        VK_API_VERSION_1_3,                \
     true)                                                                       \
   X(maintenance4,                 "maintenance4",                              \
     "VK_KHR_maintenance4",                 VK_API_VERSION_1_3,                \
     q.f13.maintenance4)                                                         \
   X(descriptor_buffer,            "descriptorBuffer",                          \
     "VK_EXT_descriptor_buffer",            SZ_CORE_NONE,                      \
     q.descriptor_buffer.descriptorBuffer)                                       \
   X(dynamic_rendering_local_read, "dynamicRenderingLocalRead",                 \
     "VK_KHR_dynamic_rendering_local_read", VK_API_VERSION_1_4,                \
     cap_dynamic_rendering_local_read)                                           \
   X(host_image_copy,              "hostImageCopy",                             \
     "VK_EXT_host_image_copy",              VK_API_VERSION_1_4,                \
     cap_host_image_copy)                                                        \
   X(push_descriptor,              "pushDescriptor",                            \
     "VK_KHR_push_descriptor",              VK_API_VERSION_1_4,                \
     cap_push_descriptor)                                                        \
   X(present_wait,                 "presentWait",                               \
     "VK_KHR_present_wait",                 SZ_CORE_NONE,                      \
     q.present_wait.presentWait)                                                 \
   X(graphics_pipeline_library,    "graphicsPipelineLibrary",                   \
     "VK_EXT_graphics_pipeline_library",    SZ_CORE_NONE,                      \
     q.gpl.graphicsPipelineLibrary)

/* Capabilities whose absence only disables an optimisation.
 *
 * X(field, printed name, primary extension, core version, feature expression) */
#define SZ_CAP_OPTIONAL_LIST(X)                                                \
   X(local_read_depth_stencil,     "localReadDepthStencil",                     \
     "VK_KHR_dynamic_rendering_local_read", VK_API_VERSION_1_4,                \
     cap_local_read_depth_stencil)                                               \
   X(local_read_multisampled,      "localReadMultisampled",                     \
     "VK_KHR_dynamic_rendering_local_read", VK_API_VERSION_1_4,                \
     cap_local_read_multisampled)                                                \
   X(msrtss,                       "multisampledRenderToSingleSampled",         \
     "VK_EXT_multisampled_render_to_single_sampled", SZ_CORE_NONE,             \
     q.msrtss.multisampledRenderToSingleSampled)                                 \
   X(shader_object,                "shaderObject",                              \
     "VK_EXT_shader_object",                SZ_CORE_NONE,                      \
     q.shader_object.shaderObject)                                               \
   X(fragment_shading_rate,        "fragmentShadingRate",                       \
     "VK_KHR_fragment_shading_rate",        SZ_CORE_NONE,                      \
     (q.fsr.pipelineFragmentShadingRate ||                                      \
      q.fsr.primitiveFragmentShadingRate ||                                     \
      q.fsr.attachmentFragmentShadingRate))                                     \
   X(shader_image_int64_atomics,   "shaderImageInt64Atomics",                   \
     "VK_EXT_shader_image_atomic_int64",    SZ_CORE_NONE,                      \
     q.image_int64.shaderImageInt64Atomics)

struct sz_caps {
#define SZ_CAP_FIELD(field, name, ext, core, expr) bool field;
   SZ_CAP_REQUIRED_LIST(SZ_CAP_FIELD)
   SZ_CAP_OPTIONAL_LIST(SZ_CAP_FIELD)
#undef SZ_CAP_FIELD

   uint32_t instance_api_version;
   uint32_t device_api_version;

   /* True when the device is genuinely 1.4+, i.e. at least one capability was
    * resolved from core instead of from an extension. */
   bool uses_vk14_core;
};

/* One row of the diagnostic table. */
struct sz_cap_row {
   const char *name;
   const char *extension;
   uint32_t core_version;
   size_t offset;
   bool required;
};

/* The full table, terminated by a row with name == NULL. */
const struct sz_cap_row *sz_caps_table(void);

/* Fills caps from a single vkGetPhysicalDeviceFeatures2 pass.
 *
 * `enum_ext`, `get_features2` and `get_props2` are resolved by the caller and
 * may be NULL when the corresponding query is not needed. */
void sz_caps_init(struct sz_caps *caps,
                  VkPhysicalDevice pdev,
                  uint32_t instance_api_version,
                  PFN_vkEnumerateDeviceExtensionProperties enum_ext,
                  PFN_vkGetPhysicalDeviceFeatures2 get_features2,
                  PFN_vkGetPhysicalDeviceProperties2 get_props2);

unsigned sz_caps_missing_required_count(const struct sz_caps *caps);

/* Writes a comma-separated list of missing required capabilities into buf.
 * Returns buf. */
const char *sz_caps_missing_required_str(const struct sz_caps *caps,
                                         char *buf, size_t buflen);

/* Prints the whole snapshot at info level. No-op unless SZ_DEBUG=info. */
void sz_caps_dump(const struct sz_caps *caps);

#endif /* SZ_CAPS_H */
