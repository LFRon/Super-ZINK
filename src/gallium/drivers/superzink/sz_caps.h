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
   /* Constant buffers are declared as they arrive from Mesa -- a plain word     \
    * array with a 4 byte stride -- so the layout needs the scalar rules that    \
    * this feature legalises. Without it every UBO would have to be re-declared  \
    * with std140 padding, and every load offset recomputed. */                  \
   X(uniform_buffer_standard_layout, "uniformBufferStandardLayout",             \
     "VK_KHR_uniform_buffer_standard_layout", VK_API_VERSION_1_2,              \
     q.f12.uniformBufferStandardLayout)                                           \
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
     q.image_int64.shaderImageInt64Atomics)                                      \
   X(multi_draw,                   "multiDraw",                                  \
     "VK_EXT_multi_draw",                   SZ_CORE_NONE,                      \
     q.multi_draw.multiDraw)                                                    \
   X(transform_feedback,           "transformFeedback",                          \
     "VK_EXT_transform_feedback",           SZ_CORE_NONE,                      \
     q.transform_feedback.transformFeedback)                                    \
   /* Without this, Vulkan forbids primitive restart on the five list topologies   \
    * (VUID-VkPipelineInputAssemblyStateCreateInfo-topology-06252), while GL        \
    * applies glPrimitiveRestartIndex to them like any other topology. */          \
   X(list_restart,                 "primitiveTopologyListRestart",               \
     "VK_EXT_primitive_topology_list_restart", SZ_CORE_NONE,                   \
     q.list_restart.primitiveTopologyListRestart)

/* Every VkPhysicalDevice*Features structure Super-ZINK ever chains, in one
 * place. Querying (sz_caps_init) and enabling (sz_device_create) both go
 * through this, so the two can never drift: a feature that the caps table
 * reports as "ok" but that device creation forgets to enable would otherwise
 * fail silently at the first vkCmd* call that relies on it.
 *
 * The members are not pre-linked. sz_vk_features_prepare() only stamps sType;
 * sz_vk_features_link() chains the subset the device actually offers. */
struct sz_vk_features {
   /* Not chained: VkPhysicalDeviceFeatures2 carries this one directly. */
   VkPhysicalDeviceFeatures core;

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
   VkPhysicalDeviceMultiDrawFeaturesEXT multi_draw;
   VkPhysicalDevicePrimitiveTopologyListRestartFeaturesEXT list_restart;
   VkPhysicalDeviceTransformFeedbackFeaturesEXT transform_feedback;

   /* Not a feature: VkPhysicalDeviceVulkan14Properties carries the two
    * local-read attachment bits, and vk.xml gives them no extension-side
    * equivalent, so they only exist on a 1.4 device. Kept here because it is
    * queried in the same pass. */
   VkPhysicalDeviceVulkan14Properties props14;
};

/* Which of the optional extension structs to chain. Filled from the advertised
 * extension list when querying, and from the capability snapshot when enabling;
 * the same flags therefore mean "can ask" and "may enable". */
struct sz_vk_feature_selection {
   bool local_read;
   bool host_image_copy;
   bool descriptor_buffer;
   bool present_wait;
   bool graphics_pipeline_library;
   bool msrtss;
   bool shader_object;
   bool fragment_shading_rate;
   bool shader_image_int64;
   bool multi_draw;
   bool list_restart;
   bool transform_feedback;
};

void sz_vk_features_prepare(struct sz_vk_features *f, uint32_t api_version);
void sz_vk_features_link(struct sz_vk_features *f, uint32_t api_version,
                         const struct sz_vk_feature_selection *sel,
                         VkPhysicalDeviceFeatures2 *features2);

/* Capabilities that are nothing but an extension name: vk.xml marks them
 * "nofeatures", there is no VkPhysicalDevice*Features member to test.
 *
 * They stay out of the two lists above because those lists' contract is
 * "feature bit && (core || extension)". Writing a placeholder `true` for the
 * feature expression would report a capability the device may not have, which is
 * exactly the failure ADR-001 exists to prevent. Here the rule collapses to
 * "the extension is advertised", and the version plays no part because none of
 * these was ever promoted to core.
 *
 * X(field, extension) */
#define SZ_CAP_EXT_ONLY_LIST(X)                                                \
   X(external_memory_fd,          "VK_KHR_external_memory_fd")                  \
   X(external_memory_dma_buf,     "VK_EXT_external_memory_dma_buf")             \
   X(drm_format_modifier,         "VK_EXT_image_drm_format_modifier")

struct sz_caps {
#define SZ_CAP_FIELD(field, name, ext, core, expr) bool field;
   SZ_CAP_REQUIRED_LIST(SZ_CAP_FIELD)
   SZ_CAP_OPTIONAL_LIST(SZ_CAP_FIELD)
#undef SZ_CAP_FIELD
#define SZ_CAP_EXT_FIELD(field, ext) bool field;
   SZ_CAP_EXT_ONLY_LIST(SZ_CAP_EXT_FIELD)
#undef SZ_CAP_EXT_FIELD

   /* The 1.0 feature set verbatim, so device creation can gate on it instead
    * of asking the driver a second, possibly different, question. */
   VkPhysicalDeviceFeatures core_features;

   /* VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors, i.e. how
    * many writes one vkCmdPushDescriptorSet call may carry. Zero when the
    * device does not report the property. */
   uint32_t max_push_descriptors;

   /* VkPhysicalDeviceMultiDrawPropertiesEXT::maxMultiDrawCount, or zero when
    * the extension is absent. The draw stream never batches more than this. */
   uint32_t max_multi_draw_count;

   /* VkPhysicalDeviceTransformFeedbackPropertiesEXT::maxTransformFeedbackBuffers,
    * or zero when the extension is absent. GL only needs four. */
   uint32_t max_transform_feedback_buffers;

   /* VK_EXT_descriptor_buffer properties. The descriptor sizes are properties
    * because the in-memory layout of a descriptor is implementation-defined:
    * vkGetDescriptorEXT writes that many bytes and nothing else knows how many
    * it will be. The alignment applies to every bound buffer offset. */
   uint64_t descriptor_buffer_offset_alignment;
   size_t descriptor_size_uniform_buffer;
   size_t descriptor_size_combined_image_sampler;

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

/* One row of the extension-only table: no core version, never required. */
struct sz_cap_ext_row {
   const char *name;
   const char *extension;
   size_t offset;
};

/* The full table, terminated by a row with name == NULL. */
const struct sz_cap_row *sz_caps_table(void);

/* The extension-only table, terminated by a row with name == NULL. */
const struct sz_cap_ext_row *sz_caps_ext_table(void);

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
