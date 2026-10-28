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
#include "sz_compiler.h"
#include "sz_context.h"
#include "sz_descriptor.h"
#include "sz_pipeline.h"
#include "sz_device.h"
#include "sz_dmabuf.h"
#include "sz_format.h"
#include "sz_resource.h"
#include "sz_types.h"
#include "sz_debug.h"

#include "util/u_debug.h"
#include "util/u_dl.h"
#include "util/u_memory.h"
#include "util/u_screen.h"
#include "util/os_file.h"
#include "util/os_time.h"

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

/* VK_KHR_surface is not used for anything -- presentation goes through the
 * winsys -- but VK_KHR_swapchain names it as a dependency, and swapchain is only
 * enabled because VK_KHR_present_wait needs it (see sz_ext_deps). A device
 * extension list that names a dependent extension without its dependency is a
 * validation error (VUID 01387), so it is enabled here. It is asked for rather
 * than assumed: a headless setup can legitimately lack it, and instance creation
 * would fail with VK_ERROR_EXTENSION_NOT_PRESENT instead of falling back. */
#define SZ_SURFACE_EXTENSION VK_KHR_SURFACE_EXTENSION_NAME

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
   /* Optional: only used to decide whether an optional instance extension may be
    * asked for. A loader that does not export it answers "not available". */
   screen->vk.EnumerateInstanceExtensionProperties =
      (PFN_vkEnumerateInstanceExtensionProperties)
         screen->vk_GetInstanceProcAddr(VK_NULL_HANDLE,
                                        "vkEnumerateInstanceExtensionProperties");
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

/* True when the loader advertises `name` as an instance extension. */
static bool
sz_instance_extension_available(struct sz_screen *screen, const char *name)
{
   if (!screen->vk.EnumerateInstanceExtensionProperties)
      return false;

   uint32_t count = 0;
   if (screen->vk.EnumerateInstanceExtensionProperties(NULL, &count, NULL) !=
           VK_SUCCESS ||
       count == 0)
      return false;

   VkExtensionProperties *props = CALLOC(count, sizeof(*props));
   if (!props)
      return false;

   bool found = false;
   if (screen->vk.EnumerateInstanceExtensionProperties(NULL, &count, props) ==
       VK_SUCCESS) {
      for (uint32_t i = 0; i < count; i++) {
         if (strcmp(props[i].extensionName, name) == 0) {
            found = true;
            break;
         }
      }
   }
   FREE(props);
   return found;
}

static VkResult
sz_create_instance(struct sz_screen *screen)
{
   const char *exts[2] = { SZ_INSTANCE_EXTENSIONS };
   unsigned num_exts = 1;
   if (sz_instance_extension_available(screen, SZ_SURFACE_EXTENSION))
      exts[num_exts++] = SZ_SURFACE_EXTENSION;

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
      .enabledExtensionCount = num_exts,
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

static uint64_t
sz_screen_get_timestamp(struct pipe_screen *pscreen)
{
   return os_time_get_nano();
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
   struct sz_screen *screen = sz_screen(pscreen);

   /* One context at a time: the command pool and timeline live on the screen, so
    * a second context would share a submission timeline it does not expect to.
    * The state tracker only ever asks for one per pipe_screen, but a
    * multithreaded frontend could, so refuse rather than corrupt. */
   if (screen->context_alive) {
      mesa_loge("SZ: only one context per screen is supported so far");
      return NULL;
   }

   struct pipe_context *pipe = sz_context_create(screen, flags);
   if (!pipe) {
      mesa_loge("SZ: context creation failed");
      return NULL;
   }

   screen->context_alive = true;
   return pipe;
}

static void
sz_screen_destroy(struct pipe_screen *pscreen)
{
   struct sz_screen *screen = sz_screen(pscreen);

   sz_queue_state_finish(&screen->batch_state);
   /* The context destroyed the queue idle, so nothing can be reading a retired
    * resource any more. Tear them down while the device is still alive. */
   sz_resource_reclaim_all(screen);
   sz_pipeline_screen_finish(screen);
   sz_descriptor_screen_finish(screen);
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

   /* Which depth/stencil formats the device can actually back, so that the
    * pipe-format to VkFormat mapping is decided once rather than per image. */
   sz_format_probe(screen);

   sz_bo_init(screen);
   sz_resource_screen_init(screen);
   sz_screen_fence_init(screen);

   /* Descriptor set layouts are screen-owned: they are immutable once created,
    * every pipeline references them, and a per-context copy would only add ways
    * for the two to disagree with the shaders' bindings. */
   if (!sz_descriptor_screen_init(screen, screen->caps.max_push_descriptors))
      goto fail;

   /* The pipeline layout and the VkPipelineCache depend only on the descriptor
    * set layout, so they live on the screen (ADR-005c). */
   if (!sz_pipeline_screen_init(screen))
      goto fail;

   if (!sz_queue_state_init(&screen->batch_state, screen,
                            screen->queues[SZ_QUEUE_GRAPHICS].family)) {
      mesa_loge("SZ: could not create the command submission state");
      goto fail;
   }

   /* pipe_screen base */
   screen->base.destroy = sz_screen_destroy;
   screen->base.get_name = sz_screen_get_name;
   screen->base.get_vendor = sz_screen_get_vendor;
   screen->base.get_device_vendor = sz_screen_get_device_vendor;
   screen->base.is_format_supported = sz_screen_is_format_supported;
   screen->base.query_memory_info = sz_screen_query_memory_info;
   screen->base.context_create = sz_screen_context_create;
   /* GL_TIMESTAMP and GL_TIMESTAMP queries ask here first and fall back to the
    * per-context hook only if the screen declines. With neither set, the
    * fallback calls a NULL pointer. The CPU clock is not the GPU clock, but it is
    * monotonic and it answers the only question a GL app asks with it. */
   screen->base.get_timestamp = sz_screen_get_timestamp;

   /* Does this device have 1D depth/stencil images at all? RADV does not: every
    * 1D depth format answers VK_ERROR_FORMAT_NOT_SUPPORTED. GL has 1D depth and
    * 1D shadow textures, so the shape is aliased rather than the feature
    * reported missing -- see sz_image_vk_type(). This is the same trade ZINK
    * makes, and for the same reason. */
   {
      const VkPhysicalDeviceImageFormatInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
         .format = VK_FORMAT_D32_SFLOAT,
         .type = VK_IMAGE_TYPE_1D,
         .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      };
      VkImageFormatProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      };
      screen->need_2d_zs =
         screen->vk.GetPhysicalDeviceImageFormatProperties2(screen->pdev, &info,
                                                            &props) != VK_SUCCESS;
   }

   u_init_pipe_screen_caps(&screen->base, !screen->is_cpu);

   /* Every driver must answer this: st/mesa asserts on it in debug builds and
    * dereferences the NULL in release builds, which is a crash inside the GLSL
    * linker (nir_inline_functions) with no hint that the driver is at fault. */
   for (unsigned i = 0; i < MESA_SHADER_STAGES; i++)
      screen->base.nir_options[i] = sz_nir_options(i);

   /* Capabilities stay minimal until there is a compiler: without GLSL support
    * the state tracker would refuse to build a context and the failure would
    * look like a driver bug rather than a missing feature. */
   {
      struct pipe_caps *caps = (struct pipe_caps *)&screen->base.caps;
      caps->glsl_feature_level = 400;
      caps->glsl_feature_level_compatibility = 400;
      caps->max_texture_2d_size = 16384;
      caps->max_texture_cube_levels = 15;
      caps->max_texture_3d_levels = 12;
      caps->max_texture_array_layers = 2048;
      caps->max_render_targets = 8;
      caps->max_dual_source_render_targets = 1;
      /* Disable the "depth must be as wide as colour" restriction the DRI config
       * generator applies by default. Every hardware driver sets this: GLX
       * clients ask for a 16-bit depth buffer while the window is 32-bit colour,
       * and with the restriction on, the only depth configs generated are the
       * ones that match the colour width, so glXChooseVisual finds nothing. */
      caps->mixed_color_depth_bits = true;

      /* Timer queries: vkCmdWriteTimestamp2 exists whenever the device reports a
       * valid-bit count, and the query layer checks that before creating a pool. */
      caps->query_timestamp = true;
      caps->query_time_elapsed = true;

      /* Per-instance vertex attributes: the binding rate is Vulkan's, and
       * glVertexAttribDivisor's non-zero divisors all collapse onto it. A divisor
       * above one therefore fetches per instance rather than every Nth, which no
       * Vulkan device can express. */
      caps->vertex_element_instance_divisor = true;

      /* Swizzled sampler views: pipe_sampler_view carries the four component
       * swizzles and the view is created with them, so nothing else is needed
       * for GL's texture swizzle to work. */
      caps->texture_swizzle = true;

      /* textureLod in a shader maps onto OpImageQueryLod, which is what
       * GL_ARB_texture_query_lod exposes. The instruction is already emitted for
       * nir_texop_lod, so the capability was simply never claimed. */
      /* Five capabilities whose mechanism has been in place all along:
       *
       *  - the sample count of an image is nir_texop_texture_samples, already
       *    emitted (query_levels needs no capability of its own -- st grants it
       *    with GLSL 1.30);
       *  - the fine and coarse derivative forms are OpDPdxFine and friends, with
       *    the DerivativeControl capability declared when they are used;
       *  - glPolygonOffsetClamp's third value is Vulkan's depthBiasClamp, which
       *    the pipeline already passes through;
       *  - gl_CullDistance is the CullDistance builtin the translator already
       *    declares. */
      caps->texture_query_lod = true;
      caps->texture_query_samples = true;
      caps->fs_fine_derivative = true;
      caps->polygon_offset_clamp = true;
      /* Not claimed, although the mechanism looks present:
       *  - caps->cull_distance makes st keep gl_CullDistance in the shader, and
       *    the builtin is declared, but piglit's arb_cull_distance renders
       *    unclipped (8 failures), so something in the chain is still wrong;
       *  - caps->sampler_view_target makes ARB_texture_view's format
       *    reinterpretation legal, and viewing an RGBA8 texture as RGBA32F needs
       *    compatibility rules this driver has none of (536 failures). */
      /* Mirrored clamp to edge is a Vulkan address mode, gated on the device
       * feature of the same name; the sampler already maps PIPE_TEX_WRAP_MIRROR_*
       * onto it. */
      caps->texture_mirror_clamp_to_edge =
         screen->caps.sampler_mirror_clamp_to_edge;
      /* Anisotropic filtering, ditto: the sampler already carries
       * maxAnisotropy from pipe_sampler_state, and both the EXT and the ARB
       * spelling of the extension follow from these two fields. */
      caps->anisotropic_filter = screen->caps.core_features.samplerAnisotropy;
      caps->max_texture_anisotropy =
         screen->caps.core_features.samplerAnisotropy ? 16.0f : 0.0f;

      /* GL_ARB_draw_indirect: the counts come out of a buffer through
       * vkCmdDrawIndirect, which is core Vulkan. Two device features decide
       * whether that is honest -- drawIndirectFirstInstance, because GL allows a
       * first instance in the argument buffer, and multiDrawIndirect, because one
       * call can describe several draws. */
      /* Buffer textures. The descriptor is a VkBufferView (pool path) or an
       * address+format (descriptor-buffer path), the SPIR-V side is OpTypeImage
       * with Dim Buffer, and sz_format.c already asks the device whether each
       * format can be a texel buffer. The offset alignment glTexBufferRange
       * needs is the device's minTexelBufferOffsetAlignment, reported below. */
      /* Not claimed although Vulkan's robustBufferAccess would back it:
       * piglit's arb_robustness group turns a clean skip into a hang on
       * arb_robustness_client-mem-bounds, which reads out of bounds from client
       * memory -- a path this driver does not implement at all. */

      caps->texture_buffer_objects = true;
      caps->max_texel_buffer_elements = 65536;
      caps->texture_buffer_offset_alignment = 256;

      /* Not claimed: per-sample shading needs a multisampled *renderbuffer* to
       * resolve into its texture, and that resolve arrives as a blit from one
       * multisampled image to another -- which Vulkan cannot do at all, and
       * which GL rejects too, so the destination's sample count is being
       * reported wrong somewhere. The four sample builtins translate (see
       * sz_load_sample_position() and friends); the capability waits for that
       * resolve to be found and fixed. */
      caps->draw_indirect =
         screen->caps.core_features.drawIndirectFirstInstance &&
         screen->caps.core_features.multiDrawIndirect;

      /* No fixed-function clip planes: Vulkan clips on the ClipDistance the
       * shader writes, and nothing else. Reporting zero is what makes the state
       * tracker lower an application's clip planes -- and gl_ClipVertex, which
       * only exists to feed them -- into ClipDistance writes, which this driver
       * translates. Reporting the u_screen default of one is worse than saying
       * nothing: st concludes the hardware provides them, leaves gl_ClipVertex in
       * the shader, and every compat shader that uses it fails to translate.
       * MaxClipPlanes stays at GL's 8 either way (st only overrides it above 1). */
      caps->clip_planes = 0;

      /* Separate RGB/alpha blend equations are ordinary Vulkan state: every
       * attachment carries srcColorBlendFactor/colorBlendOp next to the alpha
       * pair, which is exactly what Gallium's blend state asks for. st/mesa
       * gates GL 2.0 on this cap, so leaving it unset caps the reported version
       * at 1.5 and every consumer that needs shaders refuses to start. */
      caps->blend_equation_separate = true;
      /* Same story for per-target blend: the pipeline key keeps the whole blend
       * state and each attachment gets its own VkPipelineColorBlendAttachmentState,
       * whether or not independent blending is enabled. */
      caps->indep_blend_enable = true;
      caps->indep_blend_func = true;

      /* The primitives Vulkan can rasterise, which is fewer than GL can emit.
       * A mode left out of this mask is converted by the primitive converter
       * reaches the driver, which is the only correct answer for QUADS,
       * QUAD_STRIP and POLYGON -- there is no VkPrimitiveTopology for them, and
       * quietly drawing them as triangle lists produces a garbled picture rather
       * than an error. u_screen defaults the mask to "all", so leaving it alone
       * means claiming support the driver does not have. */
      caps->supported_prim_modes =
         BITFIELD_BIT(MESA_PRIM_POINTS) |
         BITFIELD_BIT(MESA_PRIM_LINES) |
         BITFIELD_BIT(MESA_PRIM_LINE_STRIP) |
         BITFIELD_BIT(MESA_PRIM_TRIANGLES) |
         BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP) |
         BITFIELD_BIT(MESA_PRIM_LINES_ADJACENCY) |
         BITFIELD_BIT(MESA_PRIM_LINE_STRIP_ADJACENCY) |
         BITFIELD_BIT(MESA_PRIM_TRIANGLES_ADJACENCY) |
         BITFIELD_BIT(MESA_PRIM_TRIANGLE_STRIP_ADJACENCY);
      /* Vulkan's primitive restart is the largest index of the index type,
       * which is GL's GL_PRIMITIVE_RESTART_FIXED_INDEX exactly. Every mode the
       * pipeline can build restarts the same way, so the mask is the one above. */
      caps->supported_prim_modes_with_restart = caps->supported_prim_modes;
      /* An index the application chose through glPrimitiveRestartIndex is not
       * expressible in Vulkan at all. Declaring the emulation makes cso's vbuf
       * rewrite those indices to the fixed one on the way in, which is what
       * turns NV_primitive_restart into a capability rather than a lie. */
      caps->primitive_restart_fixed_index = true;
      caps->emulate_nonfixed_primitive_restart = true;
      caps->primitive_restart = true;

      /* Constant buffers arrive as real buffers rather than client memory, so
       * the driver never has to invent an upload path for them. */
      caps->prefer_real_buffer_in_constbuf0 = true;

      /* Capabilities whose Gallium meaning is already exactly what Vulkan does:
       *
       *  - Vulkan never clamps vertex colours, so unclamped is the only mode.
       *  - Dynamic rendering has no requirement that attachments share a size or
       *    format, which is the restriction this cap says the driver lacks.
       *  - Cube map sampling is always seamless.
       *  - depthClampEnable is translated per pipeline, and it is the bit this
       *    cap is about.
       *  - textureLod in a fragment shader lowers to nir_texop_txl, which the
       *    translator already emits as OpImageSample*ExplicitLod. */
      caps->vertex_color_unclamped = true;
      caps->mixed_framebuffer_sizes = true;
      caps->seamless_cube_map = true;
      caps->depth_clip_disable = true;
      caps->fragment_shader_texture_lod = true;

      /* gl_FragCoord's conventions, as Vulkan defines them: the origin is the
       * upper-left corner of the framebuffer and pixel centres are at half
       * integers (0.5, 1.5, ...). Saying so is what makes the frontend leave
       * gl_FragCoord alone; a driver that reports neither convention gets the
       * "add half a pixel" adjustment instead, and since these two flags have no
       * default, the pixel-centre selection below them is unreachable and the
       * shader gets whatever the compiler decided that means -- in practice a
       * gl_FragCoord biased by half a pixel in both axes.
       *
       * The viewport is emitted with a positive height (see sz_emit_viewport),
       * so the framebuffer keeps Vulkan's own orientation and the y flip that GL
       * expects is done here, by the frontend, through
       * gl_FbWposYTransform. */
      caps->fs_coord_origin_upper_left = true;
      caps->fs_coord_pixel_center_half_integer = true;

      /* gl_InstanceID reads Vulkan's InstanceIndex, which the draw path already
       * feeds with the instance count and the first instance. GL's own
       * gl_InstanceID is 0-based, so this is exact for every draw whose base
       * instance is zero -- which is every draw before GL 4.2's
       * ARB_base_instance, and there the shader subtracts it (see
       * sz_draw_append). */
      caps->vs_instanceid = true;

      /* GL_ARB_uniform_buffer_object is derived from this: st_extensions treats
       * a maximum uniform block below 16 KiB as "no uniform buffers at all", and
       * zero is what an unset cap reads as. The translator's block type holds
       * exactly SZ_MAX_CONSTBUF_WORDS 32-bit words, so the two must agree. */
      caps->max_constant_buffer_size = SZ_MAX_CONSTBUF_WORDS * 4;

      /* GL_ARB_texture_multisample: a multisampled image is an ordinary Vulkan
       * image with samples > 1, and reading one is OpImageFetch with a Sample
       * operand, which the translator already emits for nir_texop_txf_ms. The
       * format layer decides per format whether the device can create and
       * sample at that count, so the cap is a statement about the mechanism,
       * not about any particular format. */
      caps->texture_multisample =
         screen->props.limits.framebufferColorSampleCounts != 0;

      /* Occlusion queries map straight onto VK_QUERY_TYPE_OCCLUSION.
       * Conditional rendering is evaluated on the CPU when
       * glBeginConditionalRender is called, which the GL specification allows
       * and which works on devices without VK_EXT_conditional_rendering. */
      caps->occlusion_query = true;
      caps->conditional_render = true;

      /* GL 3.0's GL_EXT_transform_feedback is gated on this cap. Four buffers is
       * what the extension guarantees and what GL exposes. */
      caps->max_stream_output_buffers =
         screen->caps.transform_feedback
            ? MIN2(4u, screen->caps.max_transform_feedback_buffers)
            : 0;

      /* GL 3.2 core requires a geometry stage. Vulkan's own floors are far
       * above these numbers, so they are GL's minimums rather than a device
       * query -- the stage works or it does not, and it does. */
      caps->max_geometry_output_vertices = 256;
      caps->max_geometry_total_output_components = 1024;
      caps->max_gs_invocations = 32;
   }

   /* Varying slots, in vec4s. The translator places varyings at dense locations
    * from zero, so this is also the number of locations it can use -- and the
    * smallest of the three limits that have to agree: what the vertex stage may
    * write, what a geometry stage may read, and what the fragment stage may read.
    * A device can report 128 components of vertex output and only 64 of geometry
    * input; advertising the larger one lets the GLSL linker pack a shader that
    * then fails to link (or, worse, to build a pipeline).
    *
    * GL 3.2 requires 60 components, which Vulkan's 64-component floor covers. */
   ((struct pipe_caps *)&screen->base.caps)->max_varyings =
      MIN2(MIN2(MAX_VARYING, screen->props.limits.maxVertexOutputComponents / 4),
           MIN2(screen->props.limits.maxGeometryInputComponents / 4,
                screen->props.limits.maxFragmentInputComponents / 4));

   /* Shader limits. st_extensions.c reads max_instructions == 0 as "this stage
    * is unsupported" and zeroes every other limit, so a stage that links at all
    * needs it set. max_const_buffer0_size is what becomes both
    * GL_MAX_*_UNIFORM_COMPONENTS and GL_MAX_UNIFORM_BLOCK_SIZE, which is why it
    * is spelled in bytes here and in vec4s there.
    *
    * The three stages the translator implements are advertised: a non-zero
    * max_instructions for tessellation or compute makes the frontend expose
    * them and makes cso_context call their bind_* entry points, and neither has
    * a translation yet. */
   const mesa_shader_stage supported_stages[] = {MESA_SHADER_VERTEX,
                                                 MESA_SHADER_GEOMETRY,
                                                 MESA_SHADER_FRAGMENT};
   for (unsigned i = 0; i < ARRAY_SIZE(supported_stages); i++) {
      /* shader_caps is const in pipe_screen because the state tracker only
       * reads it; drivers fill it exactly once, here. */
      struct pipe_shader_caps *sc =
         (struct pipe_shader_caps *)&screen->base.shader_caps[supported_stages[i]];

      sc->max_instructions = 65535;
      sc->max_alu_instructions = 65535;
      sc->max_tex_instructions = 65535;
      sc->max_tex_indirections = 65535;
      sc->max_control_flow_depth = 32;
      /* Inputs and outputs are counted in vec4s by gl_program_constants: for the
       * vertex stage the inputs are vertex attributes (16 locations, the GL
       * minimum) and for the geometry stage they are the previous stage's
       * outputs. The output count is the location space the translator places
       * varyings in -- dense from zero, and no wider than the device allows. */
      const VkPhysicalDeviceLimits *lim = &screen->props.limits;
      const unsigned varyings = screen->base.caps.max_varyings;
      switch (supported_stages[i]) {
      case MESA_SHADER_VERTEX:
         sc->max_inputs = 16;
         sc->max_outputs = varyings;
         break;
      case MESA_SHADER_GEOMETRY:
         sc->max_inputs = varyings;
         /* A geometry shader writes its own gl_Position on top of the varyings
          * it passes on, and GL 3.2 puts MAX_GEOMETRY_OUTPUT_COMPONENTS at 128 --
          * 32 vec4s. Reporting 16 makes the GLSL constant
          * gl_MaxGeometryOutputComponents 64, below what the spec requires. */
         sc->max_outputs =
            MIN2(MAX_VARYING, lim->maxGeometryOutputComponents / 4);
         break;
      default:
         sc->max_inputs = MIN2(MAX_VARYING, lim->maxFragmentInputComponents / 4);
         sc->max_outputs = 16;
         break;
      }
      sc->max_temps = 64;
      /* Exactly the size of the array the translator declares inside a uniform
       * block, so no shader can ask for a block the block type cannot hold. */
      sc->max_const_buffer0_size = SZ_MAX_CONSTBUF_WORDS * 4;
      sc->max_const_buffers = SZ_MAX_CONST_BUFFERS;
      /* Must match the number of sampler bindings the descriptor layout
       * reserves per stage, or the frontend will hand over a unit the layout
       * cannot describe. */
      sc->max_texture_samplers = SZ_MAX_SAMPLERS;
      sc->max_sampler_views = SZ_MAX_SAMPLERS;
      sc->max_shader_buffers = 8;
      sc->max_shader_images = 8;
      sc->supported_irs = 1u << PIPE_SHADER_IR_NIR;
      sc->cont_supported = true;
      sc->indirect_const_addr = true;
   }
   ((struct pipe_shader_caps *)&screen->base.shader_caps[MESA_SHADER_FRAGMENT])
      ->max_inputs = 32;

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
         goto fail_submit;
      }
      if (sz_context_selftest(screen) != 0) {
         mesa_loge("SZ: SZ_DEBUG=selftest failed; refusing to publish a screen "
                   "whose command submission does not round-trip");
         goto fail_submit;
      }
      if (sz_render_selftest(screen) != 0) {
         mesa_loge("SZ: SZ_DEBUG=selftest failed; refusing to publish a screen "
                   "that cannot render a triangle");
         goto fail_submit;
      }
      if (sz_dmabuf_selftest(screen) != 0) {
         mesa_loge("SZ: SZ_DEBUG=selftest failed; refusing to publish a screen "
                   "that cannot hand its own frames to someone else");
         goto fail_submit;
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

fail_submit:
   sz_queue_state_finish(&screen->batch_state);
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
