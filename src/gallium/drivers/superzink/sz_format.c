/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Format capability queries.
 *
 * Every answer here comes from the Vulkan driver rather than a hand-maintained
 * table, so what we advertise can never drift from what vkCreateImage and
 * vkCreateBuffer will actually accept.
 */

#include "util/macros.h"
#include "util/u_math.h"

#include "vk_format.h"

#include "sz_format.h"
#include "sz_types.h"

/* Gallium passes sample_count == 0 both for "single sample" and for "don't
 * care". Vulkan wants a concrete count, and 1 satisfies both readings. */
VkSampleCountFlagBits
sz_vk_sample_count(unsigned sample_count)
{
   const unsigned clamped = MAX2(sample_count, 1);
   /* Vulkan only defines powers of two. A request for 3x MSAA is answered with
    * the next supported count by the caller, not here. */
   unsigned bit = 1;
   while (bit < clamped && bit < 64)
      bit <<= 1;
   return (VkSampleCountFlagBits)bit;
}

/* Feature bits that describe VkBuffer usage rather than VkImage usage. */
#define SZ_BUFFER_FEATURES                                                     \
   (VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT |                                      \
    VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |                               \
    VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT)

VkImageAspectFlags
sz_format_aspect(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   if (!desc)
      return VK_IMAGE_ASPECT_COLOR_BIT;

   const bool depth = util_format_has_depth(desc);
   const bool stencil = util_format_has_stencil(desc);

   if (depth && stencil)
      return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
   if (depth)
      return VK_IMAGE_ASPECT_DEPTH_BIT;
   if (stencil)
      return VK_IMAGE_ASPECT_STENCIL_BIT;
   return VK_IMAGE_ASPECT_COLOR_BIT;
}

VkFormatFeatureFlags
sz_format_features_from_bindings(unsigned bindings)
{
   VkFormatFeatureFlags f = 0;

   if (bindings & PIPE_BIND_SAMPLER_VIEW)
      f |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

   if (bindings & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_BLENDABLE |
                   PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT))
      f |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;

   if (bindings & PIPE_BIND_DEPTH_STENCIL)
      f |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;

   if (bindings & PIPE_BIND_SHADER_IMAGE)
      f |= VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;

   if (bindings & PIPE_BIND_VERTEX_BUFFER)
      f |= VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT;

   if (bindings & PIPE_BIND_CONSTANT_BUFFER)
      f |= VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT;

   if (bindings & PIPE_BIND_SHADER_BUFFER)
      f |= VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;

   return f;
}

VkImageUsageFlags
sz_image_usage_from_bindings(unsigned bindings, enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   const bool depth_stencil =
      (bindings & PIPE_BIND_DEPTH_STENCIL) ||
      (desc && util_format_has_depth(desc));

   VkImageUsageFlags usage = 0;

   if (bindings & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_BLENDABLE |
                   PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT))
      usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

   if (depth_stencil)
      usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

   if (bindings & PIPE_BIND_SAMPLER_VIEW)
      usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

   if (bindings & PIPE_BIND_SHADER_IMAGE)
      usage |= VK_IMAGE_USAGE_STORAGE_BIT;

   /* Images that leave our process as dmabufs, or that the winsys may resolve,
    * must be legal transfer source and destination for the copy/resolve paths
    * we use instead of rendering to them twice. */
   if (bindings & (PIPE_BIND_SHARED | PIPE_BIND_DISPLAY_TARGET |
                   PIPE_BIND_SCANOUT | PIPE_BIND_LINEAR))
      usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

   if (usage == 0)
      usage = VK_IMAGE_USAGE_SAMPLED_BIT;

   return usage;
}

VkImageCreateFlags
sz_image_create_flags(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
   default:
      return 0;
   }
}

VkImageType
sz_image_type(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_1D:
   case PIPE_TEXTURE_1D_ARRAY:
      return VK_IMAGE_TYPE_1D;
   case PIPE_TEXTURE_3D:
      return VK_IMAGE_TYPE_3D;
   default:
      return VK_IMAGE_TYPE_2D;
   }
}

/* Fetch the driver's feature bits for one tiling mode. */
bool
sz_bindings_require_linear(unsigned bindings)
{
   return !!(bindings & (PIPE_BIND_LINEAR | PIPE_BIND_DISPLAY_TARGET |
                         PIPE_BIND_SCANOUT | PIPE_BIND_SHARED));
}

static VkFormatFeatureFlags
sz_tiling_features(struct sz_screen *screen,
                   VkFormat format,
                   VkImageTiling tiling)
{
   VkFormatProperties props;
   screen->vk.GetPhysicalDeviceFormatProperties(screen->pdev, format, &props);
   return tiling == VK_IMAGE_TILING_LINEAR ? props.linearTilingFeatures
                                           : props.optimalTilingFeatures;
}

static bool
sz_format_supported_buffer(struct sz_screen *screen,
                           enum pipe_format format,
                           unsigned bindings)
{
   const VkFormat vkformat = vk_format_from_pipe_format(format);
   if (vkformat == VK_FORMAT_UNDEFINED)
      return false;

   VkFormatProperties props;
   screen->vk.GetPhysicalDeviceFormatProperties(screen->pdev, vkformat, &props);

   const VkFormatFeatureFlags required =
      sz_format_features_from_bindings(bindings) & SZ_BUFFER_FEATURES;

   return (props.bufferFeatures & required) == required;
}

static bool
sz_format_supported_image(struct sz_screen *screen,
                          enum pipe_format format,
                          enum pipe_texture_target target,
                          unsigned sample_count,
                          unsigned bindings)
{
   const VkFormat vkformat = vk_format_from_pipe_format(format);
   if (vkformat == VK_FORMAT_UNDEFINED)
      return false;

   const VkImageType type = sz_image_type(target);

   /* Scanout, display and shared images leave our process as dmabufs, so the
    * consumer cannot know our optimal layout: they must be linear. A format
    * that RADV can only render to optimally is therefore not displayable. */
   const bool needs_linear = sz_bindings_require_linear(bindings);
   const VkImageTiling tiling =
      needs_linear ? VK_IMAGE_TILING_LINEAR : VK_IMAGE_TILING_OPTIMAL;

   /* Linear tiling is 2D-only. */
   if (tiling == VK_IMAGE_TILING_LINEAR && type != VK_IMAGE_TYPE_2D)
      return false;

   /* 1D/3D images are never cube-compatible, and 3D cannot be an attachment in
    * Gallium's model. */
   const VkImageCreateFlags flags =
      type == VK_IMAGE_TYPE_2D ? sz_image_create_flags(target) : 0;
   VkImageUsageFlags usage = sz_image_usage_from_bindings(bindings, format);
   if (type == VK_IMAGE_TYPE_3D) {
      usage &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      if (usage == 0)
         usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   }

   /* Answer both questions with one call each:
    *   - is (format, type, tiling, usage, flags) a creatable image at all,
    *   - and does it survive at the requested sample count. */
   VkImageFormatProperties2 ifp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
   };
   const VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = vkformat,
      .type = type,
      .tiling = tiling,
      .usage = usage,
      .flags = flags,
   };

   if (screen->vk.GetPhysicalDeviceImageFormatProperties2(screen->pdev, &info,
                                                          &ifp) != VK_SUCCESS)
      return false;

   const unsigned samples = sz_vk_sample_count(sample_count);
   if (samples > 1 &&
       !(ifp.imageFormatProperties.sampleCounts & (1u << (samples - 1))))
      return false;

   /* sampleCounts only covers image creation. The feature bits are what decides
    * whether we can actually attach or sample it, and they differ per tiling. */
   const VkFormatFeatureFlags features =
      sz_tiling_features(screen, vkformat, tiling);
   const VkFormatFeatureFlags required =
      sz_format_features_from_bindings(bindings) & ~SZ_BUFFER_FEATURES;

   return (features & required) == required;
}

bool
sz_format_supported(struct sz_screen *screen,
                    enum pipe_format format,
                    enum pipe_texture_target target,
                    unsigned sample_count,
                    unsigned storage_sample_count,
                    unsigned bindings)
{
   if (format == PIPE_FORMAT_NONE)
      return bindings == 0;

   const bool buffer_binding =
      bindings & (PIPE_BIND_VERTEX_BUFFER | PIPE_BIND_INDEX_BUFFER |
                  PIPE_BIND_CONSTANT_BUFFER | PIPE_BIND_SHADER_BUFFER |
                  PIPE_BIND_STREAM_OUTPUT | PIPE_BIND_COMMAND_ARGS_BUFFER);
   const bool image_binding =
      bindings & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_DEPTH_STENCIL |
                  PIPE_BIND_SAMPLER_VIEW | PIPE_BIND_SHADER_IMAGE |
                  PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT |
                  PIPE_BIND_SHARED | PIPE_BIND_LINEAR | PIPE_BIND_BLENDABLE);

   /* A true image-only binding is decided by the image path; a buffer binding by
    * the buffer path. Callers that ask for both must satisfy both. */
   if (image_binding &&
       !sz_format_supported_image(screen, format, target, sample_count, bindings))
      return false;

   if (buffer_binding &&
       !sz_format_supported_buffer(screen, format, bindings))
      return false;

   if (buffer_binding || image_binding)
      return true;

   /* No bindings at all: the format merely has to exist. */
   return vk_format_from_pipe_format(format) != VK_FORMAT_UNDEFINED;
}
