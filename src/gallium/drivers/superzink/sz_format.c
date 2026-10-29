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

#include "util/log.h"
#include "util/macros.h"
#include "util/u_math.h"

#include "vk_format.h"

#include "sz_format.h"
#include "sz_types.h"

/* Whether a format can be used as a depth/stencil attachment at all. */
static bool
sz_format_has_depth_features(struct sz_screen *screen, VkFormat format)
{
   VkFormatProperties props;
   screen->vk.GetPhysicalDeviceFormatProperties(screen->pdev, format, &props);
   return ((props.linearTilingFeatures | props.optimalTilingFeatures) &
           VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
}

/* Whether the device can store into an image of this format from a shader. */
static bool
sz_format_can_store(struct sz_screen *screen, VkFormat format)
{
   if (format == VK_FORMAT_UNDEFINED)
      return false;

   VkFormatProperties props;
   screen->vk.GetPhysicalDeviceFormatProperties(screen->pdev, format, &props);
   return (props.optimalTilingFeatures &
           VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0;
}

void
sz_format_probe(struct sz_screen *screen)
{
   screen->have_d24_unorm_s8_uint =
      sz_format_has_depth_features(screen, VK_FORMAT_D24_UNORM_S8_UINT);
   screen->have_x8_d24_unorm_pack32 =
      sz_format_has_depth_features(screen, VK_FORMAT_X8_D24_UNORM_PACK32);
   screen->have_d32_sfloat =
      sz_format_has_depth_features(screen, VK_FORMAT_D32_SFLOAT);
   screen->have_d32_sfloat_s8_uint =
      sz_format_has_depth_features(screen, VK_FORMAT_D32_SFLOAT_S8_UINT);
}

void
sz_format_debug_print(struct sz_screen *screen)
{
   static const enum pipe_format probe[] = {
      PIPE_FORMAT_X24S8_UINT, PIPE_FORMAT_S8X24_UINT, PIPE_FORMAT_S8_UINT,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(probe); i++)
      mesa_logi("SZ: fmt %s -> %d (sampled %d)",
                util_format_name(probe[i]),
                (int)sz_vk_format(screen, probe[i]),
                sz_format_supported(screen, probe[i], PIPE_TEXTURE_2D, 0, 0,
                                    PIPE_BIND_DEPTH_STENCIL |
                                       PIPE_BIND_SAMPLER_VIEW));
}

/* Vulkan has no three-channel 8-, 16- or 32-bit format, and no 10/10/10/2 or
 * 5/5/5/1 format without alpha. Mesa spells those GL formats with an "X"
 * channel, whose contents GL never exposes: a sampler view of one swizzles that
 * channel to 1. The natural Vulkan spelling is therefore the matching "A"
 * format, and every one of these is the same texel size as what it replaces.
 *
 * Without the substitution the state tracker's search for a GL_RGB8 backing
 * format finds none of the X8 spellings, falls through to RGB565, and every
 * RGB8 texture is quietly stored at 16 bits -- arb_texture_view's
 * rendering-formats notices immediately. */
static enum pipe_format
sz_format_emulate_x8(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return PIPE_FORMAT_B8G8R8A8_UNORM;
   case PIPE_FORMAT_B8G8R8X8_SRGB:
      return PIPE_FORMAT_B8G8R8A8_SRGB;
   case PIPE_FORMAT_R8G8B8X8_SRGB:
      return PIPE_FORMAT_R8G8B8A8_SRGB;
   case PIPE_FORMAT_R8G8B8X8_SINT:
      return PIPE_FORMAT_R8G8B8A8_SINT;
   case PIPE_FORMAT_R8G8B8X8_SNORM:
      return PIPE_FORMAT_R8G8B8A8_SNORM;
   case PIPE_FORMAT_R8G8B8X8_UNORM:
      return PIPE_FORMAT_R8G8B8A8_UNORM;
   case PIPE_FORMAT_R16G16B16X16_FLOAT:
      return PIPE_FORMAT_R16G16B16A16_FLOAT;
   case PIPE_FORMAT_R16G16B16X16_SINT:
      return PIPE_FORMAT_R16G16B16A16_SINT;
   case PIPE_FORMAT_R16G16B16X16_SNORM:
      return PIPE_FORMAT_R16G16B16A16_SNORM;
   case PIPE_FORMAT_R16G16B16X16_UNORM:
      return PIPE_FORMAT_R16G16B16A16_UNORM;
   case PIPE_FORMAT_R32G32B32X32_FLOAT:
      return PIPE_FORMAT_R32G32B32A32_FLOAT;
   case PIPE_FORMAT_R32G32B32X32_SINT:
      return PIPE_FORMAT_R32G32B32A32_SINT;
   case PIPE_FORMAT_R10G10B10X2_SINT:
      return PIPE_FORMAT_R10G10B10A2_SINT;
   case PIPE_FORMAT_R10G10B10X2_SNORM:
      return PIPE_FORMAT_R10G10B10A2_SNORM;
   case PIPE_FORMAT_R10G10B10X2_UNORM:
      return PIPE_FORMAT_R10G10B10A2_UNORM;
   case PIPE_FORMAT_B10G10R10X2_SINT:
      return PIPE_FORMAT_B10G10R10A2_SINT;
   case PIPE_FORMAT_B10G10R10X2_SNORM:
      return PIPE_FORMAT_B10G10R10A2_SNORM;
   case PIPE_FORMAT_B10G10R10X2_UNORM:
      return PIPE_FORMAT_B10G10R10A2_UNORM;
   case PIPE_FORMAT_R5G5B5X1_UNORM:
      return PIPE_FORMAT_R5G5B5A1_UNORM;
   case PIPE_FORMAT_B5G5R5X1_UNORM:
      return PIPE_FORMAT_B5G5R5A1_UNORM;
   default:
      return format;
   }
}

VkFormat
sz_vk_format(struct sz_screen *screen, enum pipe_format format)
{
   format = sz_format_emulate_x8(format);
   const VkFormat ret = vk_format_from_pipe_format(format);

   switch (format) {
   case PIPE_FORMAT_S8_UINT:
   case PIPE_FORMAT_X24S8_UINT:
   case PIPE_FORMAT_S8X24_UINT:
   case PIPE_FORMAT_X32_S8X24_UINT:
      /* GL's stencil-only texture formats. Mesa's pipe-to-Vulkan table has no
       * entry for the two combined layouts -- they are the stencil view of a
       * depth/stencil image whose depth half the format ignores -- and Vulkan's
       * own stencil-only format is S8_UINT, which RADV can both attach and
       * sample. Backing all four with it is what makes
       * GL_DEPTH_STENCIL_TEXTURE_MODE = GL_STENCIL_INDEX expressible at all: a
       * view may name only one aspect of a combined image, and a stencil-only
       * image names exactly that one. X32_S8X24_UINT is what
       * util_format_stencil_only() makes of a D32_SFLOAT_S8_UINT texel, i.e.
       * the stencil half of the temp texture glDrawPixels builds for
       * GL_DEPTH_STENCIL -- without the mapping that view had no format and the
       * draw came back as GL_OUT_OF_MEMORY. */
      return VK_FORMAT_S8_UINT;
   default:
      break;
   }

   switch (ret) {
   case VK_FORMAT_D24_UNORM_S8_UINT:
      /* GL's DEPTH24_STENCIL8. AMD's Vulkan driver has no 24-bit depth at all,
       * so this has to become D32S8; a device that supports neither cannot back
       * the format and is answered with UNDEFINED, which is_format_supported()
       * turns into "no". */
      if (!screen->have_d24_unorm_s8_uint) {
         if (screen->have_d32_sfloat_s8_uint)
            return VK_FORMAT_D32_SFLOAT_S8_UINT;
         if (screen->have_d32_sfloat)
            return VK_FORMAT_D32_SFLOAT;
         return VK_FORMAT_UNDEFINED;
      }
      return ret;

   case VK_FORMAT_X8_D24_UNORM_PACK32:
      /* GL_DEPTH_COMPONENT24, with no stencil aspect to preserve, and the same
       * substitution the depth/stencil case above makes: RADV has no 24-bit
       * depth at all, so the choice is between a 32-bit float and whatever the
       * state tracker falls back to on its own.
       *
       * Left unsupported, that fallback is Z16_UNORM -- not Z32_FLOAT. Mesa
       * searches GL_DEPTH_COMPONENT24's list in order (Z24X8, X8Z24, then
       * DEFAULT_DEPTH_FORMATS = Z24X8, X8Z24, Z16, Z24S8, S8Z24) and Z16 is the
       * first entry a device without 24-bit depth can back. A 24-bit depth
       * buffer that is silently 16 bits wide is already a wrong picture; worse,
       * it makes the attachment format differ from the 24-bit depth/stencil
       * every other framebuffer uses, which turns glBlitFramebuffer between the
       * two into GL_INVALID_OPERATION ("depth attachment format mismatch") and
       * drops the blit entirely -- that is what
       * ext_framebuffer_multisample@no-color depth saw as an empty image.
       *
       * Storing the depth in D32 keeps every bit GL was promised, and the
       * transfer path already reads a D32 image back as Z32_FLOAT (see
       * sz_vk_depth_pipe_format), so a readback converts from float to whatever
       * the application asked for exactly as it does for DEPTH24_STENCIL8. */
      if (!screen->have_x8_d24_unorm_pack32) {
         if (screen->have_d32_sfloat)
            return VK_FORMAT_D32_SFLOAT;
         return VK_FORMAT_UNDEFINED;
      }
      return ret;

   default:
      return ret;
   }
}

enum pipe_format
sz_vk_depth_pipe_format(struct sz_screen *screen, enum pipe_format format)
{
   switch (sz_vk_format(screen, format)) {
   case VK_FORMAT_D32_SFLOAT:
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      return PIPE_FORMAT_Z32_FLOAT;
   case VK_FORMAT_D16_UNORM:
   case VK_FORMAT_D16_UNORM_S8_UINT:
      return PIPE_FORMAT_Z16_UNORM;
   case VK_FORMAT_D24_UNORM_S8_UINT:
   case VK_FORMAT_X8_D24_UNORM_PACK32:
      return PIPE_FORMAT_Z24X8_UNORM;
   default:
      return PIPE_FORMAT_NONE;
   }
}

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
   /* The image behind a stencil-only format has no depth aspect to sample. */
   switch (format) {
   case PIPE_FORMAT_S8_UINT:
   case PIPE_FORMAT_X24S8_UINT:
   case PIPE_FORMAT_S8X24_UINT:
      return VK_IMAGE_ASPECT_STENCIL_BIT;
   default:
      break;
   }

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
sz_format_features_from_bindings(enum pipe_format format, unsigned bindings)
{
   VkFormatFeatureFlags f = 0;

   if (bindings & PIPE_BIND_SAMPLER_VIEW)
      f |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;

   if (bindings & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_BLENDABLE |
                   PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT)) {
      /* Mesa uses RENDER_TARGET to ask "may a draw or a blit write into this?",
       * not "is this a colour format": its glCopyPixels depth path gates the
       * fast pipe->blit on exactly this bit. A depth format has no colour
       * attachment bit, so requiring that one answers "no" and forces that
       * path onto a meta draw instead of the blit that works. */
      const struct util_format_description *desc =
         util_format_description(format);
      const bool depth_stencil =
         desc && (util_format_has_depth(desc) || util_format_has_stencil(desc));

      f |= depth_stencil ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
                         : VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
   }

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
sz_image_usage_from_bindings(struct sz_screen *screen, unsigned bindings,
                            enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   const bool depth_stencil =
      (bindings & PIPE_BIND_DEPTH_STENCIL) ||
      (desc && util_format_has_depth(desc));

   VkImageUsageFlags usage = 0;

   /* A depth/stencil format is never a colour attachment, whatever the
    * bindings ask for; claiming both bits makes the image illegal. */
   if (!(desc && (util_format_has_depth(desc) ||
                  util_format_has_stencil(desc))) &&
       (bindings & (PIPE_BIND_RENDER_TARGET | PIPE_BIND_BLENDABLE |
                    PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT)))
      usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

   if (depth_stencil)
      usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;

   if (bindings & PIPE_BIND_SAMPLER_VIEW)
      usage |= VK_IMAGE_USAGE_SAMPLED_BIT;

   if (bindings & PIPE_BIND_SHADER_IMAGE)
      usage |= VK_IMAGE_USAGE_STORAGE_BIT;

   /* A texture can be bound to an image unit long after it was created: GL has
    * no way to say "a shader will write this" at allocation time, so an
    * application calls glTexImage2D and only later glBindImageTexture. Without
    * the bit declared here that descriptor points at an image Vulkan never
    * allowed to be written (VUID-VkDescriptorGetInfoEXT-type-12218), and there
    * is no way to add it afterwards.
    *
    * The format's own features decide it, so a format that cannot be a storage
    * image gets nothing -- the frontend rejects binding such a texture anyway. */
   if (sz_format_can_store(screen, sz_vk_format(screen, format)))
      usage |= VK_IMAGE_USAGE_STORAGE_BIT;

   /* Every image is a legal transfer source and destination. The paths that need
    * it are not conditional on the bind flags: st clears a surface through
    * vkCmdClearDepthStencilImage or vkCmdClearColorImage (VUID 02659/02660),
    * uploads a texture through vkCmdCopyBufferToImage (VUID 00177), and reads
    * any of them back the same way. Vulkan cannot widen an image's usage after
    * creation, so the flags are declared here for all of them. */
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
sz_image_vk_type(struct sz_screen *screen, enum pipe_texture_target target,
                 enum pipe_format format)
{
   if (screen->need_2d_zs &&
       util_format_is_depth_or_stencil(format) &&
       (target == PIPE_TEXTURE_1D || target == PIPE_TEXTURE_1D_ARRAY))
      return VK_IMAGE_TYPE_2D;

   return sz_image_type(target);
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

bool
sz_bindings_require_export(unsigned bindings)
{
   return !!(bindings & (PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT |
                         PIPE_BIND_SHARED));
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
   /* Vulkan's index types are 16 and 32 bits: there is no 8-bit index buffer to
    * bind. GL does have one, so it is refused here and the util layer rewrites
    * those indices to 16 bits before the draw (u_vbuf's rewrite_ubyte_ibs). The
    * alternative -- reporting support and picking VK_INDEX_TYPE_UINT16 anyway --
    * reads twice the bytes the app provided and draws wrong primitives, which is
    * what a validation error about the index buffer's size was pointing at. */
   if ((bindings & PIPE_BIND_INDEX_BUFFER) && format == PIPE_FORMAT_R8_UINT)
      return false;

   const VkFormat vkformat = sz_vk_format(screen, format);
   if (vkformat == VK_FORMAT_UNDEFINED)
      return false;

   VkFormatProperties props;
   screen->vk.GetPhysicalDeviceFormatProperties(screen->pdev, vkformat, &props);

   const VkFormatFeatureFlags required =
      sz_format_features_from_bindings(format, bindings) & SZ_BUFFER_FEATURES;

   return (props.bufferFeatures & required) == required;
}

static bool
sz_format_supported_image(struct sz_screen *screen,
                          enum pipe_format format,
                          enum pipe_texture_target target,
                          unsigned sample_count,
                          unsigned bindings)
{
   const VkFormat vkformat = sz_vk_format(screen, format);
   if (vkformat == VK_FORMAT_UNDEFINED)
      return false;

   const VkImageType type = sz_image_vk_type(screen, target, format);

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
   VkImageUsageFlags usage = sz_image_usage_from_bindings(screen, bindings, format);
   if (type == VK_IMAGE_TYPE_3D) {
      usage &= ~(VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT);
      if (usage == 0)
         usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   } else if (target == PIPE_TEXTURE_1D || target == PIPE_TEXTURE_1D_ARRAY) {
      /* A 1D shape has no depth attachment, aliased or not. */
      usage &= ~VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      if (usage == 0)
         usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   }

   /* Answer both questions with one call each:
    *   - is (format, type, tiling, usage, flags) a creatable image at all,
    *   - and does it survive at the requested sample count. */
   VkImageFormatProperties2 ifp = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
   };
   VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = vkformat,
      .type = type,
      .tiling = tiling,
      .usage = usage,
      .flags = flags,
   };

   VkResult query = screen->vk.GetPhysicalDeviceImageFormatProperties2(
      screen->pdev, &info, &ifp);

   /* No Vulkan implementation supports a 1D depth attachment -- 1D images
    * cannot have a depth/stencil aspect at all -- but GL has
    * GL_TEXTURE_1D depth textures and a shader may sample one. Asking with the
    * attachment bit set answers "no such format", and since the state tracker
    * probes for a renderable format first, the answer it hears is "this driver
    * does not support depth textures", which turns into a GL_OUT_OF_MEMORY on
    * glTexImage1D. Retrying as a sampled image is what makes 1D depth textures
    * exist; attaching one to a framebuffer stays impossible, which is correct. */
   if (query != VK_SUCCESS &&
       (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) &&
       (type == VK_IMAGE_TYPE_1D || type == VK_IMAGE_TYPE_3D)) {
      info.usage = usage & ~VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
      if (info.usage == 0)
         info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
      query = screen->vk.GetPhysicalDeviceImageFormatProperties2(screen->pdev,
                                                                 &info, &ifp);
   }

   if (query != VK_SUCCESS)
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
      sz_format_features_from_bindings(format, bindings) & ~SZ_BUFFER_FEATURES;

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

   /* A buffer texture is asked about as `PIPE_BUFFER` with PIPE_BIND_SAMPLER_VIEW,
    * and the answer comes from the buffer path alone: there is no VkImage behind
    * it, so whether the *image* tiling can sample the format says nothing about
    * it. Without this, R32G32B32_FLOAT -- which no AMD device can sample as an
    * image but every one can fetch from a buffer -- is refused and
    * ARB_texture_buffer_object_rgb32 never appears. */
   const bool is_buffer_texture = target == PIPE_BUFFER;
   const bool image_path = image_binding && !is_buffer_texture;

   /* A true image-only binding is decided by the image path; a buffer binding by
    * the buffer path. Callers that ask for both must satisfy both. */
   if (image_path &&
       !sz_format_supported_image(screen, format, target, sample_count, bindings))
      return false;

   if (buffer_binding &&
       !sz_format_supported_buffer(screen, format, bindings))
      return false;

   if (buffer_binding || image_binding)
      return true;

   /* No bindings at all: the format merely has to exist. */
   return sz_vk_format(screen, format) != VK_FORMAT_UNDEFINED;
}
