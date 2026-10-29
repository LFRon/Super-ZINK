/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef SZ_FORMAT_H
#define SZ_FORMAT_H

#include <vulkan/vulkan_core.h>

#include "pipe/p_defines.h"
#include "util/format/u_formats.h"

struct sz_screen;

/* The VkFormat a Gallium format is backed by, with the depth/stencil
 * substitutions Vulkan makes necessary.
 *
 * `vk_format_from_pipe_format()` answers the obvious mapping, but GL's
 * DEPTH24_STENCIL8 has no guaranteed Vulkan equivalent: RADV reports no features
 * at all for VK_FORMAT_D24_UNORM_S8_UINT on AMD hardware, and an image that
 * cannot be created is worse than an image with more depth precision. So D24S8
 * falls back to D32_SFLOAT_S8_UINT when the former is unavailable -- the same
 * choice zink makes, and the only way to have stencil at all on such a device.
 *
 * A format with no stencil aspect is *not* substituted: the state tracker can
 * pick a different format itself, and it picks one whose texel layout matches
 * what the transfer path copies. See sz_vk_format().
 *
 * Every path that creates an image, a view, an attachment or a pipeline must go
 * through this function rather than through vk_format_from_pipe_format(), or the
 * two will disagree about what a texture actually is. */
VkFormat sz_vk_format(struct sz_screen *screen, enum pipe_format format);

/* The pipe format that describes the *depth aspect's* bytes the way the device
 * stores them.
 *
 * sz_vk_format() may substitute a wider depth format than the pipe format names
 * -- D24S8 becomes D32_SFLOAT_S8_UINT on hardware without 24-bit depth -- and a
 * copy of one aspect carries that wider format's bytes. Anything that reads or
 * writes depth through a transfer has to interpret them as this format, not as
 * the resource's, or a depth of 0.25 comes back as the low 24 bits of the float
 * 0x3E800000. PIPE_FORMAT_NONE means the format has no depth aspect at all. */
enum pipe_format sz_vk_depth_pipe_format(struct sz_screen *screen,
                                        enum pipe_format format);

/* Probes the device for the depth formats sz_vk_format() has to choose between.
 * Called once, during screen creation. */
void sz_format_probe(struct sz_screen *screen);
void sz_format_debug_print(struct sz_screen *screen);

/* Translate Gallium usage bits to the VkImageUsageFlags they require. */
VkImageUsageFlags sz_image_usage_from_bindings(unsigned bindings,
                                              enum pipe_format format);

/* Translate Gallium bindings to the VkFormatFeatureFlags that must be present. */
VkFormatFeatureFlags sz_format_features_from_bindings(unsigned bindings);

/* Mapping tables shared with sz_resource, so image creation and image support
 * queries can never disagree about what a target means. */
VkImageType sz_image_type(enum pipe_texture_target target);

/* The Vulkan image shape a Gallium resource really gets.
 *
 * Differs from sz_image_type() for one case: a 1D or 1D_ARRAY depth/stencil
 * texture on a device where 1D depth is unsupported becomes 2D (one texel high),
 * because GL has such textures and Vulkan's 1D depth is optional in practice.
 * Every place that decides a shape, a view type or a shader image dimension has
 * to agree, so they all come through here. */
VkImageType sz_image_vk_type(struct sz_screen *screen,
                             enum pipe_texture_target target,
                             enum pipe_format format);
VkImageCreateFlags sz_image_create_flags(enum pipe_texture_target target);
VkSampleCountFlagBits sz_vk_sample_count(unsigned sample_count);

/* The VkImageAspectFlags a format's data lives in. Shared so image creation,
 * barrier recording and attachment setup cannot disagree about whether a format
 * has a stencil part. */
VkImageAspectFlags sz_format_aspect(enum pipe_format format);

/* Whether a resource created with these bindings must be linearly tiled. */
bool sz_bindings_require_linear(unsigned bindings);

/* Whether a resource with these bindings is expected to leave the process as a
 * dma-buf. A superset relation with sz_bindings_require_linear(), not the same
 * question: PIPE_BIND_LINEAR alone says "the CPU reads these pixels", which does
 * not need an exported VkDeviceMemory, while PIPE_BIND_SHARED says "another
 * process reads them", which does. */
bool sz_bindings_require_export(unsigned bindings);

/* The single entry point behind pipe_screen::is_format_supported(). */
bool sz_format_supported(struct sz_screen *screen,
                         enum pipe_format format,
                         enum pipe_texture_target target,
                         unsigned sample_count,
                         unsigned storage_sample_count,
                         unsigned bindings);

#endif /* SZ_FORMAT_H */
