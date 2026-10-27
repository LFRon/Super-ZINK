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

/* Translate Gallium usage bits to the VkImageUsageFlags they require. */
VkImageUsageFlags sz_image_usage_from_bindings(unsigned bindings,
                                              enum pipe_format format);

/* Translate Gallium bindings to the VkFormatFeatureFlags that must be present. */
VkFormatFeatureFlags sz_format_features_from_bindings(unsigned bindings);

/* Mapping tables shared with sz_resource, so image creation and image support
 * queries can never disagree about what a target means. */
VkImageType sz_image_type(enum pipe_texture_target target);
VkImageCreateFlags sz_image_create_flags(enum pipe_texture_target target);
VkSampleCountFlagBits sz_vk_sample_count(unsigned sample_count);

/* Whether a resource created with these bindings must be linearly tiled. */
bool sz_bindings_require_linear(unsigned bindings);

/* The single entry point behind pipe_screen::is_format_supported(). */
bool sz_format_supported(struct sz_screen *screen,
                         enum pipe_format format,
                         enum pipe_texture_target target,
                         unsigned sample_count,
                         unsigned storage_sample_count,
                         unsigned bindings);

#endif /* SZ_FORMAT_H */
