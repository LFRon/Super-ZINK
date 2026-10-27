/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Descriptors.
 *
 * ADR-008: no descriptor pools, no vkAllocateDescriptorSets. Everything a draw
 * needs is pushed into the command buffer with vkCmdPushDescriptorSetKHR, which
 * costs command stream space instead of a separate API call per update and has
 * no lifetime to manage.
 *
 * The layout is fixed rather than derived from the shader, and deliberately so:
 *
 *  - a single set, because a pipeline layout may contain at most one set
 *    created with the push descriptor flag;
 *  - each stage owns SZ_MAX_CONST_BUFFERS consecutive bindings inside it, and a
 *    stage's binding is its base plus the gallium constant buffer slot.
 *    nir_lower_uniforms_to_ubo() maps the default uniform block to slot 0 and
 *    shifts real uniform blocks up by one, so the NIR index and the gallium slot
 *    are the same number. That is what lets the layout be a constant instead of
 *    a per-shader object.
 *
 * A layout may declare more bindings than a shader uses. The reverse -- a
 * shader binding the layout does not declare -- is what must never happen, and
 * a fixed superset rules it out by construction.
 */

#ifndef SZ_DESCRIPTOR_H
#define SZ_DESCRIPTOR_H

#include "sz_types.h"

struct sz_context;

/* The stage flags a binding carries in the layout: exactly the stage that owns
 * the binding, never more. */
static inline VkShaderStageFlags
sz_descriptor_stage_flags(mesa_shader_stage stage)
{
   switch (stage) {
   case MESA_SHADER_VERTEX:
      return VK_SHADER_STAGE_VERTEX_BIT;
   case MESA_SHADER_FRAGMENT:
      return VK_SHADER_STAGE_FRAGMENT_BIT;
   default:
      return VK_SHADER_STAGE_COMPUTE_BIT;
   }
}

/* Creates the descriptor set layout. Called once per screen, after the VkDevice
 * exists. `max_push_descriptors` is the device's
 * VkPhysicalDevicePushDescriptorPropertiesKHR::maxPushDescriptors, or zero when
 * the device does not report it. */
bool sz_descriptor_screen_init(struct sz_screen *screen,
                               unsigned max_push_descriptors);
void sz_descriptor_screen_finish(struct sz_screen *screen);

/* Records the descriptors a draw needs, from the context's recorded constant
 * buffers. Must be called after the pipeline is bound: the push goes to the
 * layout the bound pipeline was created with. */
void sz_descriptor_push(struct sz_context *ctx, VkCommandBuffer cmd);

#endif /* SZ_DESCRIPTOR_H */
