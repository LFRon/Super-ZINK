/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Descriptors.
 *
 * Two sets, split by kind of resource rather than by stage:
 *
 *  - set 0 is the push set and holds the uniform buffers. Pushing costs command
 *    stream space instead of an API call per update and has no lifetime to
 *    manage, which is what the hot path wants, and uniform buffers are rewritten
 *    whenever a state changes.
 *  - set 1 is an ordinary descriptor set and holds the combined image samplers.
 *    It cannot also be a push set: a pipeline layout may contain at most one set
 *    created with VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT
 *    (VUID-VkPipelineLayoutCreateInfo-pSetLayouts-00293), and one push set
 *    cannot hold both kinds -- 8 + 16 bindings per stage times two stages is
 *    more than the 32 maxPushDescriptors is guaranteed to allow, which is what
 *    GL 3.1's 16 texture units per stage needs.
 *
 * Sampler sets are cached by content. A draw almost always rebinds texture and
 * sampler combinations it has used before, so the cache turns the per-draw cost
 * into a hash lookup plus vkCmdBindDescriptorSets, and a new VkDescriptorSet is
 * only allocated when the content is genuinely new. Entries are only recycled
 * once the submission that last used them has finished, so a set is never
 * rewritten while the GPU may still be reading it.
 *
 * The layout is fixed rather than derived from the shader, and deliberately so:
 * each stage owns a contiguous run of bindings, and a stage's binding is its
 * index times the slot count plus the gallium slot.
 * nir_lower_uniforms_to_ubo() maps the default uniform block to slot 0 and
 * shifts real uniform blocks up by one, so the NIR index and the gallium slot
 * are the same number. That is what lets the layout be a constant instead of a
 * per-shader object.
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

/* True when at least one of the images the draw samples is not in the layout a
 * barrel-free sample needs. The caller uses it to decide whether an already open
 * render pass has to be closed first: a layout transition is illegal inside one
 * (VUID-vkCmdPipelineBarrier2-oldLayout-01181), and the alternative -- leaving the
 * image in a layout the descriptor does not claim -- is undefined reads. */
bool sz_descriptor_sampled_needs_transition(struct sz_context *ctx);

/* Transitions every sampled image into SHADER_READ_ONLY_OPTIMAL. Must be called
 * while no render pass is open. */
void sz_descriptor_prepare_sampled(struct sz_context *ctx, VkCommandBuffer cmd);

#endif /* SZ_DESCRIPTOR_H */
