/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Descriptors. See sz_descriptor.h for the design.
 */

#include "sz_descriptor.h"

#include "sz_batch.h"
#include "sz_bo.h"
#include "sz_compiler.h"
#include "sz_context.h"
#include "sz_format.h"
#include "sz_pipeline.h"
#include "sz_renderpass.h"
#include "sz_resource.h"
#include "sz_sync.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"

#include "vk_format.h"

/* Bindings in the one push set: SZ_MAX_CONST_BUFFERS uniform buffers for each of
 * the stages sz_compiler.h reserves room for. Compute is deliberately absent --
 * the layout cannot hold three stages and still stay inside the 32 descriptors
 * every device is guaranteed (see SZ_DESCRIPTOR_STAGES). */
static const mesa_shader_stage sz_descriptor_stages[SZ_DESCRIPTOR_STAGES] = {
   MESA_SHADER_VERTEX,      MESA_SHADER_TESS_CTRL, MESA_SHADER_TESS_EVAL,
   MESA_SHADER_GEOMETRY,    MESA_SHADER_FRAGMENT,
};

static bool
sz_descriptor_layout_create(struct sz_screen *screen, VkDescriptorSetLayout *out)
{
   /* Set 0: every stage's uniform buffers and shader storage blocks, written
    * per state change. */
   VkDescriptorSetLayoutBinding bindings[SZ_DESCRIPTOR_BUFFER_BINDINGS];

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];
      const VkShaderStageFlags flags = sz_descriptor_stage_flags(stage);
      for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++) {
         const unsigned binding = sz_descriptor_binding(stage, i);
         bindings[binding] = (VkDescriptorSetLayoutBinding) {
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = flags,
         };
      }
      for (unsigned i = 0; i < SZ_MAX_SSBO; i++) {
         const unsigned binding = sz_ssbo_binding(stage, i);
         bindings[binding] = (VkDescriptorSetLayoutBinding) {
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = flags,
         };
      }
   }

   /* The two ways of getting descriptors to the device need layouts that were
    * created with different flags, and one layout may not carry both: a push
    * layout has an opaque size and no binding offsets, and a descriptor-buffer
    * layout cannot be pushed to (VUID-vkGetDescriptorSetLayoutSizeEXT-layout-11811).
    * So the screen picks a path once, at creation, and everything below follows
    * it. */
   const VkDescriptorSetLayoutCreateFlags layout_flags =
      screen->caps.descriptor_buffer
         ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT
         : VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;

   VkDescriptorSetLayoutCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      /* ADR-008: nothing is ever allocated from a pool. */
      .flags = layout_flags,
      .bindingCount = SZ_DESCRIPTOR_BUFFER_BINDINGS,
      .pBindings = bindings,
   };

   VkResult result = screen->vk.CreateDescriptorSetLayout(screen->dev, &ci, NULL,
                                                          &out[0]);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateDescriptorSetLayout(uniform buffers) failed: %d",
                result);
      return false;
   }

   /* Set 1: every stage's combined image samplers. Not a push set: 32 bindings
    * of samplers plus the uniform buffers would exceed maxPushDescriptors, and a
    * pipeline layout may only contain one push set. */
   VkDescriptorSetLayoutBinding sampler_bindings[SZ_DESCRIPTOR_SAMPLER_BINDINGS];

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];
      const VkShaderStageFlags flags = sz_descriptor_stage_flags(stage);
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         const unsigned binding = sz_sampler_binding(stage, i);
         sampler_bindings[binding] = (VkDescriptorSetLayoutBinding) {
            .binding = binding,
            /* One combined image sampler per texture unit rather than separate
             * image and sampler bindings: GL's sampler uniforms carry both, and
             * one descriptor is half the budget of two. */
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = flags,
         };
      }
   }

   ci.bindingCount = SZ_DESCRIPTOR_SAMPLER_BINDINGS;
   ci.pBindings = sampler_bindings;

   result = screen->vk.CreateDescriptorSetLayout(screen->dev, &ci, NULL,
                                                 &out[1]);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateDescriptorSetLayout(samplers) failed: %d", result);
      goto fail;
   }

   /* Set 2: the texel buffers. A buffer texture has no sampler and no image
    * view -- its descriptor is a VkBufferView -- so it cannot share a binding
    * with a combined image sampler, and VkDescriptorSetLayoutBinding carries one
    * descriptor type. */
   VkDescriptorSetLayoutBinding texel_bindings[SZ_DESCRIPTOR_TEXEL_BINDINGS];

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];
      const VkShaderStageFlags flags = sz_descriptor_stage_flags(stage);
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         const unsigned binding = sz_texel_binding(stage, i);
         texel_bindings[binding] = (VkDescriptorSetLayoutBinding) {
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            .descriptorCount = 1,
            .stageFlags = flags,
         };
      }
   }

   ci.bindingCount = SZ_DESCRIPTOR_TEXEL_BINDINGS;
   ci.pBindings = texel_bindings;

   result = screen->vk.CreateDescriptorSetLayout(screen->dev, &ci, NULL,
                                                 &out[SZ_DESCRIPTOR_SET_TEXEL_BUFFERS]);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateDescriptorSetLayout(texel buffers) failed: %d",
                result);
      goto fail;
   }

   /* Set 3: the storage images. A GL image uniform is an image the shader may
    * read and write without a sampler, which is VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    * the format is part of the view, so unlike a sampled image the descriptor
    * needs nothing else. */
   VkDescriptorSetLayoutBinding image_bindings[SZ_DESCRIPTOR_IMAGE_BINDINGS];

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];
      const VkShaderStageFlags flags = sz_descriptor_stage_flags(stage);
      for (unsigned i = 0; i < SZ_MAX_IMAGES; i++) {
         const unsigned binding = sz_image_binding(stage, i);
         image_bindings[binding] = (VkDescriptorSetLayoutBinding) {
            .binding = binding,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = flags,
         };
      }
   }

   ci.bindingCount = SZ_DESCRIPTOR_IMAGE_BINDINGS;
   ci.pBindings = image_bindings;

   result = screen->vk.CreateDescriptorSetLayout(screen->dev, &ci, NULL,
                                                 &out[SZ_DESCRIPTOR_SET_IMAGES]);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateDescriptorSetLayout(storage images) failed: %d",
                result);
      goto fail;
   }

   /* The descriptor-buffer path needs no pool at all: descriptors go straight
    * into a mapped buffer and are selected by offset. */
   if (screen->caps.descriptor_buffer)
      return true;

   /* Push path: a pool for the sampler sets. Each update allocates a set and the
    * pool is only reset once everything it handed out has been submitted, so the
    * size is the number of sampler-state changes that may be in flight at once. */
   const VkDescriptorPoolSize pool_sizes[] = {
      {
         .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = SZ_DESCRIPTOR_SAMPLER_BINDINGS * SZ_SAMPLER_SET_POOL,
      },
      {
         .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
         .descriptorCount = SZ_DESCRIPTOR_TEXEL_BINDINGS * SZ_SAMPLER_SET_POOL,
      },
   };
   const VkDescriptorPoolCreateInfo pool_ci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = SZ_SAMPLER_SET_POOL,
      .poolSizeCount = ARRAY_SIZE(pool_sizes),
      .pPoolSizes = pool_sizes,
   };

   result = screen->vk.CreateDescriptorPool(screen->dev, &pool_ci, NULL,
                                            &screen->descriptor_pool);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateDescriptorPool failed: %d", result);
      screen->vk.DestroyDescriptorSetLayout(screen->dev, out[1], NULL);
      goto fail;
   }

   return true;

fail:
   screen->vk.DestroyDescriptorSetLayout(screen->dev, out[0], NULL);
   out[0] = VK_NULL_HANDLE;
   out[1] = VK_NULL_HANDLE;
   return false;
}

/* Creates the 1x1 placeholder every unbound sampler binding is pointed at. */
static bool
sz_descriptor_dummy_create(struct sz_screen *screen)
{
   const VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = { 1, 1, 1 },
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };

   if (screen->vk.CreateImage(screen->dev, &ci, NULL,
                              &screen->dummy_sampler.image) != VK_SUCCESS) {
      sz_warn_once("could not create the placeholder texture");
      return false;
   }

   VkMemoryRequirements req;
   screen->vk.GetImageMemoryRequirements(screen->dev, screen->dummy_sampler.image,
                                         &req);
   screen->dummy_sampler.bo = sz_bo_alloc(screen, &req,
                                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   if (!screen->dummy_sampler.bo ||
       screen->vk.BindImageMemory(screen->dev, screen->dummy_sampler.image,
                                  sz_bo_memory(screen->dummy_sampler.bo),
                                  sz_bo_offset(screen->dummy_sampler.bo)) !=
          VK_SUCCESS) {
      sz_warn_once("could not back the placeholder texture");
      return false;
   }

   const VkImageViewCreateInfo view_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = screen->dummy_sampler.image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   if (screen->vk.CreateImageView(screen->dev, &view_ci, NULL,
                                  &screen->dummy_sampler.view) != VK_SUCCESS) {
      sz_warn_once("could not create the placeholder view");
      return false;
   }

   const VkSamplerCreateInfo sampler_ci = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   if (screen->vk.CreateSampler(screen->dev, &sampler_ci, NULL,
                                &screen->dummy_sampler.sampler) != VK_SUCCESS) {
      sz_warn_once("could not create the placeholder sampler");
      return false;
   }

   /* The contents are filled on first use rather than here: this runs during
    * screen creation, before the submission ring exists, and recording needs a
    * command buffer. */
   return true;
}

/* Allocates the descriptor buffers and the unbound-slot placeholders.
 *
 * Called after the layouts exist, because their sizes and binding offsets are
 * layout properties.
 *
 * Failures are fatal rather than a silent fallback: the layouts were already
 * created for this path, so a screen that cannot produce descriptors cannot
 * draw. Saying so here is far cheaper than debugging an empty window. */
static bool
sz_descriptor_db_init(struct sz_screen *screen)
{
   if (!screen->caps.descriptor_buffer)
      return true;

   if (!screen->vk.GetDescriptorSetLayoutSizeEXT ||
       !screen->vk.GetDescriptorSetLayoutBindingOffsetEXT ||
       !screen->vk.GetDescriptorEXT || !screen->vk.CmdBindDescriptorBuffersEXT ||
       !screen->vk.CmdSetDescriptorBufferOffsetsEXT) {
      mesa_loge("SZ: VK_EXT_descriptor_buffer is advertised but its entry points "
                "did not resolve");
      return false;
   }

   screen->vk.GetDescriptorSetLayoutSizeEXT(
      screen->dev, screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_UBO],
      &screen->db.ubo_size);
   screen->vk.GetDescriptorSetLayoutSizeEXT(
      screen->dev, screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_SAMPLERS],
      &screen->db.sampler_size);
   screen->vk.GetDescriptorSetLayoutSizeEXT(
      screen->dev, screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_TEXEL_BUFFERS],
      &screen->db.texel_size);
   screen->vk.GetDescriptorSetLayoutSizeEXT(
      screen->dev, screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_IMAGES],
      &screen->db.image_size);

   if (!screen->db.ubo_size || !screen->db.sampler_size ||
       !screen->db.texel_size || !screen->db.image_size) {
      mesa_loge("SZ: a descriptor set layout reports size zero");
      return false;
   }

   /* Every offset handed to vkCmdSetDescriptorBufferOffsetsEXT must be a
    * multiple of the device's alignment, so the two sets are padded to it. */
   const uint64_t align =
      MAX2(screen->caps.descriptor_buffer_offset_alignment, 1ull);
   screen->db.ubo_size = align64(screen->db.ubo_size, align);
   screen->db.sampler_size = align64(screen->db.sampler_size, align);
   screen->db.texel_size = align64(screen->db.texel_size, align);
   screen->db.image_size = align64(screen->db.image_size, align);

   /* Sized so that running out is not a practical concern: a batch is flushed
    * on every state change, and one draw costs a few hundred bytes here. */
   const VkDeviceSize slot_size = 4ull * 1024 * 1024;

   for (unsigned i = 0; i < SZ_BATCH_RING_SIZE; i++) {
      const VkBufferCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = slot_size,
         /* Both usages: sampler descriptors for the image samplers, resource
          * descriptors for the uniform buffers. There is no combined bit. */
         .usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                  VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                  /* Needed because the buffer is named by device address in
                   * vkCmdBindDescriptorBuffersEXT, and
                   * vkGetBufferDeviceAddress requires this bit on the buffer it
                   * is asked about. Without it the address is garbage and the
                   * device reads descriptors from nowhere, which hangs rather
                   * than reporting an error. */
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      if (screen->vk.CreateBuffer(screen->dev, &ci, NULL,
                                  &screen->db.ring[i].buffer) != VK_SUCCESS) {
         mesa_loge("SZ: could not create a descriptor buffer");
         return false;
      }

      VkMemoryRequirements req;
      screen->vk.GetBufferMemoryRequirements(screen->dev,
                                             screen->db.ring[i].buffer, &req);
      /* Host-visible and coherent because descriptors are produced by the CPU
       * between draws; that is what the extension is for. */
      screen->db.ring[i].bo =
         sz_bo_alloc(screen, &req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
      if (!screen->db.ring[i].bo ||
          screen->vk.BindBufferMemory(screen->dev, screen->db.ring[i].buffer,
                                      sz_bo_memory(screen->db.ring[i].bo),
                                      sz_bo_offset(screen->db.ring[i].bo)) !=
             VK_SUCCESS) {
         mesa_loge("SZ: could not back a descriptor buffer");
         return false;
      }

      screen->db.ring[i].mapped = sz_bo_map(screen->db.ring[i].bo);
      if (!screen->db.ring[i].mapped) {
         mesa_loge("SZ: could not map a descriptor buffer");
         return false;
      }

      const VkBufferDeviceAddressInfo address_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
         .buffer = screen->db.ring[i].buffer,
      };
      screen->db.ring[i].address =
         screen->vk.GetBufferDeviceAddress(screen->dev, &address_info);
      screen->db.ring[i].size = slot_size;
      screen->db.ring[i].used = 0;
   }

   /* The placeholder an unbound constant buffer slot points at. It must be a
    * real uniform buffer: pointing the slot at the descriptor buffer fails
    * VUID-VkDescriptorGetInfoEXT-type-12220, because that one was not created
    * with VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT. */
   {
      const VkBufferCreateInfo ci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = 256,
         .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      if (screen->vk.CreateBuffer(screen->dev, &ci, NULL,
                                  &screen->dummy_ubo.buffer) != VK_SUCCESS) {
         mesa_loge("SZ: could not create the unbound uniform buffer placeholder");
         return false;
      }

      VkMemoryRequirements req;
      screen->vk.GetBufferMemoryRequirements(screen->dev,
                                             screen->dummy_ubo.buffer, &req);
      screen->dummy_ubo.bo =
         sz_bo_alloc(screen, &req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
      if (!screen->dummy_ubo.bo ||
          screen->vk.BindBufferMemory(screen->dev, screen->dummy_ubo.buffer,
                                      sz_bo_memory(screen->dummy_ubo.bo),
                                      sz_bo_offset(screen->dummy_ubo.bo)) !=
             VK_SUCCESS) {
         mesa_loge("SZ: could not back the unbound uniform buffer placeholder");
         return false;
      }

      const VkBufferDeviceAddressInfo info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
         .buffer = screen->dummy_ubo.buffer,
      };
      screen->dummy_ubo.address =
         screen->vk.GetBufferDeviceAddress(screen->dev, &info);
   }

   screen->db.enabled = true;

   if (sz_debug_enabled(SZ_DEBUG_INFO))
      mesa_logi("SZ: descriptor buffers: %" PRIu64 " + %" PRIu64 " + %" PRIu64
                " bytes per set, %u slots of %" PRIu64 " bytes",
                (uint64_t)screen->db.ubo_size,
                (uint64_t)screen->db.sampler_size,
                (uint64_t)screen->db.texel_size, SZ_BATCH_RING_SIZE,
                (uint64_t)slot_size);

   return true;
}

bool
sz_descriptor_screen_init(struct sz_screen *screen, unsigned max_push_descriptors)
{
   if (!sz_descriptor_layout_create(screen, screen->descriptor_set_layout))
      return false;

   /* Without the placeholder there is no safe way to leave a binding that used
    * to name a destroyed view, so a screen without it is not usable. */
   if (!sz_descriptor_dummy_create(screen))
      return false;

   return sz_descriptor_db_init(screen);
}

void
sz_descriptor_screen_finish(struct sz_screen *screen)
{
   if (screen->dummy_ubo.buffer != VK_NULL_HANDLE)
      screen->vk.DestroyBuffer(screen->dev, screen->dummy_ubo.buffer, NULL);
   if (screen->dummy_ubo.bo)
      sz_bo_free(screen->dummy_ubo.bo);
   screen->dummy_ubo.buffer = VK_NULL_HANDLE;
   screen->dummy_ubo.bo = NULL;

   for (unsigned i = 0; i < SZ_BATCH_RING_SIZE; i++) {
      if (screen->db.ring[i].buffer != VK_NULL_HANDLE)
         screen->vk.DestroyBuffer(screen->dev, screen->db.ring[i].buffer, NULL);
      if (screen->db.ring[i].bo)
         sz_bo_free(screen->db.ring[i].bo);
      screen->db.ring[i].buffer = VK_NULL_HANDLE;
      screen->db.ring[i].bo = NULL;
      screen->db.ring[i].mapped = NULL;
   }

   if (screen->dummy_sampler.sampler != VK_NULL_HANDLE)
      screen->vk.DestroySampler(screen->dev, screen->dummy_sampler.sampler, NULL);
   if (screen->dummy_sampler.view != VK_NULL_HANDLE)
      screen->vk.DestroyImageView(screen->dev, screen->dummy_sampler.view, NULL);
   if (screen->dummy_sampler.image != VK_NULL_HANDLE)
      screen->vk.DestroyImage(screen->dev, screen->dummy_sampler.image, NULL);
   if (screen->dummy_sampler.bo)
      sz_bo_free(screen->dummy_sampler.bo);

   if (screen->descriptor_pool != VK_NULL_HANDLE) {
      screen->vk.DestroyDescriptorPool(screen->dev, screen->descriptor_pool,
                                       NULL);
      screen->descriptor_pool = VK_NULL_HANDLE;
   }

   for (unsigned i = 0; i < SZ_DESCRIPTOR_SET_COUNT; i++) {
      if (screen->descriptor_set_layout[i] != VK_NULL_HANDLE) {
         screen->vk.DestroyDescriptorSetLayout(screen->dev,
                                               screen->descriptor_set_layout[i],
                                               NULL);
         screen->descriptor_set_layout[i] = VK_NULL_HANDLE;
      }
   }
}

/* The image behind one combined image sampler slot, or NULL when the slot is not
 * usable as a descriptor: Vulkan rejects a combined image sampler with either
 * half missing, and GL allows a sampler uniform to be sampled with no texture
 * bound, which reads as black rather than as an error. */
static struct sz_resource *
sz_sampled_resource(const struct sz_context *ctx, mesa_shader_stage stage,
                    unsigned slot, const struct sz_state **state_out,
                    const struct sz_sampler_view **view_out)
{
   const struct pipe_sampler_view *view = ctx->state.sampler_views[stage][slot];
   const struct sz_state *state = ctx->state.sampler_states[stage][slot];
   if (!view || !state || state->u.sampler.sampler == VK_NULL_HANDLE)
      return NULL;

   const struct sz_sampler_view *sv = (const struct sz_sampler_view *)view;
   if (sv->view == VK_NULL_HANDLE)
      return NULL;

   struct sz_resource *res = sz_resource(view->texture);
   if (!res || res->image == VK_NULL_HANDLE)
      return NULL;

   if (state_out)
      *state_out = state;
   if (view_out)
      *view_out = sv;
   return res;
}

bool
sz_descriptor_sampled_needs_transition(struct sz_context *ctx)
{
   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         struct sz_resource *res =
            sz_sampled_resource(ctx, sz_descriptor_stages[s], i, NULL, NULL);
         if (!res)
            continue;

         if (res->undefined_contents ||
             res->layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            return true;
      }
   }

   return false;
}

/* Whether the resource is one of the attachments of the render pass that is
 * currently open. The layout alone cannot answer this: an image keeps
 * COLOR_ATTACHMENT_OPTIMAL after its pass has ended, and sampling it then is
 * perfectly legal as long as it is transitioned first. Only a texture that is
 * still attached is the undefined case. */
static bool
sz_resource_is_rendering_target(struct sz_context *ctx, struct sz_resource *res)
{
   const struct pipe_framebuffer_state *fb = &ctx->state.framebuffer;

   for (unsigned i = 0; i < fb->nr_cbufs; i++) {
      if (fb->cbufs[i].texture && sz_resource(fb->cbufs[i].texture) == res)
         return true;
   }

   return fb->zsbuf.texture && sz_resource(fb->zsbuf.texture) == res;
}

/* Copies the image a draw is about to render into so the draw can sample the
 * copy. GL leaves a feedback loop undefined, but every real driver produces the
 * pixels rendered so far, and a browser compositor relies on that: RADV's
 * compressed surfaces make the loop a screenful of noise instead. */
static bool
sz_sampled_feedback_copy(struct sz_context *ctx, VkCommandBuffer cmd,
                         struct sz_resource *res, VkImageView *view_out)
{
   struct sz_screen *screen = ctx->screen;
   const VkFormat format = sz_vk_format(screen, res->base.format);

   if (!ctx->sampled_copy || ctx->sampled_copy_format != format ||
       ctx->sampled_copy_extent[0] != res->base.width0 ||
       ctx->sampled_copy_extent[1] != res->base.height0) {
      pipe_resource_reference(&ctx->sampled_copy, NULL);

      struct pipe_resource tmpl = {
         .target = PIPE_TEXTURE_2D,
         .format = res->base.format,
         .width0 = res->base.width0,
         .height0 = res->base.height0,
         .depth0 = 1,
         .array_size = 1,
         .last_level = 0,
         .nr_samples = 1,
         .usage = PIPE_USAGE_DEFAULT,
         .bind = PIPE_BIND_SAMPLER_VIEW,
      };
      ctx->sampled_copy = screen->base.resource_create(&screen->base, &tmpl);
      if (!ctx->sampled_copy) {
         mesa_loge("SZ: out of memory allocating a feedback copy");
         return false;
      }
      ctx->sampled_copy_format = format;
      ctx->sampled_copy_extent[0] = res->base.width0;
      ctx->sampled_copy_extent[1] = res->base.height0;
   }

   struct sz_resource *copy = sz_resource(ctx->sampled_copy);
   if (copy->view == VK_NULL_HANDLE)
      return false;

   sz_resource_access(screen, cmd, res, SZ_ACCESS_TRANSFER_SRC);
   sz_resource_access(screen, cmd, copy, SZ_ACCESS_TRANSFER_DST);

   const VkImageCopy region = {
      .srcSubresource = { sz_format_aspect(res->base.format), 0, 0, 1 },
      .dstSubresource = { sz_format_aspect(res->base.format), 0, 0, 1 },
      .extent = { res->base.width0, res->base.height0, 1 },
   };
   screen->vk.CmdCopyImage(cmd, res->image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           copy->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &region);

   /* Back to an attachment: the draw path re-opens the pass on this image right
    * after this returns. The copy itself stays readable. */
   sz_resource_access(screen, cmd, res, SZ_ACCESS_COLOR_ATTACH);
   sz_resource_access(screen, cmd, copy, SZ_ACCESS_SAMPLED);

   *view_out = copy->view;
   return true;
}

/* Fills the placeholder texture the first time a draw needs it.
 *
 * It has to happen here rather than at screen creation: the clear and the layout
 * transitions are commands, so they need a command buffer, and they cannot be
 * recorded inside a render pass. This function runs before the pass opens. */
static void
sz_descriptor_dummy_prepare(struct sz_screen *screen, VkCommandBuffer cmd)
{
   if (screen->dummy_sampler.ready || screen->dummy_sampler.image == VK_NULL_HANDLE)
      return;

   const VkImageSubresourceRange range = {
      .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
      .levelCount = 1,
      .layerCount = 1,
   };
   VkImageMemoryBarrier2 barrier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
      .srcAccessMask = 0,
      .dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT,
      .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .image = screen->dummy_sampler.image,
      .subresourceRange = range,
   };
   VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier,
   };
   screen->vk.CmdPipelineBarrier2(cmd, &dep);

   /* Opaque black, which is what GL returns for a sampler that reads an unbound
    * unit. */
   const VkClearColorValue black = { .float32 = { 0.0f, 0.0f, 0.0f, 1.0f } };
   screen->vk.CmdClearColorImage(cmd, screen->dummy_sampler.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1,
                                 &range);

   barrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
   barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
   barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                          VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
   barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
   barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
   barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
   screen->vk.CmdPipelineBarrier2(cmd, &dep);

   screen->dummy_sampler.ready = true;
}

void
sz_descriptor_prepare_sampled(struct sz_context *ctx, VkCommandBuffer cmd)
{
   memset(ctx->sampled_override.view, 0, sizeof(ctx->sampled_override.view));

   sz_descriptor_dummy_prepare(ctx->screen, cmd);

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         struct sz_resource *res =
            sz_sampled_resource(ctx, sz_descriptor_stages[s], i, NULL, NULL);
         if (!res)
            continue;

         /* Sampling an image that is also one of this framebuffer's attachments
          * cannot work: the transition alone would make the render pass render
          * into a read-only image, and sampling the tiled attachment without it
          * reads noise. A copy of what has been rendered so far is what the GL
          * caller means. */
         if (sz_resource_is_rendering_target(ctx, res)) {
            if (ctx->rendering)
               sz_renderpass_end(ctx, cmd);
            if (sz_sampled_feedback_copy(ctx, cmd, res,
                                         &ctx->sampled_override.view[s][i]))
               continue;
         }

         sz_resource_access(ctx->screen, cmd, res, SZ_ACCESS_SAMPLED);
      }
   }
}

/* Finds or builds the descriptor set the sampler bindings describe.
 *
 * There is deliberately no invalidation on sampler view destruction. A cached
 * set names VkImageView handles, and a destroyed handle may be recycled -- but
 * only for a *different* view, so the content key (which compares both the view
 * and sampler handle) cannot match it unless the draw really is asking for the
 * handle that now exists, in which case the set describes it correctly. Every
 * rewrite also writes all bindings, with a placeholder for the empty ones, so a
 * reused slot can never keep a name that no longer resolves.
 *
 * The set contents are a pure function of the bound (view, sampler) pairs, and a
 * draw almost always repeats a pairing it has used before -- the same shadow map
 * at the same unit, frame after frame -- so caching turns the per-draw cost into
 * a hash comparison instead of an update. Entries are compared field by field
 * rather than by hash alone, because a collision must not bind the wrong
 * textures.
 *
 * Returns VK_NULL_HANDLE when nothing is bound at all, in which case there is no
 * set to bind and the shader would have read nothing either way. */
static VkDescriptorSet
sz_descriptor_sampler_set(struct sz_context *ctx, VkCommandBuffer cmd,
                          const VkWriteDescriptorSet *writes, unsigned count)
{
   struct sz_screen *screen = ctx->screen;

   if (count == 0 || screen->descriptor_pool == VK_NULL_HANDLE)
      return VK_NULL_HANDLE;

   /* Canonical key: one entry per binding, zero where nothing is bound, so two
    * configurations compare equal regardless of the order they were collected
    * in. */
   VkImageView views[SZ_DESCRIPTOR_SAMPLER_BINDINGS] = { 0 };
   VkSampler samplers[SZ_DESCRIPTOR_SAMPLER_BINDINGS] = { 0 };

   for (unsigned i = 0; i < count; i++)
      for (unsigned j = 0; j < writes[i].descriptorCount; j++) {
         const VkDescriptorImageInfo *info = &writes[i].pImageInfo[j];
         const unsigned binding = writes[i].dstBinding + j;
         if (binding >= SZ_DESCRIPTOR_SAMPLER_BINDINGS)
            continue;
         views[binding] = info->imageView;
         samplers[binding] = info->sampler;
      }

   uint64_t hash = 0xcbf29ce484222325ull;
   hash = sz_hash_bytes(hash, views, sizeof(views));
   hash = sz_hash_bytes(hash, samplers, sizeof(samplers));

   for (unsigned i = 0; i < ARRAY_SIZE(screen->sampler_sets); i++) {
      const struct sz_sampler_set_cache_entry *e =
         &screen->sampler_sets[i];
      if (e->set == VK_NULL_HANDLE || e->hash != hash)
         continue;
      /* Field by field, not by hash alone: a collision must not bind the wrong
       * textures. */
      if (memcmp(e->views, views, sizeof(views)) != 0 ||
          memcmp(e->samplers, samplers, sizeof(samplers)) != 0)
         continue;

      screen->sampler_sets[i].used_value = screen->batch_state.next_value + 1;
      return e->set;
   }

   /* Miss. Take a free slot, or evict the least recently used one whose
    * submission has already finished -- rewriting a set the GPU may still be
    * reading is a data race the validation layer cannot see. */
   unsigned slot = ARRAY_SIZE(screen->sampler_sets);
   uint64_t oldest = UINT64_MAX;

   for (unsigned i = 0; i < ARRAY_SIZE(screen->sampler_sets); i++) {
      if (screen->sampler_sets[i].set == VK_NULL_HANDLE) {
         slot = i;
         break;
      }
      if (screen->sampler_sets[i].used_value <= screen->batch_state.completed_value &&
          screen->sampler_sets[i].used_value < oldest) {
         oldest = screen->sampler_sets[i].used_value;
         slot = i;
      }
   }

   if (slot == ARRAY_SIZE(screen->sampler_sets)) {
      /* completed_value only moves when somebody waits, so ask the device before
       * concluding that all 256 cached sets are still being read. */
      sz_queue_poll(&screen->batch_state);
      oldest = UINT64_MAX;
      for (unsigned i = 0; i < ARRAY_SIZE(screen->sampler_sets); i++) {
         if (screen->sampler_sets[i].used_value <=
                 screen->batch_state.completed_value &&
             screen->sampler_sets[i].used_value < oldest) {
            oldest = screen->sampler_sets[i].used_value;
            slot = i;
         }
      }
   }

   if (slot == ARRAY_SIZE(screen->sampler_sets)) {
      /* Every cached set is still in flight. There are at most as many of those
       * as the batch ring is deep, so this only happens with a pool far too
       * small; say so once rather than binding a set with stale contents. */
      sz_warn_once("the sampler descriptor cache is exhausted; this draw samples "
                   "through a set that is still in flight");
      return VK_NULL_HANDLE;
   }

   if (screen->sampler_sets[slot].set == VK_NULL_HANDLE) {
      const VkDescriptorSetAllocateInfo alloc = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
         .descriptorPool = screen->descriptor_pool,
         .descriptorSetCount = 1,
         .pSetLayouts = &screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_SAMPLERS],
      };
      VkDescriptorSet fresh = VK_NULL_HANDLE;
      if (screen->vk.AllocateDescriptorSets(screen->dev, &alloc, &fresh) !=
          VK_SUCCESS) {
         sz_warn_once("could not allocate a sampler descriptor set");
         return VK_NULL_HANDLE;
      }
      screen->sampler_sets[slot].set = fresh;
   }

   /* Every binding, not only the ones with content. A set is allocated empty and
    * a slot may be reused after a different configuration, so a binding left
    * alone would keep naming an image view that no longer exists -- undefined in
    * Vulkan and a GPU hang on RADV. The placeholder makes the rewrite depend on
    * nothing but the configuration being written. */
   VkDescriptorImageInfo infos[SZ_DESCRIPTOR_SAMPLER_BINDINGS];
   VkWriteDescriptorSet full[SZ_DESCRIPTOR_SAMPLER_BINDINGS];

   for (unsigned b = 0; b < SZ_DESCRIPTOR_SAMPLER_BINDINGS; b++) {
      if (views[b] != VK_NULL_HANDLE) {
         infos[b] = (VkDescriptorImageInfo) {
            .sampler = samplers[b],
            .imageView = views[b],
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         };
      } else {
         infos[b] = (VkDescriptorImageInfo) {
            .sampler = screen->dummy_sampler.sampler,
            .imageView = screen->dummy_sampler.view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         };
      }

      full[b] = (VkWriteDescriptorSet) {
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = screen->sampler_sets[slot].set,
         .dstBinding = b,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &infos[b],
      };
   }

   screen->vk.UpdateDescriptorSets(screen->dev, SZ_DESCRIPTOR_SAMPLER_BINDINGS,
                                   full, 0, NULL);

   memcpy(screen->sampler_sets[slot].views, views, sizeof(views));
   memcpy(screen->sampler_sets[slot].samplers, samplers, sizeof(samplers));
   screen->sampler_sets[slot].hash = hash;
   screen->sampler_sets[slot].used_value = screen->batch_state.next_value + 1;

   return screen->sampler_sets[slot].set;
}

/* Writes both sets into the batch's descriptor buffer and selects them.
 *
 * The buffer is bound immediately before the offsets that index into it, in the
 * same command buffer. Binding once per command buffer would be one command
 * cheaper and is what the first version did, but a binding does not survive
 * being split across two command buffers, and the draw path does open a new one
 * mid-frame -- the offsets then reference a buffer index nothing bound
 * (VUID-vkCmdSetDescriptorBufferOffsetsEXT-pBufferIndices-08065).
 *
 * Returns false when there is no room left, which is reported rather than
 * silently writing past the end: a descriptor written out of bounds is a GPU
 * fault, not a validation error. */
static bool
sz_descriptor_db_emit(struct sz_context *ctx, VkCommandBuffer cmd)
{
   struct sz_screen *screen = ctx->screen;
   if (!screen->db.enabled || !ctx->batch)
      return false;

   const unsigned slot = ctx->batch->ring_index;
   if (slot >= SZ_BATCH_RING_SIZE)
      return false;

   const uint64_t align =
      MAX2(screen->caps.descriptor_buffer_offset_alignment, 1ull);

   VkDeviceSize ubo_offset = align64(screen->db.ring[slot].used, align);
   VkDeviceSize sampler_offset = align64(ubo_offset + screen->db.ubo_size, align);
   VkDeviceSize texel_offset =
      align64(sampler_offset + screen->db.sampler_size, align);
   VkDeviceSize image_offset = align64(texel_offset + screen->db.texel_size,
                                       align);

   if (image_offset + screen->db.image_size > screen->db.ring[slot].size)
      return false;

   screen->db.ring[slot].used = image_offset + screen->db.image_size;

   uint8_t *const mapped = screen->db.ring[slot].mapped;
   const size_t ubo_size = screen->caps.descriptor_size_uniform_buffer;
   const size_t sampler_size =
      screen->caps.descriptor_size_combined_image_sampler;
   const size_t texel_size = screen->caps.descriptor_size_uniform_texel_buffer;
   const VkDescriptorSetLayout ubo_layout =
      screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_UBO];
   const VkDescriptorSetLayout sampler_layout =
      screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_SAMPLERS];
   const VkDeviceSize texel_layout_size = screen->db.texel_size;
   const VkDescriptorSetLayout texel_layout =
      screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_TEXEL_BUFFERS];

   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];

      /* ── uniform buffers ──────────────────────────────────────────────── */
      for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++) {
         VkDeviceSize binding_offset = 0;
         screen->vk.GetDescriptorSetLayoutBindingOffsetEXT(
            screen->dev, ubo_layout, sz_descriptor_binding(stage, i),
            &binding_offset);

         const struct pipe_constant_buffer *cb =
            &ctx->state.constant_buffers[stage][i];

         VkDescriptorAddressInfoEXT address = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
            .format = VK_FORMAT_UNDEFINED,
            .address = screen->dummy_ubo.address,
            .range = 256,
         };

         if (cb->buffer) {
            struct sz_resource *res = sz_resource(cb->buffer);
            if (res->buffer != VK_NULL_HANDLE) {
               sz_resource_access(screen, cmd, res, SZ_ACCESS_UNIFORM_BUFFER);

               const uint64_t offset = cb->buffer_offset;
               uint64_t range = cb->buffer_size ? cb->buffer_size : VK_WHOLE_SIZE;
               if (range != VK_WHOLE_SIZE)
                  range = MIN2(range, (uint64_t)cb->buffer->width0 - offset);

               address.address = res->address + offset;
               address.range = range;
            }
         }

         const VkDescriptorGetInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .data.pUniformBuffer = &address,
         };
         screen->vk.GetDescriptorEXT(screen->dev, &info, ubo_size,
                                     mapped + ubo_offset + binding_offset);
      }

      /* ── shader storage blocks ────────────────────────────────────────── */
      for (unsigned i = 0; i < SZ_MAX_SSBO; i++) {
         VkDeviceSize binding_offset = 0;
         screen->vk.GetDescriptorSetLayoutBindingOffsetEXT(
            screen->dev, ubo_layout, sz_ssbo_binding(stage, i), &binding_offset);

         const struct pipe_shader_buffer *sb =
            &ctx->state.shader_buffers[stage][i];

         VkDescriptorAddressInfoEXT address = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
            .format = VK_FORMAT_UNDEFINED,
            .address = screen->dummy_ubo.address,
            .range = 256,
         };

         if (sb->buffer) {
            struct sz_resource *res = sz_resource(sb->buffer);
            if (res->buffer != VK_NULL_HANDLE) {
               sz_resource_access(screen, cmd, res, SZ_ACCESS_STORAGE_BUFFER);

               const uint64_t offset = sb->buffer_offset;
               uint64_t range = sb->buffer_size ? sb->buffer_size : VK_WHOLE_SIZE;
               if (range != VK_WHOLE_SIZE)
                  range = MIN2(range, (uint64_t)sb->buffer->width0 - offset);

               address.address = res->address + offset;
               address.range = range;
            }
         }

         const VkDescriptorGetInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .data.pStorageBuffer = &address,
         };
         screen->vk.GetDescriptorEXT(
            screen->dev, &info, screen->caps.descriptor_size_storage_buffer,
            mapped + ubo_offset + binding_offset);
      }

      /* ── combined image samplers ──────────────────────────────────────── */
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         VkDeviceSize binding_offset = 0;
         screen->vk.GetDescriptorSetLayoutBindingOffsetEXT(
            screen->dev, sampler_layout, sz_sampler_binding(stage, i),
            &binding_offset);

         const struct sz_state *state = NULL;
         const struct sz_sampler_view *sv = NULL;
         const struct sz_resource *res =
            sz_sampled_resource(ctx, stage, i, &state, &sv);

         /* An unbound unit names the placeholder: GL says it samples as black,
          * and Vulkan has no such rule -- a descriptor naming a destroyed view
          * is undefined, and on RADV it hangs. */
         VkDescriptorImageInfo image = {
            .sampler = res ? state->u.sampler.sampler
                           : screen->dummy_sampler.sampler,
            .imageView = res ? (ctx->sampled_override.view[s][i]
                                   ? ctx->sampled_override.view[s][i]
                                   : sv->view)
                             : screen->dummy_sampler.view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
         };

         const VkDescriptorGetInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .data.pCombinedImageSampler = &image,
         };
         screen->vk.GetDescriptorEXT(screen->dev, &info, sampler_size,
                                     mapped + sampler_offset + binding_offset);
      }

      /* ── texel buffers ────────────────────────────────────────────────── */
      for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
         VkDeviceSize binding_offset = 0;
         screen->vk.GetDescriptorSetLayoutBindingOffsetEXT(
            screen->dev, texel_layout, sz_texel_binding(stage, i),
            &binding_offset);

         const struct pipe_sampler_view *view = ctx->state.sampler_views[stage][i];
         /* A descriptor-buffer texel descriptor is an address, a range and a
          * format -- not a VkBufferView, which only the pool path needs. An
          * unbound unit has nothing to name and no placeholder exists, so it is
          * left alone: GL samples an unbound texture unit as zero, and st binds
          * a real one before any such sample can reach the device. */
         if (!view || !view->texture || view->texture->target != PIPE_BUFFER)
            continue;

         struct sz_resource *res = sz_resource(view->texture);
         if (res->buffer == VK_NULL_HANDLE)
            continue;

         sz_resource_access(screen, cmd, res, SZ_ACCESS_TEXEL_BUFFER);

         const uint64_t offset = res->address + view->u.buf.offset;
         VkDescriptorAddressInfoEXT address = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
            .address = offset,
            .range = view->u.buf.size ? view->u.buf.size : VK_WHOLE_SIZE,
            .format = sz_vk_format(screen, view->format),
         };

         const VkDescriptorGetInfoEXT texel_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
            .data.pUniformTexelBuffer = &address,
         };
         screen->vk.GetDescriptorEXT(screen->dev, &texel_info, texel_size,
                                     mapped + texel_offset + binding_offset);
      }
   }

   /* ── storage images ──────────────────────────────────────────────────── */
   for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
      const mesa_shader_stage stage = sz_descriptor_stages[s];
      const VkDescriptorSetLayout image_layout =
         screen->descriptor_set_layout[SZ_DESCRIPTOR_SET_IMAGES];

      for (unsigned i = 0; i < SZ_MAX_IMAGES; i++) {
         const struct pipe_image_view *view = ctx->state.shader_images[stage][i];
         if (!view || !view->resource)
            continue;

         VkImageView image_view;
         if (!sz_shader_image_view(ctx, cmd, view, &image_view))
            continue;

         VkDeviceSize binding_offset = 0;
         screen->vk.GetDescriptorSetLayoutBindingOffsetEXT(
            screen->dev, image_layout, sz_image_binding(stage, i),
            &binding_offset);

         const VkDescriptorImageInfo image = {
            .imageView = image_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
         };
         const VkDescriptorGetInfoEXT info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .data.pStorageImage = &image,
         };
         screen->vk.GetDescriptorEXT(
            screen->dev, &info, screen->caps.descriptor_size_storage_image,
            mapped + image_offset + binding_offset);
      }
   }

   const VkDescriptorBufferBindingInfoEXT binding = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT,
      .address = screen->db.ring[slot].address,
      .usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
               VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT,
   };
   screen->vk.CmdBindDescriptorBuffersEXT(cmd, 1, &binding);

   /* Every set lives in slot 0 of the ring slot's buffer, so the index array is
    * all zeroes -- but it has to have descriptorSetCount entries: Vulkan reads
    * pBufferIndices[0..count-1], and a shorter array is a stack overread that
    * shows up as pBufferIndices[2] being several billion
    * (VUID-vkCmdSetDescriptorBufferOffsetsEXT-pBufferIndices-08064). */
   const uint32_t buffer_indices[SZ_DESCRIPTOR_SET_COUNT] = { 0, 0, 0, 0 };
   const VkDeviceSize offsets[SZ_DESCRIPTOR_SET_COUNT] = { ubo_offset,
                                                           sampler_offset,
                                                           texel_offset,
                                                           image_offset };
   screen->vk.CmdSetDescriptorBufferOffsetsEXT(
      cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, screen->pipeline_layout,
      SZ_DESCRIPTOR_SET_UBO, SZ_DESCRIPTOR_SET_COUNT, buffer_indices, offsets);


   return true;
}

/* Whether the batch being recorded still has room for one more pair of sets.
 *
 * Checked before a batch is extended rather than when it is flushed, because a
 * flush that runs out has already committed to the command buffer it is
 * recording into: the only safe reaction there is to drop the draw, and the only
 * good reaction is not to get there. */
bool
sz_descriptor_has_room(struct sz_context *ctx)
{
   struct sz_screen *screen = ctx->screen;
   if (!screen->db.enabled || !ctx->batch)
      return true;

   const unsigned slot = ctx->batch->ring_index;
   if (slot >= SZ_BATCH_RING_SIZE)
      return false;

   const uint64_t align =
      MAX2(screen->caps.descriptor_buffer_offset_alignment, 1ull);
   const VkDeviceSize need = align64(screen->db.ubo_size, align) +
                             align64(screen->db.sampler_size, align);
   return align64(screen->db.ring[slot].used, align) + need <=
          screen->db.ring[slot].size;
}

void
sz_descriptor_push(struct sz_context *ctx, VkCommandBuffer cmd)
{
   struct sz_screen *screen = ctx->screen;
   /* One layout for the whole screen (ADR-005c), so the push does not depend on
    * which pipeline is bound -- only on the descriptor set layouts, which never
    * change. */
   const VkPipelineLayout layout = screen->pipeline_layout;
   if (layout == VK_NULL_HANDLE)
      return;

   /* On a device with VK_EXT_descriptor_buffer the layouts were created for that
    * path and cannot be pushed to, so the descriptor buffer is the only way to
    * get descriptors to the device. There is deliberately no fallback: pushing
    * to a layout that was not created for it is a validation error and binds
    * nothing, which shows up as a black frame rather than as a failure. */
   if (screen->db.enabled) {
      if (!sz_descriptor_db_emit(ctx, cmd))
         sz_warn_once("the descriptor buffer for this batch is full");
      return;
   }

   /* Uniform buffers: one push for both stages. A push names bindings rather than
    * a whole set, and each binding carries its own stage flags in the layout, so
    * one call covers vertex and fragment without the second disturbing the
    * first. */
   {
      VkDescriptorBufferInfo infos[SZ_DESCRIPTOR_BUFFER_BINDINGS];
      VkWriteDescriptorSet writes[SZ_DESCRIPTOR_BUFFER_BINDINGS];
      unsigned count = 0;

      for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
         const mesa_shader_stage stage = sz_descriptor_stages[s];

         for (unsigned i = 0; i < SZ_MAX_CONST_BUFFERS; i++) {
            const struct pipe_constant_buffer *cb =
               &ctx->state.constant_buffers[stage][i];
            if (!cb->buffer)
               continue;

            struct sz_resource *res = sz_resource(cb->buffer);
            if (res->buffer == VK_NULL_HANDLE)
               continue;

            sz_resource_access(screen, cmd, res, SZ_ACCESS_UNIFORM_BUFFER);

            /* A zero range means "the rest of the buffer", which Vulkan spells
             * VK_WHOLE_SIZE; passing zero through is a validation error. */
            const uint64_t offset = cb->buffer_offset;
            uint64_t range = cb->buffer_size ? cb->buffer_size
                                             : VK_WHOLE_SIZE;
            if (range != VK_WHOLE_SIZE)
               range = MIN2(range, (uint64_t)cb->buffer->width0 - offset);

            infos[count] = (VkDescriptorBufferInfo) {
               .buffer = res->buffer,
               .offset = offset,
               .range = range,
            };

            writes[count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               /* dstSet is ignored by a push; the write goes to the layout the
                * pipeline was created with, at dstBinding. */
               .dstSet = VK_NULL_HANDLE,
               .dstBinding = sz_descriptor_binding(stage, i),
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
               .pBufferInfo = &infos[count],
            };

            count++;
         }
      }

      for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
         const mesa_shader_stage stage = sz_descriptor_stages[s];

         for (unsigned i = 0; i < SZ_MAX_SSBO; i++) {
            const struct pipe_shader_buffer *sb =
               &ctx->state.shader_buffers[stage][i];
            if (!sb->buffer)
               continue;

            struct sz_resource *res = sz_resource(sb->buffer);
            if (res->buffer == VK_NULL_HANDLE)
               continue;

            sz_resource_access(screen, cmd, res, SZ_ACCESS_STORAGE_BUFFER);

            const uint64_t offset = sb->buffer_offset;
            uint64_t range = sb->buffer_size ? sb->buffer_size : VK_WHOLE_SIZE;
            if (range != VK_WHOLE_SIZE)
               range = MIN2(range, (uint64_t)sb->buffer->width0 - offset);

            infos[count] = (VkDescriptorBufferInfo) {
               .buffer = res->buffer,
               .offset = offset,
               .range = range,
            };

            writes[count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = VK_NULL_HANDLE,
               .dstBinding = sz_ssbo_binding(stage, i),
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
               .pBufferInfo = &infos[count],
            };

            count++;
         }
      }

      if (count) {
         screen->vk.CmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                            layout, SZ_DESCRIPTOR_SET_UBO, count,
                                            writes);
      }
   }

   /* Combined image samplers: set 1, cached by content. A unit is written only
    * when it has both a view and a sampler, because Vulkan rejects a combined
    * image sampler with either half missing while GL allows a sampler uniform to
    * be sampled with no texture bound (which reads as black, not as an error). */
   {
      VkDescriptorImageInfo image_infos[SZ_DESCRIPTOR_SAMPLER_BINDINGS];
      VkWriteDescriptorSet writes[SZ_DESCRIPTOR_SAMPLER_BINDINGS];
      unsigned count = 0;

      for (unsigned s = 0; s < ARRAY_SIZE(sz_descriptor_stages); s++) {
         const mesa_shader_stage stage = sz_descriptor_stages[s];

         for (unsigned i = 0; i < SZ_MAX_SAMPLERS; i++) {
            const struct sz_state *state = NULL;
            const struct sz_sampler_view *sv = NULL;
            struct sz_resource *res =
               sz_sampled_resource(ctx, stage, i, &state, &sv);
            if (!res)
               continue;

            image_infos[count] = (VkDescriptorImageInfo) {
               .sampler = state->u.sampler.sampler,
               .imageView = ctx->sampled_override.view[s][i]
                               ? ctx->sampled_override.view[s][i]
                               : sv->view,
               /* sz_descriptor_prepare_sampled() put the image in this layout
                * before the render pass opened; a descriptor that claims
                * otherwise makes the sampled reads undefined. */
               .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            writes[count] = (VkWriteDescriptorSet) {
               .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
               .dstSet = VK_NULL_HANDLE,
               .dstBinding = sz_sampler_binding(stage, i),
               .descriptorCount = 1,
               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
               .pImageInfo = &image_infos[count],
            };

            count++;
         }
      }

      const VkDescriptorSet set =
         sz_descriptor_sampler_set(ctx, cmd, writes, count);
      if (set != VK_NULL_HANDLE && set != screen->bound_sampler_set) {
         const VkDescriptorSet sets[] = { set };
         screen->vk.CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                          layout, SZ_DESCRIPTOR_SET_SAMPLERS, 1,
                                          sets, 0, NULL);
         screen->bound_sampler_set = set;
      }
   }
}
