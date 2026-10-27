/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_types.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_inlines.h"

#include <inttypes.h>
#include <string.h>

/* Returns the resolved queue for a family, or NULL when the device exposed no
 * queue in it. */
static const struct sz_queue *
sz_queue_for_family(struct sz_screen *screen, uint32_t family)
{
   for (unsigned i = 0; i < SZ_QUEUE_COUNT; i++) {
      if (screen->queues[i].family == family)
         return &screen->queues[i];
   }
   return NULL;
}

/* Reads the timeline's current value without blocking. */
static uint64_t
sz_timeline_value(struct sz_screen *screen, struct sz_queue_state *qs)
{
   uint64_t value = 0;
   VkResult result = screen->vk.GetSemaphoreCounterValue(screen->dev,
                                                          qs->timeline, &value);
   if (result != VK_SUCCESS) {
      /* Cannot happen for a live timeline semaphore. Falling back to the cached
       * value only reuses a batch slot late; using zero would reuse it early. */
      mesa_loge("SZ: vkGetSemaphoreCounterValue failed: %d", result);
      return qs->completed_value;
   }

   if (value > qs->completed_value)
      qs->completed_value = value;
   return value;
}

/* Public form of the non-blocking poll above, for callers outside the batch
 * layer that need an up-to-date completed_value without waiting. */
void
sz_queue_poll(struct sz_queue_state *qs)
{
   sz_timeline_value(qs->screen, qs);
}

/* Blocks until `value` is signalled. A value of zero is always satisfied,
 * which is how a never-submitted slot passes this. */
static bool
sz_timeline_wait(struct sz_screen *screen, struct sz_queue_state *qs,
                 uint64_t value)
{
   if (value == 0 || value <= sz_timeline_value(screen, qs))
      return true;

   const VkSemaphoreWaitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &qs->timeline,
      .pValues = &value,
   };

   VkResult result = screen->vk.WaitSemaphores(screen->dev, &info, UINT64_MAX);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkWaitSemaphores(%" PRIu64 ") failed: %d", value, result);
      return false;
   }

   qs->completed_value = MAX2(qs->completed_value, value);
   return true;
}

static bool
sz_timeline_init(struct sz_screen *screen, struct sz_queue_state *qs)
{
   const VkSemaphoreTypeCreateInfo type_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
      .initialValue = 0,
   };

   const VkSemaphoreCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &type_info,
   };

   VkResult result = screen->vk.CreateSemaphore(screen->dev, &info, NULL,
                                                 &qs->timeline);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateSemaphore(timeline) failed: %d", result);
      return false;
   }

   return true;
}

bool
sz_queue_state_init(struct sz_queue_state *qs, struct sz_screen *screen,
                    uint32_t family)
{
   memset(qs, 0, sizeof(*qs));
   qs->screen = screen;
   qs->family = family;

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      /* RESET_COMMAND_BUFFER is required because the ring resets individual
       * buffers as it recycles them. */
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = family,
   };

   VkResult result = screen->vk.CreateCommandPool(screen->dev, &pool_info, NULL,
                                                   &qs->pool);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateCommandPool(family %u) failed: %d", family,
                result);
      qs->screen = NULL;
      return false;
   }

   if (!sz_timeline_init(screen, qs))
      goto fail_pool;

   for (unsigned i = 0; i < SZ_BATCH_RING_SIZE; i++) {
      qs->ring[i].screen = screen;
      qs->ring[i].qs = qs;
      qs->ring[i].ring_index = i;
   }

   VkCommandBuffer cmds[SZ_BATCH_RING_SIZE] = {0};
   const VkCommandBufferAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = qs->pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = SZ_BATCH_RING_SIZE,
   };

   result = screen->vk.AllocateCommandBuffers(screen->dev, &alloc_info, cmds);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkAllocateCommandBuffers failed: %d", result);
      goto fail_timeline;
   }

   for (unsigned i = 0; i < SZ_BATCH_RING_SIZE; i++)
      qs->ring[i].cmd = cmds[i];

   return true;

fail_timeline:
   screen->vk.DestroySemaphore(screen->dev, qs->timeline, NULL);
   qs->timeline = VK_NULL_HANDLE;
fail_pool:
   screen->vk.DestroyCommandPool(screen->dev, qs->pool, NULL);
   qs->pool = VK_NULL_HANDLE;
   qs->screen = NULL;
   return false;
}

void
sz_queue_state_finish(struct sz_queue_state *qs)
{
   struct sz_screen *screen = qs->screen;
   if (!screen)
      return;

   if (qs->pool != VK_NULL_HANDLE) {
      /* vkDestroyCommandPool frees every buffer allocated from it. The caller
       * must have idled the queue first; doing it here would paper over a
       * lifetime bug that would come back as a GPU hang later. */
      screen->vk.DestroyCommandPool(screen->dev, qs->pool, NULL);
      qs->pool = VK_NULL_HANDLE;
   }

   if (qs->timeline != VK_NULL_HANDLE) {
      screen->vk.DestroySemaphore(screen->dev, qs->timeline, NULL);
      qs->timeline = VK_NULL_HANDLE;
   }

   qs->screen = NULL;
}

void
sz_queue_wait_idle(struct sz_queue_state *qs)
{
   struct sz_screen *screen = qs->screen;
   if (!screen)
      return;

   const struct sz_queue *queue = sz_queue_for_family(screen, qs->family);

   if (!queue) {
      /* Nothing can have been submitted without a queue, but the timeline is
       * still the authority on what completed. */
      sz_timeline_wait(screen, qs, qs->next_value);
   } else if (qs->next_value > sz_timeline_value(screen, qs)) {
      /* vkQueueWaitIdle is a full device-wide serialisation on some drivers, so
       * only pay for it when work is actually outstanding. */
      VkResult result = screen->vk.QueueWaitIdle(queue->queue);
      if (result != VK_SUCCESS)
         mesa_loge("SZ: vkQueueWaitIdle failed: %d", result);
      qs->completed_value = qs->next_value;
   } else {
      qs->completed_value = qs->next_value;
   }
}

struct sz_batch *
sz_batch_begin(struct sz_queue_state *qs)
{
   struct sz_screen *screen = qs->screen;

   struct sz_batch *batch = &qs->ring[qs->next_slot];
   qs->next_slot = (qs->next_slot + 1) % SZ_BATCH_RING_SIZE;

   /* Recycle only once the GPU is done with this slot. Waiting here is what
    * bounds the number of in-flight batches; doing it anywhere else would let the
    * ring run ahead of the device. */
   if (!sz_timeline_wait(screen, qs, batch->submit_value))
      return NULL;
   batch->submit_value = 0;

   VkResult result = screen->vk.ResetCommandBuffer(batch->cmd, 0);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkResetCommandBuffer failed: %d", result);
      return NULL;
   }

   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };

   result = screen->vk.BeginCommandBuffer(batch->cmd, &begin);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkBeginCommandBuffer failed: %d", result);
      return NULL;
   }

   batch->recording = true;
   return batch;
}

bool
sz_batch_end(struct sz_batch *batch)
{
   if (!batch->recording)
      return false;

   VkResult result = batch->screen->vk.EndCommandBuffer(batch->cmd);
   batch->recording = false;

   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkEndCommandBuffer failed: %d", result);
      return false;
   }

   return true;
}

bool
sz_batch_submit(struct sz_batch *batch, VkPipelineStageFlags2 wait_stage)
{
   struct sz_screen *screen = batch->screen;
   struct sz_queue_state *qs = batch->qs;

   if (batch->recording) {
      mesa_loge("SZ: submitted a batch that is still recording");
      return false;
   }

   const struct sz_queue *queue = sz_queue_for_family(screen, qs->family);
   if (!queue) {
      mesa_loge("SZ: no queue in family %u to submit to", qs->family);
      return false;
   }

   const uint64_t signal_value = ++qs->next_value;

   const VkCommandBufferSubmitInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = batch->cmd,
   };

   const VkSemaphoreSubmitInfo signal_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = qs->timeline,
      .value = signal_value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };

   /* Waiting on the previous value of the same timeline is exactly "everything
    * submitted before me on this queue". A single Vulkan queue already executes
    * in submission order, so this costs nothing and keeps the timeline as the
    * single source of truth for completion. */
   const VkSemaphoreSubmitInfo wait_info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = qs->timeline,
      .value = signal_value - 1,
      .stageMask = wait_stage ? wait_stage
                              : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };

   const bool has_wait = signal_value > 1;

   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = has_wait ? 1 : 0,
      .pWaitSemaphoreInfos = has_wait ? &wait_info : NULL,
      .commandBufferInfoCount = 1,
      .pCommandBufferInfos = &cmd_info,
      .signalSemaphoreInfoCount = 1,
      .pSignalSemaphoreInfos = &signal_info,
   };

   VkResult result = screen->vk.QueueSubmit2(queue->queue, 1, &submit,
                                              VK_NULL_HANDLE);
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkQueueSubmit2 failed: %d", result);
      /* The value was consumed on the CPU but never signalled on the GPU, so
       * roll it back: otherwise every later wait would block forever on a value
       * that will never be reached. */
      qs->next_value--;
      return false;
   }

   batch->submit_value = signal_value;
   return true;
}

/* ------------------------------------------------------------------------- */
/* Fences                                                                     */
/* ------------------------------------------------------------------------- */

struct sz_fence {
   struct pipe_reference reference;
};

static void
sz_fence_destroy(struct sz_fence *fence)
{
   free(fence);
}

void
sz_fence_reference(struct pipe_screen *pscreen, struct pipe_fence_handle **ptr,
                   struct pipe_fence_handle *fence)
{
   struct sz_fence *old = (struct sz_fence *)*ptr;
   struct sz_fence *new = (struct sz_fence *)fence;

   /* pipe_reference takes the addresses of the two refcounts rather than the
    * objects, so a NULL on either side has to be turned into a NULL pointer
    * before it reaches the helper. */
   if (pipe_reference(old ? &old->reference : NULL,
                      new ? &new->reference : NULL))
      sz_fence_destroy(old);

   *ptr = fence;
}

bool
sz_fence_finish(struct pipe_screen *pscreen, struct pipe_context *pipe,
                struct pipe_fence_handle *fence, uint64_t timeout)
{
   return true;
}

int
sz_fence_get_fd(struct pipe_screen *pscreen, struct pipe_fence_handle *fence)
{
   /* No syncobj export yet; -1 means "no fd", which callers handle. */
   return -1;
}

void
sz_screen_fence_init(struct sz_screen *screen)
{
   screen->base.fence_reference = sz_fence_reference;
   screen->base.fence_finish = sz_fence_finish;
   screen->base.fence_get_fd = sz_fence_get_fd;
}
