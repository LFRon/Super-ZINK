/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Command recording and submission.
 *
 * A "batch" is one recorded VkCommandBuffer together with the timeline value it
 * signals. Batches are recycled through a small ring, which is what keeps
 * recording cheap: allocating a command buffer, or resetting one the GPU is
 * still executing, costs far more than reusing one whose timeline value has
 * already passed.
 *
 * Submission uses one timeline semaphore per queue, not one binary semaphore per
 * submit. Binary semaphores have to be created, submitted and destroyed every
 * frame, and that per-submit cost dominates once the command count rises -- the
 * same "GPU pegged, whole desktop stutters" symptom Super-ZINK exists to avoid.
 * A timeline semaphore is one object for the lifetime of the screen, and waiting
 * on it is an integer comparison on the CPU.
 */

#ifndef SZ_BATCH_H
#define SZ_BATCH_H

#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

/* Forward-declared: sz_types.h includes this header for struct sz_queue_state,
 * so including it back would be a cycle. */
struct sz_screen;

/* Depth of the command buffer ring. Four is enough for the CPU to run a frame
 * ahead of the GPU without pinning four frames of memory; sz_batch_begin()
 * applies back-pressure beyond that. */
#define SZ_BATCH_RING_SIZE 4

struct sz_queue_state;

struct sz_batch {
   struct sz_screen *screen;
   struct sz_queue_state *qs;

   VkCommandBuffer cmd;
   uint32_t ring_index;

   /* Timeline value this batch signalled on submit; zero while it is free. */
   uint64_t submit_value;

   bool recording;
};

struct sz_queue_state {
   struct sz_screen *screen;

   /* Command buffers may only be submitted to a queue in their pool's family, so
    * there is one state per family, not one per queue. */
   uint32_t family;

   VkCommandPool pool;

   struct sz_batch ring[SZ_BATCH_RING_SIZE];

   /* Timeline shared by every submit on this family. next_value is the value the
    * next submit will signal. */
   VkSemaphore timeline;
   uint64_t next_value;
   uint64_t completed_value;

   /* Ring slot the next sz_batch_begin() will hand out. */
   unsigned next_slot;
};

/* Refreshes completed_value from the timeline without blocking. A caller that
 * wants to recycle something an in-flight submission may still be reading has to
 * do this first, or it will treat work the GPU finished long ago as pending. */
void sz_queue_poll(struct sz_queue_state *qs);

/* Creates the pool, the ring's command buffers and the timeline semaphore. */
bool sz_queue_state_init(struct sz_queue_state *qs, struct sz_screen *screen,
                         uint32_t family);

void sz_queue_state_finish(struct sz_queue_state *qs);

/* Acquires the next command buffer in the ring and starts recording. Blocks
 * while the slot being reused is still in flight, so at most SZ_BATCH_RING_SIZE
 * batches are outstanding.
 *
 * Returns the batch to record into, or NULL. The command buffers belong to the
 * queue state, so a caller cannot supply its own sz_batch -- doing so would hand
 * a NULL VkCommandBuffer to the first vkCmd* call. */
struct sz_batch *sz_batch_begin(struct sz_queue_state *qs);

/* Ends recording. Kept separate from submission so a caller that recorded
 * nothing useful can skip an empty submit. */
bool sz_batch_end(struct sz_batch *batch);

/* Submits the batch, ordered after everything already queued on this family.
 *
 * `wait_stage` is the stage at which earlier work must be visible before this
 * batch starts. Until there are draw calls there is nothing finer to say than
 * VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT. */
bool sz_batch_submit(struct sz_batch *batch, VkPipelineStageFlags2 wait_stage);

/* Blocks until everything submitted on this family has completed. Used by
 * pipe_context::flush and by the self-tests. */
void sz_queue_wait_idle(struct sz_queue_state *qs);

/* Fences.
 *
 * A Phase 1 fence carries no timeline value: submission is asynchronous but not
 * deferred, so "the work was handed to the queue" and "the fence is done" are the
 * same moment. Waiting on one therefore always succeeds immediately. The type
 * exists because the DRI frontend calls fence_reference unconditionally and a
 * NULL pointer there aborts the process on window close, long after anything the
 * driver did wrong. */
void sz_screen_fence_init(struct sz_screen *screen);
void sz_fence_reference(struct pipe_screen *pscreen,
                        struct pipe_fence_handle **ptr,
                        struct pipe_fence_handle *fence);
bool sz_fence_finish(struct pipe_screen *pscreen, struct pipe_context *pipe,
                     struct pipe_fence_handle *fence, uint64_t timeout);
int sz_fence_get_fd(struct pipe_screen *pscreen,
                    struct pipe_fence_handle *fence);

#endif /* SZ_BATCH_H */
