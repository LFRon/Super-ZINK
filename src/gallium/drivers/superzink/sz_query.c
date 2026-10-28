/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_query.h"

#include "sz_batch.h"
#include "sz_debug.h"

#include "util/log.h"
#include "util/macros.h"
#include "util/u_memory.h"

/* One pool per query object. A pool with one slot costs one small object per
 * query, and a query object is created once per GL query name -- not per frame.
 * The alternative, one big pool with rotating slots, needs a completion value
 * per slot to know when a slot may be reset; that is only worth it once
 * timestamp queries make queries cheap enough to issue by the thousand. */
typedef bool (*sz_query_supported_fn)(const struct sz_screen *screen);

static bool
sz_query_occlusion_supported(const struct sz_screen *screen)
{
   return true;
}

/* A timestamp query counts when the device reports a non-zero valid-bit count
 * and a period: without them vkCmdWriteTimestamp2 writes nothing useful. */
static bool
sz_query_timestamp_supported(const struct sz_screen *screen)
{
   return screen->have_timestamps &&
          screen->props.limits.timestampPeriod > 0.0f;
}

static bool
sz_query_pool_type(const struct sz_screen *screen, enum pipe_query_type type,
                   VkQueryType *out, unsigned *count)
{
   switch (type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      *out = VK_QUERY_TYPE_OCCLUSION;
      *count = 1;
      return sz_query_occlusion_supported(screen);

   case PIPE_QUERY_TIMESTAMP:
      *out = VK_QUERY_TYPE_TIMESTAMP;
      *count = 1;
      return sz_query_timestamp_supported(screen);

   case PIPE_QUERY_TIMESTAMP_DISJOINT:
      /* Nothing in Vulkan says a timestamp is contiguous or that the clock ran
       * at a constant rate, so the disjoint result is reported as "not
       * disjoint" with a period of one nanosecond per tick after conversion. */
      *out = VK_QUERY_TYPE_TIMESTAMP;
      *count = 1;
      return sz_query_timestamp_supported(screen);

   case PIPE_QUERY_TIME_ELAPSED:
      /* Two timestamps, and the answer is their difference. */
      *out = VK_QUERY_TYPE_TIMESTAMP;
      *count = 2;
      return sz_query_timestamp_supported(screen);

   default:
      return false;
   }
}

struct pipe_query *
sz_create_query(struct pipe_context *pipe, unsigned query_type, unsigned index)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_screen *screen = ctx->screen;

   VkQueryType vk_type;
   unsigned count;
   if (!sz_query_pool_type(screen, (enum pipe_query_type)query_type, &vk_type,
                           &count)) {
      /* A NULL return is the API's way of saying "this query type is not
       * supported", and st turns it into a GL error rather than a crash. */
      return NULL;
   }

   const VkQueryPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = vk_type,
      .queryCount = count,
   };

   VkQueryPool pool = VK_NULL_HANDLE;
   if (screen->vk.CreateQueryPool(screen->dev, &ci, NULL, &pool) != VK_SUCCESS) {
      mesa_loge("SZ: vkCreateQueryPool failed");
      return NULL;
   }

   struct sz_query *q = CALLOC_STRUCT(sz_query);
   if (!q) {
      screen->vk.DestroyQueryPool(screen->dev, pool, NULL);
      return NULL;
   }

   q->base.type = (enum pipe_query_type)query_type;
   q->type = (enum pipe_query_type)query_type;
   q->pool = pool;
   q->count = count;

   (void)index;
   return &q->base;
}

void
sz_destroy_query(struct pipe_context *pipe, struct pipe_query *query)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_query *q = (struct sz_query *)query;

   if (!q)
      return;

   /* The pool must outlive every submission that references it. A destroy is
    * rare (once per GL query name), so waiting is cheaper than a retirement
    * table and cannot deadlock: nothing is being waited on but this queue. */
   if (q->pending)
      sz_queue_wait_idle(&ctx->screen->batch_state);

   if (q->pool != VK_NULL_HANDLE)
      ctx->screen->vk.DestroyQueryPool(ctx->screen->dev, q->pool, NULL);

   free(q);
}

/* Records one timestamp of a timer query.
 *
 * The stage is ALL_COMMANDS rather than something finer: the value orders the
 * work around the query, and it is written outside a render pass (see the
 * comment in sz_begin_query), where targeting every stage is legal. */
static void
sz_query_timestamp(struct sz_context *ctx, VkCommandBuffer cmd,
                   struct sz_query *q, unsigned index)
{
   ctx->screen->vk.CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                      q->pool, index);
}

/* Nanoseconds out of raw ticks: timestampPeriod is the number of nanoseconds one
 * tick takes, as a float, and GL reports nanoseconds. */
static uint64_t
sz_query_ticks_to_ns(const struct sz_screen *screen, uint64_t ticks)
{
   return (uint64_t)((double)ticks * (double)screen->props.limits.timestampPeriod);
}

/* Makes a query writable again: after a result was read it is "available", and
 * writing or beginning an available query is what VUID-vkCmdBeginQuery-None-00807
 * and VUID-vkCmdWriteTimestamp2-None-03864 reject. Resetting is a host call, but
 * it may only happen once the device is done with the previous use. */
static void
sz_query_prepare(struct sz_context *ctx, struct sz_query *q)
{
   if (q->pending)
      sz_queue_wait_idle(&ctx->screen->batch_state);

   ctx->screen->vk.ResetQueryPool(ctx->screen->dev, q->pool, 0, q->count);
   q->pending = false;
   q->has_result = false;
}

bool
sz_begin_query(struct pipe_context *pipe, struct pipe_query *query)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_query *q = (struct sz_query *)query;

   if (!q)
      return false;

   /* Queued draws belong before the query. */
   sz_draw_flush(ctx);

   sz_query_prepare(ctx, q);
   q->begun = true;

   /* The query has to start outside a render pass: Vulkan requires begin and end
    * to be on the same side of the boundary, and the draws between them are
    * inside one. Closing the pass here and in sz_end_query keeps both outside,
    * which is always legal. */
   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return false;

   switch (q->type) {
   case PIPE_QUERY_TIME_ELAPSED:
      /* The first of the two timestamps whose difference is the answer. */
      sz_query_timestamp(ctx, cmd, q, 0);
      break;
   case PIPE_QUERY_TIMESTAMP:
   case PIPE_QUERY_TIMESTAMP_DISJOINT:
      /* A single timestamp is written by end_query -- glQueryCounter has no
       * begin at all -- and the disjointness is a property of the device. */
      break;
   default:
      ctx->screen->vk.CmdBeginQuery(cmd, q->pool, 0, 0);
      break;
   }

   return true;
}

bool
sz_end_query(struct pipe_context *pipe, struct pipe_query *query)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_query *q = (struct sz_query *)query;

   if (!q)
      return false;

   /* Everything recorded between begin and end belongs to the query. */
   sz_draw_flush(ctx);

   /* glQueryCounter writes a timestamp with no begin before it, and a query must
    * be reset before it is written (VUID-vkCmdWriteTimestamp2-None-03864). */
   if (!q->begun)
      sz_query_prepare(ctx, q);
   q->begun = false;

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return false;

   switch (q->type) {
   case PIPE_QUERY_TIMESTAMP:
      sz_query_timestamp(ctx, cmd, q, 0);
      break;
   case PIPE_QUERY_TIME_ELAPSED:
      sz_query_timestamp(ctx, cmd, q, 1);
      break;
   case PIPE_QUERY_TIMESTAMP_DISJOINT:
      /* Nothing to record: the answer is a property of the device, not of the
       * work between begin and end. */
      break;
   default:
      ctx->screen->vk.CmdEndQuery(cmd, q->pool, 0);
      break;
   }

   q->pending = true;
   return true;
}

static bool
sz_query_read(struct sz_context *ctx, struct sz_query *q, bool wait,
              uint64_t *out)
{
   const struct sz_screen *screen = ctx->screen;

   /* The query only becomes readable after the command buffer carrying it has
    * been submitted, so a caller that never flushed still gets a result. */
   sz_context_submit(ctx);

   if (wait)
      sz_queue_wait_idle(&ctx->screen->batch_state);

   uint64_t values[2] = { 0, 0 };
   const VkResult result = screen->vk.GetQueryPoolResults(
      screen->dev, q->pool, 0, q->count, sizeof(values), values,
      sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
   if (result == VK_NOT_READY)
      return false;
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkGetQueryPoolResults failed: %d", result);
      return false;
   }

   uint64_t value;
   switch (q->type) {
   case PIPE_QUERY_TIME_ELAPSED:
      /* Two timestamps around the work; the answer is how long it took. */
      value = sz_query_ticks_to_ns(screen, values[1] - values[0]);
      break;
   case PIPE_QUERY_TIMESTAMP_DISJOINT:
      /* Only "was the clock continuous" is read from it, and it always was. */
      value = 0;
      break;
   default:
      value = sz_query_ticks_to_ns(screen, values[0]);
      break;
   }

   q->pending = false;
   q->has_result = true;
   q->value = value;
   *out = value;
   return true;
}

bool
sz_query_result_u64(struct sz_context *ctx, struct pipe_query *query,
                    uint64_t *out, bool wait)
{
   struct sz_query *q = (struct sz_query *)query;

   if (!q)
      return false;
   if (!q->pending && !q->has_result)
      return false;
   /* A result that was already read is still valid: do not submit again. */
   if (!q->pending) {
      *out = q->value;
      return true;
   }

   return sz_query_read(ctx, q, wait, out);
}

bool
sz_get_query_result(struct pipe_context *pipe, struct pipe_query *query,
                    bool wait, union pipe_query_result *result)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_query *q = (struct sz_query *)query;

   uint64_t value = 0;
   if (!sz_query_result_u64(ctx, query, &value, wait))
      return false;

   switch (q->type) {
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      result->b = value != 0;
      break;
   case PIPE_QUERY_TIMESTAMP_DISJOINT:
      /* A nanosecond per tick, and the clock never was discontinuous: Vulkan
       * offers nothing that would tell us it had been. */
      result->timestamp_disjoint.frequency = 1000000000;
      result->timestamp_disjoint.disjoint = false;
      break;
   default:
      result->u64 = value;
      break;
   }

   return true;
}

void
sz_get_query_result_resource(struct pipe_context *pipe, struct pipe_query *query,
                             enum pipe_query_flags flags,
                             enum pipe_query_value_type result_type, int index,
                             struct pipe_resource *resource, unsigned offset)
{
   /* ARB_query_buffer_object is a GL 4.4 feature; advertising it before this
    * works would make glGetQueryBufferObject silently write garbage. */
   sz_warn_once("storing a query result into a buffer is not implemented");
}
