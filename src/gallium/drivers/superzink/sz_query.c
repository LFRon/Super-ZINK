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
static bool
sz_query_pool_type(enum pipe_query_type type, VkQueryType *out)
{
   switch (type) {
   case PIPE_QUERY_OCCLUSION_COUNTER:
   case PIPE_QUERY_OCCLUSION_PREDICATE:
   case PIPE_QUERY_OCCLUSION_PREDICATE_CONSERVATIVE:
      *out = VK_QUERY_TYPE_OCCLUSION;
      return true;
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
   if (!sz_query_pool_type((enum pipe_query_type)query_type, &vk_type)) {
      /* A NULL return is the API's way of saying "this query type is not
       * supported", and st turns it into a GL error rather than a crash. */
      return NULL;
   }

   const VkQueryPoolCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = vk_type,
      .queryCount = 1,
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

bool
sz_begin_query(struct pipe_context *pipe, struct pipe_query *query)
{
   struct sz_context *ctx = sz_context(pipe);
   struct sz_query *q = (struct sz_query *)query;

   if (!q)
      return false;

   /* Queued draws belong before the query. */
   sz_draw_flush(ctx);

   /* A query must be reset before it can be begun again: after a result was
    * read it is "available", and starting an available query is what
    * VUID-vkCmdBeginQuery-None-00807 rejects. Resetting is cheap (a host call),
    * but it may only happen once the device is done with the previous use. */
   if (q->pending)
      sz_queue_wait_idle(&ctx->screen->batch_state);

   ctx->screen->vk.ResetQueryPool(ctx->screen->dev, q->pool, 0, 1);
   q->pending = false;
   q->has_result = false;

   /* The query has to start outside a render pass: Vulkan requires begin and end
    * to be on the same side of the boundary, and the draws between them are
    * inside one. Closing the pass here and in sz_end_query keeps both outside,
    * which is always legal. */
   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return false;

   ctx->screen->vk.CmdBeginQuery(cmd, q->pool, 0, 0);
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

   VkCommandBuffer cmd = sz_context_cmd_no_pass(ctx);
   if (cmd == VK_NULL_HANDLE)
      return false;

   ctx->screen->vk.CmdEndQuery(cmd, q->pool, 0);
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

   uint64_t value = q->value;
   const VkResult result =
      screen->vk.GetQueryPoolResults(screen->dev, q->pool, 0, 1, sizeof(value),
                                     &value, sizeof(value),
                                     VK_QUERY_RESULT_64_BIT);
   if (result == VK_NOT_READY)
      return false;
   if (result != VK_SUCCESS) {
      mesa_loge("SZ: vkGetQueryPoolResults failed: %d", result);
      return false;
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
