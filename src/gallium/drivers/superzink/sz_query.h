/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Query objects.
 *
 * Only the two groups GL 3.0 needs are implemented: occlusion queries and the
 * conditional rendering built on top of them. Everything else (timestamps,
 * transform feedback statistics, pipeline statistics) arrives with the version
 * that requires it, because each maps to a different VkQueryType and a
 * different result layout.
 */

#ifndef SZ_QUERY_H
#define SZ_QUERY_H

#include "sz_context.h"

/* Gallium leaves `struct pipe_query` to the driver, so this is its definition.
 * It has to be the first member of the driver's own object: the API hands the
 * pointer back verbatim. */
struct pipe_query {
   enum pipe_query_type type;
};

struct sz_query {
   struct pipe_query base;
   enum pipe_query_type type;

   VkQueryPool pool;

   /* A begin/end pair was recorded but its result has not been read yet, so the
    * pool may still be in use by the device. Resetting it has to wait. */
   bool pending;

   /* A result has been read at least once since the last begin. GL lets the same
    * result be queried repeatedly (and conditional rendering reads the very
    * query the app already read), so "already read" must not mean "unavailable". */
   bool has_result;

   /* Last value read, so a repeated GL_QUERY_RESULT does not submit again. */
   uint64_t value;
};

/* Reads a query result as a 64-bit value. `wait` blocks until the result is
 * available; without it the call returns false when the device is not done.
 * Before reading, everything recorded so far is submitted. */
bool sz_query_result_u64(struct sz_context *ctx, struct pipe_query *query,
                         uint64_t *out, bool wait);

#endif /* SZ_QUERY_H */
