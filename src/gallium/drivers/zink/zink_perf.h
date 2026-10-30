/*
 * Copyright © 2026 The Mesa authors
 * SPDX-License-Identifier: MIT
 */

#ifndef ZINK_PERF_H
#define ZINK_PERF_H

#include <stdbool.h>
#include <stdint.h>

#include "util/os_time.h"

/* Lightweight counters for A/B work.
 *
 * These exist because zink has several paths that only pay off for specific
 * application shapes (vertex-state draws, zero-color-write passes, ...). A/B
 * numbers from a benchmark that never enters a path are meaningless, so being
 * able to ask "did this run enter the path at all, and how often" has to come
 * before deciding whether an optimization is worth keeping.
 *
 * ZINK_PERF=1              enable (counters still compile in, this only turns
 *                          the periodic dump on)
 * ZINK_PERF_PERIOD=<draws> draws between dumps, default 1000
 *
 * Every dump is a single stderr line so two runs can be diffed directly.
 */

/* Why a render pass was closed.
 *
 * Every place that ends a pass has to name its reason. This exists so the audit
 * table ("how many passes per frame, and who closed them") can be produced at
 * all: a break point that cannot be named is one nobody can reason about.
 *
 * Keep this list ordered and append-only; the dump prints it by index.
 */
enum zink_rp_closure_reason {
   /* context flush / swapbuffers */
   ZINK_RP_CLOSE_FLUSH,
   /* set_framebuffer_state */
   ZINK_RP_CLOSE_FRAMEBUFFER,
   /* zsbuf changed */
   ZINK_RP_CLOSE_ZSBUF,
   /* attachment imageView changed */
   ZINK_RP_CLOSE_ATTACHMENT_VIEW,
   /* begin_rendering found the rendering info changed */
   ZINK_RP_CLOSE_STATE_CHANGE,
   /* memory or image barrier that must happen outside a pass */
   ZINK_RP_CLOSE_BARRIER,
   /* a command needs the ordered cmdbuf, so the pass cannot stay open */
   ZINK_RP_CLOSE_ORDERED_CMD,
   /* a resource lost unordered access and has to be flushed in order */
   ZINK_RP_CLOSE_UNORDERED_OFF,
   /* query end / qbo update / render condition */
   ZINK_RP_CLOSE_QUERY,
   /* fbfetch turned on or off */
   ZINK_RP_CLOSE_FBFETCH,
   /* clear needs a layout change, or the fallback clear path */
   ZINK_RP_CLOSE_CLEAR,
   /* blitting finished */
   ZINK_RP_CLOSE_BLIT_END,
   /* the blit destination is the pending resolve target */
   ZINK_RP_CLOSE_RESOLVE_TARGET,
   /* sample locations need re-evaluation */
   ZINK_RP_CLOSE_ZS_EVALUATE,
   /* compute dispatch */
   ZINK_RP_CLOSE_COMPUTE,
   /* provoking vertex mode change without per-pipeline support */
   ZINK_RP_CLOSE_PROVOKING_VERTEX,
   /* swapchain readback for the next present */
   ZINK_RP_CLOSE_SWAPCHAIN_READBACK,
   /* ZINK_DEBUG=sync */
   ZINK_RP_CLOSE_DEBUG_SYNC,
   ZINK_RP_CLOSE_COUNT,
};

enum zink_perf_counter {
   ZINK_PERF_DRAWS,
   /* draws that went through the vertex-state (display list) path */
   ZINK_PERF_VSTATE_DRAWS,
   /* vertex input / vertex buffer binds actually emitted for vertex-state draws */
   ZINK_PERF_VSTATE_BINDS,
   /* vertex-state draws where those binds were skipped as redundant */
   ZINK_PERF_VSTATE_SKIPS,
   /* gfx pipeline lookups that had to consult the hash table */
   ZINK_PERF_PIPE_LOOKUPS,
   /* gfx pipelines created (i.e. lookups that missed) */
   ZINK_PERF_PIPE_MISSES,
   /* descriptor set updates */
   ZINK_PERF_DESC_UPDATES,
   /* batch submits */
   ZINK_PERF_SUBMITS,
   /* batches started */
   ZINK_PERF_BATCHES,
   /* gfx program variants compiled with all color outputs dropped */
   ZINK_PERF_NO_COLOR_OUTPUTS,
   /* gfx program variants compiled with the extra color outputs dropped */
   ZINK_PERF_EXTRA_COLOR_OUTPUTS,
   ZINK_PERF_LAST,
};

struct zink_perf_counters {
   uint64_t c[ZINK_PERF_LAST];
   /* cpu cycles attributed to the whole zink_draw call */
   uint64_t draw_cycles;
   /* cpu cycles attributed to gfx pipeline lookup */
   uint64_t pipeline_cycles;
   /* render passes begun (mirrors the submits-per-frame HUD) */
   uint64_t rp_opens;
   /* render passes closed, per reason */
   uint64_t rp_closes[ZINK_RP_CLOSE_COUNT];
   /* bytes that a closed pass had to write back because storeOp != DONT_CARE */
   uint64_t rp_store_bytes;
   uint64_t draws_at_last_dump;
   uint64_t cycles_at_last_dump;
   uint64_t wall_at_last_dump;
   bool enabled;
   unsigned period;
};

#ifdef __cplusplus
extern "C" {
#endif

extern struct zink_perf_counters zink_perf;

void zink_perf_init(void);
/* call once per draw; dumps when the period has elapsed */
void zink_perf_tick(void);
/* named renderpass break points, for the audit dump */
extern const char *const zink_rp_closure_reason_str[ZINK_RP_CLOSE_COUNT];
#ifdef __cplusplus
}
#endif

static inline void
zink_perf_count(enum zink_perf_counter i)
{
   zink_perf.c[i]++;
}

static inline void
zink_perf_add(enum zink_perf_counter i, uint64_t v)
{
   zink_perf.c[i] += v;
}

static inline void
zink_perf_rp_open(void)
{
   zink_perf.rp_opens++;
}

static inline void
zink_perf_rp_close(enum zink_rp_closure_reason reason, uint64_t store_bytes)
{
   zink_perf.rp_closes[reason]++;
   zink_perf.rp_store_bytes += store_bytes;
}

/* Monotonic, cheap, comparable between two builds on the same machine. */
static inline uint64_t
zink_perf_clock(void)
{
#if defined(__x86_64__) || defined(__i386__)
   return __builtin_ia32_rdtsc();
#else
   /* portable fallback for non-x86: coarser, but only paid when the counters
    * are actually consulted */
   return (uint64_t)os_time_get_nano();
#endif
}

#endif /* ZINK_PERF_H */
