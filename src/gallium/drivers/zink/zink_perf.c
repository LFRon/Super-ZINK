/*
 * Copyright © 2026 The Mesa authors
 * SPDX-License-Identifier: MIT
 */

#include "zink_perf.h"

#include <inttypes.h>
#include <stdio.h>

#include "util/os_time.h"
#include "util/u_debug.h"

struct zink_perf_counters zink_perf;

const char *const zink_rp_closure_reason_str[ZINK_RP_CLOSE_COUNT] = {
   [ZINK_RP_CLOSE_FLUSH] = "flush",
   [ZINK_RP_CLOSE_FRAMEBUFFER] = "framebuffer",
   [ZINK_RP_CLOSE_ZSBUF] = "zsbuf",
   [ZINK_RP_CLOSE_ATTACHMENT_VIEW] = "attachment_view",
   [ZINK_RP_CLOSE_STATE_CHANGE] = "state_change",
   [ZINK_RP_CLOSE_BARRIER] = "barrier",
   [ZINK_RP_CLOSE_ORDERED_CMD] = "ordered_cmd",
   [ZINK_RP_CLOSE_TRANSFER] = "transfer",
   [ZINK_RP_CLOSE_IMAGE_BARRIER] = "image_barrier",
   [ZINK_RP_CLOSE_UNORDERED_OFF] = "unordered_off",
   [ZINK_RP_CLOSE_QUERY] = "query",
   [ZINK_RP_CLOSE_FBFETCH] = "fbfetch",
   [ZINK_RP_CLOSE_CLEAR] = "clear",
   [ZINK_RP_CLOSE_BLIT_END] = "blit_end",
   [ZINK_RP_CLOSE_RESOLVE_TARGET] = "resolve_target",
   [ZINK_RP_CLOSE_ZS_EVALUATE] = "zs_evaluate",
   [ZINK_RP_CLOSE_COMPUTE] = "compute",
   [ZINK_RP_CLOSE_PROVOKING_VERTEX] = "provoking_vertex",
   [ZINK_RP_CLOSE_SWAPCHAIN_READBACK] = "swapchain_readback",
   [ZINK_RP_CLOSE_DEBUG_SYNC] = "debug_sync",
};

void
zink_perf_init(void)
{
   static int8_t initialized = -1;
   if (initialized >= 0)
      return;
   initialized = 1;

   zink_perf.enabled = debug_get_bool_option("ZINK_PERF", false);
   int period = debug_get_num_option("ZINK_PERF_PERIOD", 1000);
   zink_perf.period = period > 0 ? (unsigned)period : 0;
   zink_perf.wall_at_last_dump = (uint64_t)os_time_get_nano();
}

void
zink_perf_tick(void)
{
   static uint64_t rp_opens_at_last_dump;
   static uint64_t rp_closes_at_last_dump[ZINK_RP_CLOSE_COUNT];
   static uint64_t rp_store_at_last_dump;

   if (!zink_perf.enabled)
      return;

   const uint64_t draws = zink_perf.c[ZINK_PERF_DRAWS];
   const uint64_t now_wall = (uint64_t)os_time_get_nano();
   /* draws are the primary trigger, but a caller that merges everything into a
    * handful of calls (u_threaded_context does) would never reach the period, so
    * also dump on a wall-clock cadence */
   const bool by_draws = zink_perf.period && draws - zink_perf.draws_at_last_dump >= zink_perf.period;
   const bool by_time = now_wall - zink_perf.wall_at_last_dump >= 1000000000ull;
   if (!by_draws && !by_time)
      return;
   const uint64_t ds = draws - zink_perf.draws_at_last_dump;
   if (!ds)
      return;

   const uint64_t rp_opens = zink_perf.rp_opens;
   const uint64_t rp_opens_delta = rp_opens - rp_opens_at_last_dump;
   const uint64_t store_delta = zink_perf.rp_store_bytes - rp_store_at_last_dump;

   const double wall_s = (double)(now_wall - zink_perf.wall_at_last_dump) / 1e9;
   const double draw_ns = wall_s * 1e9 / (double)ds;
   const uint64_t dcycles = zink_perf.draw_cycles - zink_perf.cycles_at_last_dump;

   fprintf(stderr,
           "ZINKPERF draws=%" PRIu64
           "  ns/draw=%.1f"
           "  zink_draw=%.0fcyc"
           "  pipeline=%.0fcyc"
           "  | vstate=%" PRIu64 " bind=%" PRIu64 " skip=%" PRIu64
           "  pLookup=%" PRIu64 " pMiss=%" PRIu64
           "  desc=%" PRIu64 " batch=%" PRIu64 " submit=%" PRIu64
           "  nocolor=%" PRIu64 "/%" PRIu64 "\n",
           draws, draw_ns, ds ? (double)dcycles / (double)ds : 0.0,
           ds ? (double)(zink_perf.pipeline_cycles) / (double)ds : 0.0,
           zink_perf.c[ZINK_PERF_VSTATE_DRAWS], zink_perf.c[ZINK_PERF_VSTATE_BINDS],
           zink_perf.c[ZINK_PERF_VSTATE_SKIPS], zink_perf.c[ZINK_PERF_PIPE_LOOKUPS],
           zink_perf.c[ZINK_PERF_PIPE_MISSES], zink_perf.c[ZINK_PERF_DESC_UPDATES],
           zink_perf.c[ZINK_PERF_BATCHES], zink_perf.c[ZINK_PERF_SUBMITS],
           zink_perf.c[ZINK_PERF_NO_COLOR_OUTPUTS], zink_perf.c[ZINK_PERF_EXTRA_COLOR_OUTPUTS]);

   zink_perf.draws_at_last_dump = draws;
   zink_perf.cycles_at_last_dump = zink_perf.draw_cycles;
   zink_perf.wall_at_last_dump = now_wall;

   /* renderpass audit: passes per draw, who closed them, bytes written back */
   uint64_t closes_delta = 0;
   for (unsigned i = 0; i < ZINK_RP_CLOSE_COUNT; i++)
      closes_delta += zink_perf.rp_closes[i] - rp_closes_at_last_dump[i];
   if (rp_opens_delta || closes_delta) {
      fprintf(stderr,
              "ZINKPERF rp passes=%" PRIu64 " closes=%" PRIu64
              "  pass/draw=%.3f  store=%.2fMB\n",
              rp_opens_delta, closes_delta,
              (double)rp_opens_delta / (double)ds,
              (double)store_delta / (1024.0 * 1024.0));
      for (unsigned i = 0; i < ZINK_RP_CLOSE_COUNT; i++) {
         const uint64_t d = zink_perf.rp_closes[i] - rp_closes_at_last_dump[i];
         if (!d)
            continue;
         fprintf(stderr, "ZINKPERF rp   %-20s %" PRIu64 "  (%.1f%%)\n",
                 zink_rp_closure_reason_str[i], d,
                 closes_delta ? 100.0 * (double)d / (double)closes_delta : 0.0);
      }
   }
   rp_opens_at_last_dump = rp_opens;
   for (unsigned i = 0; i < ZINK_RP_CLOSE_COUNT; i++)
      rp_closes_at_last_dump[i] = zink_perf.rp_closes[i];
   rp_store_at_last_dump = zink_perf.rp_store_bytes;
}
