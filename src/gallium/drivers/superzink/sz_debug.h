/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Debug channels. Combine with commas:
 *   SZ_DEBUG=info,stats,dump_cmd
 */

#ifndef SZ_DEBUG_H
#define SZ_DEBUG_H

#include <stdbool.h>
#include <stdint.h>

#include "util/log.h"
#include "util/macros.h"

enum sz_debug_flag {
   SZ_DEBUG_INFO      = (1 << 0),
   SZ_DEBUG_VALIDATE  = (1 << 1),
   SZ_DEBUG_STATS     = (1 << 2),
   SZ_DEBUG_PROFILE   = (1 << 3),
   SZ_DEBUG_DUMP_CMD  = (1 << 4),
   SZ_DEBUG_NIR       = (1 << 5),
   SZ_DEBUG_SPIRV     = (1 << 6),
   SZ_DEBUG_NO_PACE   = (1 << 7),
   SZ_DEBUG_FLUSHSYNC = (1 << 8),
   SZ_DEBUG_QUIET     = (1 << 9),
   SZ_DEBUG_SELFTEST  = (1 << 10),
   SZ_DEBUG_TRACE     = (1 << 11),
   SZ_DEBUG_SERIALIZE = (1 << 12),
   SZ_DEBUG_SUBMIT_EACH = (1 << 13),
   SZ_DEBUG_RACE      = (1 << 14),
   SZ_DEBUG_STREAM    = (1 << 15),
};

extern uint32_t sz_debug;

void sz_debug_init(void);

static inline bool
sz_debug_enabled(enum sz_debug_flag flag)
{
   return (sz_debug & flag) != 0;
}

extern int sz_trace_budget;

void sz_trace_(const char *fmt, ...) PRINTFLIKE(1, 2);

/* One-shot breadcrumb for the resource life cycle (create -> render -> hand to
 * the winsys). Only the first callsite budget of events is printed, because the
 * interesting window is usually the first few frames. */
#define sz_trace(fmt, ...)                                                     \
   do {                                                                        \
      if (sz_debug & SZ_DEBUG_TRACE)                                           \
         sz_trace_(fmt, ##__VA_ARGS__);                                        \
   } while (0)

/* Warnings describe a degraded-but-working path, so they are visible by default
 * and silenced by SZ_DEBUG=quiet. Failures use mesa_loge() and are never
 * silenced. */
#define sz_warn(fmt, ...)                                                      \
   do {                                                                        \
      if (!(sz_debug & SZ_DEBUG_QUIET))                                        \
         mesa_logw("SZ: " fmt, ##__VA_ARGS__);                                 \
   } while (0)

/* Same as sz_warn() but only the first time each call site reaches it. Used for
 * "this path is not implemented yet" messages that would otherwise repeat once
 * per draw and drown the log. */
#define sz_warn_once(fmt, ...)                                                 \
   do {                                                                        \
      static bool warned_;                                                     \
      if (!warned_) {                                                          \
         warned_ = true;                                                       \
         sz_warn(fmt, ##__VA_ARGS__);                                          \
      }                                                                        \
   } while (0)

#endif /* SZ_DEBUG_H */
