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
};

extern uint32_t sz_debug;

void sz_debug_init(void);

static inline bool
sz_debug_enabled(enum sz_debug_flag flag)
{
   return (sz_debug & flag) != 0;
}

/* Warnings describe a degraded-but-working path, so they are visible by default
 * and silenced by SZ_DEBUG=quiet. Failures use mesa_loge() and are never
 * silenced. */
#define sz_warn(fmt, ...)                                                      \
   do {                                                                        \
      if (!(sz_debug & SZ_DEBUG_QUIET))                                        \
         mesa_logw("SZ: " fmt, ##__VA_ARGS__);                                 \
   } while (0)

#endif /* SZ_DEBUG_H */
