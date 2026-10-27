/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_debug.h"

#include "util/u_debug.h"

static const struct debug_named_value
sz_debug_options[] = {
   { "info",      SZ_DEBUG_INFO,      "Print device/feature/extension summary" },
   { "validate",  SZ_DEBUG_VALIDATE,  "Assert resource state and barrier correctness per frame" },
   { "stats",     SZ_DEBUG_STATS,     "Per-frame counters: submits, barriers, PSOs, descriptor writes" },
   { "profile",   SZ_DEBUG_PROFILE,   "Time hot paths (draw / barrier / submit)" },
   { "dump_cmd",  SZ_DEBUG_DUMP_CMD,  "Dump the inline command stream as text" },
   { "nir",       SZ_DEBUG_NIR,       "Dump NIR at shader compile" },
   { "spirv",     SZ_DEBUG_SPIRV,     "Dump emitted SPIR-V" },
   { "no_pace",   SZ_DEBUG_NO_PACE,   "Disable the frame pacer (A/B comparisons)" },
   { "flushsync", SZ_DEBUG_FLUSHSYNC, "Force synchronous submits and presents" },
   { "quiet",     SZ_DEBUG_QUIET,     "Suppress warnings" },
   { "selftest",  SZ_DEBUG_SELFTEST,  "Round-trip buffers and images through the allocator at startup" },
   DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(sz_debug, "SZ_DEBUG", sz_debug_options, 0)

uint32_t sz_debug;

void
sz_debug_init(void)
{
   sz_debug = debug_get_option_sz_debug();
}

