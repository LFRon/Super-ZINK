/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include <stdarg.h>
#include <stdio.h>

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
   { "trace",     SZ_DEBUG_TRACE,     "Print the first resource/present events of the life cycle" },
   { "serialize", SZ_DEBUG_SERIALIZE, "Submit and wait after every draw batch (tells a race apart from a data bug)" },
   { "submit_each", SZ_DEBUG_SUBMIT_EACH, "Submit after every draw batch but do not wait" },
   { "race",      SZ_DEBUG_RACE,      "Warn when the CPU writes a buffer an uncompleted submission still reads" },
   { "stream",    SZ_DEBUG_STREAM,    "Print one line per draw with the buffers and offsets it uses" },
   DEBUG_NAMED_VALUE_END
};

DEBUG_GET_ONCE_FLAGS_OPTION(sz_debug, "SZ_DEBUG", sz_debug_options, 0)

uint32_t sz_debug;

int sz_trace_budget = 400;

void
sz_trace_(const char *fmt, ...)
{
   if (sz_trace_budget <= 0)
      return;

   sz_trace_budget--;

   char buf[512];
   va_list ap;
   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);

   mesa_logi("SZ: trace: %s", buf);
}

void
sz_debug_init(void)
{
   sz_debug = debug_get_option_sz_debug();
}

