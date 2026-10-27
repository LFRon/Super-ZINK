/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Unit test for the range allocator.
 *
 * This is the only part of the driver whose failures are invisible: handing the
 * same bytes to two resources produces valid Vulkan calls and corrupted
 * rendering, so it is worth exercising without a GPU.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sz_range.h"

static unsigned failures;

#define CHECK(cond)                                                            \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);       \
         failures++;                                                           \
      }                                                                        \
   } while (0)

#define CAP 64

struct arena {
   struct sz_range ranges[CAP];
   unsigned count;
};

static void
arena_reset(struct arena *a, uint64_t size)
{
   a->count = 1;
   a->ranges[0] = (struct sz_range) { .offset = 0, .size = size };
}

static bool
arena_alloc(struct arena *a, uint64_t size, uint64_t alignment, uint64_t *offset)
{
   return sz_range_alloc(a->ranges, &a->count, CAP, size, alignment, offset);
}

static bool
arena_free(struct arena *a, uint64_t offset, uint64_t size)
{
   return sz_range_free(a->ranges, &a->count, CAP, offset, size);
}

/* Invariants that must hold after every operation: ranges ascend, never
 * overlap, never touch (they would have been merged), and never exceed the
 * arena. */
static void
check_invariants(const struct arena *a, uint64_t arena_size)
{
   for (unsigned i = 0; i < a->count; i++) {
      CHECK(a->ranges[i].size > 0);
      CHECK(a->ranges[i].offset + a->ranges[i].size <= arena_size);
      if (i + 1 < a->count) {
         CHECK(a->ranges[i].offset + a->ranges[i].size < a->ranges[i + 1].offset);
      }
   }
}

/* A fresh arena must hand out a request from offset 0 and shrink by exactly the
 * request size. */
static void
test_alloc_from_fresh(void)
{
   struct arena a;
   arena_reset(&a, 1024);

   uint64_t off = 12345;
   CHECK(arena_alloc(&a, 256, 1, &off));
   CHECK(off == 0);
   CHECK(a.count == 1);
   CHECK(a.ranges[0].offset == 256);
   CHECK(a.ranges[0].size == 768);
   check_invariants(&a, 1024);
}

/* A request that consumes the whole arena must leave no ranges behind. */
static void
test_exact_fit(void)
{
   struct arena a;
   arena_reset(&a, 1024);
   uint64_t off;
   CHECK(arena_alloc(&a, 1024, 1, &off));
   CHECK(off == 0);
   CHECK(a.count == 0);
   /* A second request now has nothing to take. */
   CHECK(!arena_alloc(&a, 1, 1, &off));
   check_invariants(&a, 1024);
}

/* Alignment may force a gap at the front of a range; the gap must survive as
 * free space rather than being silently dropped. */
static void
test_alignment_gap(void)
{
   struct arena a;
   arena_reset(&a, 1024);

   uint64_t off;
   CHECK(arena_alloc(&a, 64, 1, &off));
   CHECK(off == 0);            /* free range is now [64, 960) */

   CHECK(arena_alloc(&a, 128, 256, &off));
   CHECK(off == 256);          /* 64 aligns up to 256 */
   /* The 192-byte gap [64,256) must still be free, and the tail [384,1024) too. */
   CHECK(a.count == 2);
   CHECK(a.ranges[0].offset == 64 && a.ranges[0].size == 192);
   CHECK(a.ranges[1].offset == 384 && a.ranges[1].size == 640);
   CHECK(sz_range_is_used(a.ranges, a.count, 64, 192) == false);
   check_invariants(&a, 1024);
}

/* Freeing in the middle of a split range must merge with both neighbours and
 * collapse back into a single range. */
static void
test_free_merges_both_sides(void)
{
   struct arena a;
   arena_reset(&a, 1024);

   uint64_t x, y, z;
   CHECK(arena_alloc(&a, 256, 1, &x));   /* [0,256)   */
   CHECK(arena_alloc(&a, 256, 1, &y));   /* [256,512) */
   CHECK(arena_alloc(&a, 256, 1, &z));   /* [512,768) */
   CHECK(x == 0 && y == 256 && z == 512);
   check_invariants(&a, 1024);

   CHECK(arena_free(&a, 256, 256));
   /* Now free [0,256) and [512,768) so the middle one can merge with both. */
   CHECK(arena_free(&a, 0, 256));
   CHECK(arena_free(&a, 512, 256));
   CHECK(a.count == 1);
   CHECK(a.ranges[0].offset == 0);
   CHECK(a.ranges[0].size == 1024);
   check_invariants(&a, 1024);
}

/* Best-fit must prefer the tighter hole: a 64-byte request belongs in the
 * 64-byte hole, not in the 4 KiB one. */
static void
test_best_fit(void)
{
   struct arena a;
   arena_reset(&a, 8192);

   uint64_t off;
   /* Carve [0,64) and [64,8192) so there is a small hole then a big one. */
   CHECK(arena_alloc(&a, 64, 1, &off));
   CHECK(arena_alloc(&a, 8128 - 128, 1, &off)); /* leaves [8128-128, 8192) */
   CHECK(a.count == 1);

   /* Free the leading 64 bytes back so both a 64-byte and a 128-byte hole exist. */
   CHECK(arena_free(&a, 0, 64));
   CHECK(a.count == 2);

   CHECK(arena_alloc(&a, 64, 1, &off));
   CHECK(off == 0);  /* must take the exact-size hole, not the larger one */
   check_invariants(&a, 8192);
}

/* Every byte must be reachable exactly once: allocate single bytes until the
 * arena is full, then check the total and that no offset repeats. */
static void
test_no_double_handout(void)
{
   struct arena a;
   arena_reset(&a, 512);

   bool seen[512] = { false };

   for (unsigned i = 0; i < 512; i++) {
      uint64_t off;
      CHECK(arena_alloc(&a, 1, 1, &off));
      if (off >= 512) {
         CHECK(!"offset out of arena");
         return;
      }
      if (seen[off]) {
         CHECK(!"the same byte was handed out twice");
         return;
      }
      seen[off] = true;
      check_invariants(&a, 512);
   }

   uint64_t off;
   CHECK(!arena_alloc(&a, 1, 1, &off));

   for (unsigned i = 0; i < 512; i++)
      CHECK(seen[i]);
}

/* Random alloc/free traffic. Exercises split/merge paths that hand-written cases
 * miss, and this is where an off-by-one in the insert index shows up. */
static void
test_random_traffic(void)
{
   struct arena a;
   arena_reset(&a, 65536);

   struct { uint64_t offset, size; } live[256];
   unsigned live_count = 0;

   bool used[65536] = { false };
   unsigned rng = 0x5eed1234;

   for (unsigned iter = 0; iter < 20000; iter++) {
      rng = rng * 1103515245u + 12345u;
      const bool do_alloc = live_count == 0 || (rng >> 16) % 3 != 0;

      if (do_alloc && live_count < 256) {
         const uint64_t size = ((rng >> 8) % 4096) + 1;
         const uint64_t align = 1u << ((rng >> 20) % 7);
         uint64_t off;
         if (arena_alloc(&a, size, align, &off)) {
            for (uint64_t b = 0; b < size; b++) {
               if (off + b >= 65536 || used[off + b]) {
                  CHECK(!"allocation overlaps live bytes");
                  return;
               }
            }
            for (uint64_t b = 0; b < size; b++)
               used[off + b] = true;

            live[live_count].offset = off;
            live[live_count].size = size;
            live_count++;
         }
      } else if (live_count) {
         const unsigned pick = (rng >> 4) % live_count;
         const uint64_t off = live[pick].offset;
         const uint64_t size = live[pick].size;

         CHECK(sz_range_is_used(a.ranges, a.count, off, size));
         CHECK(arena_free(&a, off, size));
         for (uint64_t b = 0; b < size; b++)
            used[off + b] = false;

         live[pick] = live[--live_count];
      }

      check_invariants(&a, 65536);

      const uint64_t free_bytes = sz_range_free_bytes(a.ranges, a.count);
      uint64_t live_bytes = 0;
      for (unsigned i = 0; i < live_count; i++)
         live_bytes += live[i].size;
      CHECK(free_bytes + live_bytes == 65536);
   }

   /* Drain everything; the arena must return to one whole range. */
   for (unsigned i = 0; i < live_count; i++)
      CHECK(arena_free(&a, live[i].offset, live[i].size));
   CHECK(a.count == 1);
   CHECK(a.ranges[0].offset == 0 && a.ranges[0].size == 65536);
}

/* A slot-limited list must fail cleanly rather than corrupt the table.
 *
 * A split only needs an extra slot when the aligned start lands inside the
 * range *and* something remains after the request; taking from the front of a
 * range never grows the list. So the smallest range must be too small and the
 * next one must be misaligned for its start. */
static void
test_capacity_limit(void)
{
   /* [0,4) is too small; [12,112) fits but starts at 12, so alignment 8 pushes
    * the request to 16 and leaves both a 4-byte head and a 91-byte tail. */
   struct sz_range ranges[3] = { { 0, 4 }, { 12, 100 } };
   unsigned count = 2;
   uint64_t off = UINT64_MAX;

   /* Two slots cannot hold head + tail: refuse rather than corrupt. */
   struct sz_range tight[2] = { { 0, 4 }, { 12, 100 } };
   unsigned tight_count = 2;
   CHECK(!sz_range_alloc(tight, &tight_count, 2, 5, 8, &off));
   CHECK(tight_count == 2);
   CHECK(tight[0].offset == 0 && tight[0].size == 4);
   CHECK(tight[1].offset == 12 && tight[1].size == 100);

   /* One more slot and the same request succeeds, splitting into head + tail. */
   CHECK(sz_range_alloc(ranges, &count, 3, 5, 8, &off));
   CHECK(off == 16);
   CHECK(count == 3);
   CHECK(ranges[0].offset == 0 && ranges[0].size == 4);
   CHECK(ranges[1].offset == 12 && ranges[1].size == 4);
   CHECK(ranges[2].offset == 21 && ranges[2].size == 91);
}

int
main(void)
{
   test_alloc_from_fresh();
   test_exact_fit();
   test_alignment_gap();
   test_free_merges_both_sides();
   test_best_fit();
   test_no_double_handout();
   test_random_traffic();
   test_capacity_limit();

   if (failures) {
      fprintf(stderr, "%u check%s failed\n", failures, failures == 1 ? "" : "s");
      return 1;
   }

   printf("sz_range_test: all checks passed\n");
   return 0;
}
