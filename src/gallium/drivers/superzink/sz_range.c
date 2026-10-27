/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_range.h"

#include <string.h>

static uint64_t
sz_align_up(uint64_t value, uint64_t alignment)
{
   if (alignment <= 1)
      return value;
   return (value + alignment - 1) & ~(alignment - 1);
}

bool
sz_range_alloc(struct sz_range *free_ranges, unsigned *count, unsigned capacity,
               uint64_t size, uint64_t alignment, uint64_t *out_offset)
{
   if (size == 0)
      return false;

   int best = -1;
   uint64_t best_offset = 0;
   uint64_t best_waste = UINT64_MAX;

   for (unsigned i = 0; i < *count; i++) {
      const uint64_t offset = sz_align_up(free_ranges[i].offset, alignment);
      const uint64_t skip = offset - free_ranges[i].offset;
      if (skip >= free_ranges[i].size)
         continue;

      const uint64_t avail = free_ranges[i].size - skip;
      if (avail < size)
         continue;

      const uint64_t waste = avail - size;
      if (waste < best_waste) {
         best = (int)i;
         best_offset = offset;
         best_waste = waste;
      }
   }

   if (best < 0)
      return false;

   struct sz_range *r = &free_ranges[best];
   const uint64_t skip = best_offset - r->offset;

   if (skip == 0) {
      if (r->size == size) {
         memmove(&free_ranges[best], &free_ranges[best + 1],
                 (size_t)(*count - (unsigned)best - 1) * sizeof(*free_ranges));
         (*count)--;
      } else {
         r->offset += size;
         r->size -= size;
      }
      *out_offset = best_offset;
      return true;
   }

   /* The request starts mid-range: the leading gap stays free and the tail (if
    * any) becomes a second free range, so one extra slot is needed. */
   const uint64_t tail_offset = best_offset + size;
   const uint64_t tail_size = r->offset + r->size - tail_offset;
   if (tail_size && *count == capacity)
      return false;

   r->size = skip;

   if (tail_size) {
      memmove(&free_ranges[best + 2], &free_ranges[best + 1],
              (size_t)(*count - (unsigned)best - 1) * sizeof(*free_ranges));
      free_ranges[best + 1] = (struct sz_range) {
         .offset = tail_offset,
         .size = tail_size,
      };
      (*count)++;
   }

   *out_offset = best_offset;
   return true;
}

bool
sz_range_free(struct sz_range *free_ranges, unsigned *count, unsigned capacity,
              uint64_t offset, uint64_t size)
{
   if (size == 0)
      return true;

   if (*count == capacity)
      return false;

   unsigned i = 0;
   while (i < *count && free_ranges[i].offset < offset)
      i++;

   memmove(&free_ranges[i + 1], &free_ranges[i],
           (size_t)(*count - i) * sizeof(*free_ranges));
   free_ranges[i] = (struct sz_range) { .offset = offset, .size = size };
   (*count)++;

   if (i + 1 < *count &&
       free_ranges[i].offset + free_ranges[i].size == free_ranges[i + 1].offset) {
      free_ranges[i].size += free_ranges[i + 1].size;
      memmove(&free_ranges[i + 1], &free_ranges[i + 2],
              (size_t)(*count - i - 2) * sizeof(*free_ranges));
      (*count)--;
   }

   if (i > 0 &&
       free_ranges[i - 1].offset + free_ranges[i - 1].size == free_ranges[i].offset) {
      free_ranges[i - 1].size += free_ranges[i].size;
      memmove(&free_ranges[i], &free_ranges[i + 1],
              (size_t)(*count - i - 1) * sizeof(*free_ranges));
      (*count)--;
   }

   return true;
}

bool
sz_range_is_used(const struct sz_range *free_ranges, unsigned count,
                 uint64_t offset, uint64_t size)
{
   for (unsigned i = 0; i < count; i++) {
      if (free_ranges[i].offset <= offset &&
          free_ranges[i].offset + free_ranges[i].size >= offset + size)
         return false;
   }
   return true;
}

uint64_t
sz_range_free_bytes(const struct sz_range *free_ranges, unsigned count)
{
   uint64_t total = 0;
   for (unsigned i = 0; i < count; i++)
      total += free_ranges[i].size;
   return total;
}
