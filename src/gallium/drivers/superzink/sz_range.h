/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Best-fit range allocator.
 *
 * Deliberately free of Vulkan and of the screen: this is the part of the memory
 * allocator whose bugs are silent (a double-handed-out range aliases two
 * resources onto the same bytes, which shows up as corrupted geometry much
 * later), so it is kept isolated and unit-tested.
 *
 * The free list is an array kept in ascending offset order. Insertion is O(n)
 * and split/merge never needs a tree walk; at the tens-of-ranges scale a frame
 * produces, an array beats a red-black tree on both code size and cache
 * behaviour.
 */

#ifndef SZ_RANGE_H
#define SZ_RANGE_H

#include <stdbool.h>
#include <stdint.h>

struct sz_range {
   uint64_t offset;
   uint64_t size;
};

/* Allocates `size` bytes at or above `alignment`.
 *
 * Best-fit: of every free range that can satisfy the request, the one leaving
 * the least waste is chosen. This matters because a 4 KiB uniform buffer must
 * not eat the tail of the 16 MiB block a vertex buffer is about to need.
 *
 * Returns false when nothing fits, or when splitting the chosen range needs a
 * spare slot and `capacity` is already reached. */
bool sz_range_alloc(struct sz_range *free_ranges, unsigned *count,
                    unsigned capacity, uint64_t size, uint64_t alignment,
                    uint64_t *out_offset);

/* Returns [offset, offset+size) to the list and merges it with the neighbours
 * it now touches. Returns false when the list is full. */
bool sz_range_free(struct sz_range *free_ranges, unsigned *count,
                   unsigned capacity, uint64_t offset, uint64_t size);

/* True when no single free range covers the whole region, i.e. it is in use. */
bool sz_range_is_used(const struct sz_range *free_ranges, unsigned count,
                      uint64_t offset, uint64_t size);

/* Sum of all free bytes. Used by the self-test and by SZ_DEBUG=stats. */
uint64_t sz_range_free_bytes(const struct sz_range *free_ranges, unsigned count);

#endif /* SZ_RANGE_H */
