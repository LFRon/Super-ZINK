/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef SZ_DEVICE_H
#define SZ_DEVICE_H

#include "sz_types.h"

/* Creates the VkDevice, resolves device entry points, picks queues and builds
 * the memory-type lookup. Consumes screen->caps as read-only truth (ADR-001):
 * nothing here re-derives what the device can do. */
bool sz_device_init(struct sz_screen *screen);
void sz_device_finish(struct sz_screen *screen);

/* Finds a memory type index satisfying `bits` and `props`. Returns
 * VK_MAX_MEMORY_TYPES when none matches, which callers must treat as fatal. */
uint32_t sz_device_find_memory_type(const struct sz_screen *screen,
                                    uint32_t type_bits,
                                    VkMemoryPropertyFlags props);

/* Smallest memory type whose heap is DEVICE_LOCAL, else the first host-visible
 * one. Used for transient resources before sz_bo takes over. */
uint32_t sz_device_pick_local_memory(const struct sz_screen *screen);

#endif /* SZ_DEVICE_H */
