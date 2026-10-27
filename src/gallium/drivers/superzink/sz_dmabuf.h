/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * dma-buf export and import.
 *
 * This is what presentation needs: the X server or a compositor hands us a
 * dma-buf for the window, we render into it, and it hands one back to whoever
 * scans it out. Nothing in this file knows about swapchains or surfaces -- the
 * WSI phase builds on these two calls, not the other way round.
 *
 * Only DRM_FORMAT_MOD_LINEAR is claimed in this phase. Every consumer can import
 * a linear dma-buf, so the choice is correct everywhere rather than fast
 * somewhere; driver-tiled modifiers need per-modifier format queries and arrive
 * with the frame-pacing work.
 */

#ifndef SZ_DMABUF_H
#define SZ_DMABUF_H

#include "sz_types.h"

/* Installs resource_get_handle, resource_from_handle, query_dmabuf_modifiers and
 * is_dmabuf_modifier_supported.
 *
 * The two handle functions are installed even when the device has no external
 * memory support: the DRI frontend calls them unconditionally, and a NULL there
 * is a segfault inside Mesa instead of a message. They return false, which makes
 * the buffer allocation fail where the failure can be reported. */
void sz_dmabuf_screen_init(struct sz_screen *screen);

/* Exports a rendered image, imports it back and checks that both sides agree on
 * the geometry. Returns the number of failures, so it can be run from the same
 * SZ_DEBUG=selftest path as the render and memory tests. */
unsigned sz_dmabuf_selftest(struct sz_screen *screen);

#endif /* SZ_DMABUF_H */
