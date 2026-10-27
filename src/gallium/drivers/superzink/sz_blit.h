/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef SZ_BLIT_H
#define SZ_BLIT_H

#include "pipe/p_context.h"

/* pipe_context::blit: copy, scaled/filtered blit and MSAA resolve. */
void sz_blit(struct pipe_context *pipe, const struct pipe_blit_info *info);

#endif /* SZ_BLIT_H */
