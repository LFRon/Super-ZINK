/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#ifndef SZ_PUBLIC_H
#define SZ_PUBLIC_H

struct pipe_screen;
struct pipe_screen_config;
struct renderonly;
struct sw_winsys;

struct pipe_screen *
superzink_create_screen(struct sw_winsys *winsys,
                        const struct pipe_screen_config *config);

#ifndef _WIN32
struct pipe_screen *
superzink_drm_create_screen(int fd,
                            const struct pipe_screen_config *config,
                            struct renderonly *ro);
#endif

#endif /* SZ_PUBLIC_H */
