# Copyright © 2026 Super-ZINK contributors
# SPDX-License-Identifier: MIT

files_libsuperzink = files(
  'sz_debug.c',
  'sz_format.c',
  'sz_screen.c',
)

libsuperzink = static_library(
  'superzink',
  [files_libsuperzink, vk_dispatch_table],
  gnu_symbol_visibility : 'hidden',
  include_directories : [
    inc_include, inc_src, inc_gallium, inc_gallium_aux,
    inc_vulkan_util,
  ],
  link_args : [ld_args_build_id],
  dependencies : [
    idep_mesautil,
    idep_vulkan_util_headers,
    idep_vulkan_util,
    dep_libdrm,
    dep_thread,
  ],
)

driver_superzink = declare_dependency(
  compile_args : '-DGALLIUM_SUPERZINK',
  link_with : [libsuperzink],
)
