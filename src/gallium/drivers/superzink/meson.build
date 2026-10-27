# Copyright © 2026 Super-ZINK contributors
# SPDX-License-Identifier: MIT

files_libsuperzink = files(
  'sz_batch.c',
  'sz_blit.c',
  'sz_bo.c',
  'sz_caps.c',
  'sz_compiler.c',
  'sz_nir_to_spirv.c',
  'sz_context.c',
  'sz_device.c',
  'sz_debug.c',
  'sz_descriptor.c',
  'sz_dmabuf.c',
  'sz_format.c',
  'sz_pipeline.c',
  'sz_range.c',
  'sz_renderpass.c',
  'sz_resource.c',
  'sz_screen.c',
  'sz_spirv.c',
  'sz_sync.c',
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
    idep_nir_headers,
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

if with_tests
  test(
    'sz_range_test',
    executable(
      'sz_range_test',
      ['sz_range_test.c', 'sz_range.c'],
      include_directories : [inc_src, inc_include],
    ),
  )
endif
