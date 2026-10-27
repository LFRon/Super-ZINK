/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * The Gallium context.
 *
 * Gallium's context is 149 function pointers, and st/mesa calls a large number
 * of them unconditionally (cso_unbind_context alone rebinds every state slot).
 * So Phase 1 needs the full *shape* of the interface before any of the behaviour
 * exists: the state objects are recorded into shadow structs, and draws are
 * stubs. What is real here is the submission machinery -- recording, submitting
 * and waiting -- because that is what the rest of the driver is built on.
 */

#ifndef SZ_CONTEXT_H
#define SZ_CONTEXT_H

#include "sz_sync.h"
#include "sz_types.h"

/* Gallium state objects are opaque void* to the state tracker, so every create
 * function hands back one of these. Phase 1 copies the descriptor in and stops
 * there; Phase 2 turns the copy into a VkPipeline, at which point having the
 * descriptor already recorded is what makes the pipeline cache key possible.
 *
 * One union rather than a struct per kind: they are all small, they are all
 * short-lived, and the alternative is twelve near-identical allocation
 * functions. */
enum sz_state_kind {
   SZ_STATE_BLEND,
   SZ_STATE_SAMPLER,
   SZ_STATE_RASTERIZER,
   SZ_STATE_DEPTH_STENCIL_ALPHA,
   SZ_STATE_VERTEX_ELEMENTS,
   SZ_STATE_SHADER,
};

struct sz_state {
   enum sz_state_kind kind;
   mesa_shader_stage stage;
   union {
      struct pipe_blend_state blend;
      struct pipe_sampler_state sampler;
      struct pipe_rasterizer_state rasterizer;
      struct pipe_depth_stencil_alpha_state dsa;
      struct {
         unsigned count;
         struct pipe_vertex_element *elements;
      } ve;
      struct {
         struct pipe_shader_state base;

         /* Our own copy of the IR. The state tracker keeps ownership of the one
          * it passes in and frees it whenever it likes, so a shader that is
          * bound for more than one draw cannot point at it. */
         struct nir_shader *nir;
         void *mem_ctx;

         /* Translation happens at shader creation, not at first draw: a shader
          * that cannot be translated has to fail where the state tracker can
          * report a link error, while a draw has nowhere to put one. */
         uint32_t *spirv;
         size_t spirv_words;
      } shader;
   } u;
};

/* Separate from sz_state because the state tracker hands the driver a
 * `struct pipe_sampler_view *` that points at this memory, so base has to be the
 * first member for the cast back to be valid. Burying it in a union would put
 * the pointer somewhere a free() cannot follow. */
struct sz_sampler_view {
   struct pipe_sampler_view base;
};

struct sz_gfx_pipeline;

struct sz_context {
   struct pipe_context base;
   struct sz_screen *screen;

   /* Recorded command stream. A batch is acquired on first use and submitted by
    * sz_context_flush(). NULL when nothing is being recorded. */
   struct sz_batch *batch;

   /* Dynamic rendering is active. Draws inside one render pass share it; the
    * pass is ended on flush, on a framebuffer change, and around any command
    * that is illegal inside one. */
   bool rendering;
   unsigned render_extent[2];
   VkFormat render_color_formats[PIPE_MAX_COLOR_BUFS];
   unsigned render_color_count;

   struct sz_gfx_pipeline *pipeline;

   /* Shadow state. Kept so later phases can diff it and record only what
    * changed, and so the current values are available when a pipeline object is
    * finally created. */
   struct {
      void *blend;
      void *rasterizer;
      void *depth_stencil_alpha;
      void *vertex_elements;
      /* Non-owning: the state tracker deletes every object it created through the
       * matching delete entry point, so the context never frees these. */
      void *sampler_states[MESA_SHADER_STAGES][PIPE_MAX_SAMPLERS];
      void *sampler_views[MESA_SHADER_STAGES][PIPE_MAX_SAMPLERS];
      void *shaders[MESA_SHADER_STAGES];

      struct pipe_framebuffer_state framebuffer;
      struct pipe_vertex_buffer vertex_buffers[PIPE_MAX_ATTRIBS];
      unsigned num_vertex_buffers;
      struct pipe_viewport_state viewports[PIPE_MAX_VIEWPORTS];
      unsigned num_viewports;
      struct pipe_scissor_state scissors[PIPE_MAX_VIEWPORTS];
      unsigned num_scissors;
      struct pipe_stencil_ref stencil_ref;
      struct pipe_blend_color blend_color;
      struct pipe_clip_state clip;
      float default_outer_level[4];
      float default_inner_level[2];
      unsigned patch_vertices;
      unsigned sample_mask;
   } state;
};

/* Creates the pipe_context. Returns NULL on failure. */
struct pipe_context *sz_context_create(struct sz_screen *screen,
                                       unsigned flags);

/* SZ_DEBUG=selftest: records a clear, a copy and a readback through the real
 * command submission path and verifies the pixels. Returns the number of failed
 * checks.
 *
 * This is the first test that exercises recording, submitting and waiting: the
 * three things that can be wrong without producing any Vulkan error at all.
 * Getting a timeline value wrong, or reusing a command buffer the GPU still
 * owns, hangs or returns stale pixels -- neither shows up until something
 * actually reads the result back. */
unsigned sz_context_selftest(struct sz_screen *screen);

/* SZ_DEBUG=selftest: renders one triangle through sz_draw_vbo and verifies the
 * pixels. Returns the number of failed checks.
 *
 * Separate from sz_context_selftest() on purpose: that one drives Vulkan
 * directly and proves submission works, while this one drives the Gallium entry
 * points and proves a draw reaches the attachment. A failure in either is a
 * different bug. */
unsigned sz_render_selftest(struct sz_screen *screen);

#endif /* SZ_CONTEXT_H */
