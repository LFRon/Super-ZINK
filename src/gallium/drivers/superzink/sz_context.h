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

#include "sz_compiler.h"
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

   /* Unique for the lifetime of the process. The pipeline cache keys on this
    * rather than on the address: delete_* frees the object, and malloc happily
    * hands the same address to the next state, which would make a cached
    * pipeline for the old state match the new one. */
   uint64_t id;

   union {
      struct pipe_blend_state blend;
      struct {
         struct pipe_sampler_state base;
         /* Created along with the state object and destroyed with it. A
          * VkSampler is immutable, so there is nothing to update later and
          * nothing to share between objects that describe the same filter. */
         VkSampler sampler;
      } sampler;
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

   /* The image is what the descriptor needs; the resource it came from is
    * already referenced through base.texture, so the VkImageView is the only
    * thing this object owns. Format and swizzle come from the template, so a
    * view of an RGBA texture as RED is a different view, not a different
    * resource. */
   VkImageView view;
};

struct sz_gfx_pipeline;

/* How many replaced pipelines can wait for their submission to retire before the
 * context waits for the queue. Two draws per frame with a changing pipeline key
 * need a couple of slots; the bound only exists so a pathological stream of
 * state changes cannot grow the list without limit. */
#define SZ_MAX_RETIRED_PIPELINES 32

/* Placeholder retire value for a pipeline that was replaced while the batch
 * referencing it was still being recorded. The timeline value that batch will
 * signal is not known until it is submitted, so the entry waits for
 * sz_pipeline_note_submit() to fill the real one in. Zero is never signalled. */
#define SZ_RETIRE_PENDING 0

/* How many pipelines one context keeps alive. Alternating between two states
 * (the shape of every state-churn workload) needs two; the rest is headroom for
 * a frame that touches a dozen material variants. */
#define SZ_MAX_CACHED_PIPELINES 24

/* Defined here rather than in sz_pipeline.h because the context embeds it by
 * value and sz_pipeline.h already includes this header. */
struct sz_pipeline_cache {
   struct sz_gfx_pipeline *entries[SZ_MAX_CACHED_PIPELINES];
   unsigned count;
   uint64_t clock;
};

/* How many draws may be accumulated before one vkCmdDrawMulti* call. Bounded by
 * VkPhysicalDeviceMultiDrawPropertiesEXT::maxMultiDrawCount at run time; the
 * compile-time value only sizes the array. */
#define SZ_MAX_BATCHED_DRAWS 64

/* What has to be re-emitted before the next batch of draws. A state setter marks
 * its bit and flushes the pending batch first, so the shadow state and the
 * recorded commands can never disagree about which state a draw saw. */
enum sz_dirty_bits {
   SZ_DIRTY_PIPELINE       = 1u << 0,
   SZ_DIRTY_VERTEX_BUFFERS = 1u << 1,
   SZ_DIRTY_INDEX_BUFFER   = 1u << 2,
   SZ_DIRTY_DESCRIPTORS    = 1u << 3,
   SZ_DIRTY_VIEWPORT       = 1u << 4,
   SZ_DIRTY_SCISSOR        = 1u << 5,
   SZ_DIRTY_BLEND_CONST    = 1u << 6,
   SZ_DIRTY_STENCIL_REF    = 1u << 7,
};

#define SZ_DIRTY_ALL 0xffu

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

   /* The pipeline bound in the command buffer being recorded. NULL means "no
    * binding recorded yet", which is also what a fresh command buffer reports. */
   struct sz_gfx_pipeline *pipeline;

   /* Pipelines built for this context. A hit costs one hash plus one key
    * comparison; a miss builds and evicts the least recently used entry that is
    * not the bound one. */
   struct sz_pipeline_cache pipelines;

   /* Emitted-command dirty bits. Zero means the state a draw needs is already in
    * the command buffer. */
   unsigned dirty;

   /* Draw stream: draws accumulate here and leave as one vkCmdDrawMulti* call.
    *
    * The invariant that makes this correct is that every state setter flushes
    * the pending batch before it changes the shadow state, so nothing is
    * recorded between the appends and the flush. */
   struct {
      struct {
         uint32_t count;
         uint32_t first;
         int32_t index_bias;
      } draws[SZ_MAX_BATCHED_DRAWS];
      unsigned count;
      unsigned capacity;
      bool indexed;
      int32_t instance_count;
      uint32_t start_instance;
      VkPrimitiveTopology topology;
      struct sz_resource *index_buffer;
      VkDeviceSize index_offset;
      VkIndexType index_type;
   } pending;

   /* Set for a draw that samples the very image it renders into. Vulkan leaves
    * that undefined and RADV's compressed surfaces turn it into noise on screen,
    * so the attachment is copied to a scratch image first and the descriptor for
    * that slot names the copy instead. The scratch lives until the format or the
    * extent changes, which is once per window size. */
   struct {
      VkImageView view[SZ_DESCRIPTOR_STAGES][SZ_MAX_SAMPLERS];
   } sampled_override;
   struct pipe_resource *sampled_copy;
   VkFormat sampled_copy_format;
   unsigned sampled_copy_extent[2];

   /* Pipelines replaced while a command buffer still had them bound. Vulkan
    * forbids destroying one in that state, so the objects wait here until the
    * submission that used them has retired; sz_pipeline_reclaim() drops the ones
    * whose timeline value has passed. */
   struct {
      struct sz_gfx_pipeline *pipeline;
      uint64_t retire_value;
   } retired[SZ_MAX_RETIRED_PIPELINES];
   unsigned num_retired;

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

      /* Owned: the context holds a reference to every bound buffer, because
       * the state tracker can delete a constant buffer while it is still the
       * source of the last recorded draw. */
      struct pipe_constant_buffer constant_buffers[MESA_SHADER_STAGES]
                                                  [SZ_MAX_CONST_BUFFERS];

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

/* The command buffer being recorded, acquired on first use. The no_pass variant
 * closes an open render pass first, because blits, clears and barrier-only
 * commands are illegal inside one. Shared with the blit module. */
VkCommandBuffer sz_context_cmd(struct sz_context *ctx);
VkCommandBuffer sz_context_cmd_no_pass(struct sz_context *ctx);

/* Emits the pending draw batch (and any state it needs) into the current command
 * buffer. Every state setter, every non-draw command and every submission calls
 * it first, which is what makes the batching correct: whatever a draw needs is
 * captured before anything else can change the state. */
void sz_draw_flush(struct sz_context *ctx);

/* pipe_context is the first member, so the cast is the whole accessor. */
static inline struct sz_context *
sz_context(struct pipe_context *pipe)
{
   return (struct sz_context *)pipe;
}

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
