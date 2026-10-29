/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_bo.h"

#include "sz_device.h"
#include "sz_range.h"

#include "util/list.h"
#include "util/log.h"
#include "util/macros.h"
#include "util/u_math.h"
#include "util/u_memory.h"

#include <inttypes.h>
#include <string.h>

static uint64_t
sz_align_up(uint64_t value, uint64_t alignment)
{
   if (alignment <= 1)
      return value;
   return (value + alignment - 1) & ~(alignment - 1);
}

/* Requests below this size get a full block, so a frame's worth of small
 * vertex/uniform buffers lands in one VkDeviceMemory instead of sixty. */
#define SZ_BLOCK_MIN_SIZE (16ull * 1024 * 1024)
/* How much larger than the request a block may be. Rounding a request up to the
 * next power of two already bounds the waste at that, so this is only a floor:
 * a request below SZ_BLOCK_MIN_SIZE still gets the minimum block, because the
 * point of a block is to serve a frame's worth of small allocations from one
 * VkDeviceMemory. */
#define SZ_BLOCK_SLACK 2

#define SZ_MAX_RANGES 4096

/* ── blocks ──────────────────────────────────────────────────────────────── */

struct sz_mem_block {
   struct list_head link;
   VkDeviceMemory mem;
   uint64_t size;
   uint32_t mem_type;
   void *map;
   bool coherent;
   bool dedicated;

   /* Free ranges, ascending. A dedicated block starts with none: the whole
    * block belongs to its single sz_bo. */
   unsigned range_count;
   struct sz_range ranges[SZ_MAX_RANGES];
};

struct sz_bo {
   struct sz_screen *screen;
   struct sz_mem_block *block;
   VkDeviceMemory mem;
   uint64_t offset;
   uint64_t size;
   uint32_t mem_type;
   void *map;      /* NULL unless host-visible */
   bool coherent;
   bool dedicated;
};

struct sz_bo_state {
   struct list_head blocks;
   uint64_t allocated;   /* bytes handed out to live sz_bo */
   uint64_t committed;   /* bytes of VkDeviceMemory we hold */
   uint64_t peak_allocated;
   unsigned live;
   unsigned blocks_created;
};

void
sz_bo_init(struct sz_screen *screen)
{
   screen->bo = rzalloc(screen, struct sz_bo_state);
   list_inithead(&screen->bo->blocks);
}

static const char *
sz_mem_type_name(const struct sz_screen *screen, uint32_t type)
{
   const VkMemoryPropertyFlags f = screen->mem_props.memoryTypes[type].propertyFlags;
   if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
      return "local+host";
   if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      return "local";
   if (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
      return "host";
   return "other";
}

/* VkMemoryAllocateInfo::flags shared by every allocation the driver makes.
 * VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT is only legal once the
 * bufferDeviceAddress feature is enabled, which ADR-001 makes a hard
 * requirement, but it is still spelled out rather than assumed: a device that
 * somehow lacked it would otherwise fail every allocation with a validation
 * error instead of a clear message here. */
static VkMemoryAllocateFlags
sz_device_address_alloc_flag(const struct sz_screen *screen)
{
   return screen->caps.buffer_device_address
             ? VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT
             : 0;
}

static struct sz_mem_block *
sz_block_create(struct sz_screen *screen, uint64_t size, uint32_t mem_type)
{
   const VkMemoryPropertyFlags props =
      screen->mem_props.memoryTypes[mem_type].propertyFlags;

   /* Every buffer carries VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, and a
    * buffer with that usage may only be bound to memory allocated with this flag
    * (VUID 03339). Blocks are shared between buffers and images, and the flag
    * costs nothing for an image. */
   VkMemoryAllocateFlagsInfo flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .flags = sz_device_address_alloc_flag(screen),
   };

   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &flags_info,
      .allocationSize = size,
      .memoryTypeIndex = mem_type,
   };

   struct sz_mem_block *block = CALLOC_STRUCT(sz_mem_block);
   if (!block)
      return NULL;

   VkResult res = screen->vk.AllocateMemory(screen->dev, &ai, NULL, &block->mem);
   if (res != VK_SUCCESS) {
      /* A pooled block is an optimisation, not a requirement: reporting this as
       * fatal would turn "16 MiB refused" into "the screen dies". The caller
       * falls back to a dedicated allocation, which is exactly the size it
       * asked for and therefore much more likely to succeed. */
      mesa_logw("SZ: could not allocate a %" PRIu64 " MiB %s block "
                "(VkResult %d); falling back to dedicated allocations",
                size / (1024 * 1024), sz_mem_type_name(screen, mem_type), res);
      FREE(block);
      return NULL;
   }

   block->size = size;
   block->mem_type = mem_type;
   block->coherent = props & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   block->range_count = 1;
   block->ranges[0] = (struct sz_range) { .offset = 0, .size = size };

   if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      /* Mapped once and kept: vkMapMemory per upload is measurable on RADV at
       * hundreds of calls per frame, and Vulkan allows a persistent map. */
      res = screen->vk.MapMemory(screen->dev, block->mem, 0, VK_WHOLE_SIZE, 0,
                                 &block->map);
      if (res != VK_SUCCESS) {
         mesa_logw("SZ: vkMapMemory failed (VkResult %d); host writes to this "
                   "block will be unavailable", res);
         block->map = NULL;
      }
   }

   screen->bo->committed += size;
   screen->bo->blocks_created++;
   list_addtail(&block->link, &screen->bo->blocks);
   return block;
}

static void
sz_block_destroy(struct sz_screen *screen, struct sz_mem_block *block)
{
   if (block->map)
      screen->vk.UnmapMemory(screen->dev, block->mem);
   screen->vk.FreeMemory(screen->dev, block->mem, NULL);
   screen->bo->committed -= block->size;
   list_del(&block->link);
   FREE(block);
}

/* Smallest power-of-two-ish block that comfortably holds `size`. */
static uint64_t
sz_block_size_for(uint64_t size)
{
   /* Round the request up to a power of two and no further.
    *
    * This used to start at the minimum block and double until the block was
    * four times the request, which turned a 17 MiB staging buffer into a
    * 128 MiB allocation -- the opposite of what SZ_BLOCK_SLACK was documented
    * to mean. On a desktop GPU a handful of those exhausts device-local
    * memory, and running out of VRAM on a compositor does not fail the
    * allocation politely, it hangs the display.
    *
    * A power of two keeps the range allocator's splitting cheap, and it can
    * never be more than twice the request. */
   const uint64_t block = MAX2(util_next_power_of_two64(size), SZ_BLOCK_MIN_SIZE);
   return MAX2(block, sz_align_up(size, 65536));
}

static struct sz_bo *
sz_bo_wrap(struct sz_screen *screen, struct sz_mem_block *block,
           uint64_t offset, uint64_t size)
{
   struct sz_bo *bo = CALLOC_STRUCT(sz_bo);
   if (!bo)
      return NULL;

   bo->screen = screen;
   bo->block = block;
   bo->mem = block->mem;
   bo->offset = offset;
   bo->size = size;
   bo->mem_type = block->mem_type;
   bo->coherent = block->coherent;
   bo->dedicated = block->dedicated;

   if (block->map)
      bo->map = (char *)block->map + offset;

   screen->bo->allocated += size;
   screen->bo->peak_allocated = MAX2(screen->bo->peak_allocated,
                                     screen->bo->allocated);
   screen->bo->live++;

   return bo;
}

struct sz_bo *
sz_bo_alloc(struct sz_screen *screen, const VkMemoryRequirements *req,
            VkMemoryPropertyFlags props)
{
   const uint32_t mem_type =
      sz_device_find_memory_type(screen, req->memoryTypeBits, props);
   if (mem_type == VK_MAX_MEMORY_TYPES) {
      /* Second chance: the caller wanted host-visible device-local memory that
       * this GPU does not have (RADV's small BAR case). Host visibility is the
       * part that actually matters, so drop the device-local request. */
      const uint32_t host_only = sz_device_find_memory_type(
         screen, req->memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      if (host_only == VK_MAX_MEMORY_TYPES) {
         mesa_loge("SZ: no memory type satisfies mask 0x%x with properties 0x%x",
                   req->memoryTypeBits, (unsigned)props);
         return NULL;
      }
      return sz_bo_alloc(screen, req,
                         screen->mem_props.memoryTypes[host_only].propertyFlags);
   }

   /* At least 64 bytes, because that is the alignment the screen advertises as
    * min_map_buffer_alignment: glMapBufferRange promises a pointer aligned to
    * it, a direct mapping hands out bo->map + the requested offset, and
    * bo->map is the block's page-aligned base plus this offset. Without the
    * floor, a suballocated buffer at offset 16 breaks the promise -- which is
    * what piglit's map_buffer_range-invalidate checks, and what makes a caller
    * that trusts the alignment read or write the wrong bytes. */
   const uint64_t alignment = MAX2(req->alignment, 64ull);
   const uint64_t size = sz_align_up(req->size, alignment);

   /* Try existing blocks of the same memory type. */
   list_for_each_entry(struct sz_mem_block, block, &screen->bo->blocks, link) {
      if (block->mem_type != mem_type || block->dedicated)
         continue;

      uint64_t offset;
      if (!sz_range_alloc(block->ranges, &block->range_count, SZ_MAX_RANGES,
                          size, alignment, &offset))
         continue;

      return sz_bo_wrap(screen, block, offset, size);
   }

   /* Nothing fits: grow. A dedicated allocation is the fallback for drivers
    * that refuse large pooled blocks. */
   struct sz_mem_block *block =
      sz_block_create(screen, sz_block_size_for(size), mem_type);
   if (!block)
      return sz_bo_alloc_dedicated(screen, req, props);

   uint64_t offset;
   if (!sz_range_alloc(block->ranges, &block->range_count, SZ_MAX_RANGES,
                       size, alignment, &offset)) {
      /* Cannot happen for a fresh block, but failing loudly beats corrupting. */
      mesa_loge("SZ: fresh %" PRIu64 " MiB block could not satisfy a %" PRIu64
                " byte request", block->size / (1024 * 1024), size);
      sz_block_destroy(screen, block);
      return NULL;
   }

   return sz_bo_wrap(screen, block, offset, size);
}

static struct sz_bo *
sz_bo_alloc_private(struct sz_screen *screen, const VkMemoryRequirements *req,
                    VkMemoryPropertyFlags props,
                    VkExportMemoryAllocateInfo *export_info)
{
   uint32_t mem_type =
      sz_device_find_memory_type(screen, req->memoryTypeBits, props);
   if (mem_type == VK_MAX_MEMORY_TYPES) {
      mem_type = sz_device_find_memory_type(screen, req->memoryTypeBits,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
      if (mem_type == VK_MAX_MEMORY_TYPES) {
         mesa_loge("SZ: no deducated-capable memory type for mask 0x%x",
                   req->memoryTypeBits);
         return NULL;
      }
   }

   const VkMemoryPropertyFlags real =
      screen->mem_props.memoryTypes[mem_type].propertyFlags;

   VkMemoryAllocateFlagsInfo flags_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
      .pNext = export_info,
      .flags = sz_device_address_alloc_flag(screen),
   };

   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &flags_info,
      .allocationSize = req->size,
      .memoryTypeIndex = mem_type,
   };

   struct sz_mem_block *block = CALLOC_STRUCT(sz_mem_block);
   if (!block)
      return NULL;

   VkResult res = screen->vk.AllocateMemory(screen->dev, &ai, NULL, &block->mem);
   if (res != VK_SUCCESS) {
      mesa_loge("SZ: dedicated vkAllocateMemory of %" PRIu64 " bytes failed "
                "(VkResult %d)", req->size, res);
      FREE(block);
      return NULL;
   }

   block->size = req->size;
   block->mem_type = mem_type;
   block->dedicated = true;
   block->coherent = real & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

   if (real & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
      res = screen->vk.MapMemory(screen->dev, block->mem, 0, VK_WHOLE_SIZE, 0,
                                 &block->map);
      if (res != VK_SUCCESS)
         block->map = NULL;
   }

   screen->bo->committed += req->size;
   screen->bo->blocks_created++;
   list_addtail(&block->link, &screen->bo->blocks);

   return sz_bo_wrap(screen, block, 0, req->size);
}

struct sz_bo *
sz_bo_alloc_dedicated(struct sz_screen *screen, const VkMemoryRequirements *req,
                      VkMemoryPropertyFlags props)
{
   return sz_bo_alloc_private(screen, req, props, NULL);
}

struct sz_bo *
sz_bo_alloc_export(struct sz_screen *screen, const VkMemoryRequirements *req,
                   VkMemoryPropertyFlags props, VkImage image)
{
   /* VkMemoryDedicatedAllocateInfo is chained unconditionally: an external
    * allocation belongs to exactly one image by definition, and a driver that
    * requires the struct to be present fails the allocation without it. */
   VkMemoryDedicatedAllocateInfo dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .image = image,
   };
   VkExportMemoryAllocateInfo export_info = {
      .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .pNext = &dedicated,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };

   /* Exportable memory is shared with another process, so it must be a whole
    * VkDeviceMemory of its own: neither end may point into the middle of an
    * allocation the other one has no handle to. */
   return sz_bo_alloc_private(screen, req, props, &export_info);
}

void
sz_bo_free(struct sz_bo *bo)
{
   if (!bo)
      return;

   struct sz_screen *screen = bo->screen;
   struct sz_mem_block *block = bo->block;

   screen->bo->allocated -= bo->size;
   screen->bo->live--;

   if (block->dedicated) {
      sz_block_destroy(screen, block);
      FREE(bo);
      return;
   }

   if (!sz_range_free(block->ranges, &block->range_count, SZ_MAX_RANGES,
                      bo->offset, bo->size)) {
      /* The range table is full. Leaking the range is the only safe response:
       * handing the memory to a second owner would alias. */
      sz_warn("range table full, leaking %" PRIu64 " bytes", bo->size);
   }

   /* Reclaim a block that has gone completely idle, but keep one warm block per
    * memory type so a steady-state frame does not churn VkDeviceMemory. */
   if (block->range_count == 1 && block->ranges[0].size == block->size) {
      unsigned same_type = 0;
      list_for_each_entry(struct sz_mem_block, b, &screen->bo->blocks, link) {
         if (b->mem_type == block->mem_type && !b->dedicated)
            same_type++;
      }
      if (same_type > 1)
         sz_block_destroy(screen, block);
   }

   FREE(bo);
}

void
sz_bo_finish(struct sz_screen *screen)
{
   if (!screen->bo)
      return;

   /* VkDevice destruction frees the memory anyway; this exists so leaks show up
    * in the log instead of disappearing with the process. */
   if (sz_debug_enabled(SZ_DEBUG_INFO) && screen->bo->live)
      sz_warn("%u allocation%s still live at shutdown (%" PRIu64 " bytes)",
              screen->bo->live, screen->bo->live == 1 ? "" : "s",
              screen->bo->allocated);

   list_for_each_entry_safe(struct sz_mem_block, block, &screen->bo->blocks,
                            link) {
      sz_block_destroy(screen, block);
   }

   ralloc_free(screen->bo);
   screen->bo = NULL;
}

void *
sz_bo_map(struct sz_bo *bo)
{
   return bo ? bo->map : NULL;
}

void
sz_bo_flush(struct sz_bo *bo, uint64_t offset, uint64_t size)
{
   if (!bo || bo->coherent || !bo->map || !bo->screen->vk.FlushMappedMemoryRanges)
      return;

   VkMappedMemoryRange range = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = bo->mem,
      .offset = bo->offset + offset,
      .size = size == VK_WHOLE_SIZE ? VK_WHOLE_SIZE : size,
   };
   bo->screen->vk.FlushMappedMemoryRanges(bo->screen->dev, 1, &range);
}

void
sz_bo_invalidate(struct sz_bo *bo, uint64_t offset, uint64_t size)
{
   if (!bo || bo->coherent || !bo->map ||
       !bo->screen->vk.InvalidateMappedMemoryRanges)
      return;

   VkMappedMemoryRange range = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = bo->mem,
      .offset = bo->offset + offset,
      .size = size == VK_WHOLE_SIZE ? VK_WHOLE_SIZE : size,
   };
   bo->screen->vk.InvalidateMappedMemoryRanges(bo->screen->dev, 1, &range);
}

uint64_t
sz_bo_size(const struct sz_bo *bo)
{
   return bo ? bo->size : 0;
}

uint64_t
sz_bo_offset(const struct sz_bo *bo)
{
   return bo ? bo->offset : 0;
}

VkDeviceMemory
sz_bo_memory(const struct sz_bo *bo)
{
   return bo ? bo->mem : VK_NULL_HANDLE;
}

void
sz_bo_get_stats(const struct sz_screen *screen, struct sz_bo_stats *out)
{
   memset(out, 0, sizeof(*out));
   if (!screen->bo)
      return;

   out->allocated = screen->bo->allocated;
   out->committed = screen->bo->committed;
   out->peak_allocated = screen->bo->peak_allocated;
   out->live = screen->bo->live;
   out->blocks = screen->bo->blocks_created;
}

void
sz_bo_dump_stats(const struct sz_screen *screen)
{
   if (!sz_debug_enabled(SZ_DEBUG_INFO) || !screen->bo)
      return;

   const struct sz_bo_state *s = screen->bo;
   mesa_logi("SZ: bo: %u live (%" PRIu64 " KiB), %" PRIu64 " KiB committed in "
             "%u block%s, peak %" PRIu64 " KiB",
             s->live, s->allocated / 1024, s->committed / 1024,
             s->blocks_created, s->blocks_created == 1 ? "" : "s",
             s->peak_allocated / 1024);
}
