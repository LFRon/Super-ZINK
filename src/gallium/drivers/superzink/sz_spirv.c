/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 */

#include "sz_spirv.h"

#include "util/log.h"
#include "util/u_memory.h"
#include "util/macros.h"

#include <stdlib.h>
#include <string.h>

/* Section indices, in the order SPIR-V requires them. */
enum {
   SEC_CAPABILITIES,
   SEC_EXTENSIONS,
   SEC_EXT_INST_IMPORTS,
   SEC_MEMORY_MODEL,
   SEC_ENTRY_POINTS,
   SEC_EXECUTION_MODES,
   SEC_NAMES,        /* OpName / OpMemberName */
   SEC_DECORATIONS,
   SEC_TYPES,        /* types, constants, global variables */
   SEC_FUNCTIONS,    /* function definitions */
   SEC_COUNT,
};

struct sz_spv_section {
   uint32_t *words;
   size_t used;
   size_t capacity;
};

struct sz_spv_module {
   struct sz_spv_section section[SEC_COUNT];
   uint32_t next_id;

   /* Type cache. Linear scan over a growable array: a hand-written shader
    * declares a few dozen types, and a hash table would be more code than the
    * whole emitter.
    *
    * It has to grow, because the count is not bounded by anything the driver
    * controls. A shader pack's passes declare hundreds -- Complementary
    * Reimagined's deferred pass alone overflows what used to be a fixed 256 --
    * and a full cache does not fail the shader, it makes every further type be
    * declared again, once per use, with an error line each time. */
   struct sz_spv_type_entry {
      unsigned kind;
      uint32_t a, b, c;
      uint32_t id;
   } *type_entry;
   unsigned type_count;
   unsigned type_capacity;
};

static bool
sz_spv_grow(struct sz_spv_section *s, size_t extra)
{
   if (s->used + extra <= s->capacity)
      return true;

   size_t capacity = MAX2(s->capacity * 2, s->used + extra);
   uint32_t *words = realloc(s->words, capacity * sizeof(*words));
   if (!words) {
      mesa_loge("SZ: out of memory growing a SPIR-V section");
      return false;
   }

   s->words = words;
   s->capacity = capacity;
   return true;
}

/* Appends one instruction. `count` is the total number of words including the
 * opcode word. */
static void
sz_spv_append(struct sz_spv_module *m, unsigned section, const uint32_t *words,
              size_t count)
{
   struct sz_spv_section *s = &m->section[section];

   if (!sz_spv_grow(s, count))
      return;

   memcpy(&s->words[s->used], words, count * sizeof(*words));
   s->used += count;
}

/* Builds the opcode word: word count in the high 16 bits, opcode in the low. */
static uint32_t
sz_spv_op_word(SpvOp opcode, size_t operand_words)
{
   return ((uint32_t)(operand_words + 1) << 16) | (uint32_t)opcode;
}

static void
sz_spv_emit_simple(struct sz_spv_module *m, unsigned section, SpvOp opcode,
                   const uint32_t *operands, size_t count)
{
   uint32_t header = sz_spv_op_word(opcode, count);

   if (!sz_spv_grow(&m->section[section], count + 1))
      return;

   sz_spv_append(m, section, &header, 1);
   if (count)
      sz_spv_append(m, section, operands, count);
}

struct sz_spv_module *
sz_spv_module_create(void)
{
   struct sz_spv_module *m = CALLOC_STRUCT(sz_spv_module);
   if (!m)
      return NULL;

   /* Id 0 is reserved in SPIR-V: it means "no result". Starting at 0 makes the
    * first id in the module the invalid one, and spirv_to_nir does not recover
    * from that -- it runs off the end of the module rather than reporting an
    * error. */
   m->next_id = 1;

   return m;
}

void
sz_spv_module_destroy(struct sz_spv_module *m)
{
   if (!m)
      return;

   for (unsigned i = 0; i < SEC_COUNT; i++)
      free(m->section[i].words);
   free(m->type_entry);
   free(m);
}

const uint32_t *
sz_spv_finalize(struct sz_spv_module *m, size_t *out_size)
{
   size_t total = 0;
   for (unsigned i = 0; i < SEC_COUNT; i++)
      total += m->section[i].used;

   /* The header goes in front of everything, so it becomes a section of its own
    * by being prepended to whatever ended up in slot 0. */
   struct sz_spv_section *pre = &m->section[0];
   const size_t header_words = 5;

   uint32_t *words = malloc((total + header_words) * sizeof(*words));
   if (!words) {
      mesa_loge("SZ: out of memory finalising a SPIR-V module");
      *out_size = 0;
      return NULL;
   }

   words[0] = 0x07230203; /* magic */
   words[1] = 0x00010600; /* SPIR-V 1.6 */
   words[2] = 0;          /* generator: none */
   words[3] = m->next_id; /* id bound */
   words[4] = 0;          /* reserved */

   if (pre->used)
      memcpy(&words[header_words], pre->words, pre->used * sizeof(*words));
   size_t at = header_words + pre->used;

   for (unsigned i = 1; i < SEC_COUNT; i++) {
      if (!m->section[i].used)
         continue;
      memcpy(&words[at], m->section[i].words,
             m->section[i].used * sizeof(*words));
      at += m->section[i].used;
   }

   free(pre->words);
   pre->words = words;
   pre->used = at;
   pre->capacity = at;

   *out_size = at * sizeof(*words);
   return words;
}

uint32_t
sz_spv_id(struct sz_spv_module *m)
{
   return m->next_id++;
}

static uint32_t
sz_spv_type_lookup(struct sz_spv_module *m, unsigned kind, uint32_t a, uint32_t b,
                   uint32_t c)
{
   for (unsigned i = 0; i < m->type_count; i++) {
      if (m->type_entry[i].kind == kind && m->type_entry[i].a == a &&
          m->type_entry[i].b == b && m->type_entry[i].c == c)
         return m->type_entry[i].id;
   }
   return 0;
}

static void
sz_spv_type_record(struct sz_spv_module *m, unsigned kind, uint32_t a,
                   uint32_t b, uint32_t c, uint32_t id)
{
   if (m->type_count == m->type_capacity) {
      const unsigned capacity = MAX2(m->type_capacity * 2, 256u);
      struct sz_spv_type_entry *grown =
         realloc(m->type_entry, capacity * sizeof(*grown));
      if (!grown) {
         /* Out of host memory: declare the type again rather than miscompile.
          * Duplicate OpType instructions are legal, just wasteful. */
         mesa_loge("SZ: SPIR-V type cache could not grow past %u entries",
                   m->type_count);
         return;
      }
      m->type_entry = grown;
      m->type_capacity = capacity;
   }
   struct sz_spv_type_entry *entry = &m->type_entry[m->type_count++];
   entry->kind = kind;
   entry->a = a;
   entry->b = b;
   entry->c = c;
   entry->id = id;
}

uint32_t
sz_spv_type_void(struct sz_spv_module *m)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeVoid, 0, 0, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeVoid, ops, 1);
   sz_spv_type_record(m, SpvOpTypeVoid, 0, 0, 0, id);
   return id;
}

uint32_t
sz_spv_type_bool(struct sz_spv_module *m)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeBool, 0, 0, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeBool, ops, 1);
   sz_spv_type_record(m, SpvOpTypeBool, 0, 0, 0, id);
   return id;
}

/* The cache key for an integer type has to include signedness, or int32 and
 * uint32 would collide and the second one would silently reuse the first. */
uint32_t
sz_spv_type_int(struct sz_spv_module *m, unsigned bits, bool is_signed)
{
   const uint32_t signedness = is_signed ? 1 : 0;

   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeInt, bits, signedness, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, bits, signedness };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeInt, ops, 3);
   sz_spv_type_record(m, SpvOpTypeInt, bits, signedness, 0, id);
   return id;
}

uint32_t
sz_spv_type_int32(struct sz_spv_module *m)
{
   return sz_spv_type_int(m, 32, true);
}

uint32_t
sz_spv_type_uint32(struct sz_spv_module *m)
{
   return sz_spv_type_int(m, 32, false);
}

uint32_t
sz_spv_type_float(struct sz_spv_module *m, unsigned bits)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeFloat, bits, 0, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, bits };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeFloat, ops, 2);
   sz_spv_type_record(m, SpvOpTypeFloat, bits, 0, 0, id);
   return id;
}

uint32_t
sz_spv_type_float32(struct sz_spv_module *m)
{
   return sz_spv_type_float(m, 32);
}

uint32_t
sz_spv_type_vec(struct sz_spv_module *m, uint32_t component, unsigned count)
{
   /* OpTypeVector's component count must be 2, 3 or 4: a one-component vector
    * is the component type itself. Callers reach this with a count that is
    * computed (a projected 1D coordinate, for instance), so the degenerate case
    * has to be handled here rather than trusted to every caller. */
   if (count == 1)
      return component;

   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeVector, component, count, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, component, count };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeVector, ops, 3);
   sz_spv_type_record(m, SpvOpTypeVector, component, count, 0, id);
   return id;
}

uint32_t
sz_spv_type_runtime_array(struct sz_spv_module *m, uint32_t element)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeRuntimeArray, element, 0, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, element };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeRuntimeArray, ops, 2);
   sz_spv_type_record(m, SpvOpTypeRuntimeArray, element, 0, 0, id);
   return id;
}

uint32_t
sz_spv_type_array(struct sz_spv_module *m, uint32_t element, unsigned count)
{
   const uint32_t length = sz_spv_const_uint32(m, count);
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeArray, element, length, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, element, length };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeArray, ops, 3);
   sz_spv_type_record(m, SpvOpTypeArray, element, length, 0, id);
   return id;
}

uint32_t
sz_spv_type_image(struct sz_spv_module *m, uint32_t sampled_type, SpvDim dim,
                  bool depth, bool arrayed, bool ms, unsigned sampled)
{
   /* OpTypeImage's Dim, Depth, Arrayed, MS and Sampled operands are literal
    * numbers rather than ids, so they cannot share the three-word cache key
    * directly; they are packed into it instead. */
   const uint32_t flags = (uint32_t)dim | (depth ? 1u << 8 : 0) |
                          (arrayed ? 1u << 9 : 0) | (ms ? 1u << 10 : 0) |
                          ((sampled & 3u) << 11);

   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeImage, sampled_type, flags, 0);
   if (id)
      return id;

   /* Some image shapes are not covered by the Shader capability and need their
    * own: a 1D texture (GL_TEXTURE_1D, which GLSL 1.30 still has), a buffer
    * texture, a cube array, and a multisampled array. Declaring the capability
    * is what makes spirv-val accept the type; without it the module is invalid
    * and the Vulkan driver's parser is entitled to do anything, including
    * segfault. */
   /* A storage image needs the Image* capability, a sampled image the Sampled*
    * one; `sampled` is 1 or 2, so the test is against 1 rather than truthiness. */
   const bool is_sampled = sampled == 1;

   switch (dim) {
   case SpvDim1D:
      sz_spv_capability(m, is_sampled ? SpvCapabilitySampled1D
                                      : SpvCapabilityImage1D);
      break;
   case SpvDimBuffer:
      sz_spv_capability(m, is_sampled ? SpvCapabilitySampledBuffer
                                      : SpvCapabilityImageBuffer);
      break;
   case SpvDimCube:
      if (arrayed)
         sz_spv_capability(m, SpvCapabilitySampledCubeArray);
      break;
   case SpvDimRect:
      sz_spv_capability(m, SpvCapabilitySampledRect);
      break;
   case SpvDimSubpassData:
      sz_spv_capability(m, SpvCapabilityInputAttachment);
      break;
   default:
      break;
   }
   if (ms && arrayed)
      sz_spv_capability(m, is_sampled ? SpvCapabilitySampledImageArrayDynamicIndexing
                                      : SpvCapabilityImageMSArray);

   /* Every texture in this driver is a sampled texture with an ordinary GL
    * format, so the format operand is Unknown: it only matters for storage
    * images, where the shader and the view must agree exactly. */
   id = sz_spv_id(m);
   const uint32_t ops[] = { id, sampled_type, (uint32_t)dim, depth, arrayed, ms,
                            sampled, SpvImageFormatUnknown };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeImage, ops, 8);
   sz_spv_type_record(m, SpvOpTypeImage, sampled_type, flags, 0, id);
   return id;
}

uint32_t
sz_spv_type_sampled_image(struct sz_spv_module *m, uint32_t image)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeSampledImage, image, 0, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, image };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeSampledImage, ops, 2);
   sz_spv_type_record(m, SpvOpTypeSampledImage, image, 0, 0, id);
   return id;
}

uint32_t
sz_spv_type_struct(struct sz_spv_module *m, const uint32_t *members,
                   unsigned count)
{
   const uint32_t id = sz_spv_id(m);

   uint32_t *ops = CALLOC(count + 1, sizeof(*ops));
   if (!ops)
      return 0;

   ops[0] = id;
   for (unsigned i = 0; i < count; i++)
      ops[i + 1] = members[i];

   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeStruct, ops, count + 1);
   free(ops);
   return id;
}

uint32_t
sz_spv_type_ptr(struct sz_spv_module *m, SpvStorageClass storage,
                uint32_t pointee)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypePointer, storage, pointee, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, storage, pointee };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypePointer, ops, 3);
   sz_spv_type_record(m, SpvOpTypePointer, storage, pointee, 0, id);
   return id;
}

uint32_t
sz_spv_type_function(struct sz_spv_module *m, uint32_t result,
                     const uint32_t *params, unsigned count)
{
   uint32_t id = sz_spv_id(m);

   uint32_t ops[1 + 1 + 8];
   ops[0] = id;
   ops[1] = result;
   for (unsigned i = 0; i < count && i < 8; i++)
      ops[2 + i] = params[i];

   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeFunction, ops, 2 + MIN2(count, 8));
   return id;
}

uint32_t
sz_spv_const_bits(struct sz_spv_module *m, uint32_t type, uint32_t bits)
{
   /* Keyed by (type, bits), not bits alone: 1.0f and the integer 0x3f800000 have
    * the same bit pattern, and reusing one as the other produces a module that
    * validates but computes the wrong thing. */
   uint32_t id = sz_spv_type_lookup(m, SpvOpConstant, type, bits, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { type, id, bits };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpConstant, ops, 3);
   sz_spv_type_record(m, SpvOpConstant, type, bits, 0, id);
   return id;
}

uint32_t
sz_spv_const_float32(struct sz_spv_module *m, float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return sz_spv_const_bits(m, sz_spv_type_float32(m), bits);
}

uint32_t
sz_spv_const_int32(struct sz_spv_module *m, int32_t value)
{
   return sz_spv_const_bits(m, sz_spv_type_int(m, 32, true), (uint32_t)value);
}

uint32_t
sz_spv_const_uint32(struct sz_spv_module *m, uint32_t value)
{
   return sz_spv_const_bits(m, sz_spv_type_int(m, 32, false), value);
}

uint32_t
sz_spv_const_bool(struct sz_spv_module *m, bool value)
{
   const uint32_t type = sz_spv_type_bool(m);
   uint32_t id = sz_spv_type_lookup(m, SpvOpConstantTrue, type, 0, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { type, id };
   sz_spv_emit_simple(m, SEC_TYPES,
                      value ? SpvOpConstantTrue : SpvOpConstantFalse, ops, 2);
   sz_spv_type_record(m, SpvOpConstantTrue, type, 0, 0, id);
   return id;
}

uint32_t
sz_spv_const_composite(struct sz_spv_module *m, uint32_t type,
                       const uint32_t *members, unsigned count)
{
   uint32_t id = sz_spv_id(m);

   uint32_t ops[16];
   ops[0] = type;
   ops[1] = id;
   for (unsigned i = 0; i < count && i < 14; i++)
      ops[2 + i] = members[i];

   sz_spv_emit_simple(m, SEC_TYPES, SpvOpConstantComposite, ops,
                      2 + MIN2(count, 14));
   return id;
}

void
sz_spv_capability(struct sz_spv_module *m, SpvCapability capability)
{
   const uint32_t ops[] = { capability };
   sz_spv_emit_simple(m, SEC_CAPABILITIES, SpvOpCapability, ops, 1);
}

void
sz_spv_memory_model(struct sz_spv_module *m, SpvAddressingModel addressing,
                    SpvMemoryModel memory)
{
   const uint32_t ops[] = { addressing, memory };
   sz_spv_emit_simple(m, SEC_MEMORY_MODEL, SpvOpMemoryModel, ops, 2);
}

void
sz_spv_execution_mode(struct sz_spv_module *m, uint32_t function,
                      SpvExecutionMode mode)
{
   const uint32_t ops[] = { function, mode };
   sz_spv_emit_simple(m, SEC_EXECUTION_MODES, SpvOpExecutionMode, ops, 2);
}

void
sz_spv_execution_mode_literal(struct sz_spv_module *m, uint32_t function,
                              SpvExecutionMode mode, uint32_t value)
{
   /* OpExecutionModeId would take an id; OutputVertices and Invocations are
    * literals, which is what a geometry shader's declared counts are. */
   const uint32_t ops[] = { function, mode, value };
   sz_spv_emit_simple(m, SEC_EXECUTION_MODES, SpvOpExecutionMode, ops, 3);
}

void
sz_spv_entry_point(struct sz_spv_module *m, SpvExecutionModel model,
                   uint32_t function, const char *name,
                   const uint32_t *interface, unsigned interface_count)
{
   /* OpEntryPoint takes a literal string: one word per four characters, padded
    * with NULs, and it is *not* NUL-terminated. */
   /* The string is NUL-terminated, so `len + 1` bytes have to fit; the padding
    * that follows is part of its last word. Off by one here emits a stray zero
    * word that a validator reads as an interface id of 0. */
   const size_t len = strlen(name);
   const size_t name_words = (len + 4) / 4;
   /* Operands only: the opcode word carries its own length. */
   const size_t count = 2 + name_words + interface_count;

   uint32_t *ops = calloc(count, sizeof(*ops));
   if (!ops)
      return;

   ops[0] = model;
   ops[1] = function;
   memcpy(&ops[2], name, len);

   for (unsigned i = 0; i < interface_count; i++)
      ops[2 + name_words + i] = interface[i];

   sz_spv_emit_simple(m, SEC_ENTRY_POINTS, SpvOpEntryPoint, ops, count);
   free(ops);
}

void
sz_spv_decorate(struct sz_spv_module *m, uint32_t target,
                SpvDecoration decoration, uint32_t value)
{
   const uint32_t ops[] = { target, decoration, value };
   sz_spv_emit_simple(m, SEC_DECORATIONS, SpvOpDecorate, ops, 3);
}

void
sz_spv_decorate_builtin(struct sz_spv_module *m, uint32_t target,
                        SpvBuiltIn builtin)
{
   sz_spv_decorate(m, target, SpvDecorationBuiltIn, builtin);
}

uint32_t
sz_spv_variable(struct sz_spv_module *m, uint32_t ptr_type,
                SpvStorageClass storage)
{
   return sz_spv_variable_id(m, sz_spv_id(m), ptr_type, storage);
}

uint32_t
sz_spv_variable_id(struct sz_spv_module *m, uint32_t id, uint32_t ptr_type,
                   SpvStorageClass storage)
{
   const uint32_t ops[] = { ptr_type, id, storage };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpVariable, ops, 3);
   return id;
}

uint32_t
sz_spv_ext_inst_import(struct sz_spv_module *m, const char *name)
{
   const size_t len = strlen(name);
   const size_t name_words = (len + 4) / 4;

   uint32_t *ops = calloc(1 + name_words, sizeof(*ops));
   if (!ops)
      return 0;

   const uint32_t id = sz_spv_id(m);
   ops[0] = id;
   memcpy(&ops[1], name, len);
   sz_spv_emit_simple(m, SEC_EXT_INST_IMPORTS, SpvOpExtInstImport, ops,
                      1 + name_words);
   free(ops);
   return id;
}

uint32_t
sz_spv_ext_inst(struct sz_spv_module *m, uint32_t set, uint32_t opcode,
                uint32_t type, const uint32_t *operands, unsigned count)
{
   uint32_t ops[4 + 8];
   const uint32_t id = sz_spv_id(m);
   ops[0] = type;
   ops[1] = id;
   ops[2] = set;
   ops[3] = opcode;
   for (unsigned i = 0; i < count && i < 8; i++)
      ops[4 + i] = operands[i];

   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpExtInst, ops,
                      4 + MIN2(count, 8));
   return id;
}

void
sz_spv_function(struct sz_spv_module *m, uint32_t id, uint32_t return_type,
                uint32_t fn_type)
{
   const uint32_t ops[] = { return_type, id, SpvFunctionControlMaskNone, fn_type };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpFunction, ops, 4);
}

uint32_t
sz_spv_label(struct sz_spv_module *m)
{
   return sz_spv_label_id(m, sz_spv_id(m));
}

uint32_t
sz_spv_label_id(struct sz_spv_module *m, uint32_t id)
{
   const uint32_t ops[] = { id };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpLabel, ops, 1);
   return id;
}

void
sz_spv_function_end(struct sz_spv_module *m)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpFunctionEnd, NULL, 0);
}

uint32_t
sz_spv_load(struct sz_spv_module *m, uint32_t type, uint32_t ptr)
{
   const uint32_t id = sz_spv_id(m);
   const uint32_t ops[] = { type, id, ptr };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpLoad, ops, 3);
   return id;
}

void
sz_spv_store(struct sz_spv_module *m, uint32_t ptr, uint32_t value)
{
   const uint32_t ops[] = { ptr, value };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpStore, ops, 2);
}

uint32_t
sz_spv_composite_construct(struct sz_spv_module *m, uint32_t type,
                           const uint32_t *members, unsigned count)
{
   uint32_t id = sz_spv_id(m);

   uint32_t ops[16];
   ops[0] = type;
   ops[1] = id;
   for (unsigned i = 0; i < count && i < 14; i++)
      ops[2 + i] = members[i];

   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpCompositeConstruct, ops,
                      2 + MIN2(count, 14));
   return id;
}

void
sz_spv_return(struct sz_spv_module *m)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpReturn, NULL, 0);
}

void
sz_spv_member_decorate(struct sz_spv_module *m, uint32_t type, unsigned member,
                       SpvDecoration decoration, uint32_t value)
{
   const uint32_t ops[] = { type, member, decoration, value };
   sz_spv_emit_simple(m, SEC_DECORATIONS, SpvOpMemberDecorate, ops, 4);
}

void
sz_spv_decorate_bare(struct sz_spv_module *m, uint32_t target,
                     SpvDecoration decoration)
{
   const uint32_t ops[] = { target, decoration };
   sz_spv_emit_simple(m, SEC_DECORATIONS, SpvOpDecorate, ops, 2);
}

uint32_t
sz_spv_undef(struct sz_spv_module *m, uint32_t type)
{
   const uint32_t id = sz_spv_id(m);
   const uint32_t ops[] = { type, id };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpUndef, ops, 2);
   return id;
}

uint32_t
sz_spv_access_chain(struct sz_spv_module *m, uint32_t result_type,
                    uint32_t base, const uint32_t *indices, unsigned count)
{
   uint32_t ops[3 + 8];
   const uint32_t id = sz_spv_id(m);
   ops[0] = result_type;
   ops[1] = id;
   ops[2] = base;
   for (unsigned i = 0; i < count && i < 8; i++)
      ops[3 + i] = indices[i];

   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpAccessChain, ops,
                      3 + MIN2(count, 8));
   return id;
}

uint32_t
sz_spv_composite_extract(struct sz_spv_module *m, uint32_t type,
                         uint32_t composite, const uint32_t *indices,
                         unsigned count)
{
   uint32_t ops[3 + 8];
   const uint32_t id = sz_spv_id(m);
   ops[0] = type;
   ops[1] = id;
   ops[2] = composite;
   for (unsigned i = 0; i < count && i < 8; i++)
      ops[3 + i] = indices[i];

   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpCompositeExtract, ops,
                      3 + MIN2(count, 8));
   return id;
}

uint32_t
sz_spv_alu(struct sz_spv_module *m, SpvOp opcode, uint32_t type,
           const uint32_t *operands, unsigned count)
{
   uint32_t ops[2 + 16];
   const uint32_t id = sz_spv_id(m);
   ops[0] = type;
   ops[1] = id;
   for (unsigned i = 0; i < count && i < 16; i++)
      ops[2 + i] = operands[i];

   sz_spv_emit_simple(m, SEC_FUNCTIONS, opcode, ops, 2 + MIN2(count, 16));
   return id;
}

uint32_t
sz_spv_result_only(struct sz_spv_module *m, SpvOp opcode,
                   const uint32_t *operands, unsigned count)
{
   uint32_t ops[1 + 8];
   const uint32_t id = sz_spv_id(m);
   ops[0] = id;
   for (unsigned i = 0; i < count && i < 8; i++)
      ops[1 + i] = operands[i];

   sz_spv_emit_simple(m, SEC_FUNCTIONS, opcode, ops, 1 + MIN2(count, 8));
   return id;
}

void
sz_spv_raw_function_op(struct sz_spv_module *m, SpvOp opcode,
                       const uint32_t *operands, unsigned count)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, opcode, operands, count);
}

void
sz_spv_emit_vertex(struct sz_spv_module *m)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpEmitVertex, NULL, 0);
}

void
sz_spv_end_primitive(struct sz_spv_module *m)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpEndPrimitive, NULL, 0);
}

void
sz_spv_branch(struct sz_spv_module *m, uint32_t target)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpBranch, &target, 1);
}

void
sz_spv_branch_conditional(struct sz_spv_module *m, uint32_t condition,
                          uint32_t then_id, uint32_t else_id)
{
   const uint32_t ops[] = { condition, then_id, else_id };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpBranchConditional, ops, 3);
}

void
sz_spv_selection_merge(struct sz_spv_module *m, uint32_t merge,
                       SpvSelectionControlMask control)
{
   const uint32_t ops[] = { merge, control };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpSelectionMerge, ops, 2);
}

void
sz_spv_loop_merge(struct sz_spv_module *m, uint32_t merge, uint32_t cont,
                  SpvLoopControlMask control)
{
   const uint32_t ops[] = { merge, cont, control };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpLoopMerge, ops, 3);
}

uint32_t
sz_spv_phi(struct sz_spv_module *m, uint32_t type, const uint32_t *operands,
           unsigned count, size_t *first_operand)
{
   struct sz_spv_section *s = &m->section[SEC_FUNCTIONS];
   const uint32_t id = sz_spv_id(m);
   const uint32_t header = sz_spv_op_word(SpvOpPhi, 2 + count);
   const size_t start = s->used;

   sz_spv_append(m, SEC_FUNCTIONS, &header, 1);
   sz_spv_append(m, SEC_FUNCTIONS, &type, 1);
   sz_spv_append(m, SEC_FUNCTIONS, &id, 1);
   if (count)
      sz_spv_append(m, SEC_FUNCTIONS, operands, count);

   if (first_operand)
      *first_operand = start + 3;

   return id;
}

void
sz_spv_patch_function_word(struct sz_spv_module *m, size_t offset,
                           uint32_t word)
{
   if (offset < m->section[SEC_FUNCTIONS].used)
      m->section[SEC_FUNCTIONS].words[offset] = word;
}
