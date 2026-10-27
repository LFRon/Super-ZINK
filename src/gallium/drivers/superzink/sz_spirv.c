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

   /* Type cache. Linear scan: a module declares a few dozen types at most, and a
    * hash table would be more code than the whole emitter. */
   struct {
      struct sz_spv_type_entry {
         unsigned kind;
         uint32_t a, b, c;
         uint32_t id;
      } entry[64];
      unsigned count;
   } types;
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
   for (unsigned i = 0; i < m->types.count; i++) {
      if (m->types.entry[i].kind == kind && m->types.entry[i].a == a &&
          m->types.entry[i].b == b && m->types.entry[i].c == c)
         return m->types.entry[i].id;
   }
   return 0;
}

static void
sz_spv_type_record(struct sz_spv_module *m, unsigned kind, uint32_t a,
                   uint32_t b, uint32_t c, uint32_t id)
{
   if (m->types.count == ARRAY_SIZE(m->types.entry)) {
      mesa_loge("SZ: SPIR-V type cache is full; a type would be declared twice");
      return;
   }
   struct sz_spv_type_entry *entry = &m->types.entry[m->types.count++];
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
static uint32_t
sz_spv_type_int(struct sz_spv_module *m, unsigned signedness)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeInt, 32, signedness, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, 32, signedness };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeInt, ops, 3);
   sz_spv_type_record(m, SpvOpTypeInt, 32, signedness, 0, id);
   return id;
}

uint32_t
sz_spv_type_int32(struct sz_spv_module *m)
{
   return sz_spv_type_int(m, 1);
}

uint32_t
sz_spv_type_uint32(struct sz_spv_module *m)
{
   return sz_spv_type_int(m, 0);
}

uint32_t
sz_spv_type_float32(struct sz_spv_module *m)
{
   uint32_t id = sz_spv_type_lookup(m, SpvOpTypeFloat, 32, 0, 0);
   if (id)
      return id;

   id = sz_spv_id(m);
   const uint32_t ops[] = { id, 32 };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpTypeFloat, ops, 2);
   sz_spv_type_record(m, SpvOpTypeFloat, 32, 0, 0, id);
   return id;
}

uint32_t
sz_spv_type_vec(struct sz_spv_module *m, uint32_t component, unsigned count)
{
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
sz_spv_const_float32(struct sz_spv_module *m, float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));

   uint32_t id = sz_spv_type_lookup(m, SpvOpConstant, bits, 0, 0);
   if (id)
      return id;

   const uint32_t type = sz_spv_type_float32(m);
   id = sz_spv_id(m);
   const uint32_t ops[] = { type, id, bits };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpConstant, ops, 3);
   sz_spv_type_record(m, SpvOpConstant, bits, 0, 0, id);
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
   uint32_t id = sz_spv_id(m);
   const uint32_t ops[] = { ptr_type, id, storage };
   sz_spv_emit_simple(m, SEC_TYPES, SpvOpVariable, ops, 3);
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
   uint32_t id = sz_spv_id(m);
   const uint32_t ops[] = { id };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpLabel, ops, 1);
   return id;
}

void
sz_spv_function_end(struct sz_spv_module *m)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpFunctionEnd, NULL, 0);
}

void
sz_spv_load(struct sz_spv_module *m, uint32_t type, uint32_t result,
            uint32_t ptr)
{
   const uint32_t ops[] = { type, result, ptr };
   sz_spv_emit_simple(m, SEC_FUNCTIONS, SpvOpLoad, ops, 3);
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
sz_spv_raw_function_op(struct sz_spv_module *m, SpvOp opcode,
                       const uint32_t *operands, unsigned count)
{
   sz_spv_emit_simple(m, SEC_FUNCTIONS, opcode, operands, count);
}
