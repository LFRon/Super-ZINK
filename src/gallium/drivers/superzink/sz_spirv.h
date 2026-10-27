/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * A SPIR-V module builder.
 *
 * This is not a general-purpose assembler. It covers exactly what the driver
 * emits, which is deliberate: the alternative is a 20 kLOC generator whose own
 * bugs are indistinguishable from translation bugs.
 *
 * The one piece of real design here is the section layout. SPIR-V requires its
 * sections in a fixed order -- capabilities before memory model, annotations
 * before types, types before functions -- and getting it wrong produces a module
 * that some drivers accept and others reject. Rather than making callers
 * remember the order, each emit function appends to the section it belongs to
 * and sz_spv_finalize() concatenates them. Call order therefore never matters.
 */

#ifndef SZ_SPIRV_H
#define SZ_SPIRV_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "compiler/spirv/spirv.h"

struct sz_spv_module;

struct sz_spv_module *sz_spv_module_create(void);
void sz_spv_module_destroy(struct sz_spv_module *m);

/* Fills out the sections and returns the word array. The module keeps ownership;
 * the pointer is valid until sz_spv_module_destroy(). */
const uint32_t *sz_spv_finalize(struct sz_spv_module *m, size_t *out_size);

/* Allocates a fresh result id. Ids are never reused, so no invalidation to
 * reason about. */
uint32_t sz_spv_id(struct sz_spv_module *m);

/* Types. Each is cached, so asking twice returns the same id, which matters
 * because SPIR-V requires a type to be declared exactly once. */
uint32_t sz_spv_type_void(struct sz_spv_module *m);
uint32_t sz_spv_type_bool(struct sz_spv_module *m);
uint32_t sz_spv_type_int32(struct sz_spv_module *m);
uint32_t sz_spv_type_uint32(struct sz_spv_module *m);
uint32_t sz_spv_type_float32(struct sz_spv_module *m);
uint32_t sz_spv_type_vec(struct sz_spv_module *m, uint32_t component,
                         unsigned count);
uint32_t sz_spv_type_ptr(struct sz_spv_module *m, SpvStorageClass storage,
                         uint32_t pointee);
uint32_t sz_spv_type_function(struct sz_spv_module *m, uint32_t result,
                              const uint32_t *params, unsigned count);

/* Constants. */
uint32_t sz_spv_const_float32(struct sz_spv_module *m, float value);
uint32_t sz_spv_const_composite(struct sz_spv_module *m, uint32_t type,
                                const uint32_t *members, unsigned count);

/* Preamble. */
void sz_spv_capability(struct sz_spv_module *m, SpvCapability capability);
void sz_spv_memory_model(struct sz_spv_module *m, SpvAddressingModel addressing,
                         SpvMemoryModel memory);
void sz_spv_entry_point(struct sz_spv_module *m, SpvExecutionModel model,
                        uint32_t function, const char *name,
                        const uint32_t *interface, unsigned interface_count);
void sz_spv_execution_mode(struct sz_spv_module *m, uint32_t function,
                           SpvExecutionMode mode);

/* Annotations. */
void sz_spv_decorate(struct sz_spv_module *m, uint32_t target,
                     SpvDecoration decoration, uint32_t value);
void sz_spv_decorate_builtin(struct sz_spv_module *m, uint32_t target,
                             SpvBuiltIn builtin);

/* Globals. */
uint32_t sz_spv_variable(struct sz_spv_module *m, uint32_t ptr_type,
                         SpvStorageClass storage);

/* Functions. */
/* `id` must come from sz_spv_id(): OpEntryPoint names a function by id, so the
 * function cannot allocate its own -- a mismatch is a module whose entry point
 * refers to an id that is never defined. */
void sz_spv_function(struct sz_spv_module *m, uint32_t id, uint32_t return_type,
                     uint32_t fn_type);
uint32_t sz_spv_label(struct sz_spv_module *m);
void sz_spv_function_end(struct sz_spv_module *m);

/* Instructions. `type` is the result type, `result` the result id, or zero for
 * instructions that produce neither. */
void sz_spv_load(struct sz_spv_module *m, uint32_t type, uint32_t result,
                 uint32_t ptr);
void sz_spv_store(struct sz_spv_module *m, uint32_t ptr, uint32_t value);
uint32_t sz_spv_composite_construct(struct sz_spv_module *m, uint32_t type,
                                    const uint32_t *members, unsigned count);
void sz_spv_return(struct sz_spv_module *m);

/* Raw escape hatch for instructions this API does not model yet. The opcode and
 * operands are appended verbatim to the function section. */
void sz_spv_raw_function_op(struct sz_spv_module *m, SpvOp opcode,
                            const uint32_t *operands, unsigned count);

#endif /* SZ_SPIRV_H */
