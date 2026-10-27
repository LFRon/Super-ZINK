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
 * the pointer is valid until sz_spv_module_destroy().
 *
 * `out_size` is in *bytes*, which is what VkShaderModuleCreateInfo wants; callers
 * that index the array divide by four. */
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
uint32_t sz_spv_type_float(struct sz_spv_module *m, unsigned bits);
uint32_t sz_spv_type_int(struct sz_spv_module *m, unsigned bits, bool is_signed);
uint32_t sz_spv_type_vec(struct sz_spv_module *m, uint32_t component,
                         unsigned count);
/* Structs are never cached: their identity is the whole member list, which does
 * not fit the type cache's three-word key. Each is declared once per module. */
uint32_t sz_spv_type_struct(struct sz_spv_module *m, const uint32_t *members,
                            unsigned count);
uint32_t sz_spv_type_runtime_array(struct sz_spv_module *m, uint32_t element);
/* Fixed-length array. The length is materialised as a constant, so it shares the
 * module's constant pool with every other use of the same number. */
uint32_t sz_spv_type_array(struct sz_spv_module *m, uint32_t element,
                           unsigned count);
/* OpTypeImage. `sampled_type` is the type of one texel component as the shader
 * reads it, `dim` the dimensionality, and the flags map straight onto the
 * operands of the same name. Not cached by the three-word key alone: the
 * literal operands are packed into it. */
uint32_t sz_spv_type_image(struct sz_spv_module *m, uint32_t sampled_type,
                           SpvDim dim, bool depth, bool arrayed, bool ms,
                           bool sampled);
/* OpTypeSampledImage, the type a combined image sampler variable holds. */
uint32_t sz_spv_type_sampled_image(struct sz_spv_module *m, uint32_t image);
uint32_t sz_spv_type_ptr(struct sz_spv_module *m, SpvStorageClass storage,
                         uint32_t pointee);
uint32_t sz_spv_type_function(struct sz_spv_module *m, uint32_t result,
                              const uint32_t *params, unsigned count);

/* Constants. `sz_spv_const_bits` is the primitive: the bit pattern plus the type
 * it is a constant *of*. The typed wrappers exist because passing a float's bits
 * where an int is meant is silent otherwise. */
uint32_t sz_spv_const_bits(struct sz_spv_module *m, uint32_t type, uint32_t bits);
uint32_t sz_spv_const_float32(struct sz_spv_module *m, float value);
uint32_t sz_spv_const_int32(struct sz_spv_module *m, int32_t value);
uint32_t sz_spv_const_uint32(struct sz_spv_module *m, uint32_t value);
uint32_t sz_spv_const_bool(struct sz_spv_module *m, bool value);
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

/* Decoration on one member of a struct type, such as Block's Offset. */
void sz_spv_member_decorate(struct sz_spv_module *m, uint32_t type,
                            unsigned member, SpvDecoration decoration,
                            uint32_t value);

/* A decoration that takes no extra operand, such as Flat. */
void sz_spv_decorate_bare(struct sz_spv_module *m, uint32_t target,
                          SpvDecoration decoration);

/* Globals. */
uint32_t sz_spv_variable(struct sz_spv_module *m, uint32_t ptr_type,
                         SpvStorageClass storage);
uint32_t sz_spv_variable_id(struct sz_spv_module *m, uint32_t id,
                            uint32_t ptr_type, SpvStorageClass storage);

/* Functions. */
/* `id` must come from sz_spv_id(): OpEntryPoint names a function by id, so the
 * function cannot allocate its own -- a mismatch is a module whose entry point
 * refers to an id that is never defined. */
void sz_spv_function(struct sz_spv_module *m, uint32_t id, uint32_t return_type,
                     uint32_t fn_type);
uint32_t sz_spv_label(struct sz_spv_module *m);
/* Same, with a caller-chosen id, so a branch can name a label that has not been
 * emitted yet. */
uint32_t sz_spv_label_id(struct sz_spv_module *m, uint32_t id);
void sz_spv_function_end(struct sz_spv_module *m);

/* Instructions. `type` is the result type, `result` the result id, or zero for
 * instructions that produce neither. */
uint32_t sz_spv_load(struct sz_spv_module *m, uint32_t type, uint32_t ptr);
void sz_spv_store(struct sz_spv_module *m, uint32_t ptr, uint32_t value);
void sz_spv_branch(struct sz_spv_module *m, uint32_t target);
void sz_spv_branch_conditional(struct sz_spv_module *m, uint32_t condition,
                               uint32_t then_id, uint32_t else_id);
void sz_spv_selection_merge(struct sz_spv_module *m, uint32_t merge,
                            SpvSelectionControlMask control);
void sz_spv_loop_merge(struct sz_spv_module *m, uint32_t merge, uint32_t cont,
                       SpvLoopControlMask control);

/* OpPhi. `operands` is a flat (value, block) sequence. The value of an operand
 * whose defining instruction has not been translated yet is emitted as zero and
 * its word offset is returned in `first_operand`, which the caller resolves with
 * sz_spv_patch_function_word() once the module is complete -- a loop header's
 * phi always takes one operand from the back edge, which by definition comes
 * later in the emission order. */
uint32_t sz_spv_phi(struct sz_spv_module *m, uint32_t type,
                    const uint32_t *operands, unsigned count,
                    size_t *first_operand);

/* Overwrites one word of the function section, at an offset returned by
 * sz_spv_phi(). */
void sz_spv_patch_function_word(struct sz_spv_module *m, size_t offset,
                                uint32_t word);

uint32_t sz_spv_composite_construct(struct sz_spv_module *m, uint32_t type,
                                    const uint32_t *members, unsigned count);
void sz_spv_return(struct sz_spv_module *m);

/* Generic result-producing emits. The translator needs dozens of opcodes, and a
 * named wrapper per opcode would be more code than the translator itself --
 * while a raw opcode with the operands spelled out keeps the NIR-to-SPIR-V
 * mapping readable in one place. */
uint32_t sz_spv_undef(struct sz_spv_module *m, uint32_t type);
uint32_t sz_spv_access_chain(struct sz_spv_module *m, uint32_t result_type,
                             uint32_t base, const uint32_t *indices,
                             unsigned count);
uint32_t sz_spv_composite_extract(struct sz_spv_module *m, uint32_t type,
                                  uint32_t composite, const uint32_t *indices,
                                  unsigned count);

/* Emits `opcode` with a result type and a fresh result id, followed by
 * `operands`. */
uint32_t sz_spv_alu(struct sz_spv_module *m, SpvOp opcode, uint32_t type,
                    const uint32_t *operands, unsigned count);

/* Emits `opcode` with a fresh result id and no result type. */
uint32_t sz_spv_result_only(struct sz_spv_module *m, SpvOp opcode,
                            const uint32_t *operands, unsigned count);

/* Extended instruction sets. */
uint32_t sz_spv_ext_inst_import(struct sz_spv_module *m, const char *name);
uint32_t sz_spv_ext_inst(struct sz_spv_module *m, uint32_t set, uint32_t opcode,
                         uint32_t type, const uint32_t *operands,
                         unsigned count);

/* Raw escape hatch for instructions this API does not model yet. The opcode and
 * operands are appended verbatim to the function section. */
void sz_spv_raw_function_op(struct sz_spv_module *m, SpvOp opcode,
                            const uint32_t *operands, unsigned count);

#endif /* SZ_SPIRV_H */
