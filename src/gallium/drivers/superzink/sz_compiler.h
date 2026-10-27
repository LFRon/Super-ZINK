/*
 * Copyright © 2026 Super-ZINK contributors
 * SPDX-License-Identifier: MIT
 *
 * Shader compilation.
 *
 * Two things live here that have to agree with each other: the
 * nir_shader_compiler_options we hand the state tracker, and the set of NIR we
 * can translate. The options are not tuning knobs -- each one tells the frontend
 * "lower this for me", and every option left off is a promise that the
 * translator handles it. Getting that wrong produces a shader that links and then
 * translates into nonsense, so the options describe exactly the capabilities this
 * backend has and nothing more.
 *
 * Following decision D8, no nir_opt_* pass runs here. Every Vulkan driver turns
 * SPIR-V back into NIR and optimises it again, so optimising here is work whose
 * result is discarded -- and it costs compile time on the critical path.
 */

#ifndef SZ_COMPILER_H
#define SZ_COMPILER_H

#include <stddef.h>
#include <stdint.h>

#include "compiler/nir/nir.h"

/* The options for one shader stage. The returned pointer stays valid for the
 * lifetime of the process. */
const struct nir_shader_compiler_options *sz_nir_options(mesa_shader_stage stage);

/* Correctness-only lowering, applied to the driver's own copy of the NIR before
 * translation. Idempotent; safe to call on already-lowered shaders. */
void sz_lower_nir(struct nir_shader *nir);

/* Translates a lowered NIR shader into a SPIR-V module. On success *out_words is
 * a malloc'd array of *out_size bytes, owned by the caller.
 *
 * Returns false and leaves *out_words NULL for anything outside the implemented
 * subset. Every rejection names the NIR instruction it could not handle: a
 * translator that silently skips an instruction produces a shader that renders
 * the wrong thing, which is far more expensive to debug than a refusal. */
bool sz_nir_to_spirv(struct nir_shader *nir, uint32_t **out_words,
                     size_t *out_size, char **error);

/* SZ_DEBUG=spirv: writes a module to dumpNN_<suffix>.spv in the working
 * directory, next to the bootstrap shaders' dumps. */
void sz_compiler_dump_spirv(const uint32_t *words, size_t word_count,
                            const char *suffix);

#endif /* SZ_COMPILER_H */
