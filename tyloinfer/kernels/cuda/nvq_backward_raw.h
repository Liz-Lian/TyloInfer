#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq_jit_nvq_backward {
/** Describe packed buffers and fixed FP16 backward geometry. */
struct Args {
    const void *indices, *aux, *state, *scale, *table, *banks, *gradient;
    void *output;
    std::int64_t indices_size, aux_size, state_size;
    int rows, outputs, width, groups, vectors, signs, supergroups, table_stride;
    int state_bits, sign_mode, format, output_tile;
};
/** Launch the NVQ split partials or fused tensor-core path. */
void nvq(const Args &args, bool mma, cudaStream_t stream);
/** Launch the NEPQ split partials or fused tensor-core path. */
void nepq(const Args &args, bool mma, cudaStream_t stream);
}
