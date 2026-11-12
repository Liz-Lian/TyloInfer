#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::nvq_raw {
/** Describe packed weights and their original CUDA decoding geometry. */
struct Weight {
    const std::uint8_t *indices, *aux, *state;
    std::int64_t indices_bytes, aux_bytes, state_bytes;
    const float* scale;
    const std::int8_t* table;
    std::int64_t table_bytes;
    const std::uint8_t* banks;
    int rows, width, group_size, state_bits, format, sign_mode, table_stride;
};
/** Describe caller-owned activation and output workspaces for a projection. */
struct Projection {
    Weight weight;
    const void* input;
    std::int8_t* quantized;
    float* scales;
    void* output;
    const void* residual;
    const void* gate;
    int rows, warps, gate_mode;
};
/** Apply the original quantized GEMV dispatch and optional activation quantization. */
void gemv(const Projection&, bool quantize, bool gated, int method, cudaStream_t);
/** Apply the original quantized integer-MMA projection. */
void mmq(const Projection&, bool gated, cudaStream_t);
/** Return whether a format supports the original fused residual path. */
bool group_exec(int format);
/** Apply the original aligned-group residual GEMV. */
void gemv_residual(const Projection&, cudaStream_t);
/** Run the original F16 online-dequant projection. */
void gemm(const Weight&, const void* input, void* output, int rows, cudaStream_t);
/** Decode NVQ or NPQ weights to an F16 matrix on the supplied stream. */
void decode(const Weight&, void* output, cudaStream_t);
/** Decode shared-bank NEPQ weights to an F16 matrix on the supplied stream. */
void decode_nepq(const Weight&, void* output, cudaStream_t);
/** Decode selected NVQ embedding rows to F16 on the supplied stream. */
void embedding(const Weight&, const std::int64_t* ids, int tokens, void* output, cudaStream_t);
}
