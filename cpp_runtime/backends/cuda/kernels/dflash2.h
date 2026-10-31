#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mfq::cuda::dflash2 {
inline constexpr int kQ8TileRows=8;
// Lossless Q8_0 tiles from Strata's DFlash2 GEMM: 8 weight rows by 512
// columns. Padding permits arbitrary Q8_0 dimensions without model gates.
struct Q8TiledMatrix { const void* weight; float* output; int rows, padded_rows; };
void q8_tile(const void* blocks, void* tiles, int rows, int columns, cudaStream_t stream);
void q8_untile(const void* tiles, void* blocks, int rows, int columns, cudaStream_t stream);
// Activations are row-major; reserve ceil(tokens/8)*8 padded rows of codes
// (ceil(columns/512)*512 bytes each) and FP32 scales (one per 32 codes).
void q8_quantize(const float* input, void* codes, float* scales, int tokens, int columns, cudaStream_t stream);
void q8_gemm(const Q8TiledMatrix* matrices, int count, const void* codes, const float* scales,
    int tokens, int columns, cudaStream_t stream);
void swiglu_float(const float* gate,const float* up,float* output,std::int64_t elements,cudaStream_t stream);
void swiglu_float_q8(const float* gate,const float* up,float* output,void* codes,float* scales,
    int rows,int width,cudaStream_t stream);
void rms_norm_q8(const float* input,const float* weight,float* output,void* codes,float* scales,
    int rows,int width,float epsilon,cudaStream_t stream);
void rms_norm(const float* input,const float* weight,float* output,int rows,int width,float epsilon,cudaStream_t stream);
void swiglu(const void* gate,const void* up,void* output,std::int64_t elements,cudaStream_t stream);
void swiglu_joined(const void* gate_up,void* output,std::int64_t elements,std::int64_t width,cudaStream_t stream);
void convolution(const float* input, const float* dynamic, const float* base, const float* residual,
    float* output, int rows, int width, int group, bool finish, cudaStream_t stream);
void convolution_q8(const float* input,const float* dynamic,const float* base,const float* residual,
    float* output,void* codes,float* scales,int rows,int width,int group,bool finish,cudaStream_t stream);
void rotary_table(const std::int64_t* positions,float* table,int rows,int dimension,float theta,cudaStream_t stream);
void norm_rope_tabled(const float* input,const float* norm,const float* table,float* output,
    int rows,int heads,int dimension,float epsilon,cudaStream_t stream);
void norm_rope(const float* input, const float* norm, const std::int64_t* positions,
    float* output, int rows, int heads, int dimension, float theta, float epsilon, cudaStream_t stream);
// Normalize/rotate K and write K/V into token-major F32 or BF16 rings.
void store_context(const float* key,const float* value,const float* norm,
    const float* table,const std::int64_t* positions,void* ring_key,void* ring_value,
    int rows,int heads,int dimension,int window,float epsilon,cudaStream_t stream,bool history_bf16=false);
// Token-major committed KV rings; the proposal's own block remains outside
// the ring so rejected rows never need to be removed from draft context.
void attention(const float* query, const void* ring_key, const void* ring_value,
    const float* block_key, const float* block_value, float* partial, float* output,
    int rows, int heads, int kv_heads, int dimension, int window, std::int64_t position, cudaStream_t stream,
    const std::int64_t* device_position = nullptr,bool history_bf16=false);
// Entire greedy candidate chain in one CTA; matches dense greedy's smallest
// vocabulary ID on tied scores and needs only one result transfer per block.
void selector_greedy_walk(const std::int64_t* compact,int rows,int candidates,const std::int64_t* previous,
    const std::uint16_t* predecessor,const std::uint16_t* successor,const float* hidden,
    std::int64_t* output,int rank,int vocabulary,cudaStream_t stream,bool successor_is_compact=false,
    bool pack_probability=false);
void selector_logits(const std::int64_t* compact, int candidates, const std::int64_t* previous,
    const std::uint16_t* predecessor, const std::uint16_t* successor, const float* hidden,
    float* output, int rank, int vocabulary, cudaStream_t stream,bool successor_is_compact=false);
}
