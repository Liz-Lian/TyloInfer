#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#include "engine/backends/cuda/native/profile.h"
#include "scratch.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
#endif

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA
using AttentionInput = tyloi::ops::AttentionInput;
using AttentionResult = tyloi::ops::AttentionResult;
#else
struct AttentionInput {
    ::tyloi::DenseTensor query;
    ::tyloi::DenseTensor key;
    ::tyloi::DenseTensor value;
    ::tyloi::DenseTensor positions;
    ::tyloi::DenseTensor write_positions;
    MfqOptional<::tyloi::DenseTensor> sequence_lengths = mfq_nullopt;
    MfqOptional<::tyloi::DenseTensor> attention_mask = mfq_nullopt;
    std::int64_t cache_position = 0;
    std::int64_t planned_kv_length = 0;
    std::int64_t decode_attention_parts = 0;
};
#endif

#ifdef TYLOINFER_NATIVE_CUDA
inline ::tyloi::DenseTensor dense_kv_buffer(const std::shared_ptr<::tyloi::Tensor>& tensor) {
    if (!tensor || !tensor->defined()) return {};
    const auto* dense = dynamic_cast<const ::tyloi::DenseTensor*>(tensor.get());
    if (!dense) throw std::invalid_argument("attention requires dense tensors");
    return *dense;
}
#endif

inline QuantizedKvPages quantized_pages(const tyloi::state::KvView& kv) {
#ifdef TYLOINFER_NATIVE_CUDA
    return kv.paged() ? QuantizedKvPages{dense_kv_buffer(kv.key_chunk_ptrs), dense_kv_buffer(kv.value_chunk_ptrs),
                                        dense_kv_buffer(kv.page_table), static_cast<int>(kv.page_size),
                                        static_cast<int>(kv.pages_per_chunk), static_cast<int>(kv.heads)}
                      : QuantizedKvPages{};
#else
    using DenseTensor = ::tyloi::DenseTensor;
    return kv.paged() ? QuantizedKvPages{DenseTensor(kv.key_chunk_ptrs), DenseTensor(kv.value_chunk_ptrs),
                                        DenseTensor(kv.page_table), static_cast<int>(kv.page_size),
                                        static_cast<int>(kv.pages_per_chunk), static_cast<int>(kv.heads)}
                      : QuantizedKvPages{};
#endif
}

struct AttentionKernels {
    bool bf16_flash128 = false;
    bool flash128_casts = false;
    bool bf16_gqa_decode = false;
    bool decode_aten = false;
    bool mma_prefill = false;
    bool mma_decode = false;
    bool split_k = false;
};

struct AttentionOptions {
    AttentionKernels kernels;
    bool fused_norm = true;
    bool fused_rope = true;
    bool fused_qk_rope_kv = true;
    bool fused_rope_kv = true;
    bool write_aten = false;
};

struct AttentionConfig {
    bool official_bf16 = false;
    bool sliding = false;
    std::int64_t window = 0;
    double scale = 0.0;
    AttentionKernels kernels;
};

#ifndef TYLOINFER_NATIVE_CUDA
struct AttentionResult {
    ::tyloi::DenseTensor value;
    bool token_major = false;
};

AttentionResult dispatch_attention(const AttentionInput& request,
                                   const tyloi::state::KvView& kv,
                                   const AttentionConfig& config,
                                   AttentionScratch& scratch,
                                   CudaProfiler& profiler);
#else
namespace ops {
std::int64_t attention_parts(std::int64_t planned_len);
tyloi::ops::AttentionResult attention(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
                                    const tyloi::ops::AttentionConfig&);
::tyloi::DenseTensor sparse_attention(
    ::tyloi::DenseTensor query, ::tyloi::DenseTensor cache,
    ::tyloi::DenseTensor indices, ::tyloi::DenseTensor mask,
    ::tyloi::DenseTensor sinks, double scale);
std::optional<tyloi::DenseTensor> norm_rope_kv(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
    const tyloi::ops::RopeTable&, const tyloi::DenseTensor&, const tyloi::DenseTensor&, double, double);
std::optional<tyloi::DenseTensor> rope_kv(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
                                   const tyloi::ops::RopeTable&);
}
#endif

} // namespace tyloi::cuda
