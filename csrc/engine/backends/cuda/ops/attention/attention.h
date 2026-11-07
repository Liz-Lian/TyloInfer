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

// The visible pair is the logical KV view returned by a cache append.  The
// physical pair and page metadata are kept separate because decode kernels
// consume the cache storage directly.
#ifdef TYLOINFER_NATIVE_CUDA
using AttentionKvView = tyloi::state::KvView;
using AttentionInput = tyloi::ops::AttentionInput;
using AttentionResult = tyloi::ops::AttentionResult;
#else
struct AttentionKvView {
    ::tyloi::DenseTensor visible_key;
    ::tyloi::DenseTensor visible_value;
    ::tyloi::DenseTensor physical_key;
    ::tyloi::DenseTensor physical_value;
    ::tyloi::DenseTensor key_chunk_ptrs;
    ::tyloi::DenseTensor value_chunk_ptrs;
    ::tyloi::DenseTensor page_table;
    KvQuantization quantization;
    std::int64_t page_size = 0;
    std::int64_t pages_per_chunk = 0;
    std::int64_t heads = 0;
    bool ring = false;
    std::int64_t batch = 0;
    std::int64_t head_dim = 0;
    ::tyloi::ScalarType dtype = ::tyloi::kFloat16;

    bool paged() const noexcept { return page_size > 0; }

};

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

inline QuantizedKvPages quantized_pages(const AttentionKvView& kv) {
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
                                   const AttentionKvView& kv,
                                   const AttentionConfig& config,
                                   AttentionScratch& scratch,
                                   CudaProfiler& profiler);
#else
namespace ops {
std::int64_t attention_parts(std::int64_t planned_len);
tyloi::ops::AttentionResult attention(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
                                    const tyloi::ops::AttentionConfig&);
std::optional<tyloi::DenseTensor> norm_rope_kv(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
    const tyloi::ops::RopeTable&, const tyloi::DenseTensor&, const tyloi::DenseTensor&, double, double);
std::optional<tyloi::DenseTensor> rope_kv(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
                                   const tyloi::ops::RopeTable&);
}
#endif

} // namespace tyloi::cuda
