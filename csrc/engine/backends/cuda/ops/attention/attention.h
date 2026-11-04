#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#include "engine/backends/cuda/native/profile.h"
#include "scratch.h"
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/ops.h"
#endif

#include <cstdint>
#include <utility>

namespace tyloi::cuda {

// The visible pair is the logical KV view returned by a cache append.  The
// physical pair and page metadata are kept separate because decode kernels
// consume the cache storage directly.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using AttentionKvView = tyloi::state::KvView;
using AttentionInput = tyloi::ops::AttentionInput;
using AttentionResult = tyloi::ops::AttentionResult;
#else
struct AttentionKvView {
    ::tyloi::Tensor visible_key;
    ::tyloi::Tensor visible_value;
    ::tyloi::Tensor physical_key;
    ::tyloi::Tensor physical_value;
    ::tyloi::Tensor key_chunk_ptrs;
    ::tyloi::Tensor value_chunk_ptrs;
    ::tyloi::Tensor page_table;
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
    ::tyloi::Tensor query;
    ::tyloi::Tensor key;
    ::tyloi::Tensor value;
    ::tyloi::Tensor positions;
    ::tyloi::Tensor write_positions;
    MfqOptional<::tyloi::Tensor> sequence_lengths = mfq_nullopt;
    MfqOptional<::tyloi::Tensor> attention_mask = mfq_nullopt;
    std::int64_t cache_position = 0;
    std::int64_t planned_kv_length = 0;
    std::int64_t decode_attention_parts = 0;
};
#endif

inline QuantizedKvPages quantized_pages(const AttentionKvView& kv) {
    using Tensor = ::tyloi::Tensor;
    return kv.paged() ? QuantizedKvPages{Tensor(kv.key_chunk_ptrs), Tensor(kv.value_chunk_ptrs),
                                        Tensor(kv.page_table), static_cast<int>(kv.page_size),
                                        static_cast<int>(kv.pages_per_chunk), static_cast<int>(kv.heads)}
                      : QuantizedKvPages{};
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

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
struct AttentionResult {
    ::tyloi::Tensor value;
    bool token_major = false;
};

AttentionResult dispatch_attention(const AttentionInput& request,
                                   const AttentionKvView& kv,
                                   const AttentionConfig& config,
                                   AttentionScratch& scratch,
                                   CudaProfiler& profiler);
#else
namespace ops {
tyloi::ops::AttentionResult attention(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
                                    const tyloi::ops::AttentionConfig&);
std::optional<tyloi::Tensor> norm_rope_kv(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
    const tyloi::ops::RopeTable&, const tyloi::Tensor&, const tyloi::Tensor&, double, double);
std::optional<tyloi::Tensor> rope_kv(const tyloi::ops::AttentionInput&, const tyloi::state::KvView&,
                                   const tyloi::ops::RopeTable&);
}
#endif

} // namespace tyloi::cuda
