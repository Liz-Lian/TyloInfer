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

namespace mfq::cuda {

// The visible pair is the logical KV view returned by a cache append.  The
// physical pair and page metadata are kept separate because decode kernels
// consume the cache storage directly.
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using AttentionKvView = mfq::state::KvView;
using AttentionInput = mfq::ops::AttentionInput;
using AttentionResult = mfq::ops::AttentionResult;
#else
struct AttentionKvView {
    mfq_tensor_backend::Tensor visible_key;
    mfq_tensor_backend::Tensor visible_value;
    mfq_tensor_backend::Tensor physical_key;
    mfq_tensor_backend::Tensor physical_value;
    mfq_tensor_backend::Tensor key_chunk_ptrs;
    mfq_tensor_backend::Tensor value_chunk_ptrs;
    mfq_tensor_backend::Tensor page_table;
    KvQuantization quantization;
    std::int64_t page_size = 0;
    std::int64_t pages_per_chunk = 0;
    std::int64_t heads = 0;
    bool ring = false;
    std::int64_t batch = 0;
    std::int64_t head_dim = 0;
    mfq_tensor_backend::ScalarType dtype = mfq_tensor_backend::kFloat16;

    bool paged() const noexcept { return page_size > 0; }

};

struct AttentionInput {
    mfq_tensor_backend::Tensor query;
    mfq_tensor_backend::Tensor key;
    mfq_tensor_backend::Tensor value;
    mfq_tensor_backend::Tensor positions;
    mfq_tensor_backend::Tensor write_positions;
    MfqOptional<mfq_tensor_backend::Tensor> sequence_lengths = mfq_nullopt;
    MfqOptional<mfq_tensor_backend::Tensor> attention_mask = mfq_nullopt;
    std::int64_t cache_position = 0;
    std::int64_t planned_kv_length = 0;
    std::int64_t decode_attention_parts = 0;
};
#endif

inline QuantizedKvPages quantized_pages(const AttentionKvView& kv) {
    using Tensor = mfq_tensor_backend::Tensor;
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
    mfq_tensor_backend::Tensor value;
    bool token_major = false;
};

AttentionResult dispatch_attention(const AttentionInput& request,
                                   const AttentionKvView& kv,
                                   const AttentionConfig& config,
                                   AttentionScratch& scratch,
                                   CudaProfiler& profiler);
#else
namespace ops {
mfq::ops::AttentionResult attention(const mfq::ops::AttentionInput&, const mfq::state::KvView&,
                                    const mfq::ops::AttentionConfig&);
std::optional<mfq::Tensor> norm_rope_kv(const mfq::ops::AttentionInput&, const mfq::state::KvView&,
    const mfq::ops::RopeTable&, const mfq::Tensor&, const mfq::Tensor&, double, double);
std::optional<mfq::Tensor> rope_kv(const mfq::ops::AttentionInput&, const mfq::state::KvView&,
                                   const mfq::ops::RopeTable&);
}
#endif

} // namespace mfq::cuda
