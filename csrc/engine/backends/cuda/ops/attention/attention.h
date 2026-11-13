#pragma once

#include "backend.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"
#include "engine/backends/cuda/native/profile.h"
#include "scratch.h"
#include "engine/ops.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace tyloi::cuda {

using AttentionInput = tyloi::ops::AttentionInput;
using AttentionResult = tyloi::ops::AttentionResult;

inline ::tyloi::DenseTensor dense_kv_buffer(const ::tyloi::Tensor& tensor) {
    if (!tensor || !tensor.defined()) return {};
    const auto* dense = tensor.representation<::tyloi::DenseTensor>();
    if (!dense) throw std::invalid_argument("attention requires dense tensors");
    return *dense;
}

inline QuantizedKvPages quantized_pages(const tyloi::state::KvView& kv) {
    return kv.paged() ? QuantizedKvPages{dense_kv_buffer(kv.key_chunk_ptrs), dense_kv_buffer(kv.value_chunk_ptrs),
                                        dense_kv_buffer(kv.page_table), static_cast<int>(kv.page_size),
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

} // namespace tyloi::cuda
