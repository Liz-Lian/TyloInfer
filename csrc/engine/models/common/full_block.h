#pragma once

#include "engine/models/common/block.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/attention.h"
#include "engine/state/kv_cache.h"

#include <cstdint>

struct FullBlock : Block, tyloi::cuda::FullAttention {
    ::tyloi::DenseTensor ffn_norm;
    tyloi::cuda::Ffn ffn;
    KVCache cache;

    bool supports_speculation() const noexcept override { return !sliding; }
    void reset(int64_t batch) override;
    void preallocate_kv();
    ::tyloi::DenseTensor forward(
        CudaExecutionContext& execution,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor positions,
        int64_t cache_position,
        const MfqOptional<::tyloi::DenseTensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<::tyloi::DenseTensor>& cache_positions = mfq_nullopt,
        const MfqOptional<::tyloi::DenseTensor>& attention_mask = mfq_nullopt) override;
    ::tyloi::DenseTensor forward_context(
        CudaExecutionContext& execution,
        ::tyloi::DenseTensor x,
        const Context& context,
        const RopeCache& rope) override;
    ::tyloi::DenseTensor forward_impl(
        CudaExecutionContext& execution,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor positions,
        int64_t cache_position,
        const MfqOptional<::tyloi::DenseTensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<::tyloi::DenseTensor>& cache_positions,
        const MfqOptional<::tyloi::DenseTensor>& attention_mask,
        int64_t planned_kv_length,
        int64_t decode_attention_parts);
    virtual ::tyloi::DenseTensor forward_ffn(
        CudaExecutionContext& execution,
        ::tyloi::DenseTensor residual,
        ::tyloi::DenseTensor attention_output,
        int64_t batch,
        int64_t tokens,
        int64_t hidden);
};
