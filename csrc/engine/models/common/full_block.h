#pragma once

#include "engine/models/common/block.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/attention.h"
#include "engine/state/kv_cache.h"

#include <cstdint>

struct FullBlock : Block, tyloi::cuda::FullAttention {
    ::tyloi::Tensor ffn_norm;
    tyloi::cuda::Ffn ffn;
    KVCache cache;

    bool supports_speculation() const noexcept override { return !sliding; }
    void reset(int64_t batch) override;
    void preallocate_kv();
    ::tyloi::Tensor forward(
        CudaExecutionContext& execution,
        ::tyloi::Tensor x,
        ::tyloi::Tensor positions,
        int64_t cache_position,
        const MfqOptional<::tyloi::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<::tyloi::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<::tyloi::Tensor>& attention_mask = mfq_nullopt) override;
    ::tyloi::Tensor forward_context(
        CudaExecutionContext& execution,
        ::tyloi::Tensor x,
        const Context& context,
        const RopeCache& rope) override;
    ::tyloi::Tensor forward_impl(
        CudaExecutionContext& execution,
        ::tyloi::Tensor x,
        ::tyloi::Tensor positions,
        int64_t cache_position,
        const MfqOptional<::tyloi::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<::tyloi::Tensor>& cache_positions,
        const MfqOptional<::tyloi::Tensor>& attention_mask,
        int64_t planned_kv_length,
        int64_t decode_attention_parts);
    virtual ::tyloi::Tensor forward_ffn(
        CudaExecutionContext& execution,
        ::tyloi::Tensor residual,
        ::tyloi::Tensor attention_output,
        int64_t batch,
        int64_t tokens,
        int64_t hidden);
};
