#pragma once

#include "models/common/block.h"
#include "engine/models/ffn.h"
#include "ops/attention/full.h"
#include "backends/cuda/engine/state/kv_cache.h"

#include <cstdint>

struct FullBlock : Block, mfq::cuda::FullAttention {
    mfq_tensor_backend::Tensor ffn_norm;
    mfq::cuda::Ffn ffn;
    KVCache cache;

    bool supports_speculation() const noexcept override { return !sliding; }
    void reset(int64_t batch) override;
    void preallocate_kv();
    mfq_tensor_backend::Tensor forward(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor positions,
        int64_t cache_position,
        const MfqOptional<mfq_tensor_backend::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask = mfq_nullopt) override;
    mfq_tensor_backend::Tensor forward_context(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        const Context& context,
        const RopeCache& rope) override;
    mfq_tensor_backend::Tensor forward_impl(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor x,
        mfq_tensor_backend::Tensor positions,
        int64_t cache_position,
        const MfqOptional<mfq_tensor_backend::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<mfq_tensor_backend::Tensor>& cache_positions,
        const MfqOptional<mfq_tensor_backend::Tensor>& attention_mask,
        int64_t planned_kv_length,
        int64_t decode_attention_parts);
    virtual mfq_tensor_backend::Tensor forward_ffn(
        CudaExecutionContext& execution,
        mfq_tensor_backend::Tensor residual,
        mfq_tensor_backend::Tensor attention_output,
        int64_t batch,
        int64_t tokens,
        int64_t hidden);
};
