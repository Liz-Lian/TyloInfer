#pragma once

#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/norm/rms.h"

#include <cstdint>
#include <vector>

struct CudaExecutionContext;

struct Block {
    struct Context {
        ::tyloi::Tensor token_ids;
        ::tyloi::Tensor host_token_ids;
        ::tyloi::Tensor positions;
        ::tyloi::Tensor full_positions;
        int64_t cache_position = 0;
        int64_t confirmed_prefix = 0;
        int64_t planned_kv_length = 0;
        int64_t decode_attention_parts = 0;
        MfqOptional<::tyloi::Tensor> sequence_lengths = mfq_nullopt;
        MfqOptional<::tyloi::Tensor> cache_positions = mfq_nullopt;
        MfqOptional<::tyloi::Tensor> attention_mask = mfq_nullopt;
    };

    int cuda_device = 0;
    bool cpu_offloaded = false;

    virtual ~Block() = default;
    virtual void reset(int64_t batch) = 0;
    virtual void set_token_ids(const ::tyloi::Tensor&) {}
    virtual void prefetch_token_ids(const ::tyloi::Tensor&) {}
    virtual bool supports_speculation() const noexcept { return false; }
    virtual std::vector<::tyloi::Tensor*> graph_warmup_state() { return {}; }
    virtual void begin_speculative(int64_t) {}
    virtual void commit_speculative() {}
    virtual void rollback_speculative(int64_t) {}
    virtual ::tyloi::Tensor forward(
        CudaExecutionContext& execution,
        ::tyloi::Tensor x,
        ::tyloi::Tensor positions,
        int64_t cache_position,
        const MfqOptional<::tyloi::Tensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<::tyloi::Tensor>& cache_positions = mfq_nullopt,
        const MfqOptional<::tyloi::Tensor>& attention_mask = mfq_nullopt) = 0;
    virtual ::tyloi::Tensor forward_context(
        CudaExecutionContext& execution,
        ::tyloi::Tensor x,
        const Context& context,
        const RopeCache& rope);
};

using ::tyloi::indexing::Slice;
