#pragma once

#include "engine/backends/cuda/ops/rope/rope.h"
#include "engine/backends/cuda/ops/norm/rms.h"

#include <cstdint>
#include <vector>

struct CudaExecutionContext;

struct Block {
    struct Context {
        ::tyloi::DenseTensor token_ids;
        ::tyloi::DenseTensor host_token_ids;
        ::tyloi::DenseTensor positions;
        ::tyloi::DenseTensor full_positions;
        int64_t cache_position = 0;
        int64_t confirmed_prefix = 0;
        int64_t planned_kv_length = 0;
        int64_t decode_attention_parts = 0;
        MfqOptional<::tyloi::DenseTensor> sequence_lengths = mfq_nullopt;
        MfqOptional<::tyloi::DenseTensor> cache_positions = mfq_nullopt;
        MfqOptional<::tyloi::DenseTensor> attention_mask = mfq_nullopt;
    };

    int cuda_device = 0;
    bool cpu_offloaded = false;

    virtual ~Block() = default;
    virtual void reset(int64_t batch) = 0;
    virtual void set_token_ids(const ::tyloi::DenseTensor&) {}
    virtual void prefetch_token_ids(const ::tyloi::DenseTensor&) {}
    virtual bool supports_speculation() const noexcept { return false; }
    virtual std::vector<::tyloi::DenseTensor*> graph_warmup_state() { return {}; }
    virtual void begin_speculative(int64_t) {}
    virtual void commit_speculative() {}
    virtual void rollback_speculative(int64_t) {}
    virtual ::tyloi::DenseTensor forward(
        CudaExecutionContext& execution,
        ::tyloi::DenseTensor x,
        ::tyloi::DenseTensor positions,
        int64_t cache_position,
        const MfqOptional<::tyloi::DenseTensor>& sequence_lengths,
        const RopeCache& rope,
        const MfqOptional<::tyloi::DenseTensor>& cache_positions = mfq_nullopt,
        const MfqOptional<::tyloi::DenseTensor>& attention_mask = mfq_nullopt) = 0;
    virtual ::tyloi::DenseTensor forward_context(
        CudaExecutionContext& execution,
        ::tyloi::DenseTensor x,
        const Context& context,
        const RopeCache& rope);
};

using ::tyloi::indexing::Slice;
