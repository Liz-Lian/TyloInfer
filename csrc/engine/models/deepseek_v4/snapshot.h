#pragma once

#include "engine/tensor.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace tyloi::models::deepseek_v4 {

struct PoolSnapshot {
    int64_t ratio = 0;
    int64_t head_dim = 0;
    int64_t cache_quant_mode = 0;
    int64_t capacity = 0;
    bool overlap = false;
    ::tyloi::Tensor state_kv;
    ::tyloi::Tensor state_gate;
    ::tyloi::Tensor previous_kv;
    ::tyloi::Tensor previous_gate;
    ::tyloi::Tensor pool;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (state_kv) visit(state_kv);
        if (state_gate) visit(state_gate);
        if (previous_kv) visit(previous_kv);
        if (previous_gate) visit(previous_gate);
        if (pool) visit(pool);
    }
};

struct BlockSnapshot {
    ::tyloi::Tensor local_cache;
    PoolSnapshot compressor;
    PoolSnapshot indexer_compressor;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (local_cache) visit(local_cache);
        compressor.tensors(visit);
        indexer_compressor.tensors(visit);
    }
};

} // namespace tyloi::models::deepseek_v4
