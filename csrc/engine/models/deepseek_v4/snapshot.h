#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>

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
        visit(state_kv);
        visit(state_gate);
        visit(previous_kv);
        visit(previous_gate);
        visit(pool);
    }
};

struct BlockSnapshot {
    ::tyloi::Tensor local_cache;
    PoolSnapshot compressor;
    PoolSnapshot indexer_compressor;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(local_cache);
        compressor.tensors(visit);
        indexer_compressor.tensors(visit);
    }
};

} // namespace tyloi::models::deepseek_v4
