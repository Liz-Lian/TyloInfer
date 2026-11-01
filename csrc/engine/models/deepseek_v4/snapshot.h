#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>

struct Dsv4PoolState;

namespace mfq::models::deepseek_v4 {

struct PoolSnapshot {
    int64_t ratio = 0;
    int64_t head_dim = 0;
    int64_t cache_quant_mode = 0;
    int64_t capacity = 0;
    bool overlap = false;
    mfq_tensor_backend::Tensor state_kv;
    mfq_tensor_backend::Tensor state_gate;
    mfq_tensor_backend::Tensor previous_kv;
    mfq_tensor_backend::Tensor previous_gate;
    mfq_tensor_backend::Tensor pool;

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
    mfq_tensor_backend::Tensor local_cache;
    PoolSnapshot compressor;
    PoolSnapshot indexer_compressor;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(local_cache);
        compressor.tensors(visit);
        indexer_compressor.tensors(visit);
    }
};

PoolSnapshot capture_pool(const Dsv4PoolState&, int64_t, size_t&);
void restore_pool(Dsv4PoolState&, const PoolSnapshot&);

} // namespace mfq::models::deepseek_v4
