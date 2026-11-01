#pragma once

#include "backend.h"

#include <cstdint>

namespace mfq::models::glm_dsa {

struct BlockSnapshot {
    mfq_tensor_backend::Tensor kv_cache;
    mfq_tensor_backend::Tensor index_cache;
    int64_t kv_capacity = 0;
    int64_t index_capacity = 0;
    bool full_indexer = false;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(kv_cache);
        visit(index_cache);
    }
};

} // namespace mfq::models::glm_dsa
