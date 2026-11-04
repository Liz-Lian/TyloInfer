#pragma once

#include "backend.h"

#include <cstdint>

namespace tyloi::models::glm_dsa {

struct BlockSnapshot {
    ::tyloi::Tensor kv_cache;
    ::tyloi::Tensor index_cache;
    int64_t kv_capacity = 0;
    int64_t index_capacity = 0;
    bool full_indexer = false;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(kv_cache);
        visit(index_cache);
    }
};

} // namespace tyloi::models::glm_dsa
