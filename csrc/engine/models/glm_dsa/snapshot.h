#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <memory>

namespace tyloi::models::glm_dsa {

struct BlockSnapshot {
    std::shared_ptr<const ::tyloi::Tensor> kv_cache;
    std::shared_ptr<const ::tyloi::Tensor> index_cache;
    int64_t kv_capacity = 0;
    int64_t index_capacity = 0;
    bool full_indexer = false;

    template <class Visit>
    void tensors(const Visit& visit) const {
        if (kv_cache) visit(*kv_cache);
        if (index_cache) visit(*index_cache);
    }
};

} // namespace tyloi::models::glm_dsa
