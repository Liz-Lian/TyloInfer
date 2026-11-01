#pragma once

#include "backend.h"

#include <cstdint>

namespace mfq::models::deepseek_v41 {

struct BlockSnapshot {
    mfq_tensor_backend::Tensor local_kv, compressed_kv, index_k, partial_kv, partial_score;
    int64_t compressed_length = 0, partial_length = 0;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(local_kv);
        visit(compressed_kv);
        visit(index_k);
        visit(partial_kv);
        visit(partial_score);
    }
};

} // namespace mfq::models::deepseek_v41
