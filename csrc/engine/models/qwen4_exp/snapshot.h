#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

namespace mfq::models::qwen4_exp {

struct BlockSnapshot {
    mfq_tensor_backend::Tensor keys, values, index;
    mfq_tensor_backend::Tensor convolution, recurrent, ple_convolution;
    int64_t ple_batch = 0;
    std::vector<int64_t> ple_tokens;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(keys);
        visit(values);
        visit(index);
        visit(convolution);
        visit(recurrent);
        visit(ple_convolution);
    }
};

} // namespace mfq::models::qwen4_exp
