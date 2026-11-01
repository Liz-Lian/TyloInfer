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

struct Snapshot {
    std::vector<BlockSnapshot> blocks;
    mfq_tensor_backend::Tensor positions;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(positions);
        for (const auto& block : blocks) block.tensors(visit);
    }
};

struct MtpSnapshot {
    std::vector<BlockSnapshot> blocks;
    std::vector<mfq_tensor_backend::Tensor> positions;
    std::vector<int64_t> lengths;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& block : blocks) block.tensors(visit);
        for (const auto& position : positions) visit(position);
    }
};

} // namespace mfq::models::qwen4_exp
