#pragma once

#include "backend.h"

#include <cstdint>
#include <vector>

namespace tyloi::models::qwen4_exp {

struct BlockSnapshot {
    ::tyloi::DenseTensor keys, values, index;
    ::tyloi::DenseTensor convolution, recurrent, ple_convolution;
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
    ::tyloi::DenseTensor positions;

    template <class Visit>
    void tensors(const Visit& visit) const {
        visit(positions);
        for (const auto& block : blocks) block.tensors(visit);
    }
};

struct MtpSnapshot {
    std::vector<BlockSnapshot> blocks;
    std::vector<::tyloi::DenseTensor> positions;
    std::vector<int64_t> lengths;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& block : blocks) block.tensors(visit);
        for (const auto& position : positions) visit(position);
    }
};

} // namespace tyloi::models::qwen4_exp
