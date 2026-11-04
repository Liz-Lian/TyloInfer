#pragma once

#include "backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tyloi::models::deepseek_v41 {

struct EngramHashSnapshot;

struct BlockSnapshot {
    ::tyloi::DenseTensor local_kv, compressed_kv, index_k, partial_kv, partial_score;
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

struct Snapshot {
    std::vector<BlockSnapshot> blocks;
    std::shared_ptr<const EngramHashSnapshot> engram;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& block : blocks) block.tensors(visit);
    }
};

struct MtpSnapshot {
    std::vector<::tyloi::DenseTensor> rings;

    template <class Visit>
    void tensors(const Visit& visit) const {
        for (const auto& ring : rings) visit(ring);
    }
};

} // namespace tyloi::models::deepseek_v41
