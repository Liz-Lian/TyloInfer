#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

struct MfePoolWeight;

struct MoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;

    bool operator==(const MoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
               gs == other.gs && device == other.device;
    }
};

struct MoeActivationKeyHash {
    size_t operator()(const MoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        return value * 1315423911u + (size_t)key.device;
    }
};

struct MoeActivationWorkspace {
    static constexpr int stable_row_limit = 2048;
    static int capacity_rows(int rows, bool stable) {
        if (!stable) return ((rows + 255) / 256) * 256;
        int capacity = 1;
        while (capacity < rows) capacity *= 2;
        return capacity;
    }
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
};

struct MixedMoeTransformKey {
    int block = 0;
    uint64_t seed = 0;

    bool operator==(const MixedMoeTransformKey & other) const {
        return block == other.block && seed == other.seed;
    }
};

struct MixedMoeTransformKeyHash {
    size_t operator()(const MixedMoeTransformKey & key) const {
        return ((size_t)key.block * 1315423911u) ^
            (size_t)(key.seed ^ (key.seed >> 32));
    }
};

struct MixedMoeActivationKey {
    int input_rows = 0;
    int groups = 0;
    int gs = 0;
    int device = 0;
    MixedMoeTransformKey transform;

    bool operator==(const MixedMoeActivationKey & other) const {
        return input_rows == other.input_rows && groups == other.groups &&
            gs == other.gs && device == other.device &&
            transform == other.transform;
    }
};

struct MixedMoeActivationKeyHash {
    size_t operator()(const MixedMoeActivationKey & key) const {
        size_t value = (size_t)key.input_rows;
        value = value * 1315423911u + (size_t)key.groups;
        value = value * 1315423911u + (size_t)key.gs;
        value = value * 1315423911u + (size_t)key.device;
        return value * 1315423911u +
            MixedMoeTransformKeyHash{}(key.transform);
    }
};

namespace mfq::cuda {

struct MfeScratch {
    using Buckets = std::unordered_map<MoeActivationKey, MoeActivationWorkspace, MoeActivationKeyHash>;
    using Pools = std::vector<MfePoolWeight>;
    std::map<std::weak_ptr<const Pools>, std::map<std::pair<int, std::uintptr_t>, Buckets>,
             std::owner_less<std::weak_ptr<const Pools>>> values;
    mutable std::mutex mutex;

    MoeActivationWorkspace get(const std::shared_ptr<const Pools>& pools,
                               mfq_tensor_backend::Tensor input, int rows, int groups,
                               int gs, bool stable = true);
};

} // namespace mfq::cuda
