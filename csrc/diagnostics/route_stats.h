#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"

#include <string>
#include <unordered_map>

namespace tyloi::cuda {

class RouteStats {
public:
    explicit RouteStats(std::string path = {}, bool energy = false);
    RouteStats(const RouteStats&) = delete;
    RouteStats& operator=(const RouteStats&) = delete;

    bool enabled() const { return !path_.empty(); }
    bool energy() const { return enabled() && energy_; }
    void clear() { layers_.clear(); }
    void record(int layer, const ::tyloi::DenseTensor& ids,
                const ::tyloi::DenseTensor& weights,
                const ::tyloi::DenseTensor& output, int experts);
    void write() const;

private:
    struct Layer {
        ::tyloi::DenseTensor counts, weight_sum, weight_sq_sum;
        ::tyloi::DenseTensor output_energy, weighted_output_energy;
    };
    std::string path_;
    bool energy_;
    std::unordered_map<int, Layer> layers_;
};

} // namespace mfq::cuda
