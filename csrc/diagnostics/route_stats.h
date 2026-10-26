#pragma once

#include "native/cuda/tensor/backend.h"

#include <string>
#include <unordered_map>

namespace mfq::cuda {

class RouteStats {
public:
    explicit RouteStats(std::string path = {}, bool energy = false);
    RouteStats(const RouteStats&) = delete;
    RouteStats& operator=(const RouteStats&) = delete;

    bool enabled() const { return !path_.empty(); }
    bool energy() const { return enabled() && energy_; }
    void clear() { layers_.clear(); }
    void record(int layer, const mfq_tensor_backend::Tensor& ids,
                const mfq_tensor_backend::Tensor& weights,
                const mfq_tensor_backend::Tensor& output, int experts);
    void write() const;

private:
    struct Layer {
        mfq_tensor_backend::Tensor counts, weight_sum, weight_sq_sum;
        mfq_tensor_backend::Tensor output_energy, weighted_output_energy;
    };
    std::string path_;
    bool energy_;
    std::unordered_map<int, Layer> layers_;
};

} // namespace mfq::cuda
