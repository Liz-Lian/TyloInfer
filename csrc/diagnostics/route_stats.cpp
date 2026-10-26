#include "route_stats.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda {

RouteStats::RouteStats(std::string path, bool energy)
    : path_(std::move(path)), energy_(energy) {}

void RouteStats::record(
        int layer,
        const mfq_tensor_backend::Tensor & ids,
        const mfq_tensor_backend::Tensor & weights,
        const mfq_tensor_backend::Tensor & output,
        int n_experts) {
    if (layer < 0 || !enabled()) return;
    auto found = layers_.find(layer);
    if (found == layers_.end()) {
        auto options = mfq_tensor_backend::TensorOptions()
            .device(ids.device()).dtype(mfq_tensor_backend::kFloat64);
        Layer value{
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
            mfq_tensor_backend::zeros({n_experts}, options),
        };
        found = layers_.emplace(layer, std::move(value)).first;
    }
    auto flat_ids = ids.reshape({-1}).to(mfq_tensor_backend::kInt64);
    auto flat_weights = weights.reshape({-1}).to(mfq_tensor_backend::kFloat64);
    auto ones = mfq_tensor_backend::ones_like(flat_weights);
    found->second.counts.scatter_add_(0, flat_ids, ones);
    found->second.weight_sum.scatter_add_(0, flat_ids, flat_weights);
    found->second.weight_sq_sum.scatter_add_(
        0, flat_ids, flat_weights.square());
    if (energy_) {
        auto energy = output.reshape({flat_ids.numel(), output.size(-1)})
            .to(mfq_tensor_backend::kFloat32).square().sum(1).to(mfq_tensor_backend::kFloat64);
        found->second.output_energy.scatter_add_(0, flat_ids, energy);
        found->second.weighted_output_energy.scatter_add_(
            0, flat_ids, energy * flat_weights.square());
    }
}

void RouteStats::write() const {
    if (!enabled() || layers_.empty()) return;
    mfq_cuda_synchronize();
    std::filesystem::path path(path_);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "cannot write MoE route statistics: " + path.string());
    }
    out << "layer,expert,count,weight_sum,weight_sq_sum,"
           "output_energy,weighted_output_energy\n";
    std::vector<int> layers;
    layers.reserve(layers_.size());
    for (const auto & item : layers_) layers.push_back(item.first);
    std::sort(layers.begin(), layers.end());
    out << std::setprecision(17);
    for (int layer : layers) {
        const auto & value = layers_.at(layer);
        auto counts = value.counts.to(mfq_tensor_backend::kCPU).contiguous();
        auto weight_sum = value.weight_sum.to(mfq_tensor_backend::kCPU).contiguous();
        auto weight_sq_sum = value.weight_sq_sum.to(mfq_tensor_backend::kCPU).contiguous();
        auto output_energy = value.output_energy.to(mfq_tensor_backend::kCPU).contiguous();
        auto weighted_output_energy =
            value.weighted_output_energy.to(mfq_tensor_backend::kCPU).contiguous();
        const auto n_experts = counts.numel();
        const double * count_data = counts.data_ptr<double>();
        const double * weight_data = weight_sum.data_ptr<double>();
        const double * weight_sq_data = weight_sq_sum.data_ptr<double>();
        const double * energy_data = output_energy.data_ptr<double>();
        const double * weighted_energy_data =
            weighted_output_energy.data_ptr<double>();
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            out << layer << ',' << expert << ','
                << count_data[expert] << ','
                << weight_data[expert] << ','
                << weight_sq_data[expert] << ','
                << energy_data[expert] << ','
                << weighted_energy_data[expert] << '\n';
        }
    }
    std::cout << "moe_route_stats_path=" << path.string()
              << " layers=" << layers.size()
              << " output_energy="
              << (energy_ ? 1 : 0) << "\n";
}

} // namespace mfq::cuda
