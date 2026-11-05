#pragma once

#include "engine/models/common/attention.h"
#include "engine/weights/loader.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "engine/models/common/weight_loading.h"
#include "engine/ops.h"

#include <memory>
#include <string>
#include <string_view>

namespace tyloi { class ModelSource; }

struct TransformerWeightLoader {
    const tyloi::cuda::weight_loader::Loader& weights;
    bool preserve_projection_boundaries = false;
    bool cpu_offloaded = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
    bool keep_quantized_router = false;

    bool has(const std::string& name) const;
    ::tyloi::DenseTensor dense(const std::string& name) const;
    std::shared_ptr<const ::tyloi::Tensor> router_parameter(const std::string& name) const;
    std::shared_ptr<const ::tyloi::Tensor> linear(const std::string& name) const;
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> projections(const std::vector<std::string>& names) const;
    std::vector<std::shared_ptr<const ::tyloi::Tensor>> gate_up(const std::vector<std::string>& names, const std::shared_ptr<const ::tyloi::Tensor>& down) const;
    void qkv(tyloi::cuda::FullAttention& attention, const std::vector<std::string>& names) const;
    void output(tyloi::cuda::FullAttention& attention, const std::string& name) const;
    void important_neurons(int64_t hidden, int64_t intermediate, tyloi::cuda::Ffn& ffn,
        const std::string& down, const std::string& gate, const std::string& up) const;
    void experts(tyloi::cuda::Ffn& ffn, tyloi::models::ExpertProjection projection,
                 const std::string& name, int layer) const;
    std::shared_ptr<const ::tyloi::Tensor> headwise(const std::string& name) const;
    static auto shape(const ::tyloi::DenseTensor& value) { return value.sizes().vec(); }
    static int64_t elements(const ::tyloi::DenseTensor& value) { return value.numel(); }
    static auto shape(const std::shared_ptr<const ::tyloi::Tensor>& value) {
        std::vector<int64_t> result;
        if (value) for (int64_t i = 0; i < value->dim(); ++i) result.push_back(value->size(i));
        return result;
    }
    static int64_t elements(const std::shared_ptr<const ::tyloi::Tensor>& value) { return value ? value->numel() : 0; }
};
struct MoeWeightLoadOptions {
    int layer;
    bool cpu_offloaded = false;
    bool router_bias_required = false;
    bool ungated_experts = false;
    bool ungated_shared = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
};

tyloi::cuda::Ffn load_moe_weights(
    const tyloi::cuda::weight_loader::Loader& weights,
    std::string_view prefix,
    const MoeWeightLoadOptions& options);
void load_important_neuron_branch(
    const tyloi::cuda::weight_loader::Loader& weights,
    int64_t hidden_size,
    int64_t intermediate_size,
    tyloi::cuda::Ffn& ffn,
    const std::string& down_name,
    const std::string& gate_name,
    const std::string& up_name);

void validate_native_placement(const ParallelConfig&, const LayerPlacementConfig&, int gpu_layers);

namespace tyloi::cuda {

std::string configure_yarn(tyloi::ops::YarnScaling& yarn, bool extend, std::string definition,
                           std::string_view backbone, int64_t context);

void validate_model_source(const tyloi::ModelSource& source);

} // namespace tyloi::cuda
