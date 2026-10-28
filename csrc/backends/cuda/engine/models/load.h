#pragma once

#include "ops/attention/full.h"
#include "backends/cuda/engine/weights/load.h"
#include "engine/models/ffn.h"
#include "engine/models/common/weight_loading.h"

#include <memory>
#include <string>
#include <string_view>

namespace mfq { class ModelSource; }
struct CudaExecutionContext;

struct TransformerWeightLoader {
    const mfq::cuda::weight_loader::Loader& weights;
    bool preserve_projection_boundaries = false;
    bool cpu_offloaded = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
    mutable std::shared_ptr<const QuantLinear> router_projection;
    bool keep_quantized_router = false;

    bool has(const std::string& name) const;
    mfq_tensor_backend::Tensor dense(const std::string& name) const;
    mfq_tensor_backend::Tensor router_parameter(const std::string& name) const;
    QuantLinear linear(const std::string& name) const;
    QuantLinearGroup projections(const std::vector<std::string>& names) const;
    QuantLinearGroup gate_up(const std::vector<std::string>& names, const QuantLinear& down) const;
    void qkv(mfq::cuda::FullAttention& attention, const std::vector<std::string>& names) const;
    void workspace(mfq::cuda::Ffn& ffn) const;
    void important_neurons(int64_t hidden, int64_t intermediate, mfq::cuda::Ffn& ffn,
        const std::string& down, const std::string& gate, const std::string& up) const;
    void experts(mfq::cuda::Ffn& ffn, mfq::models::ExpertProjection projection,
                 const std::string& name, int layer) const;
    mfq::cuda::Experts headwise(const std::string& name) const;
    static auto shape(const mfq_tensor_backend::Tensor& value) { return value.sizes().vec(); }
    static int64_t elements(const mfq_tensor_backend::Tensor& value) { return value.numel(); }
};
struct MoeWeightLoadOptions {
    int layer;
    bool cpu_offloaded = false;
    bool router_bias_required = false;
    bool ungated_experts = false;
    bool ungated_shared = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
};

mfq::cuda::Ffn load_moe_weights(
    const mfq::cuda::weight_loader::Loader& weights,
    std::string_view prefix,
    const MoeWeightLoadOptions& options);
void load_important_neuron_branch(
    const mfq::cuda::weight_loader::Loader& weights,
    int64_t hidden_size,
    int64_t intermediate_size,
    mfq::cuda::Ffn& ffn,
    const std::string& down_name,
    const std::string& gate_name,
    const std::string& up_name);

void validate_native_placement(const ParallelConfig&, const LayerPlacementConfig&, int gpu_layers);
void prepare_moe_ffn(CudaExecutionContext& execution, mfq::cuda::Ffn& ffn);

namespace mfq::cuda {

std::string configure_yarn(CudaExecutionContext& execution, std::string definition,
                           std::string_view backbone, int64_t context);

std::string load_model_config_json(
    const mfq::ModelSource& source,
    const std::string& external_path = {});

void validate_model_source(const mfq::ModelSource& source);

} // namespace mfq::cuda

namespace mfq::cuda {
template <typename Model>
Model load_causal_lm(CudaExecutionContext& execution, const std::string& model_path,
                    const std::string& config_path, std::int64_t context_size_override = 0,
                    bool load_blocks = true, bool defer_moe_cache_finalize = false,
                    std::shared_ptr<const mfq::ModelSource> source = {}, bool load_input_output = true);
} // namespace mfq::cuda
