#pragma once

#include "engine/models/source/model_config.h"
#include "engine/ops.h"

#include <memory>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

namespace tyloi::models {

struct Mlp;

enum class ExpertProjection { gate, up, gate_up, down };

// Canonical parameter relationships are shared; the loader materializes native
// weights and chooses packing, projection fusion and device placement.
template <class Ffn, class Loader>
Ffn load_dense_ffn(Loader& ops, const ModelConfig& config, const std::string& prefix) {
    Ffn result;
    const auto down = prefix + "down.weight", gate = prefix + "gate.weight",
               up = prefix + "up.weight";
    result.down = ops.linear(down);
    result.gate_up = ops.gate_up({gate, up}, result.down);
    ops.important_neurons(config.hidden_size, config.intermediate_size, result, down, gate, up);
    return result;
}

// Routed banks may be gated (gate/up or fused gate_up) or ungated (up only).
// An ungated expert's activation is relu^2, never a gate projection alias.
/** Fill owned routed weights so partial loads remain available to failure cleanup. */
template <class Ffn, class Loader>
void load_moe_weights_into(Ffn& result, Loader& ops, const std::string& prefix, int layer,
                    bool router_bias_required = false, bool ungated_experts = false,
                    bool ungated_shared = false) {
    const auto gate_up = prefix + "experts.gate_up.weight", gate = prefix + "experts.gate.weight",
               up = prefix + "experts.up.weight", down = prefix + "experts.down.weight";
    const bool has_up = ops.has(up), has_down = ops.has(down);
    const bool has_gate = ops.has(gate), has_gate_up = ops.has(gate_up);
    result.is_moe = true;
    result.moe_layer = layer;
    if (ungated_experts) {
        if (!has_up || !has_down || has_gate || has_gate_up)
            throw std::runtime_error(
                "ungated routed MoE requires only Up and Down at layer " +
                std::to_string(layer));
        if constexpr (requires { result.moe_ungated_experts = true; }) {
            result.moe_ungated_experts = true;
            // Up lives in the split slot so prefetch/cache code sees a real
            // single projection instead of an empty fused pair.
            result.moe_split_gate_up = true;
        } else
            throw std::runtime_error(
                "this FFN type does not support ungated routed experts at layer " +
                std::to_string(layer));
        if constexpr (requires { result.activation = tyloi::ops::MlpActivation::relu_squared; })
            result.activation = tyloi::ops::MlpActivation::relu_squared;
        ops.experts(result, ExpertProjection::up, up, layer);
    } else {
        const bool split = has_gate, fused = has_gate_up;
        if (split != ops.has(up) || fused == split || !ops.has(down))
            throw std::runtime_error(
                "routed MoE requires Down and exactly one fused or split "
                "Gate/Up representation at layer " + std::to_string(layer));
        result.moe_split_gate_up = split;
        if (split) {
            ops.experts(result, ExpertProjection::gate, gate, layer);
            ops.experts(result, ExpertProjection::up, up, layer);
        } else {
            ops.experts(result, ExpertProjection::gate_up, gate_up, layer);
        }
    }
    ops.experts(result, ExpertProjection::down, down, layer);
    const auto latent_in = prefix + "latent.down.weight";
    const auto latent_out = prefix + "latent.up.weight";
    if (ops.has(latent_in) != ops.has(latent_out))
        throw std::runtime_error(
            "latent MoE requires both Down and Up latent projections at layer " +
            std::to_string(layer));
    if (ops.has(latent_in)) {
        if constexpr (requires { result.moe_latent_in = ops.linear(latent_in); }) {
            result.moe_latent_in = ops.linear(latent_in);
            result.moe_latent_out = ops.linear(latent_out);
        } else {
            throw std::runtime_error(
                "this FFN type does not support latent MoE projections at layer " +
                std::to_string(layer));
        }
    }
    result.moe_router = ops.router_parameter(prefix + "router.weight");
    if (router_bias_required || ops.has(prefix + "router.bias"))
        result.moe_router_bias = ops.router_parameter(prefix + "router.bias");
    result.shared = std::make_unique<Ffn>();
    result.shared->down = ops.linear(prefix + "shared_expert.down.weight");
    const auto shared_gate = prefix + "shared_expert.gate.weight";
    const auto shared_up = prefix + "shared_expert.up.weight";
    if (ungated_shared) {
        if (ops.has(shared_gate) || !ops.has(shared_up))
            throw std::runtime_error(
                "ungated shared expert requires Up without a Gate at layer " +
                std::to_string(layer));
        if constexpr (requires { result.shared->activation = tyloi::ops::MlpActivation::relu_squared; })
            result.shared->activation = tyloi::ops::MlpActivation::relu_squared;
        result.shared->gate_up = ops.gate_up({shared_up}, result.shared->down);
    } else {
        result.shared->gate_up = ops.gate_up({shared_gate, shared_up}, result.shared->down);
    }
}

template <class Ffn, class Loader>
Ffn load_moe_weights(Loader& ops, const std::string& prefix, int layer,
                    bool router_bias_required = false, bool ungated_experts = false,
                    bool ungated_shared = false) {
    Ffn result;
    load_moe_weights_into(result, ops, prefix, layer, router_bias_required, ungated_experts, ungated_shared);
    return result;
}

// Geometry and projections of one canonical attention mixer. Blocks that also
// carry a separate FFN norm load it through their own adapter.
template <class Block, class Loader>
void load_attention_mixer(Block& block, Loader& ops, const ModelConfig& config, int layer,
                          const std::string& prefix, double norm_offset,
                          bool output_gate = false) {
    block.layer = layer;
    block.attention_heads = config.num_attention_heads;
    block.kv_heads = config.num_key_value_heads;
    block.attention_head_dim = config.head_dim;
    block.max_position_embeddings = config.max_position_embeddings;
    block.rms_norm_eps = config.rms_norm_eps;
    block.norm_weight_offset = norm_offset;
    block.attention_output_gate = output_gate;
    if constexpr (requires { block.no_rope = false; })
        block.no_rope = config.positional_encoding == PositionalEncoding::none;
#ifdef TYLOINFER_NATIVE_CUDA
    block.attn_norm = std::make_shared<tyloi::DenseTensor>(ops.dense(prefix + "attention.norm.weight"));
#else
    block.attn_norm = ops.dense(prefix + "attention.norm.weight");
#endif
    const auto attention = prefix + "attention.";
    ops.qkv(block, {attention + "query.weight", attention + "key.weight", attention + "value.weight"});
    ops.output(block, attention + "output.weight");
    if (ops.has(attention + "query_norm.weight"))
#ifdef TYLOINFER_NATIVE_CUDA
        block.q_norm = std::make_shared<tyloi::DenseTensor>(ops.dense(attention + "query_norm.weight"));
#else
        block.q_norm = ops.dense(attention + "query_norm.weight");
#endif
    if (ops.has(attention + "key_norm.weight"))
#ifdef TYLOINFER_NATIVE_CUDA
        block.k_norm = std::make_shared<tyloi::DenseTensor>(ops.dense(attention + "key_norm.weight"));
#else
        block.k_norm = ops.dense(attention + "key_norm.weight");
#endif
}

template <class Block, class Loader>
void load_full_attention(Block& block, Loader& ops, const ModelConfig& config, int layer,
                         const std::string& prefix, double norm_offset, bool output_gate = false) {
    load_attention_mixer(block, ops, config, layer, prefix, norm_offset, output_gate);
    block.ffn_norm = ops.dense(prefix + "mlp.norm.weight");
}

} // namespace tyloi::models

struct ParallelConfig;
struct LayerPlacementConfig;
namespace tyloi { class ModelSource; }
namespace tyloi::weights { struct Loader; }
namespace tyloi::cuda { struct FullAttention; class Ffn; }
#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::ops { struct FfnWeights; }
#endif

struct TransformerWeightLoader {
    const tyloi::weights::Loader& weights;
    bool preserve_projection_boundaries = false;
    bool cpu_offloaded = false;
    std::size_t shared_gate_up_compatible_prefix = 2;
    bool keep_quantized_router = false;

    bool has(const std::string& name) const;
    ::tyloi::DenseTensor dense(const std::string& name) const;
    ::tyloi::Tensor router_parameter(const std::string& name) const;
    ::tyloi::Tensor linear(const std::string& name) const;
    std::vector<::tyloi::Tensor> projections(const std::vector<std::string>& names) const;
    std::vector<::tyloi::Tensor> gate_up(const std::vector<std::string>& names, const ::tyloi::Tensor& down) const;
    void qkv(tyloi::cuda::FullAttention& attention, const std::vector<std::string>& names) const;
    void output(tyloi::cuda::FullAttention& attention, const std::string& name) const;
    void important_neurons(int64_t hidden, int64_t intermediate, tyloi::cuda::Ffn& ffn,
        const std::string& down, const std::string& gate, const std::string& up) const;
    void important_neurons(int64_t hidden, int64_t intermediate, tyloi::models::Mlp& ffn,
        const std::string& down, const std::string& gate, const std::string& up) const;
    void experts(tyloi::cuda::Ffn& ffn, tyloi::models::ExpertProjection projection,
                 const std::string& name, int layer) const;
#ifdef TYLOINFER_NATIVE_CUDA
    /** Load a dense FFN's optional important-neuron branch. */
    void important_neurons(int64_t hidden, int64_t intermediate, tyloi::ops::FfnWeights& ffn,
        const std::string& down, const std::string& gate, const std::string& up) const;
    /** Load one routed expert projection into public FFN weights. */
    void experts(tyloi::ops::FfnWeights& ffn, tyloi::models::ExpertProjection projection,
                 const std::string& name, int layer) const;
#endif
    ::tyloi::Tensor headwise(const std::string& name) const;
    static std::vector<std::int64_t> shape(const ::tyloi::DenseTensor& value);
    static int64_t elements(const ::tyloi::DenseTensor& value);
    static auto shape(const ::tyloi::Tensor& value) {
        std::vector<int64_t> result;
        if (value) for (int64_t i = 0; i < value.dim(); ++i) result.push_back(value.size(i));
        return result;
    }
    static int64_t elements(const ::tyloi::Tensor& value) { return value ? value.numel() : 0; }
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
    const tyloi::weights::Loader& weights,
    std::string_view prefix,
    const MoeWeightLoadOptions& options);
/** Fill an owned routed FFN while retaining partial weights if loading fails. */
void load_moe_weights_into(tyloi::cuda::Ffn&, const tyloi::weights::Loader&,
    std::string_view prefix, const MoeWeightLoadOptions& options);
#ifdef TYLOINFER_NATIVE_CUDA
/** Load routed FFN weights and validate their expert geometry. */
tyloi::ops::FfnWeights load_moe_data(const tyloi::weights::Loader& weights,
    std::string_view prefix, const MoeWeightLoadOptions& options);
/** Fill owned public FFN weights while retaining partial weights if loading fails. */
void load_moe_data_into(tyloi::ops::FfnWeights&, const tyloi::weights::Loader&,
    std::string_view prefix, const MoeWeightLoadOptions& options);
#endif
void load_important_neuron_branch(
    const tyloi::weights::Loader& weights,
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
