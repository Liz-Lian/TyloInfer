#include "engine/models/source.h"
#include "engine/ops.h"
#include "engine/models/common/load.h"
#include "engine/models/common/model_graph.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/weights/moe/load.h"
#include "engine/weights/loader.h"
#include "engine/weights/tq/load.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/backends/cuda/executor/tq_moe.h"

#include <cstdio>
#include <cmath>
#include <nlohmann/json.hpp>

void load_important_neuron_branch(const tyloi::cuda::weight_loader::Loader& weights,
                                  int64_t hidden_size, int64_t intermediate_size, tyloi::cuda::Ffn &f,
                                  const std::string &down_name, const std::string &gate_name,
                                  const std::string &up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = tyloi::has_tensor(weights.source, down_high);
    const bool has_gate = tyloi::has_tensor(weights.source, gate_high);
    const bool has_up = tyloi::has_tensor(weights.source, up_high);
    if (!has_down && !has_gate && !has_up) {
        return;
    }
    if (!has_down || !has_gate || !has_up) {
        throw std::runtime_error(
            "important-neuron FFN requires matching gate/up/down .in_high records");
    }
    if (f.is_moe) {
        throw std::runtime_error("important-neuron records are unsupported on routed MoE FFNs");
    }

    auto high = std::make_unique<tyloi::cuda::Ffn>();
    high->down = weights.weight(down_high, TensorParallelAxis::Input);
    high->gate_up = weights.build_gate_up({gate_high, up_high}, *high->down);
    high->geglu = f.geglu;
    high->swiglu_limit = f.swiglu_limit;

    if (f.gate_up.outs.size() != 2 || high->gate_up.outs.size() != 2 ||
        f.gate_up.outs[0] != f.gate_up.outs[1] || high->gate_up.outs[0] != high->gate_up.outs[1] ||
        f.down->size(0) != hidden_size || high->down->size(0) != hidden_size ||
        f.down->size(1) != f.gate_up.outs[0] ||
        high->down->size(1) != high->gate_up.outs[0] ||
        f.down->size(1) + high->down->size(1) != intermediate_size) {
        throw std::runtime_error("important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}

::tyloi::DenseTensor TransformerWeightLoader::dense(const std::string& name) const {
    return weights.tensor(name);
}
std::shared_ptr<const ::tyloi::Tensor> TransformerWeightLoader::linear(const std::string& name) const {
    return weights.weight(name);
}
bool TransformerWeightLoader::has(const std::string& name) const { return tyloi::has_tensor(weights.source, name); }
::tyloi::DenseTensor TransformerWeightLoader::router_parameter(const std::string& name) const {
    if(keep_quantized_router && name.ends_with("router.weight") && tyloi::require_tensor(weights.source,name).dtype=="TQ") {
        router_projection=linear(name);
        return {};
    }
    return dense(name).to(::tyloi::kFloat32).contiguous();
}
QuantLinearGroup TransformerWeightLoader::projections(const std::vector<std::string>& names) const {
    return weights.build_group(names);
}
QuantLinearGroup TransformerWeightLoader::gate_up(const std::vector<std::string>& names,
                                                 const std::shared_ptr<const ::tyloi::Tensor>& down) const {
    return weights.build_gate_up(names, *down, shared_gate_up_compatible_prefix,
                              preserve_projection_boundaries);
}
void TransformerWeightLoader::qkv(tyloi::cuda::FullAttention& block, const std::vector<std::string>& names) const {
    const bool mirror_kv = block.attention_output_gate &&
        tyloi::cuda::env_flag("TYLOINFER_TP_MIRROR_QWEN35_ATTENTION_KV", true) &&
        is_quant_dtype(tyloi::require_tensor(weights.source, names[1]).dtype) &&
        is_quant_dtype(tyloi::require_tensor(weights.source, names[2]).dtype);
    if (mirror_kv) {
        block.split_q_kv_projections = true;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        block.q_projection = weights.weight(names[0]);
        block.k_projection = weights.weight(names[1], TensorParallelAxis::Mirrored);
        block.v_projection = weights.weight(names[2], TensorParallelAxis::Mirrored);
#else
        block.q_projection = linear(names[0]);
        block.k_projection = weights.weight(names[1], TensorParallelAxis::Mirrored);
        block.v_projection = weights.weight(names[2], TensorParallelAxis::Mirrored);
#endif
    } else {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        block.qkv = weights.group(names, 2, nullptr, preserve_projection_boundaries);
#else
        block.qkv = weights.build_group(names, 2, nullptr,
                                    preserve_projection_boundaries);
#endif
    }
}
void TransformerWeightLoader::output(tyloi::cuda::FullAttention& block, const std::string& name) const {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    block.o = weights.weight(name);
#else
    block.o = linear(name);
#endif
}
void TransformerWeightLoader::important_neurons(int64_t hidden, int64_t intermediate, tyloi::cuda::Ffn& ffn,
    const std::string& down, const std::string& gate, const std::string& up) const {
    load_important_neuron_branch(weights, hidden, intermediate, ffn, down, gate, up);
}
void TransformerWeightLoader::experts(tyloi::cuda::Ffn& ffn, tyloi::models::ExpertProjection projection,
                                     const std::string& name, int layer) const {
    using Role = tyloi::models::ExpertProjection;
    const auto& record = tyloi::require_tensor(weights.source, name);
    if(record.dtype=="TQ") {
        if(projection==Role::gate_up)
            throw std::invalid_argument("TQ MoE stores separate Gate, Up and Down tensors: "+name);
        auto weight=load_tq_routed_weight(weights.cache,weights.tq,weights.source,name);
        if(!ffn.moe_tq)ffn.moe_tq=std::make_shared<TqMoeBanks>();
        switch(projection) {
            case Role::gate: ffn.moe_tq->gate=std::move(weight);break;
            case Role::up: ffn.moe_tq->up=std::move(weight);break;
            case Role::gate_up: break;
            case Role::down: ffn.moe_tq->down=std::move(weight);break;
        }
        return;
    }
    if (record.dtype == "F32" || record.dtype == "F16" || record.dtype == "BF16") {
        // Source-precision banks stay dense instead of entering an MFE cohort.
        if (cpu_offloaded)
            throw std::runtime_error(
                "CPU-offloaded MoE cannot materialize a dense expert bank at layer " +
                std::to_string(layer));
        if (projection == Role::gate_up)
            throw std::runtime_error(
                "dense fused Gate/Up expert banks require separate Gate and Up records at layer " +
                std::to_string(layer));
        auto bank = weights.native(name);
        if (bank.dim() != 3)
            throw std::runtime_error(
                "dense routed expert tensor must be [experts,out,in]: " + name);
        switch (projection) {
            case Role::gate: ffn.moe_dense_gate = std::move(bank); break;
            case Role::up: ffn.moe_dense_up = std::move(bank); break;
            case Role::down: ffn.moe_dense_down = std::move(bank); break;
            case Role::gate_up: break;
        }
        return;
    }

#if TYLOINFER_ENABLE_MFQ_COMPAT
    tyloi::cuda::Experts* weight = nullptr;
    std::shared_ptr<::tyloi::MfeTensor>* cpu = nullptr;
    const char* role = nullptr;
    switch (projection) {
        case Role::gate: weight = &ffn.moe_gate; cpu = &ffn.cpu_moe_gate; role = "gate"; break;
        case Role::up: weight = &ffn.moe_up; cpu = &ffn.cpu_moe_up; role = "up"; break;
        case Role::gate_up: weight = &ffn.moe_gate_up; cpu = &ffn.cpu_moe_gate_up; role = "gate_up"; break;
        case Role::down: weight = &ffn.moe_down; cpu = &ffn.cpu_moe_down; role = "down"; break;
    }
    if (cpu_offloaded) {
        *cpu = load_mfe_cpu_offloaded(weights.source, name);
        *weight = cpu_mixed_moe_metadata(*cpu);
    } else {
        *weight = weights.build_experts(name, true, layer, role);
    }

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
tyloi::cuda::Experts TransformerWeightLoader::headwise(const std::string& name) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT

    return weights.build_experts(name);

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
tyloi::cuda::Ffn load_moe_weights(const tyloi::cuda::weight_loader::Loader& weights,
                     std::string_view prefix, const MoeWeightLoadOptions& options) {
    TransformerWeightLoader loader{weights};
    loader.keep_quantized_router=true;
    loader.cpu_offloaded = options.cpu_offloaded;
    loader.shared_gate_up_compatible_prefix = options.shared_gate_up_compatible_prefix;
    auto result=tyloi::models::load_moe_weights<tyloi::cuda::Ffn>(
        loader, std::string(prefix), options.layer, options.router_bias_required,
        options.ungated_experts, options.ungated_shared);
    if(result.moe_tq) {
        auto& banks=*result.moe_tq;
        TYLOINFER_RUNTIME_CHECK(banks.up && banks.down && (options.ungated_experts || banks.gate),
            "TQ expert banks cannot be mixed with legacy routed formats");
        TYLOINFER_RUNTIME_CHECK(banks.up->experts==banks.down->experts &&
            banks.up->out==banks.down->input && banks.up->input==banks.down->out &&
            (!banks.gate || (banks.gate->experts==banks.up->experts &&
                banks.gate->out==banks.up->out && banks.gate->input==banks.up->input)),
            "TQ Gate/Up/Down expert dimensions disagree");
        const auto& parallel = weights.placement.experts;
        if (parallel.enabled())
            banks.slices = plan_moe_expert_parallel_slices(parallel, banks.up->experts, std::string(prefix));
    }
    result.moe_router_projection=std::move(loader.router_projection);
    return result;
}

namespace tyloi::cuda {

std::string configure_yarn(tyloi::ops::YarnScaling& yarn, bool extend, std::string definition,
                           std::string_view backbone, int64_t context) {
    yarn = {};
    if(extend) {
        if(backbone!="qwen3_5" && backbone!="qwen4_exp")
            throw std::invalid_argument("YaRN extension requires a Qwen model");
        auto config=nlohmann::json::parse(definition);
        auto& text=config.contains("text_config") ? config["text_config"] : config.contains("language_config") ? config["language_config"] : config;
        auto rope=text.value("rope_parameters",text.value("rope_scaling",nlohmann::json::object()));
        if(rope.contains("full_attention"))rope=rope["full_attention"];
        auto native=text.value("max_position_embeddings",int64_t(0));
        const auto kind=rope.value("rope_type",rope.value("type",std::string("default")));
        if(kind=="yarn")native=rope.value("original_max_position_embeddings",native);
        const auto maximum=rope.value("factor",4.0);
        if((kind!="default" && kind!="yarn") || native<=0 || !std::isfinite(maximum) || maximum<1 ||
            context>std::floor(native*maximum))throw std::invalid_argument("context exceeds Qwen YaRN limit");
        yarn.original_context=native;
        yarn.factor=std::max(1.0,double(context)/native);
        yarn.beta_fast=rope.value("beta_fast",32.0);
        yarn.beta_slow=rope.value("beta_slow",1.0);
        if(yarn.enabled())text["max_position_embeddings"]=context;
        definition=config.dump();
    }
    return definition;
}

void validate_model_source(const tyloi::ModelSource& source) {
    if (source.find_tensor("model.token_embedding.weight") == nullptr) {
        throw std::runtime_error(
            "canonical text tensor inventory has no token embedding");
    }
    for (const auto& component : tyloi::models::graph(source).components) {
        if (component.tensor_root != "runtime" &&
                !has_tensor_prefix(source, component.tensor_root)) {
            throw std::runtime_error(
                "model graph declares " + component.kind +
                " but its canonical tensor inventory is missing");
        }
    }
}

} // namespace tyloi::cuda

void validate_native_placement(const ParallelConfig& tensor, const LayerPlacementConfig& layers,
                               int gpu_layers) {
    if (tensor.enabled() || layers.enabled() || gpu_layers >= 0)
        throw std::runtime_error(
            "native attention adapter supports expert parallelism, but "
            "tensor/layer parallelism and offload require a different placement path");
}
