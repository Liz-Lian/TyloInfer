#include "backends/cuda/engine/models/load.h"
#include "native/cuda/env.h"
#include "engine/executor/ffn.h"
#include "backends/cuda/engine/weights/moe/load.h"
#include "backends/cuda/engine/weights/load.h"
#include "backends/cuda/engine/weights/tq/load.h"
#include "moe.h"
#include "backends/cuda/engine/executor/moe_ffn_pipeline.h"
#include "backends/cuda/engine/executor/tq_moe.h"

#include <cstdio>
#include <cmath>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iterator>

void load_important_neuron_branch(CudaExecutionContext &execution, const mfq::ModelSource &mfq,
                                  int64_t hidden_size, int64_t intermediate_size, mfq::cuda::Ffn &f,
                                  const std::string &down_name, const std::string &gate_name,
                                  const std::string &up_name) {
    const std::string down_high = down_name + ".in_high";
    const std::string gate_high = gate_name + ".in_high";
    const std::string up_high = up_name + ".in_high";
    const bool has_down = has_tensor(mfq, down_high);
    const bool has_gate = has_tensor(mfq, gate_high);
    const bool has_up = has_tensor(mfq, up_high);
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

    auto high = std::make_unique<mfq::cuda::Ffn>();
    high->down = load_quant_linear(execution, mfq, down_high, TensorParallelAxis::Input);
    high->gate_up = load_paired_gate_up(execution, mfq, {gate_high, up_high}, high->down);
    high->geglu = f.geglu;
    high->swiglu_limit = f.swiglu_limit;

    if (f.gate_up.outs.size() != 2 || high->gate_up.outs.size() != 2 ||
        f.gate_up.outs[0] != f.gate_up.outs[1] || high->gate_up.outs[0] != high->gate_up.outs[1] ||
        f.down.out() != hidden_size || high->down.out() != hidden_size ||
        f.down.neuron_len() != f.gate_up.outs[0] ||
        high->down.neuron_len() != high->gate_up.outs[0] ||
        f.down.neuron_len() + high->down.neuron_len() != intermediate_size) {
        throw std::runtime_error("important-neuron FFN tensor shapes disagree with model config");
    }
    f.important_neurons = std::move(high);
}

mfq_tensor_backend::Tensor TransformerWeightLoader::dense(const std::string& name) const {
    return load_dense_gpu(execution, source, name);
}
QuantLinear TransformerWeightLoader::linear(const std::string& name) const {
    return load_quant_linear(execution, source, name);
}
bool TransformerWeightLoader::has(const std::string& name) const { return has_tensor(source, name); }
mfq_tensor_backend::Tensor TransformerWeightLoader::router_parameter(const std::string& name) const {
    if(keep_quantized_router && name.ends_with("router.weight") && require_tensor(source,name).dtype=="TQ") {
        router_projection=std::make_shared<QuantLinear>(linear(name));
        return {};
    }
    return dense(name).to(mfq_tensor_backend::kFloat32).contiguous();
}
QuantLinearGroup TransformerWeightLoader::projections(const std::vector<std::string>& names) const {
    return load_quant_group(execution, source, names);
}
QuantLinearGroup TransformerWeightLoader::gate_up(const std::vector<std::string>& names,
                                                 const QuantLinear& down) const {
    return load_paired_gate_up(execution, source, names, down, shared_gate_up_compatible_prefix,
                              preserve_projection_boundaries);
}
void TransformerWeightLoader::qkv(mfq::cuda::FullAttention& block, const std::vector<std::string>& names) const {
    const bool mirror_kv = block.attention_output_gate &&
        mfq::cuda::env_flag("TYLOINFER_TP_MIRROR_QWEN35_ATTENTION_KV", true) &&
        is_quant_dtype(require_tensor(source, names[1]).dtype) &&
        is_quant_dtype(require_tensor(source, names[2]).dtype);
    if (mirror_kv) {
        block.split_q_kv_projections = true;
        block.q_projection = linear(names[0]);
        block.k_projection = load_quant_linear(execution, source, names[1], TensorParallelAxis::Mirrored);
        block.v_projection = load_quant_linear(execution, source, names[2], TensorParallelAxis::Mirrored);
    } else {
        block.qkv = load_quant_group(execution, source, names, 2, nullptr,
                                    preserve_projection_boundaries);
    }
}
void TransformerWeightLoader::workspace(mfq::cuda::Ffn& ffn) const { prepare_ffn(execution, ffn); }
void TransformerWeightLoader::important_neurons(int64_t hidden, int64_t intermediate, mfq::cuda::Ffn& ffn,
    const std::string& down, const std::string& gate, const std::string& up) const {
    load_important_neuron_branch(execution, source, hidden, intermediate, ffn, down, gate, up);
}
void TransformerWeightLoader::experts(mfq::cuda::Ffn& ffn, mfq::models::ExpertProjection projection,
                                     const std::string& name, int layer) const {
    using Role = mfq::models::ExpertProjection;
    const auto& record = require_tensor(source, name);
    if(record.dtype=="TQ") {
        if(projection==Role::gate_up)
            throw std::invalid_argument("TQ MoE stores separate Gate, Up and Down tensors: "+name);
        auto weight=load_tq_routed_weight(execution.stores.weights,execution.tq,source,name);
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
        auto bank = load_dense_native_gpu(execution, source, name);
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
    mfq::cuda::Experts* weight = nullptr;
    std::shared_ptr<MixedMoeRuntime>* cpu = nullptr;
    const char* role = nullptr;
    switch (projection) {
        case Role::gate: weight = &ffn.moe_gate; cpu = &ffn.cpu_moe_gate; role = "gate"; break;
        case Role::up: weight = &ffn.moe_up; cpu = &ffn.cpu_moe_up; role = "up"; break;
        case Role::gate_up: weight = &ffn.moe_gate_up; cpu = &ffn.cpu_moe_gate_up; role = "gate_up"; break;
        case Role::down: weight = &ffn.moe_down; cpu = &ffn.cpu_moe_down; role = "down"; break;
    }
    if (cpu_offloaded) {
        *cpu = load_mfe_cpu_offloaded(source, name);
        *weight = cpu_mixed_moe_metadata(*cpu);
    } else {
        *weight = load_mfe_gpu(execution, source, name, true, layer, role);
    }

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
mfq::cuda::Experts TransformerWeightLoader::headwise(const std::string& name) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT

    return load_mfe_gpu(execution, source, name);

#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}
mfq::cuda::Ffn load_moe_weights(CudaExecutionContext& execution, const mfq::ModelSource& source,
                     std::string_view prefix, const MoeWeightLoadOptions& options) {
    TransformerWeightLoader loader{execution, source};
    loader.keep_quantized_router=true;
    loader.cpu_offloaded = options.cpu_offloaded;
    loader.shared_gate_up_compatible_prefix = options.shared_gate_up_compatible_prefix;
    auto result=mfq::models::load_moe_weights<mfq::cuda::Ffn>(
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
        const auto& parallel = moe_parallel_config(execution);
        if (parallel.enabled())
            banks.slices = plan_moe_expert_parallel_slices(parallel, banks.up->experts, std::string(prefix));
    }
    result.moe_router_projection=std::move(loader.router_projection);
    return result;
}

void prepare_ffn(CudaExecutionContext &execution, mfq::cuda::Ffn &f) {
    if (execution.loading_cpu_layer)
        return;
    if (f.down.tensor_parallel())
        return;
    if (mfq::mfq_compat_enabled && f.gate_up.nvq_prefix2 && f.gate_up.layers.size() == 2 && f.down.is_nvq() &&
        f.gate_up.outs.size() == 2 && f.gate_up.outs[0] == f.gate_up.outs[1] &&
        f.gate_up.outs[0] == f.down.nvq.neuron_len) {
        NvqWorkspace &ws = f.gate_up.layers[0].nvq.workspace(1);
        ws.swiglu_scratch = mfq_tensor_backend::empty({f.gate_up.outs[0]},
                                                      mfq_tensor_backend::TensorOptions()
                                                          .device(mfq_tensor_backend::kCUDA)
                                                          .dtype(mfq_tensor_backend::kFloat32));
        (void)f.down.nvq.workspace(1);
    }
    if (f.important_neurons) {
        prepare_ffn(execution, *f.important_neurons);
    }
}

void prepare_moe_ffn(CudaExecutionContext& execution, mfq::cuda::Ffn& f) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if(!f.moe_pipeline.expired() || !mfq::cuda::uses_expert_cache(f) || mfq::cuda::expert_parallel(f) ||
            f.cpu_moe_gate || f.cpu_moe_up || f.cpu_moe_gate_up || f.cpu_moe_down ||
            execution.routes.enabled())return;
    std::vector<std::shared_ptr<mfq::cuda::Experts>> gate_up;
    // An ungated bank publishes Up in both slots; the pipeline detects the
    // single-image layout from the projection geometry, not the model name.
    if(f.moe_ungated_experts)
        gate_up={std::make_shared<mfq::cuda::Experts>(f.moe_up),std::make_shared<mfq::cuda::Experts>(f.moe_up)};
    else if(f.moe_split_gate_up)gate_up={std::make_shared<mfq::cuda::Experts>(f.moe_gate),std::make_shared<mfq::cuda::Experts>(f.moe_up)};
    else gate_up={std::make_shared<mfq::cuda::Experts>(f.moe_gate_up)};
    auto pipeline=make_moe_ffn_pipeline(
        execution.profiler,execution.kl_mmq,execution.config.tasks.projection.moe,execution.config.moe_ffn,
        gate_up,{std::make_shared<mfq::cuda::Experts>(f.moe_down)},
        {},{},{},f.swiglu_limit);
    if(!pipeline)return;
    execution.register_moe_pipeline(pipeline);
    f.moe_pipeline=pipeline;
    std::fprintf(stderr,"moe_generic_pipeline_dispatch layer=%d clipped_swiglu=%g\n",
        f.moe_layer,f.swiglu_limit);
#else
    return;
#endif
}

namespace mfq::cuda {

std::string configure_yarn(CudaExecutionContext& execution, std::string definition,
                           std::string_view backbone, int64_t context) {
    execution.yarn={};
    if(execution.yarn_context_extension) {
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
        execution.yarn.original_context=native;
        execution.yarn.factor=std::max(1.0,double(context)/native);
        execution.yarn.beta_fast=rope.value("beta_fast",32.0);
        execution.yarn.beta_slow=rope.value("beta_slow",1.0);
        if(execution.yarn.enabled())text["max_position_embeddings"]=context;
        definition=config.dump();
    }
    return definition;
}

std::string load_model_config_json(
        const mfq::ModelSource& source,
        const std::string& external_path) {
    if (external_path.empty()) return source.model_config_json();
    std::ifstream input(external_path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open config: " + external_path);
    return {std::istreambuf_iterator<char>(input), {}};
}

void validate_model_source(const mfq::ModelSource& source) {
    if (source.find_tensor("model.token_embedding.weight") == nullptr) {
        throw std::runtime_error(
            "canonical text tensor inventory has no token embedding");
    }
    for (const auto& component : source.resolved_model_graph().components) {
        if (component.tensor_root != "runtime" &&
                !has_tensor_prefix(source, component.tensor_root)) {
            throw std::runtime_error(
                "model graph declares " + component.kind +
                " but its canonical tensor inventory is missing");
        }
    }
}

} // namespace mfq::cuda
