#pragma once

#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/models/components.h"
#include "backends/cuda/engine/models/load.h"
#include "models/qwen35/load.h"
#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/state/kv_memory.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "models/qwen35/components.h"
#include "models/qwen35/runtime.h"
#include "models/qwen35/state.h"
#include "models/minicpmo45/state.h"
#include "cuda_runtime_config.h"
#include "mfq_cuda_kv_quantization.h"

#include <chrono>
#include <iostream>
#include <optional>
#include <type_traits>
#include <utility>

namespace mfq::cuda::internal {

template <class M, class S = M>
struct Loaded {
    using Model = M;
    using State = S;
    using Parts = RuntimeComponents<M>;

    std::shared_ptr<Model> model;
    Parts components;
    std::unique_ptr<State> owned_state;
    std::unique_ptr<Executor> executor;

    Model& language() { return components.language(*model); }
    State& state() {
        if constexpr (std::is_same_v<Model, State>) return language();
        else return *owned_state;
    }
    std::optional<KvQuantization> kv_quantization() const {
        if constexpr (std::is_same_v<Model, Qwen4CausalLm>) return cuda_kv_quantization();
        else return std::nullopt;
    }
};

template <typename F>
auto with_loaded_cuda_model(
        CudaExecutionContext& execution,
        const CudaLoadOptions& options,
        bool load_optional_components,
        F&& run) {
    CudaScope scope(execution);
    auto source = mfq::open_model_source(options.model_path);
    auto finish = [&](auto& runtime, auto started) {
        auto& model = *runtime.model;
        auto& runtime_components = runtime.components;
        using Model = std::remove_cvref_t<decltype(model)>;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            prepare_sampling_runtime(model.vocab_size());
        if(execution.stores.kv_memory) {
            execution.stores.kv_memory->finalize(moe_expert_cache_minimum_bytes(execution.stores.weights.moe));
            set_moe_kv_expert_loan(execution.stores.weights.moe,execution.stores.kv_memory->expert_loan());
            if(auto offload=execution.stores.kv_memory->offload())execution.stores.qsa_kv_store=std::move(offload);
        }
        if (moe_expert_cache_has_sources(execution.stores.weights.moe) &&
                !moe_expert_cache_finalized(execution.stores.weights.moe)) {
            std::size_t graph_reserve=0;
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                graph_reserve=qwen4_mtp_graph_reserve(model,runtime_components.mtp.get());
                if(graph_reserve)std::cerr<<"runtime_mtp_flexible_budget maximum_depth="
                    <<runtime_components.mtp->maximum_draft_depth()<<" bytes="<<graph_reserve
                    <<" initial_depth=3 lend_unused_to_experts=1\n";
            }
            finalize_moe_expert_cache(execution.stores.weights.moe,
                (execution.stores.kv_memory?execution.stores.kv_memory->growth_reserve():0)+graph_reserve,
                execution.stores.kv_memory?execution.stores.kv_memory->expert_budget_charge():0);
        }
        mfq_cuda_synchronize();
        const auto loaded = std::chrono::steady_clock::now();
        report_cuda_memory(execution.config.report_cuda_memory, "loaded");
        return run(runtime, started, loaded);
    };
    auto load = [&]<typename Model>() {
        auto started = std::chrono::steady_clock::now();
        auto owner = std::make_shared<Model>(load_causal_lm<Model>(
                execution,
                options.model_path,
                options.config_path,
                options.context_size,
                true,
                load_optional_components,
                source));
        auto& model = *owner;
        weight_loader::Loader weights{*model.source, execution.config.weights,
            {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
             execution.tensor_parallel, moe_parallel_config(execution)},
            execution.stores.weights, execution.tq};
        auto runtime_components =
            load_runtime_components(model, weights, load_optional_components);
        Loaded<Model> runtime{std::move(owner), std::move(runtime_components), {}, {}};
        return finish(runtime, started);
    };
    auto load_qwen = [&] {
        auto started = std::chrono::steady_clock::now();
        weight_loader::Loader weights{*source, execution.config.weights,
            {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
             execution.tensor_parallel, moe_parallel_config(execution)},
            execution.stores.weights, execution.tq};
        auto model = mfq::cuda::qwen35::load(
            source, weights, execution.layer_placement, execution.yarn,
            execution.dense_cpu_layer_count,
            mfq::cuda::qwen35::LoadOptions{
                .config_path = options.config_path,
                .context_size = options.context_size,
                .gpu_layers = execution.n_gpu_layers,
                .extend_yarn = execution.yarn_context_extension,
                .expert_parallel = execution.expert_parallel.enabled(),
                .mirror_scalars = execution.config.tensor_parallel_mirror_linear_attention_scalars});
        const bool mtp_allowed =
            !execution.layer_placement.enabled() &&
            execution.dense_cpu_layer_count == 0 &&
            execution.dsv4_cpu_offload_layers.empty() &&
            !execution.stores.weights.moe;
        mfq::cuda::qwen35::Runtime runtime(
            std::move(model), weights, execution.make_executor(),
            execution.collectives, execution.config.gdn_transposed_state,
            execution.kl_mmq.kv_cache_capacity, execution.gemma_stage_trace,
            execution.gemma_trace_layer, load_optional_components, mtp_allowed,
            mfq::cuda::runtime_options::kv_vram_preallocated());
        return finish(runtime, started);
    };

    auto load_tts = [&] {
        auto started = std::chrono::steady_clock::now();
        auto definition = load_model_config_json(*source, options.config_path);
        definition = configure_yarn(execution.yarn, execution.yarn_context_extension, std::move(definition),
            source->resolved_model_graph().backbone, options.context_size);
        auto owner = std::make_shared<minicpmo45::Model>(
            minicpmo45::read(source, definition, options.context_size));
        auto& model = *owner;
        if (execution.expert_parallel.enabled())
            throw std::runtime_error("--expert-parallel requires a model with routed experts");
        execution.layer_placement.prepare(model.config.num_hidden_layers);
        execution.dense_cpu_layer_count = execution.n_gpu_layers >= 0
            ? static_cast<int>(std::max<int64_t>(model.config.num_hidden_layers - execution.n_gpu_layers, 0)) : 0;
        if (execution.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution) || execution.layer_placement.enabled())
                throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            std::cerr << "dense_layer_placement cpu=0-" << execution.dense_cpu_layer_count - 1
                << " gpu=" << execution.dense_cpu_layer_count << '-' << model.config.num_hidden_layers - 1
                << " cpu_threads=" << mfq_get_num_threads() << '\n';
        }
        weight_loader::Loader weights{*source, execution.config.weights,
            {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
             execution.tensor_parallel, moe_parallel_config(execution)},
            execution.stores.weights, execution.tq};
        minicpmo45::load(model, weights, execution.layer_placement,
            execution.dense_cpu_layer_count, execution.yarn);
        minicpmo45::State state(model);
        if (runtime_options::kv_vram_preallocated()) state.preallocate();
        auto executor = execution.make_executor();
        auto components = load_runtime_components(model, weights, load_optional_components);
        Loaded<minicpmo45::Model, minicpmo45::State> runtime{
            std::move(owner), std::move(components),
            std::make_unique<minicpmo45::State>(std::move(state)),
            std::move(executor)};
        return finish(runtime, started);
    };

    switch (cuda_backbone(source->resolved_model_graph().backbone)) {
        case CudaBackbone::generic_qwen:
            return load_qwen();
        case CudaBackbone::minicpmo45:
            return load.template operator()<MiniCPMO45CausalLm>();
        case CudaBackbone::minicpmo_tts:
            return load_tts();
        case CudaBackbone::gemma4:
            return load.template operator()<Gemma4CausalLm>();
        case CudaBackbone::glm_dsa:
            return load.template operator()<GlmDsaCausalLm>();
        case CudaBackbone::glm5_next:
            return load.template operator()<Glm5CausalLm>();
        case CudaBackbone::qwen4_exp:
            return load.template operator()<Qwen4CausalLm>();
        case CudaBackbone::deepseek_v4:
            return load.template operator()<DeepseekV4CausalLm>();
        case CudaBackbone::deepseek_v41:
            return load.template operator()<DeepseekV41CausalLm>();
        case CudaBackbone::nemotron_h:
            return load.template operator()<NemotronHCausalLm>();
        case CudaBackbone::unsupported:
            throw std::runtime_error("unsupported CUDA model backbone");
    }
    throw std::runtime_error("invalid CUDA model backbone");
}

} // namespace mfq::cuda::internal
