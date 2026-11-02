#pragma once

#include "engine/models/common/loaded.h"
#include "engine/load.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/common/components.h"
#include "engine/models/common/load.h"
#include "engine/models/common/model_config.h"
#include "engine/models/common/model_graph.h"
#include "engine/models/qwen35/load.h"
#include "engine/models/gemma4/load.h"
#include "engine/models/gemma4/state.h"
#include "engine/models/glm_dsa/load.h"
#include "engine/models/glm_dsa/state.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/glm5_next/load.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/nemotron_h/state.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/state/kv_memory.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/models/qwen35/components.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/minicpmo45/state.h"
#include "engine/models/minicpmo45/text/load.h"
#include "engine/models/minicpmo45/text/state.h"

#include <chrono>
#include <iostream>
#include <type_traits>
#include <utility>

namespace mfq::cuda {

template <bool IncludeQwen = true>
inline constexpr auto with_model = []<typename F>(
        CudaExecutionContext& execution,
        const CudaLoadOptions& options,
        bool load_optional_components,
        F&& run) -> auto {
    CudaScope scope(execution.layer_placement.primary_device(), execution.contexts);
    auto source = mfq::open_model_source(options.model_path);
    auto finish = [&](auto& loaded, auto started) {
        auto& model = *loaded.model;
        auto& runtime_components = loaded.components;
        using Model = std::remove_cvref_t<decltype(model)>;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            Executor::prepare(model.vocab_size());
        mfq::cuda::finalize_kv(execution.kv.get(),
            execution.weights, execution.offload);
        std::size_t graph_reserve=0;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
            if (moe_expert_cache_has_sources(execution.weights.moe) &&
                    !moe_expert_cache_finalized(execution.weights.moe)) {
                graph_reserve=qwen4_mtp_graph_reserve(model,runtime_components.mtp.get());
                if(graph_reserve)std::cerr<<"runtime_mtp_flexible_budget maximum_depth="
                    <<runtime_components.mtp->maximum_draft_depth()<<" bytes="<<graph_reserve
                    <<" initial_depth=3 lend_unused_to_experts=1\n";
            }
        }
        const auto loaded_at = mfq::cuda::finish_load(
            execution.kv.get(), execution.weights,
            graph_reserve, execution.config.report_cuda_memory);
        return run(loaded, started, loaded_at);
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
            execution.weights, execution.tq};
        auto runtime_components =
            load_runtime_components(model, weights, load_optional_components);
        Loaded<Model> loaded{std::move(owner), {}, execution.make_executor(), std::move(runtime_components)};
        return finish(loaded, started);
    };
    auto load_tts = [&] {
        auto started = std::chrono::steady_clock::now();
        auto definition = mfq::models::config_json(*source, options.config_path);
        definition = configure_yarn(execution.yarn, execution.yarn_context_extension, std::move(definition),
            mfq::models::graph(*source).backbone, options.context_size);
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
            execution.weights, execution.tq};
        minicpmo45::load(model, weights, execution.layer_placement,
            execution.dense_cpu_layer_count, execution.yarn);
        minicpmo45::State state(model);
        if (runtime_options::kv_vram_preallocated()) state.preallocate();
        auto executor = execution.make_executor();
        auto components = load_runtime_components(model, weights, load_optional_components);
        Loaded<minicpmo45::Model, minicpmo45::State> loaded{
            std::move(owner),
            std::make_unique<minicpmo45::State>(std::move(state)),
            std::move(executor), std::move(components)};
        return finish(loaded, started);
    };

    switch (cuda_backbone(mfq::models::graph(*source).backbone)) {
        case CudaBackbone::generic_qwen:
            if constexpr (IncludeQwen) {
                auto started = std::chrono::steady_clock::now();
                weight_loader::Loader weights{*source, execution.config.weights,
                    {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
                     execution.tensor_parallel, moe_parallel_config(execution)},
                    execution.weights, execution.tq};
                auto loaded = qwen35::assemble(
                    source, weights, execution.layer_placement, execution.yarn,
                    execution.dense_cpu_layer_count,
                    mfq::cuda::qwen35::LoadOptions{
                        .config_path = options.config_path,
                        .context_size = options.context_size,
                        .gpu_layers = execution.n_gpu_layers,
                        .extend_yarn = execution.yarn_context_extension,
                        .expert_parallel = execution.expert_parallel.enabled(),
                        .mirror_scalars = execution.config.tensor_parallel_mirror_linear_attention_scalars},
                    execution.make_executor(), execution.collectives,
                    execution.config.gdn_transposed_state,
                    execution.kl_mmq.kv_cache_capacity, execution.gemma_stage_trace,
                    execution.gemma_trace_layer, load_optional_components,
                    !execution.dsv4_cpu_offload_layers.empty(),
                    runtime_options::kv_vram_preallocated());
                return finish(loaded, started);
            } else {
                throw std::runtime_error("unsupported CUDA model backbone");
            }
        case CudaBackbone::minicpmo45: {
            auto started = std::chrono::steady_clock::now();
            auto definition = mfq::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.yarn, execution.yarn_context_extension,
                std::move(definition), mfq::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<minicpmo45::text::Model>(
                minicpmo45::text::read(source, definition, options.context_size));
            weight_loader::Loader weights{*source, execution.config.weights,
                {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
                 execution.tensor_parallel, moe_parallel_config(execution)},
                execution.weights, execution.tq};
            execution.dense_cpu_layer_count = execution.n_gpu_layers >= 0
                ? static_cast<int>(std::max<int64_t>(owner->config.num_hidden_layers - execution.n_gpu_layers, 0)) : 0;
            minicpmo45::text::load(*owner, weights, execution.layer_placement,
                execution.n_gpu_layers, execution.expert_parallel.enabled(), true, true, execution.yarn);
            auto state = std::make_unique<minicpmo45::text::State>(*owner);
            state->preallocate();
            auto executor = execution.make_executor();
            auto components = load_runtime_components(*owner, *state, *executor, weights,
                load_optional_components, execution);
            Loaded<minicpmo45::text::Model, minicpmo45::text::State> loaded{
                std::move(owner), std::move(state), std::move(executor), std::move(components)};
            return finish(loaded, started);
        }
        case CudaBackbone::minicpmo_tts:
            return load_tts();
        case CudaBackbone::gemma4: {
            auto started = std::chrono::steady_clock::now();
            auto definition = mfq::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.yarn, execution.yarn_context_extension,
                std::move(definition), mfq::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<gemma4::Model>(
                gemma4::read(source, definition, options.context_size));
            weight_loader::Loader weights{*source, execution.config.weights,
                {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
                 execution.tensor_parallel, moe_parallel_config(execution)},
                execution.weights, execution.tq};
            execution.dense_cpu_layer_count = 0;
            gemma4::load(*owner, weights, execution.layer_placement, execution.n_gpu_layers,
                         execution.expert_parallel.enabled());
            auto state = std::make_unique<gemma4::State>(*owner);
            state->preallocate();
            auto executor = execution.make_executor();
            auto components = load_runtime_components(*owner, weights, load_optional_components);
            Loaded<gemma4::Model, gemma4::State> loaded{
                std::move(owner), std::move(state), std::move(executor), std::move(components)};
            return finish(loaded, started);
        }
        case CudaBackbone::glm_dsa: {
            auto started = std::chrono::steady_clock::now();
            auto definition = mfq::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.yarn, execution.yarn_context_extension,
                std::move(definition), mfq::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<glm_dsa::Model>(
                glm_dsa::read(source, definition, options.context_size));
            weight_loader::Loader weights{*source, execution.config.weights,
                {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
                 execution.tensor_parallel, moe_parallel_config(execution)},
                execution.weights, execution.tq};
            execution.dense_cpu_layer_count = 0;
            glm_dsa::load(*owner, weights, execution.layer_placement,
                          execution.n_gpu_layers, true, true, execution.yarn);
            auto state = std::make_unique<glm_dsa::State>(*owner);
            state->preallocate();
            auto executor = execution.make_executor();
            auto components = load_runtime_components(*owner, weights, load_optional_components);
            Loaded<glm_dsa::Model, glm_dsa::State> loaded{
                std::move(owner), std::move(state), std::move(executor), std::move(components)};
            return finish(loaded, started);
        }
        case CudaBackbone::glm5_next: {
            auto started = std::chrono::steady_clock::now();
            auto definition = mfq::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.yarn, execution.yarn_context_extension,
                std::move(definition), mfq::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<glm5_next::Model>(
                glm5_next::read(source, definition, options.context_size));
            weight_loader::Loader weights{*source, execution.config.weights,
                {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
                 execution.tensor_parallel, moe_parallel_config(execution)},
                execution.weights, execution.tq};
            execution.dense_cpu_layer_count = 0;
            glm5_next::load(*owner, weights, execution.layer_placement, execution.n_gpu_layers);
            auto state = std::make_unique<glm5_next::State>(*owner);
            state->preallocate();
            auto executor = execution.make_executor();
            auto components = load_runtime_components(*owner, weights, load_optional_components,
                execution.projector(), execution.collectives, execution.profiler);
            Loaded<glm5_next::Model, glm5_next::State> loaded{
                std::move(owner), std::move(state), std::move(executor), std::move(components)};
            return finish(loaded, started);
        }
        case CudaBackbone::qwen4_exp:
            return load.template operator()<Qwen4CausalLm>();
        case CudaBackbone::deepseek_v4: {
            auto started = std::chrono::steady_clock::now();
            auto definition = mfq::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.yarn, execution.yarn_context_extension,
                std::move(definition), mfq::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<deepseek_v4::Model>(
                deepseek_v4::read(source, definition, options.context_size));
            deepseek_v4::validate_offload(owner->config, execution.dsv4_cpu_offload_layers);
            if (!execution.dsv4_cpu_offload_layers.empty()) {
                execution.dsv4_cpu_offload_host_bytes = 0;
                execution.drop_file_cache = true;
            }
            weight_loader::Loader weights{*source, execution.config.weights,
                {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
                 execution.tensor_parallel, moe_parallel_config(execution)},
                execution.weights, execution.tq};
            execution.dense_cpu_layer_count = 0;
            deepseek_v4::load(*owner, weights, execution.layer_placement, execution.collectives,
                model_parallel_primary_device(execution), execution.dsv4_cpu_offload_layers,
                execution.dsv4_cpu_offload_host_bytes, execution.n_gpu_layers);
            auto state = std::make_unique<deepseek_v4::State>(*owner);
            state->preallocate();
            auto executor = execution.make_executor();
            auto components = load_runtime_components(*owner, weights, load_optional_components);
            Loaded<deepseek_v4::Model, deepseek_v4::State> loaded{
                std::move(owner), std::move(state), std::move(executor), std::move(components)};
            return finish(loaded, started);
        }
        case CudaBackbone::deepseek_v41:
            return load.template operator()<DeepseekV41CausalLm>();
        case CudaBackbone::nemotron_h: {
            auto started = std::chrono::steady_clock::now();
            auto definition = mfq::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.yarn, execution.yarn_context_extension,
                std::move(definition), mfq::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<nemotron_h::Model>(
                nemotron_h::read(source, definition, options.context_size));
            weight_loader::Loader weights{*source, execution.config.weights,
                {execution.layer_placement.primary_device(), false, execution.drop_file_cache,
                 execution.tensor_parallel, moe_parallel_config(execution)},
                execution.weights, execution.tq};
            execution.dense_cpu_layer_count = 0;
            nemotron_h::load(*owner, weights, execution.pipelines, execution.layer_placement,
                             execution.n_gpu_layers, execution.expert_parallel.enabled());
            auto state = std::make_unique<nemotron_h::State>(*owner);
            state->preallocate();
            auto executor = execution.make_executor();
            auto components = load_runtime_components(*owner, weights, load_optional_components, execution);
            Loaded<nemotron_h::Model, nemotron_h::State> loaded{
                std::move(owner), std::move(state), std::move(executor), std::move(components)};
            return finish(loaded, started);
        }
        case CudaBackbone::unsupported:
            throw std::runtime_error("unsupported CUDA model backbone");
    }
    throw std::runtime_error("invalid CUDA model backbone");
};

} // namespace mfq::cuda
