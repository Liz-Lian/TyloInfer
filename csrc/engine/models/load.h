#pragma once

#include "engine/load.h"
#include "engine/backends/cuda/native/scope.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/models/common/components.h"
#include "engine/models/qwen4_exp/components.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/glm5_next/components.h"
#include "engine/models/deepseek_v41/components.h"
#include "engine/models/nemotron_h/components.h"
#include "engine/models/common/load.h"
#include "engine/models/common/model_config.h"
#include "engine/models/common/model_graph.h"
#include "engine/models/qwen35/load.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/models/gemma4/load.h"
#include "engine/models/gemma4/state.h"
#endif
#include "engine/models/glm_dsa/load.h"
#include "engine/models/glm_dsa/state.h"
#include "engine/models/deepseek_v4/load.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/deepseek_v41/state.h"
#include "engine/models/glm5_next/load.h"
#include "engine/models/nemotron_h/load.h"
#include "engine/models/nemotron_h/state.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/state/kv_memory.h"
#include "engine/weights/cache.h"
#include "engine/models/qwen35/components.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/minicpmo45/state.h"
#include "engine/models/minicpmo45/text/load.h"
#include "engine/models/minicpmo45/text/state.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

namespace tyloi::cuda {

template <bool IncludeQwen = true>
inline constexpr auto with_model = []<typename F>(
        CudaExecutionContext& execution,
        const CudaLoadOptions& options,
        bool load_optional_components,
        F&& run) -> auto {
    CudaScope scope(execution.config.layer_placement.primary_device(), execution.executor().contexts());
    auto source = tyloi::open_model_source(options.model_path);
    auto finish = [&](auto& owner, auto& state, auto& executor, auto& components, auto started) {
        auto& model = *owner;
        using Model = std::remove_cvref_t<decltype(model)>;
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>)
            Executor::prepare(model.metadata.vocab_size);
        tyloi::cuda::finalize_kv(execution.kv.get(),
            execution.weights, execution.offload);
        std::size_t graph_reserve=0;
        if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
            if (execution.weights.needs_finalize()) {
                graph_reserve=qwen4_mtp_graph_reserve(model,components.mtp.get());
                if(graph_reserve)std::cerr<<"runtime_mtp_flexible_budget maximum_depth="
                    <<components.mtp->maximum_draft_depth()<<" bytes="<<graph_reserve
                    <<" initial_depth=3 lend_unused_to_experts=1\n";
            }
        }
        const auto loaded_at = tyloi::cuda::finish_load(
            execution.kv.get(), execution.weights,
            graph_reserve, execution.config.report_cuda_memory);
        return run(owner, state, executor, components, started, loaded_at, source);
    };
    auto load_tts = [&] {
        auto started = std::chrono::steady_clock::now();
        auto definition = tyloi::models::config_json(*source, options.config_path);
        definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension, std::move(definition),
            tyloi::models::graph(*source).backbone, options.context_size);
        auto owner = std::make_shared<minicpmo45::Model>(
            minicpmo45::read(source, definition, options.context_size));
        auto& model = *owner;
        if (execution.config.expert_parallel.enabled())
            throw std::runtime_error("--expert-parallel requires a model with routed experts");
        execution.config.layer_placement.prepare(model.config.num_hidden_layers);
        execution.config.dense_cpu_layer_count = execution.config.n_gpu_layers >= 0
            ? static_cast<int>(std::max<int64_t>(model.config.num_hidden_layers - execution.config.n_gpu_layers, 0)) : 0;
        if (execution.config.dense_cpu_layer_count > 0) {
            if (model_parallel_enabled(execution.config) || execution.config.layer_placement.enabled())
                throw std::runtime_error("--n-gpu-layers cannot be combined with tensor/expert/layer parallelism");
            std::cerr << "dense_layer_placement cpu=0-" << execution.config.dense_cpu_layer_count - 1
                << " gpu=" << execution.config.dense_cpu_layer_count << '-' << model.config.num_hidden_layers - 1
                << " cpu_threads=" << mfq_get_num_threads() << '\n';
        }
        auto weights = execution.weights.loader(*source, execution.config.weights,
            {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
             execution.config.tensor_parallel, moe_parallel_config(execution.config)});
        minicpmo45::load(model, weights, execution.config.layer_placement,
            execution.config.dense_cpu_layer_count, execution.config.yarn);
        minicpmo45::State state(model);
        if (::tyloi::cuda::options::kv_vram_preallocated()) state.preallocate();
        std::unique_ptr<minicpmo45::State> owned_state;
        auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
        std::monostate components;
        owned_state = std::make_unique<minicpmo45::State>(std::move(state));
        return finish(owner, owned_state, executor, components, started);
    };

    switch (cuda_backbone(tyloi::models::graph(*source).backbone)) {
        case CudaBackbone::generic_qwen:
            if constexpr (IncludeQwen) {
                auto started = std::chrono::steady_clock::now();
                auto weights = execution.weights.loader(*source, execution.config.weights,
                    {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                     execution.config.tensor_parallel, moe_parallel_config(execution.config)});
                std::shared_ptr<qwen35::Model> owner;
                std::unique_ptr<qwen35::State> state;
                auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
                const bool preallocate = ::tyloi::cuda::options::kv_vram_preallocated();
                owner = std::make_shared<qwen35::Model>(qwen35::load(
                    source, weights, execution.config.layer_placement, execution.config.yarn,
                    execution.config.dense_cpu_layer_count,
                    tyloi::cuda::qwen35::LoadOptions{
                        .config_path = options.config_path,
                        .context_size = options.context_size,
                        .gpu_layers = execution.config.n_gpu_layers,
                        .extend_yarn = execution.config.yarn_context_extension,
                        .expert_parallel = execution.config.expert_parallel.enabled(),
                        .mirror_scalars = execution.config.tensor_parallel_mirror_linear_attention_scalars}));
                state = std::make_unique<qwen35::State>(*owner, execution.executor().mode().transposed);
                if (preallocate) state->preallocate();
                auto components = qwen35::Components::load(
                    *owner, weights, execution.weights, execution.config.layer_placement, execution.config.dense_cpu_layer_count,
                    *executor, execution.executor().collectives(),
                    execution.executor().mmq().kv_cache_capacity, execution.executor().mode().trace,
                    execution.executor().mode().trace_layer, load_optional_components,
                    !execution.config.dsv4_cpu_offload_layers.empty(), preallocate);
                return finish(owner, state, executor, components, started);
            } else {
                throw std::runtime_error("unsupported CUDA model backbone");
            }
        case CudaBackbone::minicpmo45: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<minicpmo45::text::Model>(
                minicpmo45::text::read(source, definition, options.context_size));
            auto weights = execution.weights.loader(*source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            execution.config.dense_cpu_layer_count = execution.config.n_gpu_layers >= 0
                ? static_cast<int>(std::max<int64_t>(owner->config.num_hidden_layers - execution.config.n_gpu_layers, 0)) : 0;
            minicpmo45::text::load(*owner, weights, execution.config.layer_placement,
                execution.config.n_gpu_layers, execution.config.expert_parallel.enabled(), true, true, execution.config.yarn);
            auto state = std::make_unique<minicpmo45::text::State>(*owner);
            state->preallocate();
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            auto components = minicpmo45::load_components(*owner, *state, *executor, weights,
                load_optional_components, execution);
            return finish(owner, state, executor, components, started);
        }
        case CudaBackbone::minicpmo_tts:
            return load_tts();
        case CudaBackbone::gemma4: {
#ifdef TYLOINFER_NATIVE_CUDA
            throw std::runtime_error("unsupported CUDA model backbone");
#else
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<gemma4::Model>(
                gemma4::read(source, definition, options.context_size));
            auto weights = execution.weights.loader(*source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            execution.config.dense_cpu_layer_count = 0;
            gemma4::load(*owner, weights, execution.config.layer_placement, execution.config.n_gpu_layers,
                         execution.config.expert_parallel.enabled());
            auto state = std::make_unique<gemma4::State>(*owner);
            state->preallocate();
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            if (load_optional_components && (owner->plan.vision != CudaVisionAdapter::none ||
                    owner->plan.predictor != CudaPredictorAdapter::none))
                throw std::runtime_error("CUDA component adapter is unsupported for this model");
            std::monostate components;
            return finish(owner, state, executor, components, started);
#endif
        }
        case CudaBackbone::glm_dsa: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<glm_dsa::Model>(
                glm_dsa::read(source, definition, options.context_size));
            auto weights = execution.weights.loader(*source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            execution.config.dense_cpu_layer_count = 0;
            glm_dsa::load(*owner, weights, execution.config.layer_placement,
                          execution.config.n_gpu_layers, true, true, execution.config.yarn);
            auto state = std::make_unique<glm_dsa::State>(*owner);
            state->preallocate();
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            if (load_optional_components && (owner->plan.vision != CudaVisionAdapter::none ||
                    owner->plan.predictor != CudaPredictorAdapter::none))
                throw std::runtime_error("CUDA component adapter is unsupported for this model");
            std::monostate components;
            return finish(owner, state, executor, components, started);
        }
        case CudaBackbone::glm5_next: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<glm5_next::Model>(
                glm5_next::read(source, definition, options.context_size));
            auto weights = execution.weights.loader(*source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            execution.config.dense_cpu_layer_count = 0;
            glm5_next::load(*owner, weights, execution.config.layer_placement, execution.config.n_gpu_layers);
            auto state = std::make_unique<glm5_next::State>(*owner);
            state->preallocate();
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            auto components = glm5_next::load_components(*owner, weights, load_optional_components,
                execution.executor().projector(execution.executor().mode().ffn, model_parallel_primary_device(execution.config)), execution.executor().collectives(), execution.executor().profiler());
            return finish(owner, state, executor, components, started);
        }
        case CudaBackbone::qwen4_exp: {
            auto started = std::chrono::steady_clock::now();
            auto owner = std::make_shared<qwen4_exp::Model>();
            qwen4_exp::read(*owner, source, tyloi::models::config_json(*source, options.config_path),
                            options.context_size);
            owner->execution = &execution;
            qwen4_exp::load(*owner, execution, options.context_size, true, true);
            auto weights = execution.weights.loader(*owner->source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            auto components = qwen4_exp::load_components(*owner, weights, load_optional_components);
            auto state = std::unique_ptr<qwen4_exp::State>(static_cast<qwen4_exp::State*>(
                owner->make_state().release()));
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            return finish(owner, state, executor, components, started);
        }
        case CudaBackbone::deepseek_v4: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<deepseek_v4::Model>(
                deepseek_v4::read(source, definition, options.context_size));
            deepseek_v4::validate_offload(owner->config, execution.config.dsv4_cpu_offload_layers);
            if (!execution.config.dsv4_cpu_offload_layers.empty()) {
                execution.config.dsv4_cpu_offload_host_bytes = 0;
                execution.config.drop_file_cache = true;
            }
            auto weights = execution.weights.loader(*source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            execution.config.dense_cpu_layer_count = 0;
            deepseek_v4::load(*owner, weights, execution.config.layer_placement, execution.executor().collectives(),
                model_parallel_primary_device(execution.config), execution.config.dsv4_cpu_offload_layers,
                execution.config.dsv4_cpu_offload_host_bytes, execution.config.n_gpu_layers);
            auto state = std::make_unique<deepseek_v4::State>(*owner);
            state->preallocate();
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            if (load_optional_components && (owner->plan.vision != CudaVisionAdapter::none ||
                    owner->plan.predictor != CudaPredictorAdapter::none))
                throw std::runtime_error("CUDA component adapter is unsupported for this model");
            std::monostate components;
            return finish(owner, state, executor, components, started);
        }
        case CudaBackbone::deepseek_v41: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            auto owner = std::make_shared<deepseek_v41::Model>();
            deepseek_v41::read(*owner, source, definition, options.context_size);
            auto weights = execution.weights.loader(*owner->source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            deepseek_v41::load(*owner, execution, options.context_size, false, true, false);
            auto components = deepseek_v41::load_components(*owner, weights, load_optional_components);
            auto state = std::unique_ptr<deepseek_v41::State>(static_cast<deepseek_v41::State*>(
                owner->make_state().release()));
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            return finish(owner, state, executor, components, started);
        }
        case CudaBackbone::nemotron_h: {
            auto started = std::chrono::steady_clock::now();
            auto definition = tyloi::models::config_json(*source, options.config_path);
            definition = configure_yarn(execution.config.yarn, execution.config.yarn_context_extension,
                std::move(definition), tyloi::models::graph(*source).backbone, options.context_size);
            auto owner = std::make_shared<nemotron_h::Model>(
                nemotron_h::read(source, definition, options.context_size));
            auto weights = execution.weights.loader(*source, execution.config.weights,
                {execution.config.layer_placement.primary_device(), false, execution.config.drop_file_cache,
                 execution.config.tensor_parallel, moe_parallel_config(execution.config)});
            execution.config.dense_cpu_layer_count = 0;
            nemotron_h::load(*owner, weights, execution.executor().pipelines(), execution.config.layer_placement,
                             execution.config.n_gpu_layers, execution.config.expert_parallel.enabled());
            auto state = std::make_unique<nemotron_h::State>(*owner);
            state->preallocate();
            auto executor = execution.executor().fork(model_parallel_primary_device(execution.config));
            auto components = nemotron_h::load_components(*owner, weights, load_optional_components, execution);
            return finish(owner, state, executor, components, started);
        }
        case CudaBackbone::unsupported:
            throw std::runtime_error("unsupported CUDA model backbone");
    }
    throw std::runtime_error("invalid CUDA model backbone");
};

} // namespace tyloi::cuda
