#pragma once

#include "backends/cuda/engine/models/components.h"
#include "engine/executor/qwen35.h"
#include "backends/cuda/engine/models/qwen35/load.h"
#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/state/kv_memory.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "cuda_runtime_config.h"

#include <chrono>
#include <iostream>
#include <type_traits>
#include <utility>

namespace mfq::cuda::internal {

template <typename F>
auto with_loaded_cuda_model(
        CudaExecutionContext& execution,
        const CudaLoadOptions& options,
        bool load_optional_components,
        F&& run) {
    CudaScope scope(execution);
    auto source = mfq::open_model_source(options.model_path);
    auto finish = [&](auto& model, auto& runtime_components,
                      auto started) {
        using Model = std::decay_t<decltype(model)>;
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            prepare_sampling_runtime(model.vocab_size());
        if(execution.stores.kv_memory) {
            execution.stores.kv_memory->finalize(moe_expert_cache_minimum_bytes(execution.stores.moe_expert_cache));
            set_moe_kv_expert_loan(execution.stores.moe_expert_cache,execution.stores.kv_memory->expert_loan());
            if(auto offload=execution.stores.kv_memory->offload())execution.stores.qsa_kv_store=std::move(offload);
        }
        if (moe_expert_cache_has_sources(execution.stores.moe_expert_cache) &&
                !moe_expert_cache_finalized(execution.stores.moe_expert_cache)) {
            std::size_t graph_reserve=0;
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                graph_reserve=qwen4_mtp_graph_reserve(model,runtime_components.mtp.get());
                if(graph_reserve)std::cerr<<"runtime_mtp_flexible_budget maximum_depth="
                    <<runtime_components.mtp->maximum_draft_depth()<<" bytes="<<graph_reserve
                    <<" initial_depth=3 lend_unused_to_experts=1\n";
            }
            finalize_moe_expert_cache(execution.stores.moe_expert_cache,
                (execution.stores.kv_memory?execution.stores.kv_memory->growth_reserve():0)+graph_reserve,
                execution.stores.kv_memory?execution.stores.kv_memory->expert_budget_charge():0);
        }
        mfq_cuda_synchronize();
        const auto loaded = std::chrono::steady_clock::now();
        report_cuda_memory(execution.config, "loaded");
        return run(model, runtime_components, started, loaded);
    };
    auto load = [&]<typename Model>() {
        auto started = std::chrono::steady_clock::now();
        Model model = load_causal_lm<Model>(
            execution,
            options.model_path,
            options.config_path,
            options.context_size,
            true,
            load_optional_components,
            source);
        auto runtime_components =
            load_runtime_components(model, load_optional_components);
        return finish(model, runtime_components, started);
    };
    auto load_qwen = [&] {
        auto started = std::chrono::steady_clock::now();
        auto loaded_model = mfq::cuda::qwen35::load(
            execution, options.model_path, options.config_path, options.context_size,
            true, true, source);
        auto weights = std::make_shared<const mfq::cuda::qwen35::Model>(
            std::move(loaded_model));
        mfq::cuda::qwen35::Run model(std::move(weights), execution);
        model.preallocate();
        auto runtime_components =
            load_runtime_components(model, load_optional_components);
        return finish(model, runtime_components, started);
    };

    switch (cuda_backbone(source->resolved_model_graph().backbone)) {
        case CudaBackbone::generic_qwen:
            return load_qwen();
        case CudaBackbone::minicpmo45:
            return load.template operator()<MiniCPMO45CausalLm>();
        case CudaBackbone::minicpmo_tts:
            return load.template operator()<MiniCPMOTtsCausalLm>();
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
