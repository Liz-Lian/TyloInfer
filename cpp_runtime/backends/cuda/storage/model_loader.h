#pragma once

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/deepseek_v41/vision.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/ops.h"
#include "models/qwen35/ops.h"
#include "models/qwen4_exp/ops.h"
#include "core/mtp.h"
#include "tyloinfer/gguf_model_source.h"
#include <cstdlib>
#include "core/grid_vision_component.h"
#include "models/nemotron_h/radio_vision.h"
#include "cuda_runtime_config.h"
#include "storage/weight_loader.h"
#include "storage/moe_expert_cache.h"
#include "runtime/kv_memory.h"

#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

template <typename Model>
struct RuntimeComponents {
    mfq::ModelGraph graph;
    mfq::cuda::CudaModelPlan plan;
    std::optional<
        mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::optional<mfq::cuda::deepseek_v41_runtime::Vision> deepseek_vision;
    std::optional<mfq::cuda::nemotron_h::CudaRadioPromptComponent>
        radio_vision;
    std::unique_ptr<MtpModule> mtp;
    Model* language_override = nullptr;
    std::unique_ptr<mfq::cuda::minicpmo45::Components> composite;
    bool vision_available = false;
    bool mtp_available = false;

    Model& language(Model& fallback) {
        return language_override ? *language_override : fallback;
    }

    mfq::cuda::CudaComponentState state() const noexcept {
        auto result = mfq::cuda::cuda_component_state(
            graph, plan, vision_available, mtp_available);
        // A separately loaded draft model need not be declared in the target
        // checkpoint. Its actual loaded module determines runtime support.
        if (mtp) { result.mtp_supported = true; result.mtp_available = mtp_available; }
        return result;
    }
};

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
    Model& model,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::Qwen35CausalLm>
load_runtime_components(
    mfq::cuda::Qwen35CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::Qwen4CausalLm>
load_runtime_components(
    mfq::cuda::Qwen4CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::Glm5CausalLm>
load_runtime_components(
    mfq::cuda::Glm5CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::DeepseekV41CausalLm>
load_runtime_components(
    mfq::cuda::DeepseekV41CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::NemotronHCausalLm>
load_runtime_components(
    mfq::cuda::NemotronHCausalLm& model,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMO45CausalLm& model,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMOTtsCausalLm& model,
    bool load_optional_components);

namespace mfq::cuda::internal {

void prepare_sampling_runtime(int64_t vocabulary);

template <typename F>
auto with_loaded_cuda_model(
        CudaExecutionContext& execution,
        const CudaLoadOptions& options,
        bool load_optional_components,
        F&& run) {
    std::shared_ptr<const mfq::ModelSource> source;
    const auto* native_mtp=std::getenv("TYLOINFER_MTP_NATIVE_GGUF");
    if(native_mtp && *native_mtp) {
        if(!execution.dflash2_draft_path.empty())
            throw std::invalid_argument("native MTP and DFlash2 drafts cannot both be selected");
        if(!mfq::is_gguf_file(options.model_path))
            throw std::invalid_argument("an external MTP GGUF requires a GGUF target");
        source=std::make_shared<mfq::GgufModelSource>(options.model_path,
            std::vector<std::filesystem::path>{std::filesystem::path(native_mtp)});
        if(!source->find_tensor("predictor.embedding_norm.weight"))
            throw std::invalid_argument("the MTP GGUF companion has no canonical predictor tensors");
    } else source=mfq::open_model_source(options.model_path);
    if(!execution.dflash2_draft_path.empty() &&
            cuda_backbone(source->resolved_model_graph().backbone)!=CudaBackbone::qwen4_exp)
        throw std::invalid_argument("DFlash2 requires a Flash-Next target model");
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
        if constexpr(std::is_same_v<Model,Qwen4CausalLm>)
            prepare_sampling_runtime(model.vocab_size());
        if(execution.kv_memory) {
            execution.kv_memory->finalize(moe_expert_cache_minimum_bytes(execution.moe_expert_cache));
            set_moe_kv_expert_loan(execution.moe_expert_cache,execution.kv_memory->expert_loan());
            if(auto offload=execution.kv_memory->offload())execution.qsa_kv_store=std::move(offload);
        }
        if (moe_expert_cache_has_sources(execution.moe_expert_cache) &&
                !moe_expert_cache_finalized(execution.moe_expert_cache)) {
            std::size_t graph_reserve=0;
            if constexpr(std::is_same_v<Model,Qwen4CausalLm>) {
                graph_reserve=qwen4_mtp_graph_reserve(model,runtime_components.mtp.get());
                if(graph_reserve)std::cerr<<"runtime_mtp_flexible_budget maximum_depth="
                    <<runtime_components.mtp->maximum_draft_depth()<<" bytes="<<graph_reserve
                    <<" initial_depth=3 lend_unused_to_experts=1\n";
            }
            finalize_moe_expert_cache(execution.moe_expert_cache,
                (execution.kv_memory?execution.kv_memory->growth_reserve():0)+graph_reserve,
                execution.kv_memory?execution.kv_memory->expert_budget_charge():0);
        }
        finalize_gguf_expert_cache(execution.gguf_expert_cache,
            execution.kv_memory ? execution.kv_memory->growth_reserve() : 0);
        mfq_cuda_synchronize();
        auto loaded = std::chrono::steady_clock::now();
        report_cuda_memory(execution.config, "loaded");
        return run(model, runtime_components, started, loaded);
    };

    switch (cuda_backbone(source->resolved_model_graph().backbone)) {
        case CudaBackbone::generic_qwen:
            return load.template operator()<Qwen35CausalLm>();
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
