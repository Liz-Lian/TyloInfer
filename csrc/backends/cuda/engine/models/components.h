#pragma once

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/deepseek_v41/vision.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/ops.h"
#include "models/nemotron_h/radio_vision.h"
#include "models/qwen35/model.h"
#include "models/qwen4_exp/ops.h"
#include "models/common/grid_vision.h"
#include "engine/executor/mtp.h"
#include "cuda_model_plan.h"

#include <memory>
#include <optional>

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
        return mfq::cuda::cuda_component_state(
            graph, plan, vision_available, mtp_available);
    }
};

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
    Model& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);


RuntimeComponents<mfq::cuda::qwen35::Model>
load_runtime_components(
    mfq::cuda::qwen35::Model& model,
        const mfq::cuda::weight_loader::Loader& weights,
    CudaExecutionContext& execution,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::Qwen4CausalLm>
load_runtime_components(
    mfq::cuda::Qwen4CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::Glm5CausalLm>
load_runtime_components(
    mfq::cuda::Glm5CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::DeepseekV41CausalLm>
load_runtime_components(
    mfq::cuda::DeepseekV41CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::NemotronHCausalLm>
load_runtime_components(
    mfq::cuda::NemotronHCausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::MiniCPMO45CausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMO45CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

template <>
RuntimeComponents<mfq::cuda::MiniCPMOTtsCausalLm>
load_runtime_components(
    mfq::cuda::MiniCPMOTtsCausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);
