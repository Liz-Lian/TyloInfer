#pragma once

#include "models/deepseek_v41/vision.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/radio_vision.h"
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
