#pragma once

#include "models/deepseek_v41/vision.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/radio_vision.h"
#include "models/qwen4_exp/ops.h"
#include "models/common/grid_vision.h"
#include "engine/executor/mtp.h"
#include "engine/executor/engine.h"
#include "cuda_model_plan.h"
#include "mfq_cuda_kv_quantization.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

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

    static constexpr bool streamed_kv() noexcept {
        return std::is_same_v<Model, mfq::cuda::Qwen4CausalLm>;
    }

    mfq::engine::Metrics metrics() const {
        std::optional<mfq::cuda::KvQuantization> quantization;
        if constexpr (streamed_kv()) quantization = mfq::cuda::cuda_kv_quantization();
        const auto component_state = state();
        return {
            {"kv_memory_supported", streamed_kv() ? 1.0 : 0.0},
            {"kv_quantization_supported", quantization.has_value() ? 1.0 : 0.0},
            {"kv_quantization_bits", quantization ? quantization->bits : 0.0},
            {"vision_declared", double(component_state.vision_declared)},
            {"vision_supported", double(component_state.vision_supported)},
            {"vision_available", double(component_state.vision_available)},
            {"mtp_declared", double(component_state.mtp_declared)},
            {"mtp_supported", double(component_state.mtp_supported)},
            {"mtp_available", double(component_state.mtp_available)}};
    }

    mfq::engine::Metrics profile_metrics(const Model& language) const {
        if constexpr (std::is_same_v<Model, mfq::cuda::Qwen4CausalLm>)
            return mfq::cuda::qwen4_decode_graph_profile_metrics(language, mtp.get());
        return {};
    }

    std::optional<mfq::engine::Metrics> memory_metrics(const Model& language) const {
        if constexpr (std::is_same_v<Model, mfq::cuda::Qwen4CausalLm>)
            return mfq::cuda::qwen4_cache_memory_metrics(language, mtp.get());
        return std::nullopt;
    }

    void describe(mfq::engine::EngineInfo& info) const {
        const auto component_state = state();
        const bool duplex = bool(composite);
        info.multimodal = component_state.vision_available || duplex;
        const auto caps = mfq::cuda::cuda_runtime_capabilities(
            graph, plan, component_state, info.multimodal, duplex);
        info.duplex = caps.full_duplex;
        info.capabilities = {graph.architecture, caps.text,
            caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
            caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
    }

    mfq::StepSequence<mfq::cuda::PreparedPrompt> prepare(
        Model& language, mfq::cuda::Projector projector,
        std::shared_ptr<MoeExpertCache> cache, mfq::cuda::Collectives& collectives,
        int primary, mfq::cuda::Executor* executor,
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media) {
        if (composite) return composite->prepare(tokens, media);
        if (grid_vision)
            return grid_vision->prepare(language, projector, std::move(cache),
                collectives, primary, executor, tokens, media);
        if constexpr (std::is_same_v<Model, mfq::cuda::DeepseekV41CausalLm>)
            if (deepseek_vision)
                return deepseek_vision->prepare(language, projector, tokens, media);
        if constexpr (std::is_same_v<Model, mfq::cuda::NemotronHCausalLm>)
            if (radio_vision)
                return radio_vision->prepare(language, projector, tokens, media);
        throw std::invalid_argument("model has no multimodal component");
    }

    template <class T>
    mfq::engine::ControlResult control(const T& value) {
        if (!composite)
            throw std::invalid_argument("model has no duplex component");
        if constexpr (std::is_same_v<T, MfqDuplexSessionParams>) {
            composite->start(value);
            return std::monostate{};
        } else if constexpr (std::is_same_v<T, MfqDuplexStepInput>) {
            return composite->step(value);
        } else {
            composite->stop();
            return std::monostate{};
        }
    }
};
