#pragma once

#include "engine/models/minicpmo45/components.h"
#include "engine/models/minicpmo45/text/model.h"
#include "engine/models/nemotron_h/radio_vision.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/models/common/vision/grid_encoder.h"
#include "engine/models/deepseek_v4/model.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/gemma4/model.h"
#include "engine/models/causal_model.h"
#include "engine/models/glm5_next/model.h"
#include "engine/models/glm_dsa/model.h"
#include "engine/models/minicpmo45/model.h"
#include "engine/models/nemotron_h/model.h"
#include "engine/models/common/mtp.h"
#include "engine/engine.h"
#include "engine/models/common/capabilities.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

struct CudaExecutionContext;

template <class Model, class Parts>
void describe_model(const Model& language, const Parts& components, tyloi::engine::EngineInfo& info) {
    if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(language); }) {
        info.capabilities = static_cast<const tyloi::models::CausalModel&>(language).capabilities();
        const auto& caps = info.capabilities;
        info.duplex = caps.full_duplex;
        info.multimodal = caps.image_input || caps.video_input || caps.audio_input ||
            caps.audio_output || caps.full_duplex;
    } else {
        components.describe(info);
    }
}

template <typename Model>
struct RuntimeComponents {
    tyloi::ModelGraph graph;
    tyloi::cuda::CudaModelPlan plan;
    std::optional<
        tyloi::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::optional<tyloi::cuda::nemotron_h::CudaRadioPromptComponent>
        radio_vision;
    std::unique_ptr<tyloi::models::Mtp> mtp;
    std::unique_ptr<tyloi::cuda::minicpmo45::Components> composite;
    bool vision_available = false;
    bool mtp_available = false;

    tyloi::cuda::CudaComponentState state() const noexcept {
        return tyloi::cuda::cuda_component_state(
            graph, plan, vision_available, mtp_available);
    }

    static constexpr bool streamed_kv() noexcept {
        return std::is_same_v<Model, tyloi::cuda::qwen4_exp::Model>;
    }

    tyloi::engine::Metrics metrics() const {
        std::optional<tyloi::cuda::KvQuantization> quantization;
        if constexpr (streamed_kv()) quantization = tyloi::cuda::cuda_kv_quantization();
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

    tyloi::engine::Metrics profile_metrics(const Model& language) const {
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen4_exp::Model>)
            return tyloi::cuda::qwen4_decode_graph_profile_metrics(language, mtp.get());
        return {};
    }

    std::optional<tyloi::engine::Metrics> memory_metrics(const Model& language) const {
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen4_exp::Model>)
            return tyloi::cuda::qwen4_cache_memory_metrics(language, mtp.get());
        return std::nullopt;
    }

    void describe(tyloi::engine::EngineInfo& info) const {
        const auto component_state = state();
        const bool duplex = bool(composite);
        info.multimodal = component_state.vision_available || duplex;
        const auto caps = tyloi::cuda::cuda_runtime_capabilities(
            graph, plan, component_state, info.multimodal, duplex);
        info.duplex = caps.full_duplex;
        info.capabilities = {graph.architecture, caps.text,
            caps.image_input, caps.video_input, caps.audio_input, caps.audio_output,
            caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
    }

    tyloi::StepSequence<tyloi::cuda::PreparedPrompt> prepare(
        Model& language, tyloi::cuda::MediaCache& images,
#ifndef TYLOINFER_NATIVE_CUDA
        tyloi::cuda::Projector projector, tyloi::cuda::Collectives& collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA
        tyloi::cuda::Executor* executor,
#endif
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media) {
        if (composite) return composite->prepare(tokens, media);
#ifdef TYLOINFER_NATIVE_CUDA
        if constexpr (std::is_same_v<Model, tyloi::cuda::qwen4_exp::Model>)
#endif
        if (grid_vision)
            return grid_vision->prepare(language, images,
#ifndef TYLOINFER_NATIVE_CUDA
                projector, collectives,
#endif
                primary,
#ifndef TYLOINFER_NATIVE_CUDA
                executor,
#endif
                tokens, media);
        if constexpr (std::is_same_v<Model, tyloi::cuda::nemotron_h::Model>)
            if (radio_vision)
                return radio_vision->prepare(language,
#ifndef TYLOINFER_NATIVE_CUDA
                                             projector, *executor, collectives,
#endif
                                             primary, tokens, media);
        throw std::invalid_argument("model has no multimodal component");
    }

    template <class T>
    tyloi::engine::ControlResult control(const T& value) {
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

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
    Model& model,
        const tyloi::weights::Loader& weights,
    bool load_optional_components);


template <>
RuntimeComponents<tyloi::cuda::qwen4_exp::Model>
load_runtime_components(
    tyloi::cuda::qwen4_exp::Model& model,
        const tyloi::weights::Loader& weights,
    bool load_optional_components);

RuntimeComponents<tyloi::cuda::glm5_next::Model>
load_runtime_components(
    tyloi::cuda::glm5_next::Model& model,
    const tyloi::weights::Loader& weights,
    bool load_optional_components, tyloi::cuda::Projector projector,
    tyloi::cuda::Collectives& copies, CudaProfiler& profiler);

template <>
RuntimeComponents<tyloi::cuda::deepseek_v41::Model>
load_runtime_components(
    tyloi::cuda::deepseek_v41::Model& model,
        const tyloi::weights::Loader& weights,
    bool load_optional_components);

RuntimeComponents<tyloi::cuda::nemotron_h::Model>
load_runtime_components(
    tyloi::cuda::nemotron_h::Model& model,
        const tyloi::weights::Loader& weights,
    bool load_optional_components,
    CudaExecutionContext& execution);


RuntimeComponents<tyloi::cuda::minicpmo45::text::Model>
load_runtime_components(
    tyloi::cuda::minicpmo45::text::Model& model,
    tyloi::cuda::minicpmo45::text::State& state,
    tyloi::cuda::Executor& executor,
    const tyloi::weights::Loader& weights,
    bool load_optional_components, CudaExecutionContext& execution);

template <>
RuntimeComponents<tyloi::cuda::minicpmo45::Model>
load_runtime_components(
    tyloi::cuda::minicpmo45::Model& model,
        const tyloi::weights::Loader& weights,
    bool load_optional_components);
