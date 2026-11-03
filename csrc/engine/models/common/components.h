#pragma once

#include "engine/models/deepseek_v41/vision.h"
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
void describe_model(const Model& language, const Parts& components, mfq::engine::EngineInfo& info) {
    if constexpr (requires { static_cast<const mfq::models::CausalModel&>(language); }) {
        info.capabilities = static_cast<const mfq::models::CausalModel&>(language).capabilities();
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
    mfq::ModelGraph graph;
    mfq::cuda::CudaModelPlan plan;
    std::optional<
        mfq::cuda::grid_vision_runtime::CudaGridVisionPromptComponent>
        grid_vision;
    std::optional<mfq::cuda::deepseek_v41_runtime::Vision> deepseek_vision;
    std::optional<mfq::cuda::nemotron_h::CudaRadioPromptComponent>
        radio_vision;
    std::unique_ptr<MtpModule> mtp;
    std::unique_ptr<mfq::cuda::minicpmo45::Components> composite;
    bool vision_available = false;
    bool mtp_available = false;

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
        Model& language, mfq::cuda::MediaCache& images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        mfq::cuda::Projector projector, mfq::cuda::Collectives& collectives,
#endif
        int primary,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        mfq::cuda::Executor* executor,
#endif
        const std::vector<int64_t>& tokens, const MfqMultimodalInput& media) {
        if (composite) return composite->prepare(tokens, media);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if constexpr (std::is_same_v<Model, mfq::cuda::Qwen4CausalLm>)
#endif
        if (grid_vision)
            return grid_vision->prepare(language, images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                projector, collectives,
#endif
                primary,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                executor,
#endif
                tokens, media);
        if constexpr (std::is_same_v<Model, mfq::cuda::DeepseekV41CausalLm>)
            if (deepseek_vision)
                return deepseek_vision->prepare(language, images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                    projector,
#endif
                    tokens, media);
        if constexpr (std::is_same_v<Model, mfq::cuda::nemotron_h::Model>)
            if (radio_vision)
                return radio_vision->prepare(language,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                                             projector, *executor, collectives,
#endif
                                             primary, tokens, media);
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

template <typename Model>
RuntimeComponents<Model> load_runtime_components(
    Model& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);


template <>
RuntimeComponents<mfq::cuda::Qwen4CausalLm>
load_runtime_components(
    mfq::cuda::Qwen4CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

RuntimeComponents<mfq::cuda::glm5_next::Model>
load_runtime_components(
    mfq::cuda::glm5_next::Model& model,
    const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components, mfq::cuda::Projector projector,
    mfq::cuda::Collectives& copies, CudaProfiler& profiler);

template <>
RuntimeComponents<mfq::cuda::DeepseekV41CausalLm>
load_runtime_components(
    mfq::cuda::DeepseekV41CausalLm& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);

RuntimeComponents<mfq::cuda::nemotron_h::Model>
load_runtime_components(
    mfq::cuda::nemotron_h::Model& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components,
    CudaExecutionContext& execution);


RuntimeComponents<mfq::cuda::minicpmo45::text::Model>
load_runtime_components(
    mfq::cuda::minicpmo45::text::Model& model,
    mfq::cuda::minicpmo45::text::State& state,
    mfq::cuda::Executor& executor,
    const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components, CudaExecutionContext& execution);

template <>
RuntimeComponents<mfq::cuda::minicpmo45::Model>
load_runtime_components(
    mfq::cuda::minicpmo45::Model& model,
        const mfq::cuda::weight_loader::Loader& weights,
    bool load_optional_components);
