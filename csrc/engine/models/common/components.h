#pragma once

#ifndef TYLOINFER_NATIVE_CUDA

#include "engine/engine.h"
#include "engine/models/causal_model.h"
#include "engine/models/metadata.h"
#include "engine/models/common/mtp.h"
#include "engine/models/minicpmo45/components.h"
#include "engine/backends/cuda/kernels/mfq_cuda_kv_quantization.h"

#include <memory>
#include <type_traits>
#include <variant>

namespace tyloi::cuda {

/** Borrow predictor weights from the public model or its remaining assembly owner. */
template <class Model, class Parts>
tyloi::models::Mtp* component_predictor(const Model& model, const Parts& parts) {
    if constexpr (std::is_same_v<Parts, std::monostate> &&
                  std::is_base_of_v<tyloi::models::CausalModel, Model>)
        return model.mtp();
    else if constexpr (requires { static_cast<tyloi::models::Mtp*>(parts.get()); })
        return parts.get();
    else if constexpr (requires { static_cast<tyloi::models::Mtp*>(parts.mtp.get()); })
        return parts.mtp.get();
    else
        return nullptr;
}

template <class Model, class Parts>
bool component_vision(const Model& model, const Parts& parts) {
    if constexpr (requires { parts.grid_vision; }) return bool(parts.grid_vision);
    else if constexpr (requires { parts.radio_vision; }) return bool(parts.radio_vision);
    else if constexpr (std::is_same_v<Parts, std::unique_ptr<minicpmo45::Components>>) return bool(parts);
    else if constexpr (requires { model.vision; }) return bool(model.vision);
    else return false;
}

/** Describe assembled capabilities through the public model contract when no optional parts are held. */
template <class Model, class Parts>
void describe_model(const Model& model, const Parts& parts, tyloi::engine::EngineInfo& info) {
    if constexpr (std::is_same_v<Parts, std::monostate> &&
                  std::is_base_of_v<tyloi::models::CausalModel, Model>) {
        info.capabilities = model.capabilities();
        const auto& caps = info.capabilities;
        info.multimodal = caps.image_input || caps.video_input || caps.audio_input ||
            caps.audio_output || caps.full_duplex;
        info.duplex = caps.full_duplex;
    } else if constexpr (requires { parts.describe(info); }) {
        parts.describe(info);
    } else {
        const bool vision = component_vision(model, parts);
        const bool duplex = std::is_same_v<Parts, std::unique_ptr<minicpmo45::Components>> && vision;
        const auto state = ::tyloi::models::component_state(model.graph, model.plan, vision, component_predictor(model, parts));
        const auto caps = ::tyloi::models::runtime_capabilities(model.graph, model.plan, state, vision || duplex, duplex);
        info.multimodal = vision || duplex;
        info.duplex = caps.full_duplex;
        info.capabilities = {model.graph.architecture, caps.text, caps.image_input, caps.video_input,
            caps.audio_input, caps.audio_output, caps.full_duplex, caps.mtp, "model-graph+cuda-adapters"};
    }
}

/** Collect model-owned capability metrics and any remaining optional-part accounting. */
template <class Model, class Parts>
tyloi::engine::Metrics component_metrics(const Model& model, const Parts& parts) {
    if constexpr (std::is_same_v<Parts, std::monostate> &&
                  std::is_base_of_v<tyloi::models::CausalModel, Model>) {
        return model.metrics();
    } else if constexpr (requires { parts.metrics(); }) {
        return parts.metrics();
    } else {
        tyloi::engine::Metrics metrics;
        if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(model); })
            metrics = static_cast<const tyloi::models::CausalModel&>(model).metrics();
        else
            metrics = {{"kv_memory_supported", 0.0}, {"kv_quantization_supported", 0.0},
                       {"kv_quantization_bits", 0.0}};
        const auto state = ::tyloi::models::component_state(model.graph, model.plan,
            component_vision(model, parts), component_predictor(model, parts));
        const auto set = [&](const char* name, double value) {
            for (auto& metric : metrics) if (metric.first == name) { metric.second = value; return; }
            metrics.emplace_back(name, value);
        };
        if constexpr (requires { Parts::streamed_kv(); }) {
            if constexpr (Parts::streamed_kv()) {
                const auto quantization = cuda_kv_quantization();
                set("kv_memory_supported", 1.0);
                set("kv_quantization_supported", 1.0);
                set("kv_quantization_bits", double(quantization.bits));
            }
        }
        set("vision_declared", double(state.vision_declared));
        set("vision_supported", double(state.vision_supported));
        set("vision_available", double(state.vision_available));
        set("mtp_declared", double(state.mtp_declared));
        set("mtp_supported", double(state.mtp_supported));
        set("mtp_available", double(state.mtp_available));
        return metrics;
    }
}

} // namespace tyloi::cuda
#endif
