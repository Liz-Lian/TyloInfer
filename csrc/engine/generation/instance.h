#pragma once

#include "engine/text.h"
#include "engine/generation/generation.h"

namespace tyloi::cuda::engine_detail {

template <class M, class S, class P>
tyloi::StepSequence<tyloi::models::Prepared> Text<M, S, P>::prepare(InferenceRequest& input, State& state) {
#ifdef TYLOINFER_NATIVE_CUDA
    if constexpr (std::is_same_v<Parts, std::monostate> &&
                  std::is_base_of_v<tyloi::models::CausalModel, Model>) {
        const tyloi::models::CausalModel& causal = language;
        return causal.prepare(input.prompt, *input.vision, state);
    } else
#endif
    if constexpr (std::is_same_v<Parts, std::unique_ptr<tyloi::cuda::minicpmo45::Components>>) {
        if (!components) throw std::invalid_argument("model has no multimodal component");
        return components->prepare(input.prompt, *input.vision);
    } else if constexpr (std::is_same_v<Model, qwen4_exp::Model> ||
#ifndef TYLOINFER_NATIVE_CUDA
                         std::is_same_v<Model, nemotron_h::Model> ||
#endif
                         std::is_same_v<Model, qwen35::Model>) {
        return components.prepare(language, images,
#ifndef TYLOINFER_NATIVE_CUDA
        execution->executor().projector(execution->executor().mode().ffn, model_parallel_primary_device(execution->config)),
        execution->executor().collectives(),
#endif
        execution->config.layer_placement.primary_device(),
#ifndef TYLOINFER_NATIVE_CUDA
        executor.get(),
#endif
        input.prompt, *input.vision);
    }
#ifdef TYLOINFER_NATIVE_CUDA
    else if constexpr (requires { static_cast<const tyloi::models::CausalModel&>(language); }) {
        const tyloi::models::CausalModel& causal = language;
        return causal.prepare(input.prompt, *input.vision, state);
    }
#endif
    else throw std::invalid_argument("model has no multimodal component");
}

template <class M, class S, class P>
Generation Text<M, S, P>::generate_text(InferenceRequest& input, InferenceOutput& output, State& state,
                                       std::optional<Prepared> prepared, const RequestId& id, bool batched) {
    return internal::generate(
        language, state, executor.get(), execution->executor().collectives(),
        execution->executor().forward_mode(execution->config.layer_placement.primary_device()), execution->executor().profiler(), weights, *budget,
        execution->executor().mode().ffn.serial, execution->config.report_cuda_memory,
        execution->config.dsv4_cpu_offload_layers.empty() && execution->config.dense_cpu_layer_count == 0 &&
            model_parallel_cuda_graph_enabled(execution->config, execution->executor(), execution->weights),
        [this] {
            drain_cuda(root_executor, weights, offload.get(), execution->config.layer_placement,
                       execution->config.tensor_parallel, execution->config.expert_parallel);
        },
        info, source.get(), graph, cache, config, input, output,
        mtp_state,
        batched ? nullptr : component_predictor(language, components), std::move(prepared),
        batched ? batching.get() : nullptr, id);
}

template <class M, class S, class P>
Generation Text<M, S, P>::generate(const RequestId& id, InferenceRequest& input, InferenceOutput& output, bool batched) {
    if (batched) {
        graph.invalidate();
        if constexpr (std::is_same_v<Model,qwen4_exp::Model>) {
            // Exclusive MTP generation has ended before a batch starts.
            // Its graph consumers must retire before idle batch buffers.
            qwen4_release_decode_graphs(language,state);
            if (auto* predictor=component_predictor(language,components)) predictor->reset(predictor_state(), 1);
        }
    }
    return generate_prepared(std::ref(*this), state, *executor, input, output, id, batched);
}

} // namespace tyloi::cuda::engine_detail
