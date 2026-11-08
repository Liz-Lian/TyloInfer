#pragma once

#include "engine/text.h"
#include "engine/generation/generation.h"

namespace tyloi::cuda::engine_detail {

template <class M, class S, class P>
void Text<M, S, P>::configure_decode_budget() {
    std::size_t spare = 0;
#ifdef TYLOINFER_NATIVE_CUDA
    if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
        if (components.mtp) {
            const auto maximum=qwen4_mtp_graph_reserve(language,components.mtp.get());
            const auto active=qwen4_mtp_graph_reserve(language,components.mtp.get(),graph_draft_depth);
            if(active<maximum)spare=maximum-active;
        }
    }
#endif
    weights.configure_decode(spare,graph_draft_depth);
}

template <class M, class S, class P>
void Text<M, S, P>::prepare_graph_depth(const InferenceRequest& input) {
    if constexpr(std::is_same_v<Model,qwen4_exp::Model>) {
        if(!components.mtp)return;
        const auto* warmup=tyloi::cuda::env_lookup("TYLOINFER_QWEN4_WARMUP_GRAPHS");
        if(!state.reuse_decode_graphs || !execution->config.cache.pipeline || (warmup && warmup[0]=='0'))return;
        const auto& sampling=input.sampling;
        const bool wide=sampling.temperature<=0 || sampling.top_k==1 ||
            (sampling.top_k>0 && sampling.top_k<=64);
        const int requested=sampling.enable_mtp && wide
            ?std::clamp(int(sampling.mtp_max_draft_tokens),1,components.mtp->maximum_draft_depth()):1;
        // Short prefill shares the existing four-row graphs. Only draft
        // caps above three need additional verification storage.
        const int depth=std::max(3,requested);
        if(depth==graph_draft_depth && state.graph_registry)return;
        const auto started=std::chrono::steady_clock::now();
        preparing_graph_depth=true;
        try {
            weights.release_workspace();
            graph.invalidate();
            qwen4_release_decode_graphs(language,state,components.mtp.get());
            weights.retire_decode();
            state.reset(1);
            components.mtp->reset(predictor_state(), 1);
            auto context=tyloi::cuda::default_context(execution->config.layer_placement.primary_device());
            context->reclaim_idle_memory();
            budget->reclaim((std::size_t(2)<<30)+qwen4_mtp_graph_reserve(language,components.mtp.get(),depth));
            qwen4_warmup_decode_graphs(language,state,components.mtp.get(),depth);
            graph_draft_depth=depth;
            preparing_graph_depth=false;
            configure_decode_budget();
            weights.prepare(::tyloi::weights::Cache::Stage::loaded,
                tyloi::Device(tyloi::kCUDA, execution->config.layer_placement.primary_device()), source.get());
        } catch(...) {
            graph_draft_depth=0;preparing_graph_depth=false;
            throw;
        }
        std::cerr<<"runtime_mtp_depth_plan requested_depth="<<requested<<" prepared_depth="<<depth
            <<" ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<'\n';
    }
}

template <class M, class S, class P>
tyloi::StepSequence<tyloi::models::Prepared> Text<M, S, P>::prepare(InferenceRequest& input, State& state) {
    if constexpr (std::is_same_v<Parts, std::unique_ptr<tyloi::cuda::minicpmo45::Components>>) {
        if (!components) throw std::invalid_argument("model has no multimodal component");
        return components->prepare(input.prompt, *input.vision);
    } else if constexpr (std::is_same_v<Model, qwen4_exp::Model> ||
                         std::is_same_v<Model, nemotron_h::Model> ||
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
    if(!batched)prepare_graph_depth(input);
    bool decode_ready = false;
    if (!preparing_graph_depth) configure_decode_budget();
#ifdef TYLOINFER_NATIVE_CUDA
    decode_ready = !preparing_graph_depth;
#endif
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
        batched ? nullptr : component_predictor(components), std::move(prepared),
        batched ? batching.get() : nullptr, id, decode_ready);
}

template <class M, class S, class P>
Generation Text<M, S, P>::generate(const RequestId& id, InferenceRequest& input, InferenceOutput& output, bool batched) {
    if (batched) {
        graph.invalidate();
        if constexpr (std::is_same_v<Model,qwen4_exp::Model>) {
            // Exclusive MTP generation has ended before a batch starts.
            // Its graph consumers must retire before idle batch buffers.
            qwen4_release_decode_graphs(language,state,components.mtp.get());
            if (components.mtp) components.mtp->reset(predictor_state(), 1);
        }
    }
    return generate_prepared(std::ref(*this), state, *executor, input, output, id, batched);
}

} // namespace tyloi::cuda::engine_detail
