#pragma once

#include "engine/models/common/mtp_ops.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/weights/cache.h"

namespace tyloi::cuda::deepseek_v41 {
using Model = deepseek_v41::Model;

template<class Ops, class Chunk, class Tensor, class Timer, class Finish>
std::optional<typename Ops::PrefillResult> mtp_prefill(Model& model, Ops& ops, const Chunk& chunk,
                                                      Tensor& ids, Tensor& raw, Timer& timer, const Finish& finish_prefill) {
    if(!ops.transformed_prompt) {
        auto input=deepseek_v41_prefill_input(model,ops.state,ids,ops.prefill_compute_chunk);
        std::optional<tyloi::weights::Cache::Priority> priority;
        if(input.size(1) > model.config.sliding_window)
            priority.emplace(ops.weights, 0, static_cast<int>(model.config.causal_encoder_layers));
        const auto stopped=[&] {return ops.output && ops.output->stopped();};
        auto encoded=deepseek_v41_prefill_encode(model,ops.state,input,ops.prefill_compute_chunk,stopped,true);
        if(encoded && priority) priority->finish();
        priority.reset();
#ifdef TYLOINFER_NATIVE_CUDA
        auto hidden=encoded ? deepseek_v41_prefill_decode(model,ops.state,std::move(*encoded),stopped,&raw)
                            : ::tyloi::Tensor{};
        finish_prefill();
        return typename Ops::PrefillResult{std::move(hidden),std::move(raw),timer.elapsed_ms()};
#else
        ::tyloi::Tensor raw_result;
        auto hidden=encoded ? deepseek_v41_prefill_decode(model,ops.state,std::move(*encoded),stopped,&raw_result)
                            : ::tyloi::Tensor{};
        if(raw_result) raw=std::move(raw_result.dense());
        finish_prefill();
        return typename Ops::PrefillResult{hidden ? std::move(hidden.dense()) : Tensor{},
                                           std::move(raw),timer.elapsed_ms()};
#endif
    }
    return std::nullopt;
}

} // namespace tyloi::cuda::deepseek_v41
