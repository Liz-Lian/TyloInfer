#pragma once

#include "engine/generation/ops.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/requests/output.h"
#include "engine/weights/cache.h"

#include <utility>

namespace tyloi::cuda::deepseek_v41 {
using Model = deepseek_v41::Model;

#ifndef TYLOINFER_NATIVE_CUDA
template<class Ops, class Finish>
std::tuple<::tyloi::DenseTensor, bool, bool> generation_prefill(
        Model& model, Ops& ops, const tyloi::engine::PrefillChunk&,
        ::tyloi::DenseTensor& ids, const Finish&) {
    ::tyloi::DenseTensor hidden;
    if(!ops.prepared || !ops.prepared->transformed()) {
        auto input=deepseek_v41_prefill_input(model,ops.state,ids,ops.prefill_compute_chunk);
        std::optional<tyloi::weights::Cache::Priority> priority;
        if(input.size(1) > model.config.sliding_window)
            priority.emplace(ops.weights, 0, static_cast<int>(model.config.causal_encoder_layers));
        const auto stopped=[&] {return ops.output && ops.output->stopped();};
        auto encoded=deepseek_v41_prefill_encode(model,ops.state,input,ops.prefill_compute_chunk,stopped,false);
        if(encoded && priority) priority->finish();
        priority.reset();
        auto result=encoded ? deepseek_v41_prefill_decode(model,ops.state,std::move(*encoded),stopped)
                            : ::tyloi::Tensor{};
        if(result) hidden=result.dense();
    }
    return {std::move(hidden),false,false};
}
#endif

inline bool generation_prefill_begin(Model&, DecodeGraphCache& graph, bool streamed) {
    if(streamed)graph.invalidate();
    return streamed;
}

} // namespace tyloi::cuda::deepseek_v41
