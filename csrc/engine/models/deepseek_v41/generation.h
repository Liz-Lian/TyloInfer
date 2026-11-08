#pragma once

#include "engine/models/common/generation_ops.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/requests/output.h"

#include <utility>

namespace tyloi::cuda::deepseek_v41 {
using Model = deepseek_v41::Model;

template<class Ops, class Finish>
#ifdef TYLOINFER_NATIVE_CUDA
std::tuple<::tyloi::Tensor, bool, bool> generation_prefill(
        Model& model, Ops& ops, const tyloi::engine::PrefillChunk&,
        ::tyloi::Tensor& ids, const Finish&) {
    ::tyloi::Tensor hidden;
#else
std::tuple<::tyloi::DenseTensor, bool, bool> generation_prefill(
        Model& model, Ops& ops, const tyloi::engine::PrefillChunk&,
        ::tyloi::DenseTensor& ids, const Finish&) {
    ::tyloi::DenseTensor hidden;
#endif
    if(!ops.prepared || !ops.prepared->transformed()) {
        auto result=deepseek_v41_prefill_hidden(model,ops.state,ids,ops.prefill_compute_chunk,
            [&] {return ops.output && ops.output->stopped();});
#ifdef TYLOINFER_NATIVE_CUDA
        if(result) hidden=std::move(result);
#else
        if(result) hidden=result.dense();
#endif
    }
    return {std::move(hidden),false,false};
}

inline bool generation_prefill_begin(Model&, DecodeGraphCache& graph, bool streamed) {
    if(streamed)graph.invalidate();
    return streamed;
}

} // namespace tyloi::cuda::deepseek_v41
