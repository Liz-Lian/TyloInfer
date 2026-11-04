#pragma once

#include "engine/models/common/generation_ops.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/backends/cuda/executor/decode_graph.h"
#include "engine/requests/output.h"
#include "engine/weights/moe/expert_cache.h"

#include <utility>

namespace tyloi::cuda {

template<class Ops, class Finish>
std::tuple<::tyloi::DenseTensor, bool, bool> generation_prefill(
        DeepseekV41CausalLm& model, Ops& ops, const tyloi::engine::PrefillChunk&,
        ::tyloi::DenseTensor& ids, const Finish&) {
    ::tyloi::DenseTensor hidden;
    if(!ops.prepared || !ops.prepared->transformed())
        hidden=deepseek_v41_prefill_hidden(model,ids,ops.prefill_compute_chunk,
            [&] {return ops.output && ops.output->stopped();});
    return {std::move(hidden),false,false};
}

inline void generation_prefill_begin(DeepseekV41CausalLm&, DecodeGraphCache& graph, bool streamed) {
    if(streamed)graph.invalidate();
}

template<class Ops>
void generation_prefill_end(DeepseekV41CausalLm&, Ops& ops, bool streamed, bool retained) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    if(streamed && !retained) {
        // Private FFN graphs retain their transfer arenas after a
        // request. Retire consumers before a new grouped prefill;
        // otherwise these arenas can reduce its ring to one expert.
        release_moe_decode_buffers(ops.weights.moe);
        tyloi::cuda::default_context(mfq_current_cuda_device())->reclaim_idle_memory();
    }
#endif
}

} // namespace tyloi::cuda
