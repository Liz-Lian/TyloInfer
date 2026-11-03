#pragma once

#include "engine/models/common/generation_ops.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/generation/sampling.h"
#include "engine/requests/output.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/routed_cache.h"

#include <algorithm>
#include <utility>

namespace mfq::cuda {

template<class Ops, class Finish>
std::tuple<mfq_tensor_backend::Tensor, bool, bool> generation_prefill(
        Qwen4CausalLm& model, Ops& ops, const mfq::engine::PrefillChunk& chunk,
        mfq_tensor_backend::Tensor& ids, const Finish& finish_prefill) {
    using Tensor = mfq_tensor_backend::Tensor;
    auto& weights=ops.weights;
    auto& state=ops.state;
    auto* prepared=ops.prepared;
    auto* output=ops.output;
    Tensor hidden;
    bool layer_major=false,short_windows=false;
    // TQ groups all prompt rows by expert without expanding an entire
    // layer. Splitting these batches into verification windows would
    // repeatedly transfer the same expert when it is not resident.
    const bool verification_windows=!weights.tq || chunk.count<=8;
    if(verification_windows && (!prepared || !prepared->transformed()) && model.reuse_decode_graphs &&
            model.graph_registry && chunk.count<=128) {
        // The same bounded prompt windows as MTP avoid materializing
        // a whole expert layer before a short prompt has any scratch.
        short_windows=true;
        for(int64_t offset=0;offset<chunk.count;) {
            if(output && output->stopped())break;
            const auto rows=state.cache_pos==0?int64_t(1):std::min<int64_t>(4,chunk.count-offset);
            auto part=ids.narrow(1,offset,rows);
            Tensor window_logits;
            if(auto result=qwen4_decode_graph_hidden(model,part,nullptr,rows>1?1:0,
                    nullptr,{},&window_logits))
                hidden=std::move(*result);
            else hidden=model.hidden_forward(part,mfq_nullopt,mfq_nullopt,nullptr,
                mfq_nullopt,nullptr,rows>1?1:0);
            if(rows>1)state.commit_speculative();
            offset+=rows;
        }
    } else if((!prepared || !prepared->transformed()) &&
        qwen4_layer_major_prefill_eligible(model,chunk.count)) {
        layer_major=true;
        hidden=qwen4_layer_major_prefill_hidden(model,ids,ops.prefill_compute_chunk,true,
            [&] {return output && output->stopped();},nullptr,false,finish_prefill);
    }
    return {std::move(hidden),layer_major,short_windows};
}

inline bool generation_workspace(Qwen4CausalLm& model, MoePrefillWorkspaceLoan& workspace,
                                 int64_t rows) {
    return workspace.begin((std::size_t(4)<<30)+
        std::size_t(rows)*model.hidden_size()*model.hc_mult()*sizeof(float)*4)>0;
}

template<class Ops>
std::optional<int64_t> generation_decode(Qwen4CausalLm& model, Ops& ops) {
    if(ops.known_token_host.defined()) {
        const bool greedy=ops.sampler.greedy() && !ops.has_penalties && !ops.constraint;
        if(auto result=qwen4_decode_graph_logits(model,ops.pending.reshape({1,1}),greedy?2:1,ops.known_token_host)) {
            ops.pending=greedy?std::move(*result):mfq::engine::sample_logits(ops.sampler,
                result->contiguous().view({1,-1}),ops.counts,ops.constraint);
            return ops.pending.template item<int64_t>();
        }
    }
    return std::nullopt;
}

template<class Ops>
void generation_request(Qwen4CausalLm& model, Ops& ops, const mfq::engine::InferenceRequest& input) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    model.decode_reserve_until=std::min<int64_t>(ops.info.max_context,
        int64_t(input.prompt.size())+std::min<int64_t>(input.sampling.max_tokens,256));
    if(!ops.batching && input.prompt.size()>1024) {
        finish_moe_expert_exchanges(ops.weights.moe);
        if(ops.state.cache_pos==0 && !model.graph_registry)
            release_moe_decode_buffers(ops.weights.moe);
        else release_moe_prefill_buffers(ops.weights.moe);
        mfq::cuda::default_context(mfq_current_cuda_device())->reclaim_idle_memory();
    }
#endif
}

inline bool generation_borrow(Qwen4CausalLm& model, std::size_t tokens, const PreparedPrompt* prepared) {
    return qwen4_layer_major_prefill_eligible(model,std::min<std::size_t>(tokens,8192)) &&
        (!prepared || !prepared->transformed());
}

} // namespace mfq::cuda
