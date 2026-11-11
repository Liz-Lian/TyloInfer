#pragma once

#include "engine/generation/ops.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/generation/sampling.h"
#include "engine/requests/output.h"
#include "engine/weights/cache.h"

#include <algorithm>
#include <utility>

namespace tyloi::cuda {

/** Prefill a legacy request with the available Qwen4 execution path. */
template<class Ops, class Finish>
std::tuple<::tyloi::DenseTensor, bool, bool> generation_prefill(
        qwen4_exp::Model& model, Ops& ops, const tyloi::engine::PrefillChunk& chunk,
        ::tyloi::DenseTensor& ids, const Finish& finish_prefill) {
    auto& weights=ops.weights;
    auto& state=ops.state;
    auto* prepared=ops.prepared;
    auto* output=ops.output;
    ::tyloi::DenseTensor hidden;
    bool layer_major=false,short_windows=false;
    // TQ groups all prompt rows by expert without expanding an entire
    // layer. Splitting these batches into verification windows would
    // repeatedly transfer the same expert when it is not resident.
    const bool verification_windows=weights.split_prefill(chunk.count);
    const bool layer_eligible=(!prepared || !prepared->transformed()) &&
        qwen4_layer_major_prefill_eligible(model,state,chunk.count);
    if(verification_windows && (!prepared || !prepared->transformed()) && ops.state.reuse_decode_graphs &&
            ops.state.graph_registry && chunk.count<=128) {
        // The same bounded prompt windows as MTP avoid materializing
        // a whole expert layer before a short prompt has any scratch.
        short_windows=true;
        for(int64_t offset=0;offset<chunk.count;) {
            if(output && output->stopped())break;
            const auto rows=state.cache_pos==0?int64_t(1):std::min<int64_t>(4,chunk.count-offset);
            auto part=ids.narrow(1,offset,rows);
            ::tyloi::Tensor window_logits;
            if(auto result=qwen4_decode_graph_hidden(model,ops.state,part,nullptr,rows>1?1:0,
                    nullptr,{},&window_logits))
                hidden=result->dense();
            else
                hidden = model.forward(tyloi::models::Input{part, {}, {}, {}, {}, nullptr, nullptr,
                                                            false, rows>1?1:0, 0, 0},
                                       ops.state).dense();
            if(rows>1)state.commit_speculative();
            offset+=rows;
        }
    } else if(layer_eligible) {
        layer_major=true;
        auto result=qwen4_layer_major_prefill_hidden(model,ops.state,ids,ops.prefill_compute_chunk,true,
            [&] {return output && output->stopped();},nullptr,false,finish_prefill);
        if (result) hidden=result.dense();
    }
    return {std::move(hidden),layer_major,short_windows};
}

/** Borrow workspace for transformed prefill rows. */
inline bool generation_workspace(qwen4_exp::Model& model, tyloi::weights::Loan& workspace,
                                 int64_t rows) {
    return workspace.begin((std::size_t(4)<<30)+
        std::size_t(rows)*model.metadata.hidden_size*model.metadata.hc_mult*sizeof(float)*4)>0;
}

/** Decode a token using a retained Qwen4 graph when available. */
template<class Ops>
std::optional<int64_t> generation_decode(qwen4_exp::Model& model, Ops& ops) {
    if(ops.known_token_host.defined()) {
        const bool greedy=ops.sampler.greedy() && !ops.has_penalties && !ops.constraint;
        if(auto result=qwen4_decode_graph_logits(model,ops.state,ops.pending.reshape({1,1}),greedy?2:1,ops.known_token_host)) {
            ops.pending=greedy?result->dense():tyloi::engine::sample_logits(ops.sampler,
                result->dense().contiguous().view({1,-1}),ops.counts,ops.constraint);
            return ops.pending.template item<int64_t>();
        }
    }
    return std::nullopt;
}

/** Check whether the request can borrow prefill workspace. */
inline bool generation_borrow(qwen4_exp::Model& model, qwen4_exp::State& state, std::size_t tokens, const tyloi::models::Prepared* prepared) {
    return qwen4_layer_major_prefill_eligible(model,state,std::min<std::size_t>(tokens,8192)) &&
        (!prepared || !prepared->transformed());
}

} // namespace tyloi::cuda
