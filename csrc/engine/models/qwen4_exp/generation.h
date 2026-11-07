#pragma once

#include "engine/models/common/generation_ops.h"
#include "engine/models/qwen4_exp/ops.h"
#include "engine/budget.h"
#include "engine/generation/sampling.h"
#include "engine/requests/output.h"
#include "engine/weights/cache_internal.h"

#include <algorithm>
#include <utility>

namespace tyloi::cuda {

template<class Ops, class Finish>
std::tuple<::tyloi::DenseTensor, bool, bool> generation_prefill(
        qwen4_exp::Model& model, Ops& ops, const tyloi::engine::PrefillChunk& chunk,
        ::tyloi::DenseTensor& ids, const Finish& finish_prefill) {
    using DenseTensor = ::tyloi::DenseTensor;
    auto& weights=ops.weights;
    auto& state=ops.state;
    auto* prepared=ops.prepared;
    auto* output=ops.output;
    DenseTensor hidden;
    bool layer_major=false,short_windows=false;
    // TQ groups all prompt rows by expert without expanding an entire
    // layer. Splitting these batches into verification windows would
    // repeatedly transfer the same expert when it is not resident.
    const bool verification_windows=!::tyloi::weights::detail::cache(weights).tq || chunk.count<=8;
    if(verification_windows && (!prepared || !prepared->transformed()) && model.reuse_decode_graphs &&
            model.graph_registry && chunk.count<=128) {
        // The same bounded prompt windows as MTP avoid materializing
        // a whole expert layer before a short prompt has any scratch.
        short_windows=true;
        for(int64_t offset=0;offset<chunk.count;) {
            if(output && output->stopped())break;
            const auto rows=state.cache_pos==0?int64_t(1):std::min<int64_t>(4,chunk.count-offset);
            auto part=ids.narrow(1,offset,rows);
            DenseTensor window_logits;
            if(auto result=qwen4_decode_graph_hidden(model,part,nullptr,rows>1?1:0,
                    nullptr,{},&window_logits))
                hidden=std::move(*result);
            else
                hidden = dynamic_cast<const DenseTensor&>(*model.forward(tyloi::models::Input{std::make_shared<DenseTensor>(part), {}, {}, {}, {}, nullptr, nullptr,
                                                            false, rows>1?1:0, 0, 0},
                                       ops.state));
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

inline bool generation_workspace(qwen4_exp::Model& model, tyloi::weights::Loan& workspace,
                                 int64_t rows) {
    return workspace.begin((std::size_t(4)<<30)+
        std::size_t(rows)*model.metadata.hidden_size*model.metadata.hc_mult*sizeof(float)*4)>0;
}

template<class Ops>
std::optional<int64_t> generation_decode(qwen4_exp::Model& model, Ops& ops) {
    if(ops.known_token_host.defined()) {
        const bool greedy=ops.sampler.greedy() && !ops.has_penalties && !ops.constraint;
        if(auto result=qwen4_decode_graph_logits(model,ops.pending.reshape({1,1}),greedy?2:1,ops.known_token_host)) {
            ops.pending=greedy?std::move(*result):tyloi::engine::sample_logits(ops.sampler,
                result->contiguous().view({1,-1}),ops.counts,ops.constraint);
            return ops.pending.template item<int64_t>();
        }
    }
    return std::nullopt;
}

template<class Ops>
void generation_request(qwen4_exp::Model& model, Ops& ops, const tyloi::engine::InferenceRequest& input) {
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    model.decode_reserve_until=std::min<int64_t>(ops.info.max_context,
        int64_t(input.prompt.size())+std::min<int64_t>(input.sampling.max_tokens,256));
    if(!ops.batching && input.prompt.size()>1024) {
        ops.budget.request(tyloi::Device(tyloi::kCUDA, mfq_current_cuda_device()),
            ops.state.position()==0 && !model.graph_registry);
    }
#endif
}

inline bool generation_borrow(qwen4_exp::Model& model, std::size_t tokens, const PreparedPrompt* prepared) {
    return qwen4_layer_major_prefill_eligible(model,std::min<std::size_t>(tokens,8192)) &&
        (!prepared || !prepared->transformed());
}

} // namespace tyloi::cuda
