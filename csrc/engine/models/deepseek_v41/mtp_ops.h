#pragma once

#include "engine/models/common/mtp_ops.h"
#include "engine/models/deepseek_v41/ops.h"

namespace tyloi::cuda::deepseek_v41 {
using Model = deepseek_v41::Model;

template<class Ops, class Chunk, class DenseTensor, class Timer, class Finish>
std::optional<typename Ops::PrefillResult> mtp_prefill(Model& model, Ops& ops, const Chunk& chunk,
                                                      DenseTensor& ids, DenseTensor& raw, Timer& timer, const Finish& finish_prefill) {
    if(!ops.transformed_prompt) {
        ::tyloi::Tensor raw_result;
        auto hidden=deepseek_v41_prefill_hidden(model,ops.state,ids,ops.prefill_compute_chunk,
            [&] {return ops.output && ops.output->stopped();},&raw_result);
        if(raw_result) raw=std::move(raw_result.dense());
        finish_prefill();
        return typename Ops::PrefillResult{hidden ? std::move(hidden.dense()) : DenseTensor{},
                                           std::move(raw),timer.elapsed_ms()};
    }
    return std::nullopt;
}

} // namespace tyloi::cuda::deepseek_v41
