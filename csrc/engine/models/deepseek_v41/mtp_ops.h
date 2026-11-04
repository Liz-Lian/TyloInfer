#pragma once

#include "engine/models/common/mtp_ops.h"
#include "engine/models/deepseek_v41/ops.h"

namespace tyloi::cuda {

template<class Ops, class Chunk, class Tensor, class Timer, class Finish>
std::optional<typename Ops::PrefillResult> mtp_prefill(DeepseekV41CausalLm& model, Ops& ops, const Chunk& chunk,
                                                      Tensor& ids, Tensor& raw, Timer& timer, const Finish& finish_prefill) {
    if(!ops.transformed_prompt) {
        auto hidden=deepseek_v41_prefill_hidden(model,ids,ops.prefill_compute_chunk,
            [&] {return ops.output && ops.output->stopped();},&raw);
        finish_prefill();
        return typename Ops::PrefillResult{std::move(hidden),std::move(raw),timer.elapsed_ms()};
    }
    return std::nullopt;
}

} // namespace tyloi::cuda
