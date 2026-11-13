#include "engine/models/qwen4_exp/layer.h"
#include "engine/models/qwen4_exp/state.h"

#include <utility>

namespace tyloi::cuda::qwen4_exp {

DenseTensor forward(const Layer& weights, LayerState& request,
                    DenseTensor x, const DenseTensor& ids,
                    const DenseTensor& positions, const DenseTensor& full_positions,
                    int64_t confirmed, const DenseTensor& host_ids) {
    request.retained_prefix = confirmed > 0 && ids.size(1) <= 8;
    const auto measure = [&](const char* name, auto&& operation) {
        return ::tyloi::ops::measure(name, ::tyloi::Device(::tyloi::kCUDA,weights.cuda_device), std::forward<decltype(operation)>(operation));
    };
    if (weights.ple) {
        auto ple=measure("prefill.ple", [&] {
            return weights.ple->forward(request.ple, x, ids, true, confirmed, host_ids,
                                        full_positions.size(-1) - ids.size(1));
        });
        x=x.is_cuda()?tyloi::ops::add(x,ple).dense():x+ple;
    }
    auto [attention_input, attention_residual, attention_gate] = measure(
        "prefill.gr_attention", [&] { return weights.attention_gr.pre(x); });
    auto attention_output = weights.gdn
        ? measure("prefill.gdn", [&] {
            return weights.gdn->forward(request.gdn, attention_input, true, confirmed,
                                        full_positions.size(-1) - ids.size(1))
                .dense()
                ;
        })
        : measure("prefill.qsa", [&] {
            return weights.qsa->forward(*request.qsa, attention_input, positions,
                                        full_positions, true)
                .dense()
                ;
        });
    auto [ffn_input, ffn_residual, ffn_gate] = measure(
        "prefill.gr_ffn_pre", [&] {
            return weights.ffn_gr.pre_after(attention_output, attention_residual, attention_gate);
        });
    attention_output = {};
    auto ffn_output = measure("prefill.ffn", [&] {
        return tyloi::ops::ffn(weights.ffn,ffn_input).dense();
    });
    return measure("prefill.gr_ffn_post", [&] {
        return weights.ffn_gr.post(ffn_output, ffn_residual, ffn_gate)
            .dense()
            ;
    });
}

DenseTensor prefill_ple(const Layer& weights, LayerState& request,
                        DenseTensor x, const DenseTensor& ids, const DenseTensor& host_ids) {
    if (weights.ple) {
        auto ple = ::tyloi::ops::measure("prefill.ple",
            ::tyloi::Device(::tyloi::kCUDA,weights.cuda_device),
            [&] {
            return weights.ple->forward(request.ple, x, ids, true, 0, host_ids);
        });
        x=x.is_cuda()?tyloi::ops::add(x,ple).dense():x+ple;
    }
    return x;
}

DenseTensor prefill_attention_core(const Layer& weights, LayerState& request,
                                   const DenseTensor& input, const DenseTensor& positions,
                                   const DenseTensor& full_positions) {
    const auto measure = [&](const char* name, auto&& operation) {
        return ::tyloi::ops::measure(name, ::tyloi::Device(::tyloi::kCUDA,weights.cuda_device), std::forward<decltype(operation)>(operation));
    };
    return weights.gdn
        ? measure("prefill.gdn", [&] {
            return weights.gdn->forward(request.gdn, input, true, 0)
                .dense()
                ;
        })
        : measure("prefill.qsa", [&] {
            return weights.qsa->forward(*request.qsa, input, positions, full_positions, true)
                .dense()
                ;
        });
}

tyloi::ops::Residual::Result prefill_attention(const Layer& weights, LayerState& request,
                                        DenseTensor x, const DenseTensor& ids,
                                        const DenseTensor& host_ids, const DenseTensor& positions,
                                        const DenseTensor& full_positions) {
    const auto measure = [&](const char* name, auto&& operation) {
        return ::tyloi::ops::measure(name, ::tyloi::Device(::tyloi::kCUDA,weights.cuda_device), std::forward<decltype(operation)>(operation));
    };
    x = prefill_ple(weights, request, std::move(x), ids, host_ids);
    auto mix = measure("prefill.gr_attention", [&] { return weights.attention_gr.pre(x); });
    auto attention_output = prefill_attention_core(weights, request,
                                                    mix.mixed.dense(), positions, full_positions);
    return measure("prefill.gr_ffn_pre", [&] {
        return weights.ffn_gr.pre_after(attention_output, mix.residual, mix.gate);
    });
}

} // namespace tyloi::cuda::qwen4_exp
