#pragma once
#include <utility>

namespace tyloi::models {

// DeepSeek HC/MHC uses a separate residual stream and branch preparation for
// attention and FFN. MHC's next_pre travels from attention to the FFN branch.
template <class Tensor, class AttnPre, class Attention, class AttnPost, class FfnPre,
          class FeedForward, class FfnPost, class Commit>
auto hyperconnection_layer(Tensor hidden, AttnPre attention_pre, Attention attention,
                           AttnPost attention_post, FfnPre ffn_pre, FeedForward feed_forward,
                           FfnPost ffn_post, Commit commit) {
    auto attention_mix = attention_pre(hidden);
    auto attended = attention(attention_mix);
    hidden = attention_post(std::move(attended), hidden, attention_mix);
    auto ffn_mix = ffn_pre(hidden, attention_mix);
    auto output = feed_forward(ffn_mix);
    hidden = ffn_post(std::move(output), hidden, ffn_mix);
    commit(ffn_mix);
    return hidden;
}

template <class Tensor, class Layers, class Apply>
Tensor layer_stack(Tensor hidden, Layers &layers, Apply apply) {
    for (auto &layer : layers)
        hidden = apply(layer, std::move(hidden));
    return hidden;
}

template <class Tensor, class Norm, class Attention, class FeedForward, class Add>
Tensor pre_norm_layer(Tensor hidden, Norm norm, Attention attention, FeedForward ffn, Add add) {
    auto branch = attention(norm(hidden, 0));
    hidden = add(std::move(hidden), std::move(branch));
    branch = ffn(norm(hidden, 1));
    return add(std::move(hidden), std::move(branch));
}

} // namespace tyloi::models
