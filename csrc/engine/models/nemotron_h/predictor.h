#pragma once

#include "engine/models/common/predictor.h"

namespace mfq::models::nemotron_h {

enum class PredictorNorm { embedding, hidden, output };

// Nemotron-H's own multi-token-prediction head, matching the three reference
// implementations (llama.cpp src/models/nemotron-h-moe.cpp, vLLM and SGLang
// nemotron_h_mtp.py); no Hugging Face implementation builds these weights:
//   e = enorm(embed(next_ids)); h = hnorm(previous_hidden)
//   x = eh_proj(cat([e, h], -1))                         # embedding first
//   x = x + mixer(mtp_layers_block_type[0])(norm0(x))    # attention, NoPE
//   x = x + mixer(mtp_layers_block_type[1])(norm1(x))    # moe, latent relu^2
//   logits = lm_head(final_layernorm(x))
// Two details differ from the DeepSeek-V3 convention:
//   * previous_hidden is the backbone hidden state *after* output_norm, so the
//     target model hands the predictor its already-normalized hidden (never
//     normalize it twice);
//   * for further draft steps the head's own final_layernorm output is passed
//     back as previous_hidden unchanged, so the backbone norm is not applied
//     again and only hnorm renormalizes it.
//   * the final norm is RMSNorm (llama.cpp uses LayerNorm; vLLM and SGLang use
//     RMSNorm, which is also Megatron's configured normalization).
// The single predictor stage owns one attention history; the MoE sub-block is
// stateless. `next_ids` are the tokens after each hidden state, so the
// predictor history is one row shorter than the target prompt.
template <class Ops>
auto mtp_predictor(Ops &ops, typename Ops::Tensor hidden, typename Ops::Tensor ids,
                   typename Ops::Tensor positions, int64_t depth, bool cache) {
    auto &model = ops.model;
    const auto &config = model.config;
    require_model(
        ops.rank(hidden) == 3 && ops.rank(ids) == 2 && ops.size(hidden, 0) == ops.size(ids, 0) &&
            ops.size(hidden, 1) == ops.size(ids, 1) && ops.size(hidden, 2) == config.hidden_size &&
            ops.size(hidden, 1) > 0,
        "Nemotron-H MTP inputs must be matching [B,T] hidden states and [B,T] next-token IDs");
    require_model(config.num_nextn_predict_layers == 1 && !config.mtp_mixers.empty(),
                  "Nemotron-H MTP requires one declared predictor stage");
    const auto batch = ops.size(hidden, 0), tokens = ops.size(hidden, 1);
    return mfq::models::projected_predictor(
        model, batch, tokens, depth, cache, [&](int64_t next) { model.reset(next); },
        [&] { return ops.embed(std::move(ids), hidden); },
        [&](mfq::models::PredictorCursor cursor) {
            return ops.positions(std::move(positions), cursor.start, tokens);
        },
        [&](typename Ops::Tensor embeds, const typename Ops::Tensor &) {
            return ops.normalize(std::move(embeds), PredictorNorm::embedding);
        },
        [&] { return ops.normalize(hidden, PredictorNorm::hidden); },
        [&](typename Ops::Tensor e, typename Ops::Tensor h) {
            return ops.fuse(std::move(e), std::move(h));
        },
        [&](typename Ops::Tensor x, int64_t stage, const typename Ops::Tensor &pos) {
            return ops.mixer_blocks(std::move(x), stage, pos);
        },
        [&](const typename Ops::Tensor &x) {
            return ops.normalize(x, PredictorNorm::output);
        },
        [&](int64_t, const typename Ops::Tensor &) {});
}

} // namespace mfq::models::nemotron_h
