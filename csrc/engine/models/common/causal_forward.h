#pragma once
#include "causal_metadata.h"
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace mfq::models {

inline void require_model(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Model, class Guard> void reset_model(Model &model, int64_t batch, Guard guard) {
    model.cache_pos = model.decode_position_delta = 0;
    model.adapter_reset(batch);
    model.speculative_start = -1;
    model.speculative_confirmed = 0;
    model.speculative_suffix_forward = false;
    for (auto &block : model.blocks) {
        auto scope = guard(block);
        block->reset(batch);
    }
}

template <class Model, class Guard>
void begin_speculative_suffix(Model &model, int64_t tokens, Guard guard) {
    require_model(model.supports_suffix_speculation() && model.speculative_start < 0 &&
                      tokens > 0 && model.cache_pos + tokens <= model.max_position_embeddings(),
                  "invalid speculative suffix");
    model.speculative_start = model.cache_pos;
    model.speculative_confirmed = 0;
    model.adapter_begin_speculative();
    size_t begun = 0;
    try {
        for (auto &block : model.blocks) {
            auto scope = guard(block);
            block->begin_speculative(tokens);
            ++begun;
        }
    } catch (...) {
        for (size_t i = 0; i < begun; ++i) {
            try {
                auto scope = guard(model.blocks[i]);
                model.blocks[i]->commit_speculative();
            } catch (...) {
            }
        }
        try {
            model.adapter_rollback_speculative(model.cache_pos);
        } catch (...) {
        }
        model.speculative_start = -1;
        model.speculative_confirmed = 0;
        throw;
    }
}

template <class Model, class Guard>
void finish_speculative(Model &model, bool commit, int64_t accepted, Guard guard) {
    require_model(model.speculative_start >= 0 && accepted >= 0, "no speculative transaction");
    const auto keep = model.speculative_start + model.speculative_confirmed + accepted;
    for (auto &block : model.blocks) {
        auto scope = guard(block);
        if (commit)
            block->commit_speculative();
        else
            block->rollback_speculative(keep);
    }
    if (commit)
        model.adapter_commit_speculative();
    else {
        model.adapter_rollback_speculative(keep);
        model.cache_pos = keep;
    }
    model.speculative_start = -1;
    model.speculative_confirmed = 0;
}

template <class Tensor> struct CausalGeometry {
    Tensor cache_positions, positions, full_positions;
    std::optional<Tensor> effective_attention_mask;
};

// Position and mask math is shared; policy supplies the family's typed geometry hooks.
template <class Policy, class Ops>
auto prepare_causal_geometry(Policy &policy, const CausalState &state, Ops &ops, int64_t batch,
                             int64_t tokens)
    -> CausalGeometry<std::decay_t<decltype(ops.ids)>> {
    using Tensor = std::decay_t<decltype(ops.ids)>;
    CausalGeometry<Tensor> geometry{
        ops.cache_positions_override
            ? ops.device_ids(*ops.cache_positions_override)
            : ops.position_range(state.cache_pos, tokens),
        {},
        {},
        {}};
    auto current = ops.pos_override
                       ? ops.device_ids(*ops.pos_override)
                       : (state.decode_position_delta == 0
                              ? geometry.cache_positions
                              : ops.offset_positions(geometry.cache_positions,
                                                      state.decode_position_delta));
    auto prepared = policy.adapter_prepare_positions(std::move(current), batch, tokens);
    geometry.positions = std::move(prepared.positions);
    geometry.full_positions = std::move(prepared.full_positions);
    require_model((ops.rank(geometry.cache_positions) == 1 &&
                   ops.elements(geometry.cache_positions) == tokens) ||
                      (ops.rank(geometry.cache_positions) == 2 &&
                       ops.size(geometry.cache_positions, 0) == batch &&
                       ops.size(geometry.cache_positions, 1) == tokens),
                  "cache_positions must have shape [tokens] or [batch,tokens]");
    policy.adapter_validate_positions(geometry.positions, batch, tokens, ops.has_mrope());
    if (ops.attention_mask.has_value()) {
        const auto &mask = *ops.attention_mask;
        require_model(ops.rank(mask) == 2 && ops.size(mask, 0) == batch &&
                          ops.size(mask, 1) >= state.cache_pos + tokens,
                      "attention_mask must cover [batch,cached_plus_current_tokens]");
    }
    geometry.effective_attention_mask =
        policy.adapter_attention_mask(ops.attention_mask, tokens, state.cache_pos);
    return geometry;
}

// The model's logical cache and layer traversal are independent of storage.
// Ops places tensors on devices and supplies each family's native operators.
template <class Ops> auto causal_forward(Ops &ops) {
    auto &model = ops.model;
    const auto &state = static_cast<const CausalState &>(model);
    ops.prepare_inputs();
    const auto B = ops.size(ops.ids, 0), T = ops.size(ops.ids, 1);
    const auto &embeddings = ops.input_embeddings;
    require_model(ops.rank(embeddings) == 3 && ops.size(embeddings, 0) == B &&
                      ops.size(embeddings, 1) == T &&
                      ops.size(embeddings, 2) == model.hidden_size(),
                  "inputs_embeds shape must match [batch,tokens,hidden_size]");
    if (model.adapter_requires_batch_reset(B))
        model.reset(B);
    model.adapter_validate_forward(B, T, model.cache_pos, ops.pos_override.has_value(),
                                   ops.cache_positions_override.has_value(),
                                   ops.attention_mask.has_value());
    require_model(model.speculative_start < 0 || model.speculative_suffix_forward,
                  "commit or roll back the pending speculative pass before forwarding");
    const auto confirmed = ops.confirmed_prefix;
    require_model(
        confirmed >= 0 &&
            (confirmed == 0 || (confirmed < T && B == 1 && model.cache_pos > 0 &&
                                (!ops.pos_override.has_value() ||
                                 model.adapter_allows_speculative_position_override()) &&
                                !ops.cache_positions_override.has_value() &&
                                !ops.attention_mask.has_value() && model.supports_speculation())),
        "unsupported speculative backbone geometry");
    if (confirmed > 0) {
        require_model(model.cache_pos + T <= model.max_position_embeddings(),
                      "speculative pass exceeds context capacity");
        model.speculative_start = model.cache_pos;
        model.speculative_confirmed = confirmed;
    }
    if (model.cache_pos == 0)
        model.reset(B);
    auto geometry = prepare_causal_geometry(model, state, ops, B, T);
    ops.cache_positions = std::move(geometry.cache_positions);
    ops.pos = std::move(geometry.positions);
    ops.full_positions = std::move(geometry.full_positions);
    ops.effective_attention_mask = std::move(geometry.effective_attention_mask);
    auto hidden = ops.prepare_hidden(B, T);
    model.adapter_begin_forward(ops.raw_hidden != nullptr);
    ops.trace(hidden);
    for (auto &block : model.blocks) {
        hidden = ops.layer(block, std::move(hidden));
        ops.trace(hidden);
    }
    model.adapter_finish_forward(ops.full_positions, B, T);
    if (model.adapter_force_cache_advance() || !ops.pos_override.has_value() ||
        ops.advance_cache_with_position_ids)
        model.cache_pos += T;
    return ops.finish(std::move(hidden), B, T);
}
} // namespace mfq::models
