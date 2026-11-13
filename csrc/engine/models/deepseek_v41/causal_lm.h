#pragma once
#include "config.h"
#include "engine/models/metadata.h"
#include "../common/check.h"
#include <optional>
#include <utility>
#include <vector>

namespace tyloi::models::deepseek_v41 {

template <class Gather, class Project, class Gate, class Add>
auto engram(Gather gather, Project project, Gate gate, Add add) {
    auto key_value = project(gather());
    auto gated = gate(key_value);
    return add(std::move(gated), key_value);
}

template <class Embed, class Expand, class InitialPre, class Layer, class Collapse, class Logits,
          class Slice>
auto dspark_draft(int64_t width, int64_t requested, size_t stages, Embed embed, Expand expand,
                  InitialPre initial_pre, Layer layer, Collapse collapse, Logits logits,
                  Slice slice) {
    require_model(requested > 0 && requested <= width && stages > 0, "invalid DSpark draft width");
    auto embedded = embed(width);
    auto hidden = expand(embedded);
    auto previous_pre = initial_pre(embedded);
    for (size_t index = 0; index < stages; ++index)
        hidden = layer(index, std::move(hidden), previous_pre,
                       index + 1 == stages ? requested : width);
    // Predictor stages attend across the full physical block. After their
    // final collapse, vocabulary projection is independent for each row;
    // discard unused rows before normalization and the large output head.
    return logits(slice(collapse(std::move(hidden), previous_pre), requested));
}

template <class Project, class Keys, class Store>
void dspark_context(int64_t &position, int64_t tokens, size_t stages, Project project, Keys keys,
                    Store store) {
    auto main = project();
    for (size_t index = 0; index < stages; ++index)
        store(index, keys(index, main));
    position += tokens;
}

template <class Query, class Key, class Cache, class Plan, class Attend, class Inverse,
          class Output>
auto dspark_attention(int64_t context, Query query, Key key, Cache cache, Plan plan, Attend attend,
                      Inverse inverse, Output output) {
    require_model(context > 0, "DeepSeek-V4.1 DSpark draft requires committed context");
    auto q = query();
    auto keys = cache(key());
    auto indices = plan(keys);
    return output(inverse(attend(std::move(q), std::move(keys), indices)));
}

} // namespace tyloi::models::deepseek_v41
