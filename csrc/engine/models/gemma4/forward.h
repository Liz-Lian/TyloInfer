#pragma once

#include "engine/models/gemma4/model.h"
#include "engine/models/gemma4/state.h"

namespace tyloi::cuda::gemma4 {

/** Run one Gemma4 layer, allocating and updating its request-owned KV cache as needed. */
::tyloi::Tensor forward_layer(const Layer&, tyloi::state::KvView&, const tyloi::Tensor&, const AttentionStep&,
                     const Model::Options&);

} // namespace tyloi::cuda::gemma4
