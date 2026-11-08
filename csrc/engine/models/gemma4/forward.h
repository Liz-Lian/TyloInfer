#pragma once

#include "engine/models/gemma4/model.h"
#include "engine/models/gemma4/state.h"

namespace tyloi::cuda::gemma4 {

std::shared_ptr<tyloi::Tensor> forward_layer(const Layer&, tyloi::state::KvView&, const tyloi::Tensor&, const AttentionStep&,
                     const Model::Options&);

} // namespace tyloi::cuda::gemma4
