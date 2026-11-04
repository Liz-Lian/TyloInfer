#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/input.h"
#include "engine/models/glm_dsa/model.h"
#include "engine/models/glm_dsa/state.h"

namespace tyloi::cuda::glm_dsa {
Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids, const Executor::Mode& = {});
Tensor normalize(const Model&, Executor&, Tensor hidden, int64_t batch, int64_t tokens);
Tensor softcap(const Model&, Executor&, Tensor logits);
Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true, const Executor::Mode& = {});
Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
Tensor forward_layer(const Layer&, LayerState&, Selection&, Executor&, Tensor, const AttentionStep&,
                     const RopeCache&, const Executor::Mode&);
} // namespace tyloi::cuda::glm_dsa
