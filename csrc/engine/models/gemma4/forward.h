#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/gemma4/model.h"
#include "engine/models/gemma4/state.h"

#include <cstddef>
#include <cstdint>

namespace mfq::cuda::gemma4 {

Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids,
             const Executor::Mode& = {});
Tensor normalize(const Model&, Executor&, Tensor hidden, std::int64_t batch, std::int64_t tokens);
Tensor softcap(const Model&, Executor&, Tensor logits);
Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true,
              const Executor::Mode& = {});
Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
Tensor forward_layer(const Layer&, KVCache&, Executor&, std::size_t scratch_index, Tensor,
                     const AttentionStep&, const Executor::Mode&);

} // namespace mfq::cuda::gemma4
