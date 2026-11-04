#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/input.h"
#include "engine/models/nemotron_h/model.h"
#include "engine/models/nemotron_h/state.h"

#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::nemotron_h {

Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids,
             const Executor::Mode& = {});
Tensor normalize(const Model&, Executor&, Tensor hidden, std::int64_t batch, std::int64_t tokens);
Tensor softcap(const Model&, Executor&, Tensor logits);
Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true,
              const Executor::Mode& = {});
Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
Tensor forward_layer(const Layer&, LayerState&, Executor&, std::size_t scratch_index, Tensor,
                     const AttentionStep&, std::int64_t confirmed, const Executor::Mode&);
bool supports_speculation(const Model&) noexcept;
void replay(const Model&, State&, Executor&, std::int64_t accepted = 0);

} // namespace tyloi::cuda::nemotron_h
