#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/glm5_next/model.h"

namespace mfq::cuda::glm5_next {
Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids, const Executor::Mode& = {});
Tensor normalize(const Model&, Executor&, Tensor hidden, int64_t batch, int64_t tokens);
Tensor softcap(const Model&, Executor&, Tensor logits);
Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true, const Executor::Mode& = {});
Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
bool supports_speculation(const Model&) noexcept;
void replay(const Model&, State&, Executor&, int64_t accepted = 0);
} // namespace mfq::cuda::glm5_next
