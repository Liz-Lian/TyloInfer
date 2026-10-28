#pragma once

#include "models/minicpmo45/model.h"
#include "models/minicpmo45/state.h"
#include "engine/executor/run.h"

namespace mfq::cuda::minicpmo45 {

Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids,
             const Executor::Mode& = {});
Tensor normalize(const Model&, Executor&, Tensor hidden, int64_t batch, int64_t tokens);
Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true,
              const Executor::Mode& = {});
Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);

} // namespace mfq::cuda::minicpmo45
