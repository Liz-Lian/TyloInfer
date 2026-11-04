#pragma once

#include "engine/models/minicpmo45/model.h"
#include "engine/models/minicpmo45/state.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/input.h"

namespace tyloi::cuda::minicpmo45 {

DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);

} // namespace tyloi::cuda::minicpmo45
