#pragma once

#include "engine/models/minicpmo45/model.h"
#include "engine/models/minicpmo45/state.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/input.h"

namespace tyloi::cuda::minicpmo45 {

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor embed(const Model&, Executor&, Collectives&, int primary, const ::tyloi::Tensor& ids,
             const Executor::Mode& = {});
::tyloi::Tensor normalize(const Model&, Executor&, const ::tyloi::Tensor& hidden, int64_t batch, int64_t tokens);
::tyloi::Tensor logits(const Model&, Executor&, const ::tyloi::Tensor& hidden, bool capped = true,
              const Executor::Mode& = {});
::tyloi::Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
#else
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
#endif

} // namespace tyloi::cuda::minicpmo45
