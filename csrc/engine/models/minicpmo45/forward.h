#pragma once

#include "engine/models/minicpmo45/model.h"
#include "engine/models/minicpmo45/state.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#endif
#include "engine/models/input.h"

namespace tyloi::cuda::minicpmo45 {

#ifdef TYLOINFER_NATIVE_CUDA
/** Embed token IDs on the requested primary device. */
::tyloi::Tensor embed(const Model&, int primary, const ::tyloi::Tensor& ids);
/** Normalize hidden states for the given batch and token geometry. */
::tyloi::Tensor normalize(const Model&, const ::tyloi::Tensor&, int64_t batch, int64_t tokens);
/** Project hidden states to logits. */
::tyloi::Tensor logits(const Model&, const ::tyloi::Tensor&, bool capped = true);
/** Run a request using its explicit input and forward policy. */
::tyloi::Tensor forward(const Model&, State&, tyloi::models::Input, const tyloi::models::ForwardOptions& = {});
#else
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#endif

} // namespace tyloi::cuda::minicpmo45
