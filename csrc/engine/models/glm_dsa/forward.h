#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/input.h"
#include "engine/models/glm_dsa/model.h"
#include "engine/models/glm_dsa/state.h"

namespace tyloi::cuda::glm_dsa {
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids, const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true, const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
DenseTensor forward_layer(const Layer&, LayerState&, Selection&, Executor&, DenseTensor, const AttentionStep&,
#ifdef TYLOINFER_NATIVE_CUDA
                     const tyloi::ops::RopeTable&,
#else
                     const RopeCache&,
#endif
                     const Executor::Mode&);
} // namespace tyloi::cuda::glm_dsa
