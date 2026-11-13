#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/common/attention.h"
#include "engine/models/glm_dsa/model.h"
#include "engine/models/glm_dsa/state.h"

namespace tyloi::cuda::glm_dsa {
DenseTensor forward_layer(const Layer&, LayerState&, Selection&, Executor&, DenseTensor, const AttentionStep&,
                     const tyloi::ops::RopeTable&,
                     const Executor::Mode&);
} // namespace tyloi::cuda::glm_dsa
