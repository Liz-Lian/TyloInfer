#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/nemotron_h/model.h"
#include "engine/models/nemotron_h/state.h"

#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::nemotron_h {

DenseTensor forward_layer(const Layer&, LayerState&, DenseTensor,
                     const AttentionStep&, std::int64_t confirmed,
                     const tyloi::models::ForwardOptions&);
bool supports_speculation(const Model&) noexcept;

} // namespace tyloi::cuda::nemotron_h
