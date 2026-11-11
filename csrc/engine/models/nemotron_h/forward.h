#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/nemotron_h/model.h"
#include "engine/models/nemotron_h/state.h"

#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::nemotron_h {

#ifndef TYLOINFER_NATIVE_CUDA
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, std::int64_t batch, std::int64_t tokens);
DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
DenseTensor forward_layer(const Layer&, LayerState&, DenseTensor,
                     const AttentionStep&, std::int64_t confirmed,
                     const tyloi::models::ForwardOptions&);
#else
DenseTensor forward_layer(const Layer&, LayerState&, Executor&, std::size_t scratch_index, DenseTensor,
                     const AttentionStep&, std::int64_t confirmed, const Executor::Mode&);
#endif
bool supports_speculation(const Model&) noexcept;

} // namespace tyloi::cuda::nemotron_h
