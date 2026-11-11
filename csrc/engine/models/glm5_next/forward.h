#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/glm5_next/model.h"

namespace tyloi::cuda::glm5_next {
#ifndef TYLOINFER_NATIVE_CUDA
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids, const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true, const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#endif
bool supports_speculation(const Model&) noexcept;
} // namespace tyloi::cuda::glm5_next
