#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/input.h"
#include "engine/models/glm5_next/model.h"

namespace tyloi::cuda::glm5_next {
#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor embed(const Model&, Executor&, Collectives&, int primary, ::tyloi::Tensor ids, const Executor::Mode& = {});
::tyloi::Tensor normalize(const Model&, Executor&, ::tyloi::Tensor hidden, int64_t batch, int64_t tokens);
::tyloi::Tensor softcap(const Model&, Executor&, ::tyloi::Tensor logits);
::tyloi::Tensor logits(const Model&, Executor&, ::tyloi::Tensor hidden, bool capped = true, const Executor::Mode& = {});
::tyloi::Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#else
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids, const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true, const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#endif
bool supports_speculation(const Model&) noexcept;
void replay(const Model&, State&, Executor&, int64_t accepted = 0);
} // namespace tyloi::cuda::glm5_next
