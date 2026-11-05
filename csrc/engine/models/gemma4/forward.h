#pragma once

#include "engine/models/gemma4/model.h"
#include "engine/models/gemma4/state.h"

#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/input.h"
#endif

#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::gemma4 {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
std::shared_ptr<tyloi::Tensor> forward_layer(const Layer&, tyloi::state::KvView&, const tyloi::Tensor&, const AttentionStep&,
                     const Model::Options&);
#else
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, std::int64_t batch, std::int64_t tokens);
DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
DenseTensor forward_layer(const Layer&, KVCache&, Executor&, std::size_t scratch_index, DenseTensor,
                     const AttentionStep&, const Executor::Mode&);
#endif

} // namespace tyloi::cuda::gemma4
