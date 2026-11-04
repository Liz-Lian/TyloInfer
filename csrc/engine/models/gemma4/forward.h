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
Tensor forward_layer(const Layer&, tyloi::state::KvView&, Tensor, const AttentionStep&,
                     const Model::Options&);
#else
Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids,
             const Executor::Mode& = {});
Tensor normalize(const Model&, Executor&, Tensor hidden, std::int64_t batch, std::int64_t tokens);
Tensor softcap(const Model&, Executor&, Tensor logits);
Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true,
              const Executor::Mode& = {});
Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
Tensor forward_layer(const Layer&, KVCache&, Executor&, std::size_t scratch_index, Tensor,
                     const AttentionStep&, const Executor::Mode&);
#endif

} // namespace tyloi::cuda::gemma4
