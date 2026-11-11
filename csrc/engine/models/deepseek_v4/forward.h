#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/deepseek_v4/model.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/input.h"

extern bool g_dsv4_fused_hc;
extern bool g_dsv4_compare_hc_ops;

namespace tyloi::cuda::deepseek_v4 {
#ifdef TYLOINFER_NATIVE_CUDA
/** Embed token IDs into model hidden states. */
::tyloi::Tensor embed(const Model&, Executor&, Collectives&, int primary, const ::tyloi::Tensor& ids, const Executor::Mode& = {});
/** Normalize the model output hidden states. */
::tyloi::Tensor normalize(const Model&, Executor&, const ::tyloi::Tensor& hidden, int64_t batch, int64_t tokens);
/** Apply the model logit softcap. */
::tyloi::Tensor softcap(const Model&, Executor&, const ::tyloi::Tensor& logits);
/** Project hidden states to logits, optionally applying the softcap. */
::tyloi::Tensor logits(const Model&, Executor&, const ::tyloi::Tensor& hidden, bool capped = true, const Executor::Mode& = {});
::tyloi::Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#else
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids, const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true, const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#endif
DenseTensor forward_layer(const Layer&, LayerState&, Executor&, DenseTensor input, DenseTensor token_ids,
                     const AttentionStep&, const Executor::Mode&);
} // namespace tyloi::cuda::deepseek_v4
