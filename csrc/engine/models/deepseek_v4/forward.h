#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/deepseek_v4/model.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/input.h"

extern bool g_dsv4_fused_hc;
extern bool g_dsv4_compare_hc_ops;

namespace tyloi::cuda::deepseek_v4 {
#ifndef TYLOINFER_NATIVE_CUDA
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids, const Executor::Mode& = {});
DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, int64_t batch, int64_t tokens);
DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true, const Executor::Mode& = {});
DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);
#endif
#ifdef TYLOINFER_NATIVE_CUDA
/** Run one layer in the caller's active execution scope. */
::tyloi::Tensor forward_layer(const Layer&, LayerState&, const ::tyloi::Tensor& input,
    const ::tyloi::Tensor& token_ids, const AttentionStep&, const tyloi::models::ForwardOptions&);
/** Reduce HC streams and apply the original final normalization. */
::tyloi::Tensor finalize(const Model&, const ::tyloi::Tensor&, int64_t batch, int64_t tokens);
#else
DenseTensor forward_layer(const Layer&, LayerState&, Executor&, DenseTensor input, DenseTensor token_ids,
                     const AttentionStep&, const Executor::Mode&);
#endif
} // namespace tyloi::cuda::deepseek_v4
