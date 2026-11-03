#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/deepseek_v4/model.h"
#include "engine/models/deepseek_v4/state.h"
#include "engine/models/common/input.h"

extern bool g_dsv4_fused_hc;
extern bool g_dsv4_compare_hc_ops;

namespace mfq::cuda::deepseek_v4 {
Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids, const Executor::Mode& = {});
Tensor normalize(const Model&, Executor&, Tensor hidden, int64_t batch, int64_t tokens);
Tensor softcap(const Model&, Executor&, Tensor logits);
Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true, const Executor::Mode& = {});
Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);
Tensor forward_layer(const Layer&, LayerState&, Executor&, Tensor input, Tensor token_ids,
                     const AttentionStep&, const Executor::Mode&);
} // namespace mfq::cuda::deepseek_v4
