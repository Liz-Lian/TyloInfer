#pragma once

#include "backends/cuda/engine/executor/run.h"
#include "engine/executor/input.h"
#include "models/qwen35/model.h"
#include "models/qwen35/state.h"
#include "ops/attention/full.h"

#include <cstddef>
#include <cstdint>

namespace mfq::cuda::qwen35 {

Tensor embed(const Model&, Executor&, Collectives&, int primary, Tensor ids,
             const Executor::Mode& = {});

Tensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);

Tensor normalize(const Model&, Executor&, Tensor hidden, std::int64_t batch,
                 std::int64_t tokens);

Tensor logits(const Model&, Executor&, Tensor hidden, bool capped = true,
              const Executor::Mode& = {});

Tensor softcap(const Model&, Executor&, Tensor logits);

bool supports_speculation(const Model&) noexcept;

void replay(const Model&, State&, Executor&, std::int64_t accepted = 0);

Tensor forward_layer(
    Collectives& collectives,
    const Layer& layer,
    LayerState& state,
    mfq::cuda::Executor& executor,
    std::size_t scratch_offset,
    Tensor input,
    const AttentionStep& step,
    std::int64_t confirmed_prefix,
    const RopeCache& rope,
    const Executor::Mode& options);

} // namespace mfq::cuda::qwen35
