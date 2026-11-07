#pragma once

#include "engine/backends/cuda/executor/executor.h"
#include "engine/models/common/input.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/common/attention.h"

#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::qwen35 {

DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});

DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, Input);

DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, std::int64_t batch,
                 std::int64_t tokens);

DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});

DenseTensor softcap(const Model&, Executor&, DenseTensor logits);

bool supports_speculation(const Model&) noexcept;

void replay(const Model&, State&, Executor&, std::int64_t accepted = 0);

DenseTensor forward_layer(
    Collectives& collectives,
    const Layer& layer,
    LayerState& state,
    tyloi::cuda::Executor& executor,
    std::size_t scratch_offset,
    DenseTensor input,
    const AttentionStep& step,
    std::int64_t confirmed_prefix,
#ifdef TYLOINFER_NATIVE_CUDA
    const tyloi::ops::RopeTable& rope,
#else
    const RopeCache& rope,
#endif
    const Executor::Mode& options);

} // namespace tyloi::cuda::qwen35
