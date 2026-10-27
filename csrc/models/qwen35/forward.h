#pragma once

#include "backends/cuda/engine/executor/run.h"
#include "models/qwen35/model.h"
#include "models/qwen35/state.h"
#include "ops/attention/full.h"

#include <cstddef>
#include <cstdint>

namespace mfq::cuda::qwen35 {

struct Options {
    FfnExec::Mode ffn;
    LinearAttention::Options linear;
    std::int64_t kv_capacity = 0;
    bool transposed = true;
    Trace* trace = nullptr;
    int trace_layer = -1;
};

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
    const Options& options);

} // namespace mfq::cuda::qwen35
