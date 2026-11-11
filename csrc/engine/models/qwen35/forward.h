#pragma once

#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#endif
#include "engine/models/input.h"
#include "engine/models/qwen35/model.h"
#include "engine/models/qwen35/state.h"
#include "engine/models/common/attention.h"

#include <cstddef>
#include <cstdint>

namespace tyloi::cuda::qwen35 {

#ifdef TYLOINFER_NATIVE_CUDA
/** Embed token IDs on the requested primary device. */
::tyloi::Tensor embed(const Model&, int primary, const ::tyloi::Tensor& ids);
/** Execute a request using its explicit positions and forward options. */
::tyloi::Tensor forward(const Model&, State&, tyloi::models::Input,
                        const tyloi::models::ForwardOptions& = {});
/** Apply the final hidden-state normalization. */
::tyloi::Tensor normalize(const Model&, const ::tyloi::Tensor& hidden, std::int64_t batch,
                          std::int64_t tokens);
/** Project hidden states to logits, optionally applying the configured cap. */
::tyloi::Tensor logits(const Model&, const ::tyloi::Tensor& hidden, bool capped = true);
/** Apply the configured logit cap. */
::tyloi::Tensor softcap(const Model&, const ::tyloi::Tensor& logits);
#else
DenseTensor embed(const Model&, Executor&, Collectives&, int primary, DenseTensor ids,
             const Executor::Mode& = {});

DenseTensor forward(const Model&, State&, Executor&, Collectives&, const Executor::Mode&, tyloi::models::Input);

DenseTensor normalize(const Model&, Executor&, DenseTensor hidden, std::int64_t batch,
                 std::int64_t tokens);

DenseTensor logits(const Model&, Executor&, DenseTensor hidden, bool capped = true,
              const Executor::Mode& = {});

DenseTensor softcap(const Model&, Executor&, DenseTensor logits);
#endif

/** Return whether the model can replay an accepted speculative prefix. */
bool supports_speculation(const Model&) noexcept;

#ifdef TYLOINFER_NATIVE_CUDA
/** Restore and replay the accepted speculative prefix. */
void replay(const Model&, State&, std::int64_t accepted = 0);
#else
void replay(const Model&, State&, Executor&, std::int64_t accepted = 0);
#endif

#ifdef TYLOINFER_NATIVE_CUDA
/** Execute one layer using its placement and the current execution scope. */
::tyloi::Tensor forward_layer(const Layer&, LayerState&, ::tyloi::Tensor input,
    const AttentionStep&, std::int64_t confirmed_prefix, const tyloi::ops::RopeTable&,
    const tyloi::models::ForwardOptions&);
#else
DenseTensor forward_layer(
    Collectives& collectives,
    const Layer& layer,
    LayerState& state,
    tyloi::cuda::Executor& executor,
    std::size_t scratch_offset,
    DenseTensor input,
    const AttentionStep& step,
    std::int64_t confirmed_prefix,
    const RopeCache& rope,
    const Executor::Mode& options);
#endif

} // namespace tyloi::cuda::qwen35
