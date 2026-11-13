#pragma once

#include "engine/models/minicpmo45/text/model.h"
#include "engine/models/minicpmo45/text/state.h"
#include "engine/models/input.h"

namespace tyloi::cuda::minicpmo45::text {
/** Embed token IDs on the requested primary device. */
::tyloi::Tensor embed(const Model&, int primary, const ::tyloi::Tensor& ids);
/** Normalize hidden states for the given batch and token geometry. */
::tyloi::Tensor normalize(const Model&, const ::tyloi::Tensor&, int64_t batch, int64_t tokens);
/** Project hidden states to logits. */
::tyloi::Tensor logits(const Model&, const ::tyloi::Tensor&, bool capped = true);
/** Run a request using its explicit input and forward policy. */
::tyloi::Tensor forward(const Model&, State&, tyloi::models::Input, const tyloi::models::ForwardOptions& = {});
} // namespace tyloi::cuda::minicpmo45::text
