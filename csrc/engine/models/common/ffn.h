#pragma once

#include "engine/ops.h"

namespace tyloi::cuda { class Ffn; }

namespace tyloi::models {

Tensor full_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output = nullptr);

Tensor linear_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset);

} // namespace tyloi::models
