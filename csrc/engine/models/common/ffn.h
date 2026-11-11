#pragma once

#include "engine/models/common/mlp.h"

namespace tyloi::cuda { class Ffn; }
namespace tyloi::ops { struct FfnWeights; }

namespace tyloi::models {

Tensor full_ffn(const Mlp& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output = nullptr);

Tensor full_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output = nullptr);

/** Apply full-attention residual normalization and FFN using public weights. */
Tensor full_ffn(const ops::FfnWeights& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset, bool bf16, bool trace,
    ops::Trace* trace_output = nullptr);

Tensor linear_ffn(const cuda::Ffn& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset);

/** Apply recurrent residual normalization and FFN using public weights. */
Tensor linear_ffn(const ops::FfnWeights& weights, const Tensor& norm,
    const Tensor& residual, const Tensor& attention_output,
    double epsilon, double offset);

} // namespace tyloi::models
