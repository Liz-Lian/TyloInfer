#pragma once
#include "backend.h"
#include <optional>

// One BF16 scalar weight row, Half input, 1..8 tokens. Unsupported views
// return nullopt so the caller keeps its ordinary dense projection.
std::optional<::tyloi::Tensor> try_shared_gate_sigmoid_cuda(
    const ::tyloi::Tensor& input,
    const ::tyloi::Tensor& weight);

struct NintWeight;
std::optional<::tyloi::Tensor> try_nint_shared_gate_sigmoid_cuda(
    const ::tyloi::Tensor& input,const NintWeight& weight,
    const ::tyloi::Tensor& quantized={},const ::tyloi::Tensor& scales={});
