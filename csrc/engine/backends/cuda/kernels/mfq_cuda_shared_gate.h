#pragma once
#include "backend.h"
#include <optional>

// One BF16 scalar weight row, Half input, 1..8 tokens. Unsupported views
// return nullopt so the caller keeps its ordinary dense projection.
std::optional<::tyloi::DenseTensor> try_shared_gate_sigmoid_cuda(
    const ::tyloi::DenseTensor& input,
    const ::tyloi::DenseTensor& weight);

#include "engine/weights/encoding/nint.h"
std::optional<::tyloi::DenseTensor> try_nint_shared_gate_sigmoid_cuda(
    const ::tyloi::DenseTensor& input,const tyloi::NintTensor& weight,
    const ::tyloi::DenseTensor& quantized={},const ::tyloi::DenseTensor& scales={});
