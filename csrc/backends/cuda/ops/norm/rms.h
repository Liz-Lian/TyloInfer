#pragma once

#include "backend.h"

mfq_tensor_backend::Tensor qwen_rms_norm(
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);

mfq_tensor_backend::Tensor qwen_rms_norm_bf16(
    bool fused,
    mfq_tensor_backend::Tensor x,
    mfq_tensor_backend::Tensor weight,
    double eps,
    double weight_offset);
