#pragma once
#include "mfq_tensor_backend.h"

// Cached causal windows use device-resident sequence lengths, including
// changing lengths during Graph replay. Empty means the MMA resources or
// operand geometry are unsupported; callers retain their ordinary kernel.
mfq_tensor_backend::Tensor try_attention_mma_verify_cuda(
    mfq_tensor_backend::Tensor query,mfq_tensor_backend::Tensor keys,
    mfq_tensor_backend::Tensor values,mfq_tensor_backend::Tensor lengths,
    double scale,int64_t planned_length);
