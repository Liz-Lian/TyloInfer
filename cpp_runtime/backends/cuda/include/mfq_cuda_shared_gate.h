#pragma once
#include "mfq_tensor_backend.h"
#include <optional>

// One BF16 scalar weight row and Half inputs. The dot follows the dense
// adapter's BF16 rounding, with FP16 activation/output boundaries. Unsupported
// views return nullopt so the caller keeps its ordinary dense projection.
std::optional<mfq_tensor_backend::Tensor> try_shared_gate_sigmoid_cuda(
    const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& weight);

// FP32 scalar weights: native FP32 dot/sigmoid and expert-output scaling in
// one launch. Keep the ordinary FP32 result; no weight/input requantization.
std::optional<mfq_tensor_backend::Tensor> try_shared_gate_multiply_cuda(
    const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& weight,
    const mfq_tensor_backend::Tensor& expert_output);

struct NintWeight;
std::optional<mfq_tensor_backend::Tensor> try_nint_shared_gate_sigmoid_cuda(
    const mfq_tensor_backend::Tensor& input,const NintWeight& weight,
    const mfq_tensor_backend::Tensor& quantized={},const mfq_tensor_backend::Tensor& scales={});
