#pragma once
#include "mfq_tensor_backend.h"
#include "gated_residual.h"
struct GgufWeight;
// Empty asks the caller to retain its general projection+mix path.
mfq_tensor_backend::Tensor try_gguf_residual_mix(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& normalized,
    std::int64_t streams);
std::vector<mfq_tensor_backend::Tensor> try_gguf_gated_residual_pre(
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& norm,
    const GgufWeight& down,const GgufWeight& up,const GgufWeight& injection,
    std::int64_t streams,double eps,const mfq_tensor_backend::Tensor& branch={},
    const mfq_tensor_backend::Tensor& prior_injection={},const GatedResidualMarker& marker={});
