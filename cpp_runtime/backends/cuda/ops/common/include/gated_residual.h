#pragma once
#include "mfq_tensor_backend.h"
#include <functional>
#include <vector>

struct CudaExecutionContext;
using GatedResidualProjection = std::function<mfq_tensor_backend::Tensor(
    CudaExecutionContext&, const mfq_tensor_backend::Tensor&)>;
using GatedResidualMixProjection = std::function<mfq_tensor_backend::Tensor(
    CudaExecutionContext&, const mfq_tensor_backend::Tensor&,
    const mfq_tensor_backend::Tensor&, int64_t)>;
using GatedResidualMarker = std::function<void(const char*)>;
using GatedResidualActivationProjection = std::function<mfq_tensor_backend::Tensor(
    CudaExecutionContext&, const mfq_tensor_backend::Tensor&,int64_t,bool)>;

std::vector<mfq_tensor_backend::Tensor> try_gated_residual_bf16_prefill_cuda(
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& norm,
    const mfq_tensor_backend::Tensor& down,const mfq_tensor_backend::Tensor& up,
    const mfq_tensor_backend::Tensor& injection,int64_t streams,double eps,
    const mfq_tensor_backend::Tensor& branch,const mfq_tensor_backend::Tensor& prior_injection,
    const GatedResidualMarker& marker = {});

// Original BF16 matrices, shared across token rows. An empty result asks the
// caller to use its general projection path; no permanent weights are expanded.
std::vector<mfq_tensor_backend::Tensor> try_gated_residual_bf16_cuda(
    const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& norm,
    const mfq_tensor_backend::Tensor& down,const mfq_tensor_backend::Tensor& up,
    const mfq_tensor_backend::Tensor& injection,int64_t streams,double eps,
    const mfq_tensor_backend::Tensor& branch = {},const mfq_tensor_backend::Tensor& prior_injection = {},
    const GatedResidualMarker& marker = {});

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_projected(
    CudaExecutionContext&, const mfq_tensor_backend::Tensor& input,
    const mfq_tensor_backend::Tensor& norm, const GatedResidualProjection& down,
    const GatedResidualProjection& up, const GatedResidualProjection& injection,
    int64_t hidden, int64_t streams, double eps,
    const GatedResidualMixProjection& mixed_projection = {},
    const GatedResidualMarker& marker = {},
    const GatedResidualActivationProjection& activated_down = {},
    const GatedResidualActivationProjection& activated_injection = {});

std::vector<mfq_tensor_backend::Tensor> gated_residual_pre_after_projected(
    CudaExecutionContext&, const mfq_tensor_backend::Tensor& branch,
    const mfq_tensor_backend::Tensor& residual, const mfq_tensor_backend::Tensor& prior_injection,
    const mfq_tensor_backend::Tensor& norm, const GatedResidualProjection& down,
    const GatedResidualProjection& up, const GatedResidualProjection& injection,
    int64_t hidden, int64_t streams, double eps,
    const GatedResidualMixProjection& mixed_projection = {},
    const GatedResidualMarker& marker = {},
    const GatedResidualActivationProjection& activated_down = {},
    const GatedResidualActivationProjection& activated_injection = {});
