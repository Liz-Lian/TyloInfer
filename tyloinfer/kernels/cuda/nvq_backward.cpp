#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include "packed_backward.h"
#include "nvq_check.cuh"
#include "nvq_backward_raw.h"

using namespace tyloi::cuda::packed_nvq;
using mfq_jit_nvq::check_common;
using mfq_jit_nvq::check_nepq_common;

torch::Tensor nvq_dequant_cuda(torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);
torch::Tensor nepq_dequant_cuda(torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);

/** Compute NVQ input gradients using the original format-specific path. */
torch::Tensor nvq_backward_input_cuda(
        torch::Tensor indices,
        torch::Tensor aux,
        torch::Tensor sub_scale,
        torch::Tensor neuron_scale,
        torch::Tensor codebook,
        torch::Tensor output_gradient,
        int64_t neuron_len,
        int64_t gs,
        int64_t sub_bits,
        int64_t format,
        int64_t sign_mode) {
    check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == neuron_scale.numel(),
        "NVQ backward output gradient geometry mismatch");
    TORCH_CHECK(
        indices.device() == output_gradient.device() &&
        aux.device() == output_gradient.device() &&
        sub_scale.device() == output_gradient.device() &&
        neuron_scale.device() == output_gradient.device() &&
        codebook.device() == output_gradient.device(),
        "NVQ backward tensors must share one CUDA device");
    const auto dtype = output_gradient.scalar_type();
    TORCH_CHECK(
        dtype == at::kHalf ||
        dtype == at::kFloat ||
        dtype == at::kBFloat16,
        "NVQ backward output gradient must be fp16, bf16, or fp32");
    const int M = static_cast<int>(output_gradient.size(0));
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int nvec = (K + (is_d4_format(format) ? 3 : 7)) /
        (is_d4_format(format) ? 4 : 8);
    const int nsign = (K + 7) / 8;
    auto result = torch::empty({M, K}, output_gradient.options());
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(M) * K;
    if (total == 0) {
        return result;
    }
    c10::cuda::CUDAGuard guard(output_gradient.device());
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    // Wide down projections expose enough independent K tiles to fuse
    // decoding with WMMA; output-wide projections retain the dense path.
    if (dtype == at::kHalf && M >= 5 && M <= 8 &&
            static_cast<int64_t>(K) >= 2 * static_cast<int64_t>(N)) {
        mfq_jit_nvq_backward::Args args{indices.data_ptr(), aux.data_ptr(), sub_scale.data_ptr(),
            neuron_scale.data_ptr(), codebook.data_ptr(), nullptr, output_gradient.data_ptr(), result.data_ptr(),
            indices.numel(), aux.numel(), sub_scale.numel(), M, N, K, ng, nvec, nsign, 0, 0,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), static_cast<int>(format), 0};
        mfq_jit_nvq_backward::nvq(args, true, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    if (dtype == at::kHalf && M <= 4) {
        constexpr int output_tile = 16;
        const int splits = (N + output_tile - 1) / output_tile;
        auto partials = torch::empty(
            {splits, M, K},
            output_gradient.options());
        mfq_jit_nvq_backward::Args args{indices.data_ptr(), aux.data_ptr(), sub_scale.data_ptr(),
            neuron_scale.data_ptr(), codebook.data_ptr(), nullptr, output_gradient.data_ptr(), partials.data_ptr(),
            indices.numel(), aux.numel(), sub_scale.numel(), M, N, K, ng, nvec, nsign, 0, 0,
            static_cast<int>(sub_bits), static_cast<int>(sign_mode), static_cast<int>(format), output_tile};
        mfq_jit_nvq_backward::nvq(args, false, stream);
        mfq_jit_backward::launch_split_half_reduce_to_half(
            reinterpret_cast<const __half *>(partials.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(result.data_ptr<at::Half>()),
            M, K, splits, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = nvq_dequant_cuda(
        indices, aux, sub_scale, neuron_scale, codebook,
        neuron_len, gs, sub_bits, format, sign_mode);
    mfq_jit_backward::launch_dense_half_weight(
        output_gradient, weight, result, M, N, K, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

/** Compute NEPQ input gradients using the original shared-bank path. */
torch::Tensor nepq_backward_input_cuda(
        torch::Tensor indices,
        torch::Tensor aux,
        torch::Tensor state_stream,
        torch::Tensor neuron_scale,
        torch::Tensor table_pool,
        torch::Tensor bank_ids,
        torch::Tensor output_gradient,
        int64_t neuron_len,
        int64_t state_bits,
        int64_t format) {
    check_nepq_common(
        indices, aux, state_stream, neuron_scale, table_pool, bank_ids,
        neuron_len, state_bits, format);
    TORCH_CHECK(
        output_gradient.is_cuda() && output_gradient.is_contiguous() &&
        output_gradient.dim() == 2 &&
        output_gradient.size(1) == neuron_scale.numel(),
        "NEPQ backward output gradient geometry mismatch");
    TORCH_CHECK(
        indices.device() == output_gradient.device() &&
        aux.device() == output_gradient.device() &&
        state_stream.device() == output_gradient.device() &&
        neuron_scale.device() == output_gradient.device() &&
        table_pool.device() == output_gradient.device() &&
        bank_ids.device() == output_gradient.device(),
        "NEPQ backward tensors must share one CUDA device");
    const auto dtype = output_gradient.scalar_type();
    TORCH_CHECK(
        dtype == at::kHalf ||
        dtype == at::kFloat ||
        dtype == at::kBFloat16,
        "NEPQ backward output gradient must be fp16, bf16, or fp32");
    const int M = static_cast<int>(output_gradient.size(0));
    const int N = static_cast<int>(neuron_scale.numel());
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + kGroupSize - 1) / kGroupSize;
    const int nvec = K / 8;
    const int nsign = (K + 7) / 8;
    const int nsuper = (ng + kNepqGroupsPerSupergroup - 1) /
        kNepqGroupsPerSupergroup;
    const int table_stride = static_cast<int>(table_pool.size(1));
    auto result = torch::empty({M, K}, output_gradient.options());
    constexpr int threads = 256;
    const int64_t total = static_cast<int64_t>(M) * K;
    if (total == 0) {
        return result;
    }
    c10::cuda::CUDAGuard guard(output_gradient.device());
    const int blocks = static_cast<int>(std::min<int64_t>(
        (total + threads - 1) / threads, 65535));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    // Fuse dynamic table selection, vector decode, and WMMA when K tiles
    // provide enough parallel work; retain split reduction for output-wide MLPs.
    if (dtype == at::kHalf && M >= 5 && M <= 8 &&
            static_cast<int64_t>(K) >= 2 * static_cast<int64_t>(N)) {
        mfq_jit_nvq_backward::Args args{indices.data_ptr(), aux.data_ptr(), state_stream.data_ptr(),
            neuron_scale.data_ptr(), table_pool.data_ptr(), bank_ids.data_ptr(), output_gradient.data_ptr(), result.data_ptr(),
            indices.numel(), aux.numel(), state_stream.numel(), M, N, K, ng, nvec, nsign, nsuper, table_stride,
            static_cast<int>(state_bits), 0, static_cast<int>(format), 0};
        mfq_jit_nvq_backward::nepq(args, true, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    if (dtype == at::kHalf && M <= 8) {
        const int output_tile = M <= 4 ? 16 : 32;
        const int splits = (N + output_tile - 1) / output_tile;
        auto partials = torch::empty(
            {splits, M, K}, output_gradient.options());
        mfq_jit_nvq_backward::Args args{indices.data_ptr(), aux.data_ptr(), state_stream.data_ptr(),
            neuron_scale.data_ptr(), table_pool.data_ptr(), bank_ids.data_ptr(), output_gradient.data_ptr(), partials.data_ptr(),
            indices.numel(), aux.numel(), state_stream.numel(), M, N, K, ng, nvec, nsign, nsuper, table_stride,
            static_cast<int>(state_bits), 0, static_cast<int>(format), output_tile};
        mfq_jit_nvq_backward::nepq(args, false, stream);
        mfq_jit_backward::launch_split_half_reduce_to_half(
            reinterpret_cast<const __half *>(
                partials.data_ptr<at::Half>()),
            reinterpret_cast<__half *>(result.data_ptr<at::Half>()),
            M, K, splits, stream);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return result;
    }
    auto weight = nepq_dequant_cuda(
        indices, aux, state_stream, neuron_scale, table_pool, bank_ids,
        neuron_len, state_bits, format);
    mfq_jit_backward::launch_dense_half_weight(
        output_gradient, weight, result, M, N, K, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
