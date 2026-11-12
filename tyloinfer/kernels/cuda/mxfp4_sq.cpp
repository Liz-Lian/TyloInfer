#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "mxfp4_sq.h"
#include "env.h"
#include "mxfp4_sq_raw.h"
#include "mxfp4_sq_check.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstdlib>
namespace raw = ::tyloi::cuda::mxfp4_sq_raw;
namespace {
using mfq_jit_sq4::validate;

}
torch::Tensor mxfp4_sq_dequant_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor row_auxiliary,
        std::int64_t bits, std::int64_t outputs,
        std::int64_t width, std::int64_t base,
        std::int64_t q_sum, std::int64_t sq4_rows,
        bool fp32) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    const c10::cuda::CUDAGuard guard(blob.device());
    auto out = torch::empty({outputs, width}, blob.options().dtype(
        fp32 ? at::kFloat : at::kHalf));
    raw::dequant(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),row_auxiliary.data_ptr<std::int32_t>(),out.data_ptr(),q,fp32,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

torch::Tensor mxfp4_sq_matmul_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor row_auxiliary,
        torch::Tensor input,
        std::int64_t bits, std::int64_t outputs,
        std::int64_t width, std::int64_t base,
        std::int64_t q_sum, std::int64_t sq4_rows) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    TORCH_CHECK(input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() && input.size(1) == width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == at::kHalf ||
         input.scalar_type() == at::kFloat),
        "MXFP4-SQ activation requires contiguous rank-2 CUDA FP16/FP32 on the weight device");
    const c10::cuda::CUDAGuard guard(blob.device());
    auto out = torch::empty({input.size(0), outputs}, input.options());
    const int rows = int(input.size(0));
    if (!rows) return out;
    const auto stream=at::cuda::getCurrentCUDAStream().stream();
    const bool half=input.scalar_type()==at::kHalf;
    bool tensor_core=false;
    if(half) {
        const char* disable=tyloi::cuda::env_lookup("TYLOINFER_DISABLE_SQ_TENSOR_CORE");
        const char* force=tyloi::cuda::env_lookup("TYLOINFER_FORCE_SQ_TENSOR_CORE");
        const bool profitable=rows>=32 && std::int64_t(rows)*outputs>=131072;
        tensor_core=rows>8 && (profitable || (force && force[0]=='1')) && (!disable || disable[0]!='1');
    }
    raw::matmul(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),row_auxiliary.data_ptr<std::int32_t>(),input.data_ptr(),out.data_ptr(),q,rows,half?::tyloi::cuda::NumericType::f16 : ::tyloi::cuda::NumericType::f32,tensor_core,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void mxfp4_sq_moe_matmul_cuda(
        torch::Tensor blob,
        torch::Tensor row_q,
        torch::Tensor row_symbol_byte_offsets,
        torch::Tensor row_auxiliary,
        torch::Tensor input,
        torch::Tensor expert_ids,
        torch::Tensor expert_local,
        std::int64_t bits,
        std::int64_t n_experts,
        std::int64_t local_experts,
        std::int64_t out_per_expert,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows,
        torch::Tensor output) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, local_experts * out_per_expert, width, base,
        q_sum, sq4_rows);
    TORCH_CHECK(
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.is_contiguous() && input.scalar_type() == at::kHalf &&
        (input.dim() == 2 || input.dim() == 3) && input.size(-1) == width &&
        expert_ids.is_cuda() && expert_ids.get_device() == blob.get_device() &&
        expert_ids.is_contiguous() && expert_ids.scalar_type() == at::kInt &&
        expert_ids.dim() == 2 && expert_ids.size(0) == input.size(0) &&
        expert_local.is_cuda() && expert_local.get_device() == blob.get_device() &&
        expert_local.is_contiguous() && expert_local.scalar_type() == at::kInt &&
        expert_local.dim() == 1 && expert_local.numel() == n_experts &&
        output.is_cuda() && output.get_device() == blob.get_device() &&
        output.is_contiguous() && output.scalar_type() == at::kHalf &&
        output.dim() == 3 && output.size(0) == expert_ids.size(0) &&
        output.size(1) == expert_ids.size(1) && output.size(2) == out_per_expert &&
        (input.dim() == 2 || input.size(1) == expert_ids.size(1)) &&
        n_experts > 0 && local_experts > 0 && out_per_expert > 0 &&
        n_experts <= std::numeric_limits<int>::max() &&
        local_experts <= std::numeric_limits<int>::max() &&
        out_per_expert <= std::numeric_limits<int>::max(),
        "MXFP4-SQ routed matmul requires matching contiguous CUDA FP16 tensors");
    const auto route_count64 = expert_ids.numel();
    TORCH_CHECK(
        route_count64 <= std::numeric_limits<int>::max(),
        "MXFP4-SQ route count exceeds CUDA limits");
    if (route_count64 == 0) return;
    const c10::cuda::CUDAGuard guard(blob.device());
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    raw::routed(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),row_auxiliary.data_ptr<std::int32_t>(),input.data_ptr(),output.data_ptr(),expert_ids.data_ptr<std::int32_t>(),expert_local.data_ptr<std::int32_t>(),q,int(route_count64),int(expert_ids.size(1)),int(n_experts),int(local_experts),int(out_per_expert),input.dim()==2,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
