#include "backend.h"
#include "env.h"
#include "mxfp4_sq_raw.h"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <cstdlib>
namespace raw = ::tyloi::cuda::mxfp4_sq_raw;
namespace {
tyloi::sq::Layout validate(
        const ::tyloi::DenseTensor& blob,
        const ::tyloi::DenseTensor& row_q,
        const ::tyloi::DenseTensor& row_symbol_byte_offsets,
        const ::tyloi::DenseTensor& row_auxiliary,
        std::int64_t bits,
        std::int64_t outputs,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows) {
    const auto q = bits == 0
        ? tyloi::sq::adaptive_layout(outputs, width, base, q_sum, sq4_rows)
        : tyloi::sq::layout(bits, outputs, width, base);
    TYLOINFER_RUNTIME_CHECK(blob.is_cuda() && blob.scalar_type() == ::tyloi::kUInt8 &&
        blob.dim() == 1 && blob.is_contiguous() && std::size_t(blob.numel()) == q.bytes,
        "MXFP4-SQ blob requires contiguous rank-1 CUDA uint8 and exact payload length");
    TYLOINFER_RUNTIME_CHECK(
        row_q.is_cuda() && row_q.get_device() == blob.get_device() &&
        row_q.scalar_type() == ::tyloi::kUInt8 &&
        row_q.dim() == 1 && row_q.is_contiguous() && row_q.numel() == outputs &&
        row_symbol_byte_offsets.is_cuda() &&
        row_symbol_byte_offsets.get_device() == blob.get_device() &&
        row_symbol_byte_offsets.scalar_type() == ::tyloi::kInt32 &&
        row_symbol_byte_offsets.dim() == 1 &&
        row_symbol_byte_offsets.is_contiguous() &&
        row_symbol_byte_offsets.numel() == outputs &&
        row_auxiliary.is_cuda() && row_auxiliary.get_device() == blob.get_device() &&
        row_auxiliary.scalar_type() == ::tyloi::kInt32 &&
        row_auxiliary.dim() == 1 && row_auxiliary.is_contiguous() &&
        row_auxiliary.numel() == outputs,
        "MXFP4-SQ row metadata requires matching contiguous CUDA arrays");
    TYLOINFER_RUNTIME_CHECK(reinterpret_cast<std::uintptr_t>(blob.data_ptr<std::uint8_t>()) % 4 == 0,
        "MXFP4-SQ blob must be 4-byte aligned");
    return q;
}

}
::tyloi::DenseTensor mxfp4_sq_dequant_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor row_auxiliary,
        std::int64_t bits, std::int64_t outputs,
        std::int64_t width, std::int64_t base,
        std::int64_t q_sum, std::int64_t sq4_rows,
        bool fp32) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    const MfqCudaGuard guard(blob.device());
    auto out = ::tyloi::cuda::empty({outputs, width}, blob.options().dtype(
        fp32 ? ::tyloi::kFloat32 : ::tyloi::kFloat16));
    raw::dequant(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),row_auxiliary.data_ptr<std::int32_t>(),out.data_ptr(),q,fp32,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

::tyloi::DenseTensor mxfp4_sq_matmul_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor row_auxiliary,
        ::tyloi::DenseTensor input,
        std::int64_t bits, std::int64_t outputs,
        std::int64_t width, std::int64_t base,
        std::int64_t q_sum, std::int64_t sq4_rows) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, outputs, width, base, q_sum, sq4_rows);
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.get_device() == blob.get_device() &&
        input.dim() == 2 && input.is_contiguous() && input.size(1) == width &&
        input.size(0) <= std::numeric_limits<int>::max() &&
        (input.scalar_type() == ::tyloi::kFloat16 ||
         input.scalar_type() == ::tyloi::kFloat32),
        "MXFP4-SQ activation requires contiguous rank-2 CUDA FP16/FP32 on the weight device");
    const MfqCudaGuard guard(blob.device());
    auto out = ::tyloi::cuda::empty({input.size(0), outputs}, input.options());
    const int rows = int(input.size(0));
    if (!rows) return out;
    const auto stream=mfq_current_cuda_stream();
    const bool half=input.scalar_type()==::tyloi::kFloat16;
    bool tensor_core=false;
    if(half) {
        const char* disable=tyloi::cuda::env_lookup("TYLOINFER_DISABLE_SQ_TENSOR_CORE");
        const char* force=tyloi::cuda::env_lookup("TYLOINFER_FORCE_SQ_TENSOR_CORE");
        const bool profitable=rows>=32 && std::int64_t(rows)*outputs>=131072;
        tensor_core=rows>8 && (profitable || (force && force[0]=='1')) && (!disable || disable[0]!='1');
    }
    raw::matmul(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),row_auxiliary.data_ptr<std::int32_t>(),input.data_ptr(),out.data_ptr(),q,rows,half?::tyloi::cuda::NumericType::f16 : ::tyloi::cuda::NumericType::f32,tensor_core,stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void mxfp4_sq_moe_matmul_cuda(
        ::tyloi::DenseTensor blob,
        ::tyloi::DenseTensor row_q,
        ::tyloi::DenseTensor row_symbol_byte_offsets,
        ::tyloi::DenseTensor row_auxiliary,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor expert_ids,
        ::tyloi::DenseTensor expert_local,
        std::int64_t bits,
        std::int64_t n_experts,
        std::int64_t local_experts,
        std::int64_t out_per_expert,
        std::int64_t width,
        std::int64_t base,
        std::int64_t q_sum,
        std::int64_t sq4_rows,
        ::tyloi::DenseTensor output) {
    const auto q = validate(
        blob, row_q, row_symbol_byte_offsets, row_auxiliary,
        bits, local_experts * out_per_expert, width, base,
        q_sum, sq4_rows);
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.get_device() == blob.get_device() &&
        input.is_contiguous() && input.scalar_type() == ::tyloi::kFloat16 &&
        (input.dim() == 2 || input.dim() == 3) && input.size(-1) == width &&
        expert_ids.is_cuda() && expert_ids.get_device() == blob.get_device() &&
        expert_ids.is_contiguous() && expert_ids.scalar_type() == ::tyloi::kInt32 &&
        expert_ids.dim() == 2 && expert_ids.size(0) == input.size(0) &&
        expert_local.is_cuda() && expert_local.get_device() == blob.get_device() &&
        expert_local.is_contiguous() && expert_local.scalar_type() == ::tyloi::kInt32 &&
        expert_local.dim() == 1 && expert_local.numel() == n_experts &&
        output.is_cuda() && output.get_device() == blob.get_device() &&
        output.is_contiguous() && output.scalar_type() == ::tyloi::kFloat16 &&
        output.dim() == 3 && output.size(0) == expert_ids.size(0) &&
        output.size(1) == expert_ids.size(1) && output.size(2) == out_per_expert &&
        (input.dim() == 2 || input.size(1) == expert_ids.size(1)) &&
        n_experts > 0 && local_experts > 0 && out_per_expert > 0 &&
        n_experts <= std::numeric_limits<int>::max() &&
        local_experts <= std::numeric_limits<int>::max() &&
        out_per_expert <= std::numeric_limits<int>::max(),
        "MXFP4-SQ routed matmul requires matching contiguous CUDA FP16 tensors");
    const auto route_count64 = expert_ids.numel();
    TYLOINFER_RUNTIME_CHECK(
        route_count64 <= std::numeric_limits<int>::max(),
        "MXFP4-SQ route count exceeds CUDA limits");
    if (route_count64 == 0) return;
    const MfqCudaGuard guard(blob.device());
    const auto stream = mfq_current_cuda_stream();
    raw::routed(blob.data_ptr<std::uint8_t>(),row_q.data_ptr<std::uint8_t>(),row_symbol_byte_offsets.data_ptr<std::int32_t>(),row_auxiliary.data_ptr<std::int32_t>(),input.data_ptr(),output.data_ptr(),expert_ids.data_ptr<std::int32_t>(),expert_local.data_ptr<std::int32_t>(),q,int(route_count64),int(expert_ids.size(1)),int(n_experts),int(local_experts),int(out_per_expert),input.dim()==2,stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}
