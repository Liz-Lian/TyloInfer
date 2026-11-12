#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdlib>
#include <type_traits>
#include <utility>

#include "nint_check.h"
#include "nint_raw.h"
#include "nint_short_decode.h"
#include "env.h"

/** Decode packed NINT8-zero weights to FP16. */
torch::Tensor nint8_zero_dequant_cuda(
        torch::Tensor quantized, torch::Tensor scale, int64_t width) {
    mfq_jit_nint::validate_nint8_zero(quantized, scale);
    TORCH_CHECK(width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 width is invalid");
    const int rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    auto output = torch::empty({rows, width}, quantized.options().dtype(at::kHalf));
    tyloi::cuda::nint_raw::zero_decode(
        quantized.data_ptr<uint8_t>(), scale.data_ptr<at::Half>(),
        output.data_ptr<at::Half>(), rows, groups, width, false,
        at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

/** Launch the routed NINT projection. */
void launch_nint_matmul_routed_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor quantized_input,
        torch::Tensor input_scale,
        torch::Tensor input,
        torch::Tensor route_ids,
        torch::Tensor expert_local,
        bool route_map_ready,
        torch::Tensor ids_dst,
        torch::Tensor expert_bounds,
        torch::Tensor tile_bounds,
        torch::Tensor tile_experts,
        torch::Tensor output,
        int tokens,
        int routes,
        int experts,
        int output_rows,
        int groups,
        int padded_width,
        int input_width,
        int group_size,
        int q_expert_stride,
        int route_tile_m,
        int pool_phase,
        bool masked_experts,
        int epilogue_mode,
        bool routed_input,
        cudaStream_t stream) {
    const bool compact = tokens > 8 && route_map_ready;
    const bool tiled = compact && (route_tile_m == 16 || route_tile_m == 32 ||
        route_tile_m == 64 || route_tile_m == 128);
    if (!tiled) {
        TORCH_CHECK(route_tile_m == 8,
            "NINT routed tile must be 8, 16, 32, 64, or 128");
    }
    tyloi::cuda::nint_raw::Routed a{};
    a.bitstream=bitstream.data_ptr<uint8_t>(); a.row_q_bits=row_q_bits.data_ptr<uint8_t>();
    a.row_q_bit_offsets=row_q_bit_offsets.data_ptr<int64_t>();
    a.subgroup_scale=subgroup_scale.data_ptr<uint8_t>(); a.subgroup_minimum=subgroup_minimum.data_ptr<uint8_t>();
    a.neuron_scale=neuron_scale.data_ptr<float>(); a.neuron_minimum=neuron_minimum.data_ptr<float>();
    a.output=output.data_ptr<at::Half>(); a.expert_local=expert_local.data_ptr<int32_t>();
    if (tiled) a.input=input.data_ptr<at::Half>();
    else {
        a.quantized_input=quantized_input.data_ptr<int8_t>(); a.input_scale=input_scale.data_ptr<float>();
        a.route_ids=route_ids.data_ptr<int32_t>();
    }
    if (compact) {
        a.ids_dst=ids_dst.data_ptr<int32_t>(); a.expert_bounds=expert_bounds.data_ptr<int32_t>();
        a.tile_bounds=tile_bounds.data_ptr<int32_t>(); a.tile_experts=tile_experts.data_ptr<int32_t>();
    }
    a.tokens = tokens;
    a.routes = routes;
    a.experts = experts;
    a.output_rows = output_rows;
    a.groups = groups;
    a.padded_width = padded_width;
    a.input_width = input_width;
    a.group_size = group_size;
    a.q_expert_stride = q_expert_stride;
    a.route_tile_m = route_tile_m;
    a.pool_phase = pool_phase;
    a.epilogue_mode = epilogue_mode;
    a.route_map_ready = route_map_ready;
    a.masked_experts = masked_experts;
    a.routed_input = routed_input;
    tyloi::cuda::nint_raw::routed(a, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Output>
static torch::Tensor nint_decode_impl(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    TORCH_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == at::kByte &&
        bitstream.dim() == 1,
        "NINT bitstream must be contiguous CUDA uint8 rank-1");
    TORCH_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == at::kByte &&
        row_q_bits.dim() == 1,
        "NINT q metadata must be contiguous CUDA uint8 rank-1");
    TORCH_CHECK(
        row_q_bit_offsets.is_cuda() && row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == at::kLong &&
        row_q_bit_offsets.dim() == 1,
        "NINT row offsets must be contiguous CUDA int64 rank-1");
    TORCH_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == at::kByte &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == at::kByte &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    TORCH_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == at::kFloat &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == at::kFloat,
        "NINT neuron metadata is invalid");
    const int rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    TORCH_CHECK(
        rows > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT decode dimensions are invalid");
    TORCH_CHECK(
        row_q_bits.numel() == rows && row_q_bit_offsets.numel() == rows &&
        neuron_scale.numel() == rows && neuron_minimum.numel() == rows,
        "NINT row metadata shape mismatch");
    TORCH_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device(),
        "NINT tensors must share one CUDA device");
    auto output = torch::empty(
        {rows, width},
        neuron_scale.options().dtype(
            std::is_same_v<Output, __half>
                ? at::kHalf
                : at::kFloat));
    tyloi::cuda::nint_raw::Decode a{};
    a.bitstream=bitstream.data_ptr<uint8_t>(); a.row_q_bits=row_q_bits.data_ptr<uint8_t>();
    a.row_q_bit_offsets=row_q_bit_offsets.data_ptr<int64_t>();
    a.subgroup_scale=subgroup_scale.data_ptr<uint8_t>(); a.subgroup_minimum=subgroup_minimum.data_ptr<uint8_t>();
    a.neuron_scale=neuron_scale.data_ptr<float>(); a.neuron_minimum=neuron_minimum.data_ptr<float>();
    a.output=output.data_ptr(); a.bytes=uint64_t(bitstream.numel()); a.rows=rows; a.groups=groups;
    a.group_size=static_cast<int>(group_size); a.width=width;
    a.output_float=std::is_same_v<Output,float>;
    a.dense_reference=!a.output_float && (group_size == 24 || group_size == 28) &&
        width % 4 == 0 && (reinterpret_cast<std::uintptr_t>(a.bitstream) & 3u) == 0 &&
        tyloi::cuda::env_lookup("TYLOINFER_NINT_DENSE_REFERENCE") != nullptr;
    tyloi::cuda::nint_raw::decode(a,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor nint_decode_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    return nint_decode_impl<__half>(
        std::move(bitstream), std::move(row_q_bits),
        std::move(row_q_bit_offsets), std::move(subgroup_scale),
        std::move(subgroup_minimum), std::move(neuron_scale),
        std::move(neuron_minimum), width, group_size);
}


torch::Tensor nint_decode_f32_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    return nint_decode_impl<float>(
        std::move(bitstream), std::move(row_q_bits),
        std::move(row_q_bit_offsets), std::move(subgroup_scale),
        std::move(subgroup_minimum), std::move(neuron_scale),
        std::move(neuron_minimum), width, group_size);
}

torch::Tensor cublas_gemm_nt_f32_output(
        const torch::Tensor & input,
        const torch::Tensor & weight) {
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = torch::empty(
        {rows, outputs},
        input.options().dtype(at::kFloat));
    const auto status = tyloi::cuda::nint_raw::gemm_nt(
        at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream().stream(), input.data_ptr(),
        weight.data_ptr(), result.data_ptr(), rows, outputs, width, true, true);
    TORCH_CHECK(status.stream == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream())");
    TORCH_CHECK(status.gemm == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasGemmEx(...)");
    return result;
}

torch::Tensor nint_cublas_gemm_nt_f16acc_cuda(
        torch::Tensor input,
        torch::Tensor weight) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == at::kHalf &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == at::kHalf &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    TORCH_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = torch::empty({rows, outputs}, input.options());
    const auto status = tyloi::cuda::nint_raw::gemm_nt(
        at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream().stream(), input.data_ptr(),
        weight.data_ptr(), result.data_ptr(), rows, outputs, width, false, false);
    TORCH_CHECK(status.stream == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream())");
    TORCH_CHECK(status.gemm == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasGemmEx(...)");
    return result;
}


torch::Tensor nint_cublas_gemm_nt_f32acc_cuda(
        torch::Tensor input,
        torch::Tensor weight) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == at::kHalf &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    TORCH_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == at::kHalf &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    TORCH_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = torch::empty({rows, outputs}, input.options());
    const auto status = tyloi::cuda::nint_raw::gemm_nt(
        at::cuda::getCurrentCUDABlasHandle(), at::cuda::getCurrentCUDAStream().stream(), input.data_ptr(),
        weight.data_ptr(), result.data_ptr(), rows, outputs, width, true, false);
    TORCH_CHECK(status.stream == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasSetStream(handle, at::cuda::getCurrentCUDAStream().stream())");
    TORCH_CHECK(status.gemm == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasGemmEx(...)");
    return result;
}



torch::Tensor nint8_zero_gemv_ws_cuda(
        torch::Tensor quantized,
        torch::Tensor scale,
        torch::Tensor input,
        torch::Tensor quantized_input,
        torch::Tensor input_scale) {
    mfq_jit_nint::validate_nint8_zero(quantized, scale);
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        (input.scalar_type() == at::kHalf ||
         input.scalar_type() == at::kFloat) &&
        input.dim() == 2,
        "NINT8-0 input must be contiguous CUDA fp16/fp32 rank-2");
    const int activation_rows = static_cast<int>(input.size(0));
    const int output_rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * 32;
    TORCH_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        real_width <= padded_width,
        "NINT8-0 packed matmul expects M in [1,8] and K <= packed K");
    TORCH_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == at::kChar &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == at::kFloat &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT8-0 activation workspace is invalid");
    TORCH_CHECK(
        quantized.device() == input.device() && scale.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device(),
        "NINT8-0 tensors and workspace must share one CUDA device");
    auto output = torch::empty(
        {activation_rows, output_rows}, input.options().dtype(at::kHalf));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    tyloi::cuda::nint_raw::zero_gemv(quantized.data_ptr<uint8_t>(),
        scale.data_ptr<at::Half>(), input.data_ptr(), quantized_input.data_ptr<int8_t>(),
        input_scale.data_ptr<float>(), output.data_ptr<at::Half>(), activation_rows,
        output_rows, groups, real_width, input.scalar_type() == at::kFloat, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor nint8_zero_mmq_f16_packed_cuda(
        torch::Tensor quantized,
        torch::Tensor scale,
        torch::Tensor input,
        int64_t width) {
    mfq_jit_nint::validate_nint8_zero(quantized, scale);
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == at::kHalf &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return nint_cublas_gemm_nt_f16acc_cuda(input, weight);
}


torch::Tensor nint8_zero_mmq_f32_packed_cuda(
        torch::Tensor quantized,
        torch::Tensor scale,
        torch::Tensor input,
        int64_t width) {
    mfq_jit_nint::validate_nint8_zero(quantized, scale);
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == at::kHalf &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 FP32 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return cublas_gemm_nt_f32_output(input, weight);
}

std::vector<torch::Tensor> nint8_one_quantize_reconstruct_cuda(
        torch::Tensor input) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == at::kHalf &&
        input.dim() == 2,
        "NINT8-1 input must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    TORCH_CHECK(rows > 0 && width > 0, "NINT8-1 input must be non-empty");
    const int groups = (width + 31) / 32;
    auto quantized = torch::empty(
        {rows, groups, 32},
        input.options().dtype(at::kChar));
    auto scale = torch::empty({rows, groups}, input.options());
    auto sum = torch::empty({rows, groups}, input.options());
    auto reconstructed = torch::empty_like(input);
    tyloi::cuda::nint_raw::one_quantize(input.data_ptr<at::Half>(),quantized.data_ptr<int8_t>(),
        scale.data_ptr<at::Half>(),sum.data_ptr<at::Half>(),reconstructed.data_ptr<at::Half>(),
        rows,width,groups,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {quantized, scale, sum, reconstructed};
}

static torch::Tensor nint_matmul_ws_impl(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor input,
        const torch::Tensor * gate,
        int64_t activation_mode,
        int64_t group_size,
        torch::Tensor quantized_input,
        torch::Tensor input_scale,
        bool aligned_q8 = false,int uniform_bits = 0) {
    TORCH_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        (input.scalar_type() == at::kHalf ||
         input.scalar_type() == at::kFloat) &&
        input.dim() == 2,
        "NINT input must be contiguous CUDA fp16/fp32 rank-2");
    if (gate != nullptr) {
        TORCH_CHECK(
            activation_mode == 1 || activation_mode == 2,
            "NINT input gate mode must be sigmoid or SiLU");
        TORCH_CHECK(
            gate->is_cuda() && gate->is_contiguous() &&
            (gate->scalar_type() == at::kHalf ||
             gate->scalar_type() == at::kFloat) &&
            gate->dim() == 2 && gate->sizes() == input.sizes(),
            "NINT gate must be contiguous CUDA fp16/fp32 and match input shape");
    } else {
        TORCH_CHECK(
            activation_mode == 0,
            "NINT activation mode requires a gate tensor");
    }
    TORCH_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == at::kByte &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == at::kByte &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    TORCH_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == at::kByte &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == at::kByte &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == at::kLong &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    TORCH_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == at::kFloat &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == at::kFloat,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(input.size(0));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * static_cast<int>(group_size);
    TORCH_CHECK(
        activation_rows >= 1 && activation_rows <= 8,
        "NINT packed matmul supports M in [1,8]");
    TORCH_CHECK(
        group_size >= 4 && group_size <= 64 && real_width <= padded_width,
        "NINT packed matmul dimensions are invalid");
    TORCH_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT row metadata shape mismatch");
    TORCH_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == at::kChar &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == at::kFloat &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT activation workspace is invalid");
    TORCH_CHECK(
        bitstream.device() == input.device() &&
        row_q_bits.device() == input.device() &&
        row_q_bit_offsets.device() == input.device() &&
        subgroup_scale.device() == input.device() &&
        subgroup_minimum.device() == input.device() &&
        neuron_scale.device() == input.device() &&
        neuron_minimum.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device() &&
        (gate == nullptr || gate->device() == input.device()),
        "NINT tensors and workspace must share one CUDA device");
    auto output = torch::empty(
        {activation_rows, output_rows}, input.options().dtype(at::kHalf));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const char* dense_group_setting = tyloi::cuda::env_lookup("TYLOINFER_NINT_GROUP_DOT");
    const char* whole_setting_for_sum = tyloi::cuda::env_lookup("TYLOINFER_NINT_WHOLE_GROUP");
    const char* batch_setting_for_sum = tyloi::cuda::env_lookup("TYLOINFER_NINT_BATCH_WHOLE_GROUP");
    const char* sum_setting = tyloi::cuda::env_lookup("TYLOINFER_NINT_ACTIVATION_SUM");
    const auto* interleaved_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_INTERLEAVED");
    const bool interleaved_default=activation_rows>=3 && activation_rows<=6;
    const bool interleaved_requested=interleaved_setting?interleaved_setting[0]=='1':interleaved_default;
    const auto* float_whole_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_FLOAT_WHOLE_GROUP");
    const bool whole_input=input.scalar_type()==at::kHalf ||
        (input.scalar_type()==at::kFloat && (!float_whole_setting || float_whole_setting[0]!='0'));
    const bool sum_default=nint_default_activation_sum(int(group_size),groups,activation_rows,uniform_bits,interleaved_requested);
    const char* q8_decode_option=tyloi::cuda::env_lookup("TYLOINFER_NINT_Q8_DECODE");
    const int q8_decode=q8_decode_option?std::atoi(q8_decode_option):2;
    const bool grouped_router=output_rows==512 && groups==54;
    const bool grouped_q8=aligned_q8 && group_size==48 && q8_decode>0 &&
        (reinterpret_cast<uintptr_t>(bitstream.data_ptr())&3u)==0 &&
        (activation_rows>1 || output_rows>2048 || q8_decode==3 || grouped_router);
    const auto* q8_sum_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_Q8_SUM");
    // Large vocabulary projections amortize this once-per-activation sum.
    // Tiny router/alpha/beta matrices keep their existing launch costs.
    const bool head_sum_default=output_rows>=65536 && groups==54 && activation_rows<=6;
    const bool q8_sums=grouped_q8 && q8_decode>=2 &&
        (q8_sum_setting?q8_sum_setting[0]=='1':head_sum_default);
    const bool cached_packed_sums = (sum_setting?sum_setting[0]=='1':sum_default) && !aligned_q8 &&
        (group_size==24 || group_size==28) && groups>32 &&
        whole_input &&
        (reinterpret_cast<uintptr_t>(bitstream.data_ptr())&3u)==0 &&
        (!dense_group_setting || dense_group_setting[0]!='0') &&
        (!whole_setting_for_sum || whole_setting_for_sum[0]!='0') &&
        (activation_rows==1 || !batch_setting_for_sum || batch_setting_for_sum[0]!='0') &&
        tyloi::cuda::env_lookup("TYLOINFER_NINT_DENSE_REFERENCE")==nullptr;
    const bool cached_sums=cached_packed_sums || q8_sums;
    const auto* fixed_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_FIXED_BITS");
    const bool interleaved=interleaved_requested &&
        uniform_bits==5 && group_size==28 && groups>32 && groups<=96 &&
        activation_rows>=2 && activation_rows<=6 && !aligned_q8 &&
        whole_input &&
        (reinterpret_cast<uintptr_t>(bitstream.data_ptr())&3u)==0 &&
        (!dense_group_setting || dense_group_setting[0]!='0') &&
        (!whole_setting_for_sum || whole_setting_for_sum[0]!='0') &&
        (!batch_setting_for_sum || batch_setting_for_sum[0]!='0') &&
        (!fixed_setting || fixed_setting[0]!='0') &&
        tyloi::cuda::env_lookup("TYLOINFER_NINT_DENSE_REFERENCE")==nullptr;
    const auto* exact_batch=tyloi::cuda::env_lookup("TYLOINFER_NINT_EXACT_BATCH");
    const int interleaved_columns=activation_rows<=2?2:activation_rows<=4?4:
        (exact_batch && exact_batch[0]=='0')?8:activation_rows;
    torch::Tensor interleaved_activation;
    if(interleaved)interleaved_activation=torch::empty(
        {groups,7,interleaved_columns,4},input.options().dtype(at::kChar));
    torch::Tensor activation_sums;
    if(cached_sums)activation_sums=torch::empty(
        {activation_rows,groups},input.options().dtype(at::kInt));
    {
        tyloi::cuda::nint_raw::Quantize a{};
        a.input=input.data_ptr(); a.gate=gate ? gate->data_ptr() : nullptr;
        a.codes=quantized_input.data_ptr<int8_t>(); a.scale=input_scale.data_ptr<float>();
        a.rows=activation_rows; a.width=real_width; a.padded_width=padded_width;
        a.groups=groups; a.group_size=int(group_size); a.activation_mode=int(activation_mode);
        a.input_float=input.scalar_type()==at::kFloat;
        a.gate_float=gate && gate->scalar_type()==at::kFloat;
        a.sums=activation_sums.defined()?activation_sums.data_ptr<int32_t>():nullptr;
        a.interleaved=interleaved_activation.defined()?interleaved_activation.data_ptr<int8_t>():nullptr;
        a.cached_sums=cached_sums; a.interleave=interleaved; a.columns=interleaved_columns;
        tyloi::cuda::nint_raw::quantize(a,stream);
    }
    tyloi::cuda::nint_raw::Projection a{};
    auto & q=a.q8; auto & w=q.weight;
    w.bitstream=bitstream.data_ptr<uint8_t>(); w.row_q_bits=row_q_bits.data_ptr<uint8_t>();
    w.row_q_bit_offsets=row_q_bit_offsets.data_ptr<int64_t>();
    w.subgroup_scale=subgroup_scale.data_ptr<uint8_t>(); w.subgroup_minimum=subgroup_minimum.data_ptr<uint8_t>();
    w.neuron_scale=neuron_scale.data_ptr<float>(); w.neuron_minimum=neuron_minimum.data_ptr<float>();
    w.bytes=uint64_t(bitstream.numel()); w.rows=output_rows; w.groups=groups; w.group_size=int(group_size);
    q.input=quantized_input.data_ptr<int8_t>(); q.input_scale=input_scale.data_ptr<float>();
    q.sums=activation_sums.defined()?activation_sums.data_ptr<int32_t>():nullptr;
    q.output=output.data_ptr<at::Half>(); q.rows=activation_rows; q.padded_width=padded_width;
    q.grouped=grouped_q8; q.q8_decode=q8_decode;
    a.interleaved=interleaved_activation.defined()?interleaved_activation.data_ptr<int8_t>():nullptr;
    a.dense_group_setting=dense_group_setting; a.fixed_setting=fixed_setting;
    a.real_width=real_width; a.uniform_bits=uniform_bits; a.device=input.get_device();
    a.aligned_q8=aligned_q8; a.whole_input=whole_input; a.cached_sums=cached_sums;
    const int error=tyloi::cuda::nint_raw::project(a,stream);
    TORCH_CHECK(error != 1,"NINT shared activation device query failed");
    TORCH_CHECK(error != 2,"NINT processor count query failed");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


torch::Tensor nint_matmul_fixed_bits_ws_cuda(
        torch::Tensor bitstream,torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,torch::Tensor input,
        int64_t group_size,torch::Tensor quantized_input,
        torch::Tensor input_scale,int uniform_bits) {
    return nint_matmul_ws_impl(bitstream,row_q_bits,row_q_bit_offsets,subgroup_scale,
        subgroup_minimum,neuron_scale,neuron_minimum,input,nullptr,0,group_size,
        quantized_input,input_scale,false,uniform_bits);
}

torch::Tensor nint_matmul_ws_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor input,
        int64_t group_size,
        torch::Tensor quantized_input,
        torch::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, nullptr, 0, group_size, quantized_input, input_scale);
}


torch::Tensor nint_matmul_q8_ws_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor input,
        int64_t group_size,
        torch::Tensor quantized_input,
        torch::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, nullptr, 0, group_size, quantized_input, input_scale, true);
}






torch::Tensor nint_matmul_input_mul_fixed_bits_ws_cuda(
        torch::Tensor bitstream,torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,torch::Tensor input,
        torch::Tensor gate,int64_t activation_mode,int64_t group_size,
        torch::Tensor quantized_input,torch::Tensor input_scale,int uniform_bits) {
    return nint_matmul_ws_impl(bitstream,row_q_bits,row_q_bit_offsets,subgroup_scale,
        subgroup_minimum,neuron_scale,neuron_minimum,input,&gate,activation_mode,
        group_size,quantized_input,input_scale,false,uniform_bits);
}

torch::Tensor nint_matmul_input_mul_ws_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor input,
        torch::Tensor gate,
        int64_t activation_mode,
        int64_t group_size,
        torch::Tensor quantized_input,
        torch::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, &gate, activation_mode, group_size,
        quantized_input, input_scale);
}


torch::Tensor nint_matmul_input_mul_q8_ws_cuda(
        torch::Tensor bitstream,
        torch::Tensor row_q_bits,
        torch::Tensor row_q_bit_offsets,
        torch::Tensor subgroup_scale,
        torch::Tensor subgroup_minimum,
        torch::Tensor neuron_scale,
        torch::Tensor neuron_minimum,
        torch::Tensor input,
        torch::Tensor gate,
        int64_t activation_mode,
        int64_t group_size,
        torch::Tensor quantized_input,
        torch::Tensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, &gate, activation_mode, group_size,
        quantized_input, input_scale, true);
}
