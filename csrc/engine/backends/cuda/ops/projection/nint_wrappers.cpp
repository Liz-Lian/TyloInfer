#include "backend.h"
#include "nint_raw.h"
#include "nint_short_decode.h"
#include "mfq_cuda_nint_route_hint.h"
#include "env.h"
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cuda_fp16.h>
#include <type_traits>
#include <utility>

/** Launch the routed NINT projection. */
void launch_nint_matmul_routed_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        ::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor route_ids,
        ::tyloi::DenseTensor expert_local,
        bool route_map_ready,
        ::tyloi::DenseTensor ids_dst,
        ::tyloi::DenseTensor expert_bounds,
        ::tyloi::DenseTensor tile_bounds,
        ::tyloi::DenseTensor tile_experts,
        ::tyloi::DenseTensor output,
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
        TYLOINFER_RUNTIME_CHECK(route_tile_m == 8,
            "NINT routed tile must be 8, 16, 32, 64, or 128");
    }
    tyloi::cuda::nint_raw::Routed a{};
    a.bitstream=bitstream.data_ptr<uint8_t>(); a.row_q_bits=row_q_bits.data_ptr<uint8_t>();
    a.row_q_bit_offsets=row_q_bit_offsets.data_ptr<int64_t>();
    a.subgroup_scale=subgroup_scale.data_ptr<uint8_t>(); a.subgroup_minimum=subgroup_minimum.data_ptr<uint8_t>();
    a.neuron_scale=neuron_scale.data_ptr<float>(); a.neuron_minimum=neuron_minimum.data_ptr<float>();
    a.output=output.data_ptr<mfq_half>(); a.expert_local=expert_local.data_ptr<int32_t>();
    if (tiled) a.input=input.data_ptr<mfq_half>();
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
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

static void nint_moe_grouped_matmul_dispatch_cuda(
        ::tyloi::DenseTensor pointers,::tyloi::DenseTensor params,
        ::tyloi::DenseTensor expert_pool,::tyloi::DenseTensor expert_local,
        ::tyloi::DenseTensor ids,::tyloi::DenseTensor output,
        int64_t input_width,bool routed_input,int group_hint,int bits_hint) {
    TYLOINFER_RUNTIME_CHECK(pointers.is_cuda() && pointers.is_contiguous() && pointers.scalar_type()==::tyloi::kInt64 &&
        pointers.dim()==2 && pointers.size(1)==9 && pointers.size(0)>0 &&
        params.is_cuda() && params.is_contiguous() && params.scalar_type()==::tyloi::kInt32 &&
        params.dim()==2 && params.size(0)==pointers.size(0) && params.size(1)==3,
        "mixed NINT pointer/geometry table disagrees");
    TYLOINFER_RUNTIME_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.scalar_type()==::tyloi::kInt32 && ids.dim()==2 &&
        ids.size(0)>0 && ids.size(0)<=8 && ids.size(1)>0 && ids.numel()<=65535 && input_width>0 &&
        expert_pool.is_cuda() && expert_pool.is_contiguous() && expert_pool.scalar_type()==::tyloi::kInt32 &&
        expert_pool.dim()==1 && expert_pool.numel()>0 && expert_pool.numel()<=4096 &&
        expert_local.is_cuda() && expert_local.is_contiguous() && expert_local.scalar_type()==::tyloi::kInt32 &&
        expert_local.sizes()==expert_pool.sizes() && output.is_cuda() && output.is_contiguous() &&
        output.scalar_type()==::tyloi::kFloat16 && output.dim()==3 && output.size(0)==ids.size(0) &&
        output.size(1)==ids.size(1) && output.size(2)>0 && output.size(2)<=INT_MAX &&
        pointers.device()==output.device() && params.device()==output.device() && ids.device()==output.device() &&
        expert_pool.device()==output.device() && expert_local.device()==output.device(),
        "mixed NINT route/output geometry or device disagrees");
    MfqCudaGuard guard(output.device());
    tyloi::cuda::nint_raw::Mixed a{};
    a.pointers=pointers.data_ptr<int64_t>(); a.params=params.data_ptr<int32_t>();
    a.expert_pool=expert_pool.data_ptr<int32_t>(); a.expert_local=expert_local.data_ptr<int32_t>();
    a.ids=ids.data_ptr<int32_t>(); a.output=output.data_ptr();
    a.pairs=int(ids.numel()); a.routes=int(ids.size(1)); a.experts=int(expert_pool.numel());
    a.outputs=int(output.size(2)); a.input_width=input_width; a.routed_input=routed_input;
    a.group_hint=group_hint; a.bits_hint=bits_hint;
    a.grouped=tyloi::cuda::env_lookup("TYLOINFER_NINT_GROUP_DOT");
    a.lanes=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_LANES");
    a.blocks=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_BLOCKS");
    a.rows=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_ROWS");
    a.hint=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_HINT");
    if ((!a.grouped || a.grouped[0]!='0') && (!a.hint || a.hint[0]!='0') &&
        (group_hint==24 || group_hint==28) && bits_hint>=4 && bits_hint<=6) {
        a.warps=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_WARPS");
        a.late_scale=nint_late_scale_enabled(input_width,bits_hint,output.get_device());
    }
    const auto *multi_sum=tyloi::cuda::env_lookup("TYLOINFER_CUDA_WARP_MULTI_SUM");
    a.multi_sum=multi_sum && multi_sum[0]=='1' && multi_sum[1]=='\0';
    tyloi::cuda::nint_raw::mixed(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
}

void nint_moe_grouped_matmul_hetero_cuda(
        ::tyloi::DenseTensor pointers,::tyloi::DenseTensor params,
        ::tyloi::DenseTensor expert_pool,::tyloi::DenseTensor expert_local,
        ::tyloi::DenseTensor ids,::tyloi::DenseTensor output,
        int64_t input_width,bool routed_input) {
    nint_moe_grouped_matmul_dispatch_cuda(pointers,params,expert_pool,expert_local,
        ids,output,input_width,routed_input,0,0);
}

void nint_moe_grouped_matmul_hinted_cuda(
        ::tyloi::DenseTensor pointers,::tyloi::DenseTensor params,
        ::tyloi::DenseTensor expert_pool,::tyloi::DenseTensor expert_local,
        ::tyloi::DenseTensor ids,::tyloi::DenseTensor output,
        int64_t input_width,bool routed_input,int group_hint,int bits_hint) {
    nint_moe_grouped_matmul_dispatch_cuda(pointers,params,expert_pool,expert_local,
        ids,output,input_width,routed_input,group_hint,bits_hint);
}

bool tyloi::cuda::try_nint_decode(
        NintSingleRouteWeight weight,::tyloi::DenseTensor row_metadata,::tyloi::DenseTensor expert_local,
        ::tyloi::DenseTensor ids,::tyloi::DenseTensor output,
        bool routed_input,int bits_hint,bool metadata_ready) {
    const char* setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_DIRECT_POOL");
    const char* grouped=tyloi::cuda::env_lookup("TYLOINFER_NINT_GROUP_DOT");
    const char* hint=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_HINT");
    const char* warps=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_WARPS");
    if((setting && setting[0]=='0') || (grouped && grouped[0]=='0') || (hint && hint[0]=='0') ||
            (warps && std::atoi(warps)==4) || bits_hint<4 || bits_hint>6 ||
            (weight.geometry[1]!=24 && weight.geometry[1]!=28))return false;
    TYLOINFER_RUNTIME_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() && expert_local.scalar_type()==::tyloi::kInt32 &&
        expert_local.dim()==1 && expert_local.numel()>0 && expert_local.numel()<=4096 &&
        ids.is_cuda() && ids.is_contiguous() && ids.scalar_type()==::tyloi::kInt32 && ids.dim()==2 &&
        ids.size(0)>0 && ids.size(0)<=8 && ids.size(1)>0 && ids.numel()<=65535 &&
        output.is_cuda() && output.is_contiguous() && output.scalar_type()==::tyloi::kFloat16 && output.dim()==3 &&
        output.size(0)==ids.size(0) && output.size(1)==ids.size(1) && output.size(2)>0 && output.size(2)<=INT_MAX &&
        output.device()==ids.device() && output.device()==expert_local.device() &&
        weight.local_experts>0 && weight.geometry[0]>0 && weight.geometry[2]>0,
        "single-pool NINT routed tensors disagree");
    MfqCudaGuard guard(output.device());
    const bool late_scale=nint_late_scale_enabled(
        int64_t(weight.geometry[0])*weight.geometry[1],bits_hint,output.get_device());
    const char* metadata=tyloi::cuda::env_lookup("TYLOINFER_NINT_ROUTE_METADATA");
    const auto metadata_ptr=row_metadata.defined()?reinterpret_cast<int64_t>(row_metadata.data_ptr()):0;
    weight.row_metadata=0;
    const bool packed=metadata_ptr && (metadata_ready || (metadata && metadata[0]!='0'));
    if(packed) {
        TYLOINFER_RUNTIME_CHECK(row_metadata.is_cuda() && row_metadata.device()==output.device(),
            "NINT route metadata device disagrees with output");
        weight.row_metadata=metadata_ptr;
    }
    const auto *multi_sum=tyloi::cuda::env_lookup("TYLOINFER_CUDA_WARP_MULTI_SUM");
    tyloi::cuda::nint_raw::single(weight,expert_local.data_ptr<int32_t>(),ids.data_ptr<int32_t>(),
        output.data_ptr(),int(ids.numel()),int(ids.size(1)),int(expert_local.numel()),
        int(output.size(2)),routed_input,bits_hint,packed,
        multi_sum && multi_sum[0]=='1' && multi_sum[1]=='\0',
        late_scale,
        mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

template <typename Output>
static ::tyloi::DenseTensor nint_decode_impl(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    TYLOINFER_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == ::tyloi::kUInt8 &&
        bitstream.dim() == 1,
        "NINT bitstream must be contiguous CUDA uint8 rank-1");
    TYLOINFER_RUNTIME_CHECK(
        row_q_bits.is_cuda() && row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == ::tyloi::kUInt8 &&
        row_q_bits.dim() == 1,
        "NINT q metadata must be contiguous CUDA uint8 rank-1");
    TYLOINFER_RUNTIME_CHECK(
        row_q_bit_offsets.is_cuda() && row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == ::tyloi::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT row offsets must be contiguous CUDA int64 rank-1");
    TYLOINFER_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == ::tyloi::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == ::tyloi::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    TYLOINFER_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == ::tyloi::kFloat32,
        "NINT neuron metadata is invalid");
    const int rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    TYLOINFER_RUNTIME_CHECK(
        rows > 0 && groups > 0 && group_size >= 4 && group_size <= 64 &&
        width > 0 && width <= static_cast<int64_t>(groups) * group_size,
        "NINT decode dimensions are invalid");
    TYLOINFER_RUNTIME_CHECK(
        row_q_bits.numel() == rows && row_q_bit_offsets.numel() == rows &&
        neuron_scale.numel() == rows && neuron_minimum.numel() == rows,
        "NINT row metadata shape mismatch");
    TYLOINFER_RUNTIME_CHECK(
        bitstream.device() == row_q_bits.device() &&
        bitstream.device() == row_q_bit_offsets.device() &&
        bitstream.device() == subgroup_scale.device() &&
        bitstream.device() == subgroup_minimum.device() &&
        bitstream.device() == neuron_scale.device() &&
        bitstream.device() == neuron_minimum.device(),
        "NINT tensors must share one CUDA device");
    auto output = ::tyloi::cuda::empty(
        {rows, width},
        neuron_scale.options().dtype(
            std::is_same_v<Output, __half>
                ? ::tyloi::kFloat16
                : ::tyloi::kFloat32));
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
    tyloi::cuda::nint_raw::decode(a,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


::tyloi::DenseTensor nint_decode_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    return nint_decode_impl<__half>(
        std::move(bitstream), std::move(row_q_bits),
        std::move(row_q_bit_offsets), std::move(subgroup_scale),
        std::move(subgroup_minimum), std::move(neuron_scale),
        std::move(neuron_minimum), width, group_size);
}


::tyloi::DenseTensor nint_decode_f32_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        int64_t width,
        int64_t group_size) {
    return nint_decode_impl<float>(
        std::move(bitstream), std::move(row_q_bits),
        std::move(row_q_bit_offsets), std::move(subgroup_scale),
        std::move(subgroup_minimum), std::move(neuron_scale),
        std::move(neuron_minimum), width, group_size);
}

namespace {
void validate_nint8_zero(
        const ::tyloi::DenseTensor & quantized,
        const ::tyloi::DenseTensor & scale) {
    TYLOINFER_RUNTIME_CHECK(
        quantized.is_cuda() && quantized.is_contiguous() &&
        quantized.scalar_type() == ::tyloi::kUInt8 &&
        quantized.dim() == 3 && quantized.size(2) == 32,
        "NINT8-0 q must be contiguous CUDA uint8 [N,G,32]");
    TYLOINFER_RUNTIME_CHECK(
        scale.is_cuda() && scale.is_contiguous() &&
        scale.scalar_type() == ::tyloi::kFloat16 &&
        scale.dim() == 2 && scale.size(0) == quantized.size(0) &&
        scale.size(1) == quantized.size(1),
        "NINT8-0 scale must be contiguous CUDA fp16 [N,G]");
    TYLOINFER_RUNTIME_CHECK(
        quantized.device() == scale.device(),
        "NINT8-0 tensors must share one CUDA device");
}
::tyloi::DenseTensor cublas_gemm_nt_f32_output(
        const ::tyloi::DenseTensor & input,
        const ::tyloi::DenseTensor & weight) {
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = ::tyloi::cuda::empty(
        {rows, outputs},
        input.options().dtype(::tyloi::kFloat32));
    const auto status = tyloi::cuda::nint_raw::gemm_nt(
        mfq_current_cublas_handle(), mfq_current_cuda_stream(), input.data_ptr(),
        weight.data_ptr(), result.data_ptr(), rows, outputs, width, true, true);
    TYLOINFER_RUNTIME_CHECK(status.stream == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasSetStream(handle, mfq_current_cuda_stream())");
    TYLOINFER_RUNTIME_CHECK(status.gemm == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasGemmEx(...)");
    return result;
}


}
::tyloi::DenseTensor nint_cublas_gemm_nt_f16acc_cuda(
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor weight) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == ::tyloi::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == ::tyloi::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    TYLOINFER_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = ::tyloi::cuda::empty({rows, outputs}, input.options());
    const auto status = tyloi::cuda::nint_raw::gemm_nt(
        mfq_current_cublas_handle(), mfq_current_cuda_stream(), input.data_ptr(),
        weight.data_ptr(), result.data_ptr(), rows, outputs, width, false, false);
    TYLOINFER_RUNTIME_CHECK(status.stream == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasSetStream(handle, mfq_current_cuda_stream())");
    TYLOINFER_RUNTIME_CHECK(status.gemm == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasGemmEx(...)");
    return result;
}


::tyloi::DenseTensor nint_cublas_gemm_nt_f32acc_cuda(
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor weight) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == ::tyloi::kFloat16 &&
        input.dim() == 2,
        "GEMM input must be contiguous CUDA fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(
        weight.is_cuda() && weight.is_contiguous() &&
        weight.scalar_type() == ::tyloi::kFloat16 &&
        weight.dim() == 2 && weight.size(1) == input.size(1),
        "GEMM weight must be contiguous CUDA fp16 [N,K]");
    TYLOINFER_RUNTIME_CHECK(
        input.device() == weight.device(),
        "GEMM tensors must share one CUDA device");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    const int outputs = static_cast<int>(weight.size(0));
    auto result = ::tyloi::cuda::empty({rows, outputs}, input.options());
    const auto status = tyloi::cuda::nint_raw::gemm_nt(
        mfq_current_cublas_handle(), mfq_current_cuda_stream(), input.data_ptr(),
        weight.data_ptr(), result.data_ptr(), rows, outputs, width, true, false);
    TYLOINFER_RUNTIME_CHECK(status.stream == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasSetStream(handle, mfq_current_cuda_stream())");
    TYLOINFER_RUNTIME_CHECK(status.gemm == CUBLAS_STATUS_SUCCESS,
        "cuBLAS call failed: cublasGemmEx(...)");
    return result;
}
template <typename Output>
static ::tyloi::DenseTensor nint8_zero_dequant_impl(
        ::tyloi::DenseTensor quantized,
        ::tyloi::DenseTensor scale,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    TYLOINFER_RUNTIME_CHECK(
        width > 0 && width <= quantized.size(1) * 32,
        "NINT8-0 width is invalid");
    const int rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    auto output = ::tyloi::cuda::empty(
        {rows, width},
        quantized.options().dtype(
            std::is_same_v<Output, __half>
                ? ::tyloi::kFloat16
                : ::tyloi::kFloat32));
    const auto output_data = [&]() {
        if constexpr (std::is_same_v<Output, __half>) {
            return reinterpret_cast<__half *>(output.template data_ptr<mfq_half>());
        } else {
            return output.template data_ptr<float>();
        }
    }();
    tyloi::cuda::nint_raw::zero_decode(
        quantized.data_ptr<uint8_t>(), scale.data_ptr<mfq_half>(), output_data,
        rows, groups, width, std::is_same_v<Output, float>,
        mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


::tyloi::DenseTensor nint8_zero_dequant_cuda(
        ::tyloi::DenseTensor quantized,
        ::tyloi::DenseTensor scale,
        int64_t width) {
    return nint8_zero_dequant_impl<__half>(
        std::move(quantized), std::move(scale), width);
}


::tyloi::DenseTensor nint8_zero_dequant_f32_cuda(
        ::tyloi::DenseTensor quantized,
        ::tyloi::DenseTensor scale,
        int64_t width) {
    return nint8_zero_dequant_impl<float>(
        std::move(quantized), std::move(scale), width);
}


::tyloi::DenseTensor nint8_zero_gemv_ws_cuda(
        ::tyloi::DenseTensor quantized,
        ::tyloi::DenseTensor scale,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale) {
    validate_nint8_zero(quantized, scale);
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        (input.scalar_type() == ::tyloi::kFloat16 ||
         input.scalar_type() == ::tyloi::kFloat32) &&
        input.dim() == 2,
        "NINT8-0 input must be contiguous CUDA fp16/fp32 rank-2");
    const int activation_rows = static_cast<int>(input.size(0));
    const int output_rows = static_cast<int>(quantized.size(0));
    const int groups = static_cast<int>(quantized.size(1));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * 32;
    TYLOINFER_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8 &&
        real_width <= padded_width,
        "NINT8-0 packed matmul expects M in [1,8] and K <= packed K");
    TYLOINFER_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == ::tyloi::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == ::tyloi::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT8-0 activation workspace is invalid");
    TYLOINFER_RUNTIME_CHECK(
        quantized.device() == input.device() && scale.device() == input.device() &&
        quantized_input.device() == input.device() &&
        input_scale.device() == input.device(),
        "NINT8-0 tensors and workspace must share one CUDA device");
    auto output = ::tyloi::cuda::empty(
        {activation_rows, output_rows}, input.options().dtype(::tyloi::kFloat16));
    const cudaStream_t stream = mfq_current_cuda_stream();
    tyloi::cuda::nint_raw::zero_gemv(quantized.data_ptr<uint8_t>(),
        scale.data_ptr<mfq_half>(), input.data_ptr(), quantized_input.data_ptr<int8_t>(),
        input_scale.data_ptr<float>(), output.data_ptr<mfq_half>(), activation_rows,
        output_rows, groups, real_width, input.scalar_type() == ::tyloi::kFloat32, stream);
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


::tyloi::DenseTensor nint8_zero_mmq_f16_packed_cuda(
        ::tyloi::DenseTensor quantized,
        ::tyloi::DenseTensor scale,
        ::tyloi::DenseTensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == ::tyloi::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return nint_cublas_gemm_nt_f16acc_cuda(input, weight);
}


::tyloi::DenseTensor nint8_zero_mmq_f32_packed_cuda(
        ::tyloi::DenseTensor quantized,
        ::tyloi::DenseTensor scale,
        ::tyloi::DenseTensor input,
        int64_t width) {
    validate_nint8_zero(quantized, scale);
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == ::tyloi::kFloat16 &&
        input.dim() == 2 && input.size(1) == width,
        "NINT8-0 FP32 GEMM input geometry is invalid");
    auto weight = nint8_zero_dequant_cuda(quantized, scale, width);
    return cublas_gemm_nt_f32_output(input, weight);
}

::tyloi::DenseTensor nint_matmul_online_f16_cuda(
        ::tyloi::DenseTensor bitstream,::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,::tyloi::DenseTensor input,int64_t group_size) {
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.dim()==2 &&
        input.scalar_type()==::tyloi::kFloat16 && input.size(0)>8 && input.size(0)<=32,
        "NINT online projection requires contiguous CUDA FP16 input with 9..32 rows");
    for(const auto* tensor:{&bitstream,&row_q_bits,&row_q_bit_offsets,&subgroup_scale,
            &subgroup_minimum,&neuron_scale,&neuron_minimum}) {
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->is_contiguous() && tensor->device()==input.device(),
            "NINT online projection tensors must share the input CUDA device");
    }
    TYLOINFER_RUNTIME_CHECK(bitstream.scalar_type()==::tyloi::kUInt8 && bitstream.dim()==1 &&
        row_q_bits.scalar_type()==::tyloi::kUInt8 && row_q_bits.dim()==1 &&
        row_q_bit_offsets.scalar_type()==::tyloi::kInt64 && row_q_bit_offsets.dim()==1 &&
        subgroup_scale.scalar_type()==::tyloi::kUInt8 && subgroup_scale.dim()==2 &&
        subgroup_minimum.scalar_type()==::tyloi::kUInt8 && subgroup_minimum.sizes()==subgroup_scale.sizes() &&
        neuron_scale.scalar_type()==::tyloi::kFloat32 && neuron_minimum.scalar_type()==::tyloi::kFloat32,
        "NINT online projection metadata types are invalid");
    const int rows=int(input.size(0)),outputs=int(subgroup_scale.size(0)),groups=int(subgroup_scale.size(1));
    TYLOINFER_RUNTIME_CHECK(outputs>0 && groups>0 && group_size>=4 && group_size<=64 &&
        input.size(1)>0 && input.size(1)<=int64_t(groups)*group_size && row_q_bits.numel()==outputs &&
        row_q_bit_offsets.numel()==outputs && neuron_scale.numel()==outputs && neuron_minimum.numel()==outputs,
        "NINT online projection metadata dimensions are invalid");
    auto output=::tyloi::cuda::empty({rows,outputs},input.options());
    const int output_tiles=(outputs+tyloi::cuda::nint_raw::prefill_tile_n-1)/tyloi::cuda::nint_raw::prefill_tile_n;
    int split=1;
    static thread_local int cached_device=-1,processors=1;
    if(cached_device!=input.get_device()) {
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&processors,cudaDevAttrMultiProcessorCount,input.get_device()));
        cached_device=input.get_device();
    }
    while(split<8 && output_tiles*split<processors*2 && input.size(1)>=int64_t(split*2)*tyloi::cuda::nint_raw::prefill_tile_k)split*=2;
    auto partial=split>1 ? ::tyloi::cuda::empty({split,rows,outputs},input.options().dtype(::tyloi::kFloat32))
        : ::tyloi::DenseTensor();
    tyloi::cuda::nint_raw::Decode a{};
    a.bitstream=bitstream.data_ptr<uint8_t>(); a.row_q_bits=row_q_bits.data_ptr<uint8_t>();
    a.row_q_bit_offsets=row_q_bit_offsets.data_ptr<int64_t>();
    a.subgroup_scale=subgroup_scale.data_ptr<uint8_t>(); a.subgroup_minimum=subgroup_minimum.data_ptr<uint8_t>();
    a.neuron_scale=neuron_scale.data_ptr<float>(); a.neuron_minimum=neuron_minimum.data_ptr<float>();
    a.rows=outputs; a.groups=groups; a.group_size=int(group_size); a.width=input.size(1);
    tyloi::cuda::nint_raw::online(a,input.data_ptr<mfq_half>(),output.data_ptr<mfq_half>(),
        split>1 ? partial.data_ptr<float>() : nullptr,rows,split,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

std::vector<::tyloi::DenseTensor> nint8_one_quantize_reconstruct_cuda(
        ::tyloi::DenseTensor input) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        input.scalar_type() == ::tyloi::kFloat16 &&
        input.dim() == 2,
        "NINT8-1 input must be contiguous CUDA fp16 rank-2");
    const int rows = static_cast<int>(input.size(0));
    const int width = static_cast<int>(input.size(1));
    TYLOINFER_RUNTIME_CHECK(rows > 0 && width > 0, "NINT8-1 input must be non-empty");
    const int groups = (width + 31) / 32;
    auto quantized = ::tyloi::cuda::empty(
        {rows, groups, 32},
        input.options().dtype(::tyloi::kInt8));
    auto scale = ::tyloi::cuda::empty({rows, groups}, input.options());
    auto sum = ::tyloi::cuda::empty({rows, groups}, input.options());
    auto reconstructed = ::tyloi::cuda::empty_like(input);
    tyloi::cuda::nint_raw::one_quantize(input.data_ptr<mfq_half>(),quantized.data_ptr<int8_t>(),
        scale.data_ptr<mfq_half>(),sum.data_ptr<mfq_half>(),reconstructed.data_ptr<mfq_half>(),
        rows,width,groups,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return {quantized, scale, sum, reconstructed};
}

namespace { thread_local NintInputQuantizationScope* prepared_input_scope=nullptr; }

NintInputQuantizationScope::NintInputQuantizationScope(
        ::tyloi::DenseTensor source,const std::vector<int>& sizes):input(std::move(source)) {
    if(sizes.empty())return;
    TYLOINFER_RUNTIME_CHECK(input.is_cuda() && input.is_contiguous() && input.dim()==2 &&
        input.size(0)>=1 && input.size(0)<=6 && input.size(1)>0 && input.size(1)<=INT_MAX-64 &&
        (input.scalar_type()==::tyloi::kFloat16 || input.scalar_type()==::tyloi::kFloat32) &&
        sizes.size()<=4,"invalid shared NINT quantization input");
    const MfqCudaGuard guard(input.device());
    tyloi::cuda::nint_raw::GroupQuantize args;
    args.input=input.data_ptr();args.rows=int(input.size(0));args.width=int(input.size(1));
    args.fp32=input.scalar_type()==::tyloi::kFloat32;
    const auto* exact=tyloi::cuda::env_lookup("TYLOINFER_NINT_EXACT_BATCH");
    const int columns=args.rows<=2?2:args.rows<=4?4:(exact && exact[0]=='0')?8:args.rows;
    int total=0;
    for(int gs:sizes) {
        TYLOINFER_RUNTIME_CHECK(gs==24 || gs==28 || gs==48,"unsupported shared NINT group size");
        const int ng=(args.width+gs-1)/gs;
        NintPreparedGroupInput prepared;prepared.gs=gs;
        prepared.codes=::tyloi::cuda::empty({args.rows,ng*gs},input.options().dtype(::tyloi::kInt8));
        prepared.scale=::tyloi::cuda::empty({args.rows,ng},input.options().dtype(::tyloi::kFloat32));
        prepared.sum=::tyloi::cuda::empty({args.rows,ng},input.options().dtype(::tyloi::kInt32));
        if(gs==28 && args.rows>=2)
            prepared.interleaved=::tyloi::cuda::empty({ng,7,columns,4},input.options().dtype(::tyloi::kInt8));
        args.group[args.count++]={prepared.codes.data_ptr<int8_t>(),
            prepared.interleaved.defined()?prepared.interleaved.data_ptr<int8_t>():nullptr,
            prepared.scale.data_ptr<float>(),prepared.sum.data_ptr<int32_t>(),gs,ng,total,columns};
        total+=ng;groups.push_back(std::move(prepared));
    }
    tyloi::cuda::nint_raw::group_quantize(args,total,mfq_current_cuda_stream());
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    tyloi::cuda::nint_raw::trace_group_quantize(args,mfq_current_cuda_stream());
    previous=prepared_input_scope;prepared_input_scope=this;
}
NintInputQuantizationScope::~NintInputQuantizationScope() {
    if(!groups.empty())prepared_input_scope=previous;
}

static ::tyloi::DenseTensor nint_matmul_ws_impl(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        ::tyloi::DenseTensor input,
        const ::tyloi::DenseTensor * gate,
        int64_t activation_mode,
        int64_t group_size,
        ::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale,
        bool aligned_q8 = false,int uniform_bits = 0) {
    TYLOINFER_RUNTIME_CHECK(
        input.is_cuda() && input.is_contiguous() &&
        (input.scalar_type() == ::tyloi::kFloat16 ||
         input.scalar_type() == ::tyloi::kFloat32) &&
        input.dim() == 2,
        "NINT input must be contiguous CUDA fp16/fp32 rank-2");
    if (gate != nullptr) {
        TYLOINFER_RUNTIME_CHECK(
            activation_mode == 1 || activation_mode == 2,
            "NINT input gate mode must be sigmoid or SiLU");
        TYLOINFER_RUNTIME_CHECK(
            gate->is_cuda() && gate->is_contiguous() &&
            (gate->scalar_type() == ::tyloi::kFloat16 ||
             gate->scalar_type() == ::tyloi::kFloat32) &&
            gate->dim() == 2 && gate->sizes() == input.sizes(),
            "NINT gate must be contiguous CUDA fp16/fp32 and match input shape");
    } else {
        TYLOINFER_RUNTIME_CHECK(
            activation_mode == 0,
            "NINT activation mode requires a gate tensor");
    }
    TYLOINFER_RUNTIME_CHECK(
        subgroup_scale.is_cuda() && subgroup_scale.is_contiguous() &&
        subgroup_scale.scalar_type() == ::tyloi::kUInt8 &&
        subgroup_scale.dim() == 2 && subgroup_minimum.is_cuda() &&
        subgroup_minimum.is_contiguous() &&
        subgroup_minimum.scalar_type() == ::tyloi::kUInt8 &&
        subgroup_minimum.sizes() == subgroup_scale.sizes(),
        "NINT subgroup metadata is invalid");
    TYLOINFER_RUNTIME_CHECK(
        bitstream.is_cuda() && bitstream.is_contiguous() &&
        bitstream.scalar_type() == ::tyloi::kUInt8 &&
        bitstream.dim() == 1 && row_q_bits.is_cuda() &&
        row_q_bits.is_contiguous() &&
        row_q_bits.scalar_type() == ::tyloi::kUInt8 &&
        row_q_bits.dim() == 1 && row_q_bit_offsets.is_cuda() &&
        row_q_bit_offsets.is_contiguous() &&
        row_q_bit_offsets.scalar_type() == ::tyloi::kInt64 &&
        row_q_bit_offsets.dim() == 1,
        "NINT bitstream or q metadata is invalid");
    TYLOINFER_RUNTIME_CHECK(
        neuron_scale.is_cuda() && neuron_scale.is_contiguous() &&
        neuron_scale.scalar_type() == ::tyloi::kFloat32 &&
        neuron_minimum.is_cuda() && neuron_minimum.is_contiguous() &&
        neuron_minimum.scalar_type() == ::tyloi::kFloat32,
        "NINT neuron metadata is invalid");
    const int output_rows = static_cast<int>(subgroup_scale.size(0));
    const int groups = static_cast<int>(subgroup_scale.size(1));
    const int activation_rows = static_cast<int>(input.size(0));
    const int real_width = static_cast<int>(input.size(1));
    const int padded_width = groups * static_cast<int>(group_size);
    TYLOINFER_RUNTIME_CHECK(
        activation_rows >= 1 && activation_rows <= 8,
        "NINT packed matmul supports M in [1,8]");
    TYLOINFER_RUNTIME_CHECK(
        group_size >= 4 && group_size <= 64 && real_width <= padded_width,
        "NINT packed matmul dimensions are invalid");
    TYLOINFER_RUNTIME_CHECK(
        row_q_bits.numel() == output_rows &&
        row_q_bit_offsets.numel() == output_rows &&
        neuron_scale.numel() == output_rows &&
        neuron_minimum.numel() == output_rows,
        "NINT row metadata shape mismatch");
    TYLOINFER_RUNTIME_CHECK(
        quantized_input.is_cuda() && quantized_input.is_contiguous() &&
        quantized_input.scalar_type() == ::tyloi::kInt8 &&
        quantized_input.dim() == 2 &&
        quantized_input.size(0) >= activation_rows &&
        quantized_input.size(1) >= padded_width && input_scale.is_cuda() &&
        input_scale.is_contiguous() &&
        input_scale.scalar_type() == ::tyloi::kFloat32 &&
        input_scale.dim() == 2 && input_scale.size(0) >= activation_rows &&
        input_scale.size(1) >= groups,
        "NINT activation workspace is invalid");
    TYLOINFER_RUNTIME_CHECK(
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
    auto output = ::tyloi::cuda::empty(
        {activation_rows, output_rows}, input.options().dtype(::tyloi::kFloat16));
    const cudaStream_t stream = mfq_current_cuda_stream();
    const char* dense_group_setting = tyloi::cuda::env_lookup("TYLOINFER_NINT_GROUP_DOT");
    const char* whole_setting_for_sum = tyloi::cuda::env_lookup("TYLOINFER_NINT_WHOLE_GROUP");
    const char* batch_setting_for_sum = tyloi::cuda::env_lookup("TYLOINFER_NINT_BATCH_WHOLE_GROUP");
    const char* sum_setting = tyloi::cuda::env_lookup("TYLOINFER_NINT_ACTIVATION_SUM");
    const auto* interleaved_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_INTERLEAVED");
    const bool interleaved_default=activation_rows>=3 && activation_rows<=6;
    const bool interleaved_requested=interleaved_setting?interleaved_setting[0]=='1':interleaved_default;
    const auto* float_whole_setting=tyloi::cuda::env_lookup("TYLOINFER_NINT_FLOAT_WHOLE_GROUP");
    const bool whole_input=input.scalar_type()==::tyloi::kFloat16 ||
        (input.scalar_type()==::tyloi::kFloat32 && (!float_whole_setting || float_whole_setting[0]!='0'));
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
    const NintPreparedGroupInput* prepared=nullptr;
    if(!gate)for(auto* scope=prepared_input_scope;scope && !prepared;scope=scope->previous) {
        if(scope->input.data_ptr()!=input.data_ptr() || scope->input.sizes()!=input.sizes() ||
                scope->input.scalar_type()!=input.scalar_type() || scope->input.device()!=input.device())continue;
        for(const auto& group:scope->groups)if(group.gs==group_size &&
                group.scale.size(1)==groups && (!interleaved || group.interleaved.defined())) {
            prepared=&group;break;
        }
    }
    if(prepared){quantized_input=prepared->codes;input_scale=prepared->scale;}
    ::tyloi::DenseTensor interleaved_activation;
    if(interleaved)interleaved_activation=prepared?prepared->interleaved: ::tyloi::cuda::empty(
        {groups,7,interleaved_columns,4},input.options().dtype(::tyloi::kInt8));
    ::tyloi::DenseTensor activation_sums;
    if(cached_sums)activation_sums=prepared?prepared->sum: ::tyloi::cuda::empty(
        {activation_rows,groups},input.options().dtype(::tyloi::kInt32));
    if(!prepared) {
        tyloi::cuda::nint_raw::Quantize a{};
        a.input=input.data_ptr(); a.gate=gate ? gate->data_ptr() : nullptr;
        a.codes=quantized_input.data_ptr<int8_t>(); a.scale=input_scale.data_ptr<float>();
        a.rows=activation_rows; a.width=real_width; a.padded_width=padded_width;
        a.groups=groups; a.group_size=int(group_size); a.activation_mode=int(activation_mode);
        a.input_float=input.scalar_type()==::tyloi::kFloat32;
        a.gate_float=gate && gate->scalar_type()==::tyloi::kFloat32;
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
    q.output=output.data_ptr<mfq_half>(); q.rows=activation_rows; q.padded_width=padded_width;
    q.grouped=grouped_q8; q.q8_decode=q8_decode;
    a.interleaved=interleaved_activation.defined()?interleaved_activation.data_ptr<int8_t>():nullptr;
    a.dense_group_setting=dense_group_setting; a.fixed_setting=fixed_setting;
    a.real_width=real_width; a.uniform_bits=uniform_bits; a.device=input.get_device();
    a.aligned_q8=aligned_q8; a.whole_input=whole_input; a.cached_sums=cached_sums;
    const int error=tyloi::cuda::nint_raw::project(a,stream);
    TYLOINFER_RUNTIME_CHECK(error != 1,"NINT shared activation device query failed");
    TYLOINFER_RUNTIME_CHECK(error != 2,"NINT processor count query failed");
    TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}


::tyloi::DenseTensor nint_matmul_fixed_bits_ws_cuda(
        ::tyloi::DenseTensor bitstream,::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,::tyloi::DenseTensor input,
        int64_t group_size,::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale,int uniform_bits) {
    return nint_matmul_ws_impl(bitstream,row_q_bits,row_q_bit_offsets,subgroup_scale,
        subgroup_minimum,neuron_scale,neuron_minimum,input,nullptr,0,group_size,
        quantized_input,input_scale,false,uniform_bits);
}

::tyloi::DenseTensor nint_matmul_ws_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        ::tyloi::DenseTensor input,
        int64_t group_size,
        ::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, nullptr, 0, group_size, quantized_input, input_scale);
}


::tyloi::DenseTensor nint_matmul_q8_ws_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        ::tyloi::DenseTensor input,
        int64_t group_size,
        ::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, nullptr, 0, group_size, quantized_input, input_scale, true);
}






::tyloi::DenseTensor nint_matmul_input_mul_fixed_bits_ws_cuda(
        ::tyloi::DenseTensor bitstream,::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,::tyloi::DenseTensor input,
        ::tyloi::DenseTensor gate,int64_t activation_mode,int64_t group_size,
        ::tyloi::DenseTensor quantized_input,::tyloi::DenseTensor input_scale,int uniform_bits) {
    return nint_matmul_ws_impl(bitstream,row_q_bits,row_q_bit_offsets,subgroup_scale,
        subgroup_minimum,neuron_scale,neuron_minimum,input,&gate,activation_mode,
        group_size,quantized_input,input_scale,false,uniform_bits);
}

::tyloi::DenseTensor nint_matmul_input_mul_ws_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor gate,
        int64_t activation_mode,
        int64_t group_size,
        ::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, &gate, activation_mode, group_size,
        quantized_input, input_scale);
}


::tyloi::DenseTensor nint_matmul_input_mul_q8_ws_cuda(
        ::tyloi::DenseTensor bitstream,
        ::tyloi::DenseTensor row_q_bits,
        ::tyloi::DenseTensor row_q_bit_offsets,
        ::tyloi::DenseTensor subgroup_scale,
        ::tyloi::DenseTensor subgroup_minimum,
        ::tyloi::DenseTensor neuron_scale,
        ::tyloi::DenseTensor neuron_minimum,
        ::tyloi::DenseTensor input,
        ::tyloi::DenseTensor gate,
        int64_t activation_mode,
        int64_t group_size,
        ::tyloi::DenseTensor quantized_input,
        ::tyloi::DenseTensor input_scale) {
    return nint_matmul_ws_impl(
        bitstream, row_q_bits, row_q_bit_offsets,
        subgroup_scale, subgroup_minimum, neuron_scale, neuron_minimum,
        input, &gate, activation_mode, group_size,
        quantized_input, input_scale, true);
}
