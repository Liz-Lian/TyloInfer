#include "backend.h"
#include "nvq_raw.h"
#include <cuda_runtime_api.h>
#include <cstdint>

void nvq_check_native(const ::tyloi::DenseTensor&, const ::tyloi::DenseTensor&,
    const ::tyloi::DenseTensor&, const ::tyloi::DenseTensor&, const ::tyloi::DenseTensor&,
    int64_t, int64_t, int64_t, int64_t, int64_t);
void nepq_check_native(const ::tyloi::DenseTensor&, const ::tyloi::DenseTensor&,
    const ::tyloi::DenseTensor&, const ::tyloi::DenseTensor&, const ::tyloi::DenseTensor&,
    const ::tyloi::DenseTensor&, int64_t, int64_t, int64_t);

namespace {
/** Describe the buffers after the original format checks. */
tyloi::cuda::nvq_raw::Weight weight(const ::tyloi::DenseTensor& indices, const ::tyloi::DenseTensor& aux,
    const ::tyloi::DenseTensor& state, const ::tyloi::DenseTensor& scale, const ::tyloi::DenseTensor& table,
    const std::uint8_t* banks, int width, int gs, int bits, int format, int sign, int table_stride) {
    return {indices.data_ptr<uint8_t>(), aux.data_ptr<uint8_t>(), state.data_ptr<uint8_t>(),
        indices.numel(), aux.numel(), state.numel(), scale.data_ptr<float>(), table.data_ptr<int8_t>(), table.numel(), banks,
        static_cast<int>(scale.numel()), width, gs, bits, format, sign, table_stride};
}
}

/** Decode NVQ weights into their original F16 layout. */
::tyloi::DenseTensor nvq_dequant_cuda(::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor state, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor table,
    int64_t width, int64_t gs, int64_t bits, int64_t format, int64_t sign) {
    nvq_check_native(indices,aux,state,scale,table,width,gs,bits,format,sign);
    const int n=static_cast<int>(scale.numel()), k=static_cast<int>(width);
    auto output=::tyloi::cuda::empty({n,k},scale.options().dtype(::tyloi::kFloat16));
    auto w=weight(indices,aux,state,scale,table,nullptr,k,static_cast<int>(gs),static_cast<int>(bits),static_cast<int>(format),static_cast<int>(sign),0);
    tyloi::cuda::nvq_raw::decode(w,output.data_ptr(),mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ dequant kernel launch failed");
    return output;
}

/** Decode NEPQ shared-bank weights into their original F16 layout. */
::tyloi::DenseTensor nepq_dequant_cuda(::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor state, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor table,
    ::tyloi::DenseTensor banks, int64_t width, int64_t bits, int64_t format) {
    nepq_check_native(indices,aux,state,scale,table,banks,width,bits,format);
    const int n=static_cast<int>(scale.numel()), k=static_cast<int>(width);
    auto output=::tyloi::cuda::empty({n,k},scale.options().dtype(::tyloi::kFloat16));
    auto w=weight(indices,aux,state,scale,table,banks.data_ptr<uint8_t>(),k,24,static_cast<int>(bits),static_cast<int>(format),0,static_cast<int>(table.size(1)));
    tyloi::cuda::nvq_raw::decode_nepq(w,output.data_ptr(),mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NEPQ dequant kernel launch failed");
    return output;
}

/** Decode the selected NVQ embedding rows. */
::tyloi::DenseTensor nvq_embedding_lookup_cuda(::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor state, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor table,
    ::tyloi::DenseTensor ids, int64_t width, int64_t gs, int64_t bits, int64_t format, int64_t sign) {
    nvq_check_native(indices,aux,state,scale,table,width,gs,bits,format,sign);
    TYLOINFER_RUNTIME_CHECK(ids.is_cuda() && ids.scalar_type()==::tyloi::kInt64 && ids.is_contiguous(),
        "NVQ token_ids must be CUDA contiguous int64");
    const int tokens=static_cast<int>(ids.numel()), k=static_cast<int>(width);
    auto shape=ids.sizes().vec(); shape.push_back(k);
    auto output=::tyloi::cuda::empty(shape,scale.options().dtype(::tyloi::kFloat16));
    auto w=weight(indices,aux,state,scale,table,nullptr,k,static_cast<int>(gs),static_cast<int>(bits),static_cast<int>(format),static_cast<int>(sign),0);
    tyloi::cuda::nvq_raw::embedding(w,ids.data_ptr<int64_t>(),tokens,output.data_ptr(),mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ embedding kernel launch failed");
    return output;
}
/** Apply the original online-dequant F16 projection. */
::tyloi::DenseTensor nvq_gemm_f16_cuda(::tyloi::DenseTensor indices, ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor state, ::tyloi::DenseTensor scale, ::tyloi::DenseTensor table,
    ::tyloi::DenseTensor x, int64_t width, int64_t gs, int64_t bits, int64_t format, int64_t sign) {
    nvq_check_native(indices,aux,state,scale,table,width,gs,bits,format,sign);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type()==::tyloi::kFloat16 &&
        x.is_contiguous() && x.dim()==2,"NVQ FP16 GEMM x must be CUDA contiguous fp16 rank-2");
    const int m=static_cast<int>(x.size(0)), k=static_cast<int>(width), n=static_cast<int>(scale.numel());
    TYLOINFER_RUNTIME_CHECK(m>=16,"NVQ FP16 GEMM requires M >= 16");
    TYLOINFER_RUNTIME_CHECK(x.size(1)==k,"NVQ FP16 GEMM x width must equal neuron_len");
    auto output=::tyloi::cuda::empty({m,n},x.options());
    auto w=weight(indices,aux,state,scale,table,nullptr,k,static_cast<int>(gs),static_cast<int>(bits),static_cast<int>(format),static_cast<int>(sign),0);
    tyloi::cuda::nvq_raw::gemm(w,x.data_ptr(),output.data_ptr(),m,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ FP16 GEMM kernel launch failed");
    return output;
}/** Apply the original NVQ quantized projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_gemv_m1_vec8_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1, "NVQ vec8 GEMV x must be CUDA contiguous fp16 [1,K]");
    TYLOINFER_RUNTIME_CHECK(nwarps == 1 || nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ vec8 GEMV nwarps must be 1, 2, 4, or 8");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ vec8 GEMV x width must equal neuron_len");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() &&
                qx.numel() >= K_pad, "NVQ vec8 GEMV qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= ng, "NVQ vec8 GEMV xscale workspace mismatch");
    auto output = ::tyloi::cuda::empty({1, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,1,static_cast<int>(nwarps),0};
    tyloi::cuda::nvq_raw::gemv(p,true,false,1,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_gemv_batch_vec8_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ batch vec8 GEMV x must be CUDA contiguous fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ batch vec8 GEMV nwarps must be 2, 4, or 8");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(M >= 2 && M <= 16, "NVQ batch vec8 GEMV supports M in [2,16]");
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ batch vec8 GEMV x width must equal neuron_len");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ batch vec8 GEMV qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ batch vec8 GEMV xscale workspace mismatch");
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,static_cast<int>(nwarps),0};
    tyloi::cuda::nvq_raw::gemv(p,true,false,2,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_gemv_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TYLOINFER_RUNTIME_CHECK(M >= 1 && M <= 16, "NVQ direct GEMV supports M in [1,16]");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::gemv(p,true,false,0,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,M==1 ? "NVQ vec8 GEMV kernel launch failed" : "NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_gemv_qx_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() && qx.dim() == 2,
                "NVQ prequantized qx must be CUDA contiguous int8 rank-2");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.dim() == 2, "NVQ prequantized xscale must be CUDA contiguous fp32 rank-2");
    const int M = static_cast<int>(qx.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(M >= 1 && M <= 16, "NVQ prequantized GEMV supports M in [1,16]");
    TYLOINFER_RUNTIME_CHECK(qx.size(1) >= K_pad, "NVQ prequantized qx width mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.size(0) >= M && xscale.size(1) >= ng,
                "NVQ prequantized xscale shape mismatch");
    auto output = ::tyloi::cuda::empty(
        {M, N}, neuron_scale.options().dtype(::tyloi::kFloat16));
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,nullptr,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::gemv(p,false,false,0,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ prequantized GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_gemv_qx_residual_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale,
    ::tyloi::DenseTensor residual) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(
        tyloi::cuda::nvq_raw::group_exec(static_cast<int>(format)),
        "NVQ fused residual requires an aligned group execution format");
    TYLOINFER_RUNTIME_CHECK(
        qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 &&
        qx.is_contiguous() && qx.dim() == 2 && qx.size(0) == 1,
        "NVQ fused residual qx must be CUDA contiguous int8 [1,K]");
    TYLOINFER_RUNTIME_CHECK(
        xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 &&
        xscale.is_contiguous() && xscale.dim() == 2 && xscale.size(0) >= 1,
        "NVQ fused residual xscale must be CUDA contiguous fp32 [1,ng]");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(qx.size(1) >= K_pad, "NVQ fused residual qx width mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.size(1) >= ng, "NVQ fused residual xscale width mismatch");
    TYLOINFER_RUNTIME_CHECK(
        residual.is_cuda() && residual.scalar_type() == ::tyloi::kFloat16 &&
        residual.is_contiguous() && residual.dim() == 2 &&
        residual.size(0) == 1 && residual.size(1) == N,
        "NVQ fused residual must be CUDA contiguous fp16 [1,N]");

    auto output = ::tyloi::cuda::empty_like(residual);
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,nullptr,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),residual.data_ptr(),nullptr,1,0,0};
    tyloi::cuda::nvq_raw::gemv_residual(p,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ fused residual GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_gemv_gate_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(gate.is_cuda() && gate.scalar_type() == ::tyloi::kFloat16 && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TYLOINFER_RUNTIME_CHECK(M >= 1 && M <= 16, "NVQ gated direct GEMV supports M in [1,16]");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,gate.data_ptr(),M,0,static_cast<int>(mode)};
    tyloi::cuda::nvq_raw::gemv(p,true,true,0,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,M==1 ? "NVQ gated vec8 GEMV kernel launch failed" : "NVQ gated batch vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ integer-MMA projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_mmq_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    ::tyloi::DenseTensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TYLOINFER_RUNTIME_CHECK(M >= 4 && M <= 64, "NVQ MMA MMQ supports M in [4,64]");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::mmq(p,false,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ MMA MMQ kernel launch failed" );
    return output;
}

/** Apply the original NVQ integer-MMA projection with caller-owned workspace. */
::tyloi::DenseTensor nvq_mmq_gate_ws_cuda(
    ::tyloi::DenseTensor indices,
    ::tyloi::DenseTensor aux,
    ::tyloi::DenseTensor sub_scale,
    ::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor codebook,
    ::tyloi::DenseTensor x,
    ::tyloi::DenseTensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    ::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale) {
    nvq_check_native(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TYLOINFER_RUNTIME_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.scalar_type() == ::tyloi::kFloat16 && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    TYLOINFER_RUNTIME_CHECK(gate.is_cuda() && gate.scalar_type() == ::tyloi::kFloat16 && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TYLOINFER_RUNTIME_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TYLOINFER_RUNTIME_CHECK(M >= 4 && M <= 64, "NVQ gated MMA MMQ supports M in [4,64]");
    TYLOINFER_RUNTIME_CHECK(qx.is_cuda() && qx.scalar_type() == ::tyloi::kInt8 && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TYLOINFER_RUNTIME_CHECK(xscale.is_cuda() && xscale.scalar_type() == ::tyloi::kFloat32 && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = ::tyloi::cuda::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,gate.data_ptr(),M,0,static_cast<int>(mode)};
    tyloi::cuda::nvq_raw::mmq(p,true,mfq_current_cuda_stream());
    TYLOINFER_RUNTIME_CHECK(cudaGetLastError()==cudaSuccess,"NVQ gated MMA MMQ kernel launch failed" );
    return output;
}
