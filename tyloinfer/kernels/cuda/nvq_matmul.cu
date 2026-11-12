#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "nvq_check.cuh"
#include "nvq_raw.h"
#include <cuda_runtime_api.h>
#include <cstdint>
#include <climits>
#include <limits>

namespace {
/** Describe the buffers after the original format checks. */
tyloi::cuda::nvq_raw::Weight weight(const torch::Tensor& indices, const torch::Tensor& aux,
    const torch::Tensor& state, const torch::Tensor& scale, const torch::Tensor& table,
    const std::uint8_t* banks, int width, int gs, int bits, int format, int sign, int table_stride) {
    return {indices.data_ptr<uint8_t>(), aux.data_ptr<uint8_t>(), state.data_ptr<uint8_t>(),
        indices.numel(), aux.numel(), state.numel(), scale.data_ptr<float>(), table.data_ptr<int8_t>(), table.numel(), banks,
        static_cast<int>(scale.numel()), width, gs, bits, format, sign, table_stride};
}
}

/** Decode NVQ weights into their original F16 layout. */
torch::Tensor nvq_dequant_cuda(torch::Tensor indices, torch::Tensor aux,
    torch::Tensor state, torch::Tensor scale, torch::Tensor table,
    int64_t width, int64_t gs, int64_t bits, int64_t format, int64_t sign) {
    mfq_jit_nvq::check_common(indices,aux,state,scale,table,width,gs,bits,format,sign);
    const int n=static_cast<int>(scale.numel()), k=static_cast<int>(width);
    auto output=torch::empty({n,k},scale.options().dtype(at::kHalf));
    auto w=weight(indices,aux,state,scale,table,nullptr,k,static_cast<int>(gs),static_cast<int>(bits),static_cast<int>(format),static_cast<int>(sign),0);
    tyloi::cuda::nvq_raw::decode(w,output.data_ptr(),at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ dequant kernel launch failed");
    return output;
}

/** Decode NEPQ shared-bank weights into their original F16 layout. */
torch::Tensor nepq_dequant_cuda(torch::Tensor indices, torch::Tensor aux,
    torch::Tensor state, torch::Tensor scale, torch::Tensor table,
    torch::Tensor banks, int64_t width, int64_t bits, int64_t format) {
    mfq_jit_nvq::check_nepq_common(indices,aux,state,scale,table,banks,width,bits,format);
    const int n=static_cast<int>(scale.numel()), k=static_cast<int>(width);
    auto output=torch::empty({n,k},scale.options().dtype(at::kHalf));
    auto w=weight(indices,aux,state,scale,table,banks.data_ptr<uint8_t>(),k,24,static_cast<int>(bits),static_cast<int>(format),0,static_cast<int>(table.size(1)));
    tyloi::cuda::nvq_raw::decode_nepq(w,output.data_ptr(),at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NEPQ dequant kernel launch failed");
    return output;
}

/** Decode the selected NVQ embedding rows. */
torch::Tensor nvq_embedding_lookup_cuda(torch::Tensor indices, torch::Tensor aux,
    torch::Tensor state, torch::Tensor scale, torch::Tensor table,
    torch::Tensor ids, int64_t width, int64_t gs, int64_t bits, int64_t format, int64_t sign) {
    mfq_jit_nvq::check_common(indices,aux,state,scale,table,width,gs,bits,format,sign);
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type()==at::kLong && ids.is_contiguous(),
        "NVQ token_ids must be CUDA contiguous int64");
    const int tokens=static_cast<int>(ids.numel()), k=static_cast<int>(width);
    auto shape=ids.sizes().vec(); shape.push_back(k);
    auto output=torch::empty(shape,scale.options().dtype(at::kHalf));
    auto w=weight(indices,aux,state,scale,table,nullptr,k,static_cast<int>(gs),static_cast<int>(bits),static_cast<int>(format),static_cast<int>(sign),0);
    tyloi::cuda::nvq_raw::embedding(w,ids.data_ptr<int64_t>(),tokens,output.data_ptr(),at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ embedding kernel launch failed");
    return output;
}
/** Apply the original online-dequant F16 projection. */
torch::Tensor nvq_gemm_f16_cuda(torch::Tensor indices, torch::Tensor aux,
    torch::Tensor state, torch::Tensor scale, torch::Tensor table,
    torch::Tensor x, int64_t width, int64_t gs, int64_t bits, int64_t format, int64_t sign) {
    mfq_jit_nvq::check_common(indices,aux,state,scale,table,width,gs,bits,format,sign);
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kHalf &&
        x.is_contiguous() && x.dim()==2,"NVQ FP16 GEMM x must be CUDA contiguous fp16 rank-2");
    const int m=static_cast<int>(x.size(0)), k=static_cast<int>(width), n=static_cast<int>(scale.numel());
    TORCH_CHECK(m>=16,"NVQ FP16 GEMM requires M >= 16");
    TORCH_CHECK(x.size(1)==k,"NVQ FP16 GEMM x width must equal neuron_len");
    auto output=torch::empty({m,n},x.options());
    auto w=weight(indices,aux,state,scale,table,nullptr,k,static_cast<int>(gs),static_cast<int>(bits),static_cast<int>(format),static_cast<int>(sign),0);
    tyloi::cuda::nvq_raw::gemm(w,x.data_ptr(),output.data_ptr(),m,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ FP16 GEMM kernel launch failed");
    return output;
}/** Apply the original NVQ quantized projection with caller-owned workspace. */
torch::Tensor nvq_gemv_m1_vec8_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1, "NVQ vec8 GEMV x must be CUDA contiguous fp16 [1,K]");
    TORCH_CHECK(nwarps == 1 || nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ vec8 GEMV nwarps must be 1, 2, 4, or 8");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ vec8 GEMV x width must equal neuron_len");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() &&
                qx.numel() >= K_pad, "NVQ vec8 GEMV qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.numel() >= ng, "NVQ vec8 GEMV xscale workspace mismatch");
    auto output = torch::empty({1, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,1,static_cast<int>(nwarps),0};
    tyloi::cuda::nvq_raw::gemv(p,true,false,1,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
torch::Tensor nvq_gemv_batch_vec8_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t nwarps,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2,
                "NVQ batch vec8 GEMV x must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(nwarps == 2 || nwarps == 4 || nwarps == 8,
                "NVQ batch vec8 GEMV nwarps must be 2, 4, or 8");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(M >= 2 && M <= 16, "NVQ batch vec8 GEMV supports M in [2,16]");
    TORCH_CHECK(x.size(1) == K, "NVQ batch vec8 GEMV x width must equal neuron_len");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ batch vec8 GEMV qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ batch vec8 GEMV xscale workspace mismatch");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,static_cast<int>(nwarps),0};
    tyloi::cuda::nvq_raw::gemv(p,true,false,2,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
torch::Tensor nvq_gemv_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 1 && M <= 16, "NVQ direct GEMV supports M in [1,16]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::gemv(p,true,false,0,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,M==1 ? "NVQ vec8 GEMV kernel launch failed" : "NVQ batch vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
torch::Tensor nvq_gemv_qx_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() && qx.dim() == 2,
                "NVQ prequantized qx must be CUDA contiguous int8 rank-2");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.dim() == 2, "NVQ prequantized xscale must be CUDA contiguous fp32 rank-2");
    const int M = static_cast<int>(qx.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(M >= 1 && M <= 16, "NVQ prequantized GEMV supports M in [1,16]");
    TORCH_CHECK(qx.size(1) >= K_pad, "NVQ prequantized qx width mismatch");
    TORCH_CHECK(xscale.size(0) >= M && xscale.size(1) >= ng,
                "NVQ prequantized xscale shape mismatch");
    auto output = torch::empty(
        {M, N}, neuron_scale.options().dtype(at::kHalf));
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,nullptr,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::gemv(p,false,false,0,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ prequantized GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
torch::Tensor nvq_gemv_qx_residual_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor residual) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(
        tyloi::cuda::nvq_raw::group_exec(static_cast<int>(format)),
        "NVQ fused residual requires an aligned group execution format");
    TORCH_CHECK(
        qx.is_cuda() && qx.scalar_type() == at::kChar &&
        qx.is_contiguous() && qx.dim() == 2 && qx.size(0) == 1,
        "NVQ fused residual qx must be CUDA contiguous int8 [1,K]");
    TORCH_CHECK(
        xscale.is_cuda() && xscale.scalar_type() == at::kFloat &&
        xscale.is_contiguous() && xscale.dim() == 2 && xscale.size(0) >= 1,
        "NVQ fused residual xscale must be CUDA contiguous fp32 [1,ng]");
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(qx.size(1) >= K_pad, "NVQ fused residual qx width mismatch");
    TORCH_CHECK(xscale.size(1) >= ng, "NVQ fused residual xscale width mismatch");
    TORCH_CHECK(
        residual.is_cuda() && residual.scalar_type() == at::kHalf &&
        residual.is_contiguous() && residual.dim() == 2 &&
        residual.size(0) == 1 && residual.size(1) == N,
        "NVQ fused residual must be CUDA contiguous fp16 [1,N]");

    auto output = torch::empty_like(residual);
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,nullptr,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),residual.data_ptr(),nullptr,1,0,0};
    tyloi::cuda::nvq_raw::gemv_residual(p,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ fused residual GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ quantized projection with caller-owned workspace. */
torch::Tensor nvq_gemv_gate_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    torch::Tensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == at::kHalf && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 1 && M <= 16, "NVQ gated direct GEMV supports M in [1,16]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,gate.data_ptr(),M,0,static_cast<int>(mode)};
    tyloi::cuda::nvq_raw::gemv(p,true,true,0,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,M==1 ? "NVQ gated vec8 GEMV kernel launch failed" : "NVQ gated batch vec8 GEMV kernel launch failed");
    return output;
}

/** Apply the original NVQ integer-MMA projection with caller-owned workspace. */
torch::Tensor nvq_mmq_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 4 && M <= 64, "NVQ MMA MMQ supports M in [4,64]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::mmq(p,false,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ MMA MMQ kernel launch failed" );
    return output;
}

/** Apply the original NVQ integer-MMA projection with caller-owned workspace. */
torch::Tensor nvq_mmq_gate_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    torch::Tensor gate,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    int64_t mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode);
    TORCH_CHECK(mode == 1 || mode == 2, "NVQ gate mode must be 1(sigmoid) or 2(silu)");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2,
                "NVQ x must be CUDA contiguous fp16 rank-2");
    TORCH_CHECK(gate.is_cuda() && gate.scalar_type() == at::kHalf && gate.is_contiguous() &&
                gate.sizes() == x.sizes(), "NVQ gate must be CUDA contiguous fp16 with x shape");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ x width must equal neuron_len");
    TORCH_CHECK(M >= 4 && M <= 64, "NVQ gated MMA MMQ supports M in [4,64]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad, "NVQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng, "NVQ xscale workspace mismatch");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,gate.data_ptr(),M,0,static_cast<int>(mode)};
    tyloi::cuda::nvq_raw::mmq(p,true,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ gated MMA MMQ kernel launch failed" );
    return output;
}
/** Apply the original NEPQ projection with its validated inputs. */
torch::Tensor nepq_gemv_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + 24 - 1) / 24;
    const int K_pad = ng * 24;
    TORCH_CHECK(x.size(1) == K && M >= 1 && M <= 16,
                "NEPQ GEMV input must be [M,K] with M in [1,16]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,table_pool,bank_ids.data_ptr<uint8_t>(),K,24,static_cast<int>(sub_bits),static_cast<int>(format),0,static_cast<int>(table_pool.size(1)));
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::nepq_gemv(p,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NEPQ GEMV kernel launch failed" );
    return output;
}

/** Apply the original NEPQ projection with its validated inputs. */
torch::Tensor nepq_mmq_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + 24 - 1) / 24;
    const int K_pad = ng * 24;
    TORCH_CHECK(x.size(1) == K && M >= 4 && M <= 64,
                "NEPQ MMQ input must be [M,K] with M in [4,64]");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar &&
                qx.is_contiguous() && qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NEPQ qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat &&
                xscale.is_contiguous() && xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NEPQ xscale workspace mismatch");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,table_pool,bank_ids.data_ptr<uint8_t>(),K,24,static_cast<int>(sub_bits),static_cast<int>(format),0,static_cast<int>(table_pool.size(1)));
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::nepq_mmq(p,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NEPQ MMQ kernel launch failed" );
    return output;
}

/** Apply the original NEPQ projection with its validated inputs. */
torch::Tensor nepq_gemm_f16_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format) {
    mfq_jit_nvq::check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf &&
                x.is_contiguous() && x.dim() == 2,
                "NEPQ x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int N = static_cast<int>(neuron_scale.numel());
    const int ng = (K + 24 - 1) / 24;
    TORCH_CHECK(x.size(1) == K && M >= 16 && M <= 256,
                "NEPQ online GEMM input must be [M,K] with M in [16,256]");
    auto output = torch::empty({M, N}, x.options());
    auto w=weight(indices,aux,sub_scale,neuron_scale,table_pool,bank_ids.data_ptr<uint8_t>(),K,24,static_cast<int>(sub_bits),static_cast<int>(format),0,static_cast<int>(table_pool.size(1)));
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),nullptr,nullptr,output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::nepq_gemm(p,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NEPQ online GEMM kernel launch failed" );
    return output;
}

/** Apply the original NEPQ grouped routed projection. */
torch::Tensor nepq_moe_grouped_matmul_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor grouped_table_pool,
    torch::Tensor x,
    torch::Tensor ids,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    torch::Tensor out,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor ids_dst,
    torch::Tensor expert_bounds,
    torch::Tensor tile_bounds,
    torch::Tensor tile_experts) {
    mfq_jit_nvq::check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = tyloi::cuda::nvq_raw::nepq_grouped_stride(static_cast<int>(format),static_cast<int>(table_pool.size(1)));
    TORCH_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == at::kChar &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ expert count must be in [1,4096]");
    TORCH_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == n_experts * out_per_expert,
                "NEPQ expert shape does not match the flattened row count");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == at::kInt && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == at::kHalf &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    TORCH_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == at::kHalf && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + 24 - 1) / 24;
    const int K_pad = ng * 24;
    const int input_rows = routed_input ? pairs : tokens;
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == at::kChar && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == at::kFloat && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    auto w=weight(indices,aux,sub_scale,neuron_scale,table_pool,bank_ids.data_ptr<uint8_t>(),K,24,
        static_cast<int>(sub_bits),static_cast<int>(format),0,static_cast<int>(table_pool.size(1)));
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        out.data_ptr(),nullptr,nullptr,input_rows,0,0};
    const auto stream=at::cuda::getCurrentCUDAStream().stream();
    tyloi::cuda::nvq_raw::quantize(p,stream);
    tyloi::cuda::nvq_raw::Grouped grouped{p,grouped_table_pool.data_ptr<int8_t>(),ids.data_ptr<int32_t>(),
        nullptr,nullptr,nullptr,nullptr,nullptr,tokens,routes,static_cast<int>(n_experts),static_cast<int>(n_experts),
        static_cast<int>(out_per_expert),static_cast<int>(grouped_table_pool.size(1)),routed_input};
    if(tokens>8) {
        TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == at::kInt && ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == at::kInt &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == at::kInt &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == at::kInt &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        grouped.ids_dst=ids_dst.data_ptr<int32_t>();
        grouped.expert_bounds=expert_bounds.data_ptr<int32_t>();
        grouped.tile_bounds=tile_bounds.data_ptr<int32_t>();
        grouped.tile_experts=tile_experts.data_ptr<int32_t>();
    }
    tyloi::cuda::nvq_raw::nepq_grouped(grouped,stream);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NEPQ routed matmul kernel launch failed");
    return out;
}
/** Apply the original NEPQ pooled routed projection. */
torch::Tensor nepq_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor table_pool,
    torch::Tensor bank_ids,
    torch::Tensor grouped_table_pool,
    torch::Tensor x,
    torch::Tensor ids,
    torch::Tensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t sub_bits,
    int64_t format,
    bool input_quantized,
    torch::Tensor out,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor ids_dst,
    torch::Tensor expert_bounds,
    torch::Tensor tile_bounds,
    torch::Tensor tile_experts) {
    mfq_jit_nvq::check_nepq_common(
        indices, aux, sub_scale, neuron_scale, table_pool, bank_ids,
        neuron_len, sub_bits, format);
    const int expected_grouped_stride = tyloi::cuda::nvq_raw::nepq_grouped_stride(static_cast<int>(format),static_cast<int>(table_pool.size(1)));
    TORCH_CHECK(
        grouped_table_pool.is_cuda() && grouped_table_pool.is_contiguous() &&
        grouped_table_pool.scalar_type() == at::kChar &&
        grouped_table_pool.dim() == 2 &&
        grouped_table_pool.size(0) == table_pool.size(0) &&
        grouped_table_pool.size(1) == expected_grouped_stride,
        "NEPQ grouped table pool shape mismatch");
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096,
                "NEPQ global expert count must be in [1,4096]");
    TORCH_CHECK(pool_experts > 0 &&
                pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()) &&
                out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NEPQ pool shape does not match the flattened row count");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == at::kInt &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NEPQ expert-local map must be CUDA contiguous int32");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == at::kInt && ids.dim() == 2,
                "NEPQ route IDs must be CUDA contiguous int32 [T,R]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == at::kHalf &&
                (x.dim() == 2 || x.dim() == 3) && x.size(-1) == neuron_len,
                "NEPQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(tokens > 0 && routes > 0, "NEPQ route dimensions must be nonzero");
    TORCH_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NEPQ input leading dimensions do not match routes");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == at::kHalf && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NEPQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + 24 - 1) / 24;
    const int K_pad = ng * 24;
    const int input_rows = routed_input ? pairs : tokens;
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == at::kChar && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NEPQ routed qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == at::kFloat && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NEPQ routed xscale workspace mismatch");
    auto w=weight(indices,aux,sub_scale,neuron_scale,table_pool,bank_ids.data_ptr<uint8_t>(),K,24,
        static_cast<int>(sub_bits),static_cast<int>(format),0,static_cast<int>(table_pool.size(1)));
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        out.data_ptr(),nullptr,nullptr,input_rows,0,0};
    const auto stream=at::cuda::getCurrentCUDAStream().stream();
    if(!input_quantized) tyloi::cuda::nvq_raw::quantize(p,stream);
    tyloi::cuda::nvq_raw::Grouped grouped{p,grouped_table_pool.data_ptr<int8_t>(),ids.data_ptr<int32_t>(),
        nullptr,nullptr,nullptr,nullptr,expert_local.data_ptr<int32_t>(),tokens,routes,static_cast<int>(n_experts),
        static_cast<int>(pool_experts),static_cast<int>(out_per_expert),static_cast<int>(grouped_table_pool.size(1)),routed_input};
    if(tokens>8) {
        TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == at::kInt &&
                    ids_dst.numel() >= pairs,
                    "NEPQ compact route map is missing");
        TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == at::kInt &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NEPQ expert bounds are missing");
        TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == at::kInt &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NEPQ tile bounds are missing");
        TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == at::kInt &&
                    tile_experts.numel() >= pairs,
                    "NEPQ tile expert map is missing");
        grouped.ids_dst=ids_dst.data_ptr<int32_t>();
        grouped.expert_bounds=expert_bounds.data_ptr<int32_t>();
        grouped.tile_bounds=tile_bounds.data_ptr<int32_t>();
        grouped.tile_experts=tile_experts.data_ptr<int32_t>();
    }
    tyloi::cuda::nvq_raw::nepq_grouped(grouped,stream);
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NEPQ routed pool matmul kernel launch failed");
    return out;
}
/** Apply the original online-F16 NEPQ pooled projection. */
torch::Tensor nepq_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor table_pool,
    torch::Tensor bank_ids, torch::Tensor x, torch::Tensor expert_local,
    int64_t n_experts, int64_t pool_experts, int64_t out_per_expert,
    int64_t neuron_len, int64_t sub_bits, int64_t format,
    torch::Tensor out, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts) {
    mfq_jit_nvq::check_nepq_common(indices, aux, sub_scale, neuron_scale, table_pool,
                      bank_ids, neuron_len, sub_bits, format);
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == at::kHalf &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NEPQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    auto w=weight(indices,aux,sub_scale,neuron_scale,table_pool,bank_ids.data_ptr<uint8_t>(),K,24,
        static_cast<int>(sub_bits),static_cast<int>(format),0,static_cast<int>(table_pool.size(1)));
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),nullptr,nullptr,out.data_ptr(),nullptr,nullptr,0,0,0};
    tyloi::cuda::nvq_raw::Grouped grouped{p,nullptr,nullptr,ids_dst.data_ptr<int32_t>(),
        expert_bounds.data_ptr<int32_t>(),tile_bounds.data_ptr<int32_t>(),tile_experts.data_ptr<int32_t>(),
        expert_local.data_ptr<int32_t>(),static_cast<int>(out.size(0)),routes,static_cast<int>(n_experts),
        static_cast<int>(pool_experts),static_cast<int>(out_per_expert),0,x.dim()==3};
    tyloi::cuda::nvq_raw::nepq_grouped_f16(grouped,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
/** Apply the original online-F16 NVQ pooled projection. */
torch::Tensor nvq_moe_grouped_matmul_pool_f16_cuda(
    torch::Tensor indices, torch::Tensor aux, torch::Tensor sub_scale,
    torch::Tensor neuron_scale, torch::Tensor codebook, torch::Tensor x,
    torch::Tensor expert_local, int64_t n_experts, int64_t pool_experts,
    int64_t out_per_expert, int64_t neuron_len, int64_t gs,
    int64_t sub_bits, int64_t format, int64_t sign_mode,
    torch::Tensor out, torch::Tensor ids_dst, torch::Tensor expert_bounds,
    torch::Tensor tile_bounds, torch::Tensor tile_experts) {
    mfq_jit_nvq::check_common(indices, aux, sub_scale, neuron_scale, codebook,
                 neuron_len, gs, sub_bits, format, sign_mode, true);
    TORCH_CHECK(gs == 24, "NVQ routed FP16 requires gs24");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == at::kHalf &&
                (x.dim() == 2 || x.dim() == 3) &&
                x.size(-1) == neuron_len,
                "NVQ routed FP16 input shape mismatch");
    const int routes = static_cast<int>(out.size(1));
    const int pairs = static_cast<int>(out.size(0)) * routes;
    const int K = static_cast<int>(neuron_len);
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),
        static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),nullptr,nullptr,out.data_ptr(),nullptr,nullptr,0,0,0};
    tyloi::cuda::nvq_raw::Grouped grouped{p,nullptr,nullptr,ids_dst.data_ptr<int32_t>(),
        expert_bounds.data_ptr<int32_t>(),tile_bounds.data_ptr<int32_t>(),tile_experts.data_ptr<int32_t>(),
        expert_local.data_ptr<int32_t>(),static_cast<int>(out.size(0)),routes,static_cast<int>(n_experts),
        static_cast<int>(pool_experts),static_cast<int>(out_per_expert),0,x.dim()==3};
    tyloi::cuda::nvq_raw::nvq_grouped_f16(grouped,at::cuda::getCurrentCUDAStream().stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
/** Apply the original NVQ pooled routed projection. */
torch::Tensor nvq_moe_grouped_matmul_pool_ws_cuda(
    torch::Tensor indices,
    torch::Tensor aux,
    torch::Tensor sub_scale,
    torch::Tensor neuron_scale,
    torch::Tensor codebook,
    torch::Tensor x,
    torch::Tensor ids,
    torch::Tensor expert_local,
    int64_t n_experts,
    int64_t pool_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t gs,
    int64_t sub_bits,
    int64_t format,
    int64_t sign_mode,
    bool input_quantized,
    torch::Tensor out,
    torch::Tensor qx,
    torch::Tensor xscale,
    torch::Tensor ids_dst,
    torch::Tensor expert_bounds,
    torch::Tensor tile_bounds,
    torch::Tensor tile_experts) {
    mfq_jit_nvq::check_common(
        indices, aux, sub_scale, neuron_scale, codebook,
        neuron_len, gs, sub_bits, format, sign_mode,true);
    TORCH_CHECK(n_experts > 0 && n_experts <= 4096,
                "NVQ global expert count must be in [1,4096]");
    TORCH_CHECK(
        pool_experts > 0 &&
        pool_experts <= static_cast<int64_t>(std::numeric_limits<int>::max()),
        "NVQ pool storage expert count must fit int32");
    TORCH_CHECK(out_per_expert > 0 &&
                neuron_scale.numel() == pool_experts * out_per_expert,
                "NVQ pool shape does not match the flattened row count");
    TORCH_CHECK(expert_local.is_cuda() && expert_local.is_contiguous() &&
                expert_local.scalar_type() == at::kInt &&
                expert_local.dim() == 1 && expert_local.numel() >= n_experts,
                "NVQ expert-local map must be CUDA contiguous int32");
    TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() &&
                ids.scalar_type() == at::kInt && ids.dim() == 2,
                "NVQ route IDs must be CUDA contiguous int32 [T,R]");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() &&
                x.scalar_type() == at::kHalf &&
                (x.dim() == 2 || x.dim() == 3) &&
                (input_quantized || x.size(-1) == neuron_len),
                "NVQ routed input must be CUDA contiguous fp16 [T,K] or [T,R,K]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(tokens > 0 && routes > 0, "NVQ route dimensions must be nonzero");
    TORCH_CHECK((!routed_input && x.size(0) == tokens) ||
                (routed_input && x.size(0) == tokens && x.size(1) == routes),
                "NVQ input leading dimensions do not match routes");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() &&
                out.scalar_type() == at::kHalf && out.dim() == 3 &&
                out.size(0) == tokens && out.size(1) == routes &&
                out.size(2) == out_per_expert,
                "NVQ output must be CUDA contiguous fp16 [T,R,O]");

    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int input_rows = routed_input ? pairs : tokens;
    TORCH_CHECK(qx.is_cuda() && qx.is_contiguous() &&
                qx.scalar_type() == at::kChar && qx.dim() == 2 &&
                qx.size(0) >= input_rows && qx.size(1) >= K_pad,
                "NVQ routed qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.is_contiguous() &&
                xscale.scalar_type() == at::kFloat && xscale.dim() == 2 &&
                xscale.size(0) >= input_rows && xscale.size(1) >= ng,
                "NVQ routed xscale workspace mismatch");
    auto w=weight(indices,aux,sub_scale,neuron_scale,codebook,nullptr,K,static_cast<int>(gs),
        static_cast<int>(sub_bits),static_cast<int>(format),static_cast<int>(sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        out.data_ptr(),nullptr,nullptr,input_rows,0,0};
    const auto stream=at::cuda::getCurrentCUDAStream().stream();
    if(!input_quantized) tyloi::cuda::nvq_raw::quantize(p,stream);
    tyloi::cuda::nvq_raw::Grouped grouped{p,nullptr,ids.data_ptr<int32_t>(),nullptr,nullptr,nullptr,nullptr,
        expert_local.data_ptr<int32_t>(),tokens,routes,static_cast<int>(n_experts),static_cast<int>(pool_experts),
        static_cast<int>(out_per_expert),0,routed_input};
    const bool dense=tyloi::cuda::nvq_raw::dense_format(static_cast<int>(format)) && aux.numel()==0;
    if(tokens>8 && !dense) {
        TORCH_CHECK(ids_dst.is_cuda() && ids_dst.is_contiguous() &&
                    ids_dst.scalar_type() == at::kInt &&
                    ids_dst.numel() >= pairs,
                    "NVQ compact route map is missing");
        TORCH_CHECK(expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
                    expert_bounds.scalar_type() == at::kInt &&
                    expert_bounds.numel() >= n_experts + 1,
                    "NVQ expert bounds are missing");
        TORCH_CHECK(tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
                    tile_bounds.scalar_type() == at::kInt &&
                    tile_bounds.numel() >= n_experts + 1,
                    "NVQ tile bounds are missing");
        TORCH_CHECK(tile_experts.is_cuda() && tile_experts.is_contiguous() &&
                    tile_experts.scalar_type() == at::kInt &&
                    tile_experts.numel() >= pairs,
                    "NVQ tile expert map is missing");
        grouped.ids_dst=ids_dst.data_ptr<int32_t>(); grouped.expert_bounds=expert_bounds.data_ptr<int32_t>();
        grouped.tile_bounds=tile_bounds.data_ptr<int32_t>(); grouped.tile_experts=tile_experts.data_ptr<int32_t>();
    }
    const bool used_dense=tyloi::cuda::nvq_raw::nvq_grouped(grouped,stream);
    if(used_dense) { C10_CUDA_KERNEL_LAUNCH_CHECK(); return out; }
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ routed pool matmul kernel launch failed");
    return out;
}
/** Apply the original heterogeneous F16 routed projection. */
torch::Tensor nvq_moe_grouped_matmul_hetero_f16_cuda(
    torch::Tensor weight_ptrs,
    torch::Tensor weight_sizes,
    torch::Tensor pool_params,
    torch::Tensor expert_pool,
    torch::Tensor expert_local,
    torch::Tensor x,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    int64_t route_tile_m,
    torch::Tensor out,
    torch::Tensor ids_dst,
    torch::Tensor expert_bounds,
    torch::Tensor tile_bounds,
    torch::Tensor tile_experts,
    int64_t format_group,
    bool masked_experts) {
    TORCH_CHECK(
        n_experts > 0 && n_experts <= 4096,
        "NVQ heterogeneous expert count must be in [1,4096]");
    TORCH_CHECK(
        out_per_expert > 0 && out_per_expert <= INT_MAX &&
            neuron_len > 0 && neuron_len <= INT_MAX,
        "NVQ heterogeneous dimensions must fit int32");
    TORCH_CHECK(
        route_tile_m == 8 || route_tile_m == 64 || route_tile_m == 128,
        "NVQ heterogeneous route tile must be 8, 64, or 128");
    TORCH_CHECK(
        tyloi::cuda::nvq_raw::hetero_format(format_group),
        "NVQ heterogeneous FP16 format group is invalid");
    TORCH_CHECK(
        weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
            weight_ptrs.scalar_type() == at::kLong &&
            weight_ptrs.dim() == 2 && weight_ptrs.size(0) > 0 &&
            weight_ptrs.size(1) == 5,
        "NVQ heterogeneous weight pointers must be CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    TORCH_CHECK(
        weight_sizes.is_cuda() && weight_sizes.is_contiguous() &&
            weight_sizes.scalar_type() == at::kLong &&
            weight_sizes.dim() == 2 && weight_sizes.size(0) == pools &&
            weight_sizes.size(1) == 3,
        "NVQ heterogeneous weight sizes must be CUDA int64 [pools,3]");
    TORCH_CHECK(
        pool_params.is_cuda() && pool_params.is_contiguous() &&
            pool_params.scalar_type() == at::kInt &&
            pool_params.dim() == 2 && pool_params.size(0) == pools &&
            pool_params.size(1) == 7,
        "NVQ heterogeneous pool parameters must be CUDA int32 [pools,7]");
    TORCH_CHECK(
        expert_pool.is_cuda() && expert_pool.is_contiguous() &&
            expert_pool.scalar_type() == at::kInt &&
            expert_pool.numel() == n_experts,
        "NVQ heterogeneous expert-pool map must be CUDA int32 [experts]");
    TORCH_CHECK(
        expert_local.is_cuda() && expert_local.is_contiguous() &&
            expert_local.scalar_type() == at::kInt &&
            expert_local.numel() == n_experts,
        "NVQ heterogeneous expert-local map must be CUDA int32 [experts]");
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == at::kHalf &&
            (x.dim() == 2 || x.dim() == 3) &&
            x.size(-1) == neuron_len,
        "NVQ heterogeneous FP16 input shape mismatch");
    TORCH_CHECK(
        out.is_cuda() && out.is_contiguous() &&
            out.scalar_type() == at::kHalf &&
            out.dim() == 3 && out.size(2) == out_per_expert,
        "NVQ heterogeneous output must be CUDA FP16 [tokens,routes,out]");
    const int tokens = static_cast<int>(out.size(0));
    const int routes = static_cast<int>(out.size(1));
    const int pairs = tokens * routes;
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(
        tokens > 8 && routes > 0 &&
            ((!routed_input && x.size(0) == tokens) ||
             (routed_input && x.size(0) == tokens && x.size(1) == routes)),
        "NVQ heterogeneous input leading dimensions do not match output");
    TORCH_CHECK(
        ids_dst.is_cuda() && ids_dst.is_contiguous() &&
            ids_dst.scalar_type() == at::kInt &&
            ids_dst.numel() >= pairs,
        "NVQ heterogeneous compact route map is missing");
    TORCH_CHECK(
        expert_bounds.is_cuda() && expert_bounds.is_contiguous() &&
            expert_bounds.scalar_type() == at::kInt &&
            expert_bounds.numel() >= n_experts + 1,
        "NVQ heterogeneous expert bounds are missing");
    TORCH_CHECK(
        tile_bounds.is_cuda() && tile_bounds.is_contiguous() &&
            tile_bounds.scalar_type() == at::kInt &&
            tile_bounds.numel() >= n_experts + 1,
        "NVQ heterogeneous tile bounds are missing");
    TORCH_CHECK(
        tile_experts.is_cuda() && tile_experts.is_contiguous() &&
            tile_experts.scalar_type() == at::kInt &&
            tile_experts.numel() >= pairs,
        "NVQ heterogeneous tile expert map is missing");
    for (const auto * tensor : {
             &weight_sizes, &pool_params, &expert_pool, &expert_local,
             &x, &out, &ids_dst, &expert_bounds, &tile_bounds,
             &tile_experts}) {
        TORCH_CHECK(
            tensor->device() == weight_ptrs.device(),
            "NVQ heterogeneous tensors must share one CUDA device");
    }

    tyloi::cuda::nvq_raw::Hetero a{weight_ptrs.data_ptr<int64_t>(),weight_sizes.data_ptr<int64_t>(),
        pool_params.data_ptr<int32_t>(),expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),
        x.data_ptr(),out.data_ptr(),ids_dst.data_ptr<int32_t>(),expert_bounds.data_ptr<int32_t>(),
        tile_bounds.data_ptr<int32_t>(),tile_experts.data_ptr<int32_t>(),tokens,routes,static_cast<int>(n_experts),
        pools,static_cast<int>(out_per_expert),static_cast<int>(neuron_len),static_cast<int>(route_tile_m),
        static_cast<int>(format_group),masked_experts,routed_input};
    TORCH_CHECK(tyloi::cuda::nvq_raw::hetero_f16(a,at::cuda::getCurrentCUDAStream().stream()),
        "failed to opt in to the NVQ MoE shared-memory size");
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
static torch::Tensor nvq_moe_grouped_matmul_hetero_impl(
    torch::Tensor weight_ptrs,
    torch::Tensor weight_sizes,
    torch::Tensor pool_params,
    torch::Tensor expert_pool,
    torch::Tensor expert_local,
    torch::Tensor x,
    torch::Tensor ids,
    int64_t n_experts,
    int64_t out_per_expert,
    int64_t neuron_len,
    bool input_quantized,
    torch::Tensor out,
    torch::Tensor qx,
    torch::Tensor xscale,torch::Tensor active_plan) {
    TORCH_CHECK(
        n_experts > 0 && n_experts <= 4096,
        "NVQ heterogeneous expert count must be in [1,4096]");
    TORCH_CHECK(
        out_per_expert > 0 && out_per_expert <= INT_MAX &&
            neuron_len > 0 && neuron_len <= INT_MAX,
        "NVQ heterogeneous dimensions must fit int32");
    TORCH_CHECK(
        weight_ptrs.is_cuda() && weight_ptrs.is_contiguous() &&
            weight_ptrs.scalar_type() == at::kLong &&
            weight_ptrs.dim() == 2 && weight_ptrs.size(0) > 0 &&
            weight_ptrs.size(1) == 5,
        "NVQ heterogeneous weight pointers must be CUDA int64 [pools,5]");
    const int pools = static_cast<int>(weight_ptrs.size(0));
    TORCH_CHECK(
        weight_sizes.is_cuda() && weight_sizes.is_contiguous() &&
            weight_sizes.scalar_type() == at::kLong &&
            weight_sizes.dim() == 2 && weight_sizes.size(0) == pools &&
            weight_sizes.size(1) == 3,
        "NVQ heterogeneous weight sizes must be CUDA int64 [pools,3]");
    TORCH_CHECK(
        pool_params.is_cuda() && pool_params.is_contiguous() &&
            pool_params.scalar_type() == at::kInt &&
            pool_params.dim() == 2 && pool_params.size(0) == pools &&
            pool_params.size(1) == 7,
        "NVQ heterogeneous pool parameters must be CUDA int32 [pools,7]");
    TORCH_CHECK(
        expert_pool.is_cuda() && expert_pool.is_contiguous() &&
            expert_pool.scalar_type() == at::kInt &&
            expert_pool.numel() == n_experts &&
            expert_local.is_cuda() && expert_local.is_contiguous() &&
            expert_local.scalar_type() == at::kInt &&
            expert_local.numel() == n_experts,
        "NVQ heterogeneous expert maps must be CUDA int32 [experts]");
    TORCH_CHECK(
        ids.is_cuda() && ids.is_contiguous() &&
            ids.scalar_type() == at::kInt && ids.dim() == 2,
        "NVQ heterogeneous route IDs must be CUDA int32 [tokens,routes]");
    const int tokens = static_cast<int>(ids.size(0));
    const int routes = static_cast<int>(ids.size(1));
    const int pairs = tokens * routes;
    TORCH_CHECK(
        tokens > 0 && tokens <= 8 && routes > 0,
        "NVQ heterogeneous MMVQ requires one to eight tokens");
    TORCH_CHECK(
        x.is_cuda() && x.is_contiguous() &&
            x.scalar_type() == at::kHalf &&
            (x.dim() == 2 || x.dim() == 3) &&
            (input_quantized || x.size(-1) == neuron_len),
        "NVQ heterogeneous routed input shape mismatch");
    const bool routed_input = x.dim() == 3;
    TORCH_CHECK(
        (!routed_input && x.size(0) == tokens) ||
            (routed_input && x.size(0) == tokens && x.size(1) == routes),
        "NVQ heterogeneous input leading dimensions do not match routes");
    TORCH_CHECK(
        out.is_cuda() && out.is_contiguous() &&
            out.scalar_type() == at::kHalf &&
            out.dim() == 3 && out.size(0) == tokens &&
            out.size(1) == routes && out.size(2) == out_per_expert,
        "NVQ heterogeneous output must be CUDA FP16 [tokens,routes,out]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + 24 - 1) / 24;
    const int K_pad = ng * 24;
    const int input_rows = routed_input ? pairs : tokens;
    TORCH_CHECK(
        qx.is_cuda() && qx.is_contiguous() &&
            qx.scalar_type() == at::kChar && qx.dim() == 2 &&
            qx.size(0) >= input_rows && qx.size(1) >= K_pad,
        "NVQ heterogeneous qx workspace mismatch");
    TORCH_CHECK(
        xscale.is_cuda() && xscale.is_contiguous() &&
            xscale.scalar_type() == at::kFloat &&
            xscale.dim() == 2 && xscale.size(0) >= input_rows &&
            xscale.size(1) >= ng,
        "NVQ heterogeneous xscale workspace mismatch");
    for (const auto * tensor : {
             &weight_sizes, &pool_params, &expert_pool, &expert_local,
             &x, &ids, &out, &qx, &xscale}) {
        TORCH_CHECK(
            tensor->device() == weight_ptrs.device(),
            "NVQ heterogeneous tensors must share one CUDA device");
    }

    if(active_plan.defined())TORCH_CHECK(active_plan.is_cuda() && active_plan.is_contiguous() &&
        active_plan.scalar_type()==at::kInt && active_plan.device()==x.device() &&
        active_plan.numel()==1+int64_t(pairs)*3,"NVQ active route plan geometry/device mismatch");
    tyloi::cuda::nvq_raw::Weight w{}; w.width=K; w.group_size=24;
    tyloi::cuda::nvq_raw::Projection p{w,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        out.data_ptr(),nullptr,nullptr,input_rows,0,0};
    const cudaStream_t stream=at::cuda::getCurrentCUDAStream().stream();
    if(!input_quantized) tyloi::cuda::nvq_raw::quantize(p,stream);
    tyloi::cuda::nvq_raw::Hetero route{weight_ptrs.data_ptr<int64_t>(),weight_sizes.data_ptr<int64_t>(),
        pool_params.data_ptr<int32_t>(),expert_pool.data_ptr<int32_t>(),expert_local.data_ptr<int32_t>(),
        x.data_ptr(),out.data_ptr(),nullptr,nullptr,nullptr,nullptr,tokens,routes,static_cast<int>(n_experts),
        pools,static_cast<int>(out_per_expert),K,0,0,false,routed_input};
    tyloi::cuda::nvq_raw::HeteroQuant a{route,qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),ids.data_ptr<int32_t>(),
        active_plan.defined()?active_plan.data_ptr<int32_t>():nullptr};
    tyloi::cuda::nvq_raw::hetero_quant(a,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
/** Apply the original heterogeneous quantized routed projection. */
torch::Tensor nvq_moe_grouped_matmul_hetero_ws_cuda(
    torch::Tensor weight_ptrs,torch::Tensor weight_sizes,
    torch::Tensor pool_params,torch::Tensor expert_pool,
    torch::Tensor expert_local,torch::Tensor x,torch::Tensor ids,
    int64_t n_experts,int64_t out_per_expert,int64_t neuron_len,bool input_quantized,
    torch::Tensor out,torch::Tensor qx,torch::Tensor xscale) {
    return nvq_moe_grouped_matmul_hetero_impl(weight_ptrs,weight_sizes,pool_params,expert_pool,expert_local,
        x,ids,n_experts,out_per_expert,neuron_len,input_quantized,out,qx,xscale,{});
}

/** Apply the original heterogeneous quantized routed projection. */
torch::Tensor nvq_moe_grouped_matmul_active_ws_cuda(
    torch::Tensor weight_ptrs,torch::Tensor weight_sizes,
    torch::Tensor pool_params,torch::Tensor expert_pool,
    torch::Tensor expert_local,torch::Tensor x,torch::Tensor ids,
    int64_t n_experts,int64_t out_per_expert,int64_t neuron_len,bool input_quantized,
    torch::Tensor out,torch::Tensor qx,torch::Tensor xscale,
    torch::Tensor active_plan) {
    TORCH_CHECK(active_plan.defined(),"NVQ active route plan required");
    return nvq_moe_grouped_matmul_hetero_impl(weight_ptrs,weight_sizes,pool_params,expert_pool,expert_local,
        x,ids,n_experts,out_per_expert,neuron_len,input_quantized,out,qx,xscale,active_plan);
}

/** Apply both original NVQ projections with one quantized input. */
torch::Tensor nvq_gemv_multi2_ws_cuda(
    torch::Tensor first_indices,
    torch::Tensor first_aux,
    torch::Tensor first_sub_scale,
    torch::Tensor first_neuron_scale,
    torch::Tensor first_codebook,
    torch::Tensor second_indices,
    torch::Tensor second_aux,
    torch::Tensor second_sub_scale,
    torch::Tensor second_neuron_scale,
    torch::Tensor second_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t first_sub_bits,
    int64_t first_format,
    int64_t first_sign_mode,
    int64_t second_sub_bits,
    int64_t second_format,
    int64_t second_sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(first_indices, first_aux, first_sub_scale, first_neuron_scale, first_codebook,
                 neuron_len, gs, first_sub_bits, first_format, first_sign_mode);
    mfq_jit_nvq::check_common(second_indices, second_aux, second_sub_scale, second_neuron_scale, second_codebook,
                 neuron_len, gs, second_sub_bits, second_format, second_sign_mode);
    TORCH_CHECK(first_format == second_format,
                "NVQ multi-projection requires the same NVQ format");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2,
                "NVQ multi-projection x must be CUDA contiguous fp16 rank-2");
    const int M = static_cast<int>(x.size(0));
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(M >= 1 && M <= 8, "NVQ multi-projection supports M in [1,8]");
    TORCH_CHECK(x.size(1) == K, "NVQ multi-projection x width mismatch");
    TORCH_CHECK(qx.is_cuda() && qx.scalar_type() == at::kChar && qx.is_contiguous() &&
                qx.numel() >= static_cast<int64_t>(M) * K_pad,
                "NVQ multi-projection qx workspace mismatch");
    TORCH_CHECK(xscale.is_cuda() && xscale.scalar_type() == at::kFloat && xscale.is_contiguous() &&
                xscale.numel() >= static_cast<int64_t>(M) * ng,
                "NVQ multi-projection xscale workspace mismatch");
    auto first=weight(first_indices,first_aux,first_sub_scale,first_neuron_scale,first_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(first_sub_bits),static_cast<int>(first_format),static_cast<int>(first_sign_mode),0);
    auto second=weight(second_indices,second_aux,second_sub_scale,second_neuron_scale,second_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(second_sub_bits),static_cast<int>(second_format),static_cast<int>(second_sign_mode),0);
    auto output=torch::empty({M,first.rows+second.rows},x.options());
    tyloi::cuda::nvq_raw::Projection p{first,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        output.data_ptr(),nullptr,nullptr,M,0,0};
    tyloi::cuda::nvq_raw::multi2({p,second},at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ multi-projection kernel launch failed");
    return output;
}
/** Apply the original fused NVQ projection. */
torch::Tensor nvq_gemv_swiglu_ws_cuda(
    torch::Tensor gate_indices,
    torch::Tensor gate_aux,
    torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale,
    torch::Tensor gate_codebook,
    torch::Tensor up_indices,
    torch::Tensor up_aux,
    torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale,
    torch::Tensor up_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    mfq_jit_nvq::check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TORCH_CHECK(gate_format == up_format, "NVQ SwiGLU requires matching NVQ formats");
    TORCH_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ SwiGLU gate/up output widths must match");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1,
                "NVQ SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    TORCH_CHECK(x.size(1) == K, "NVQ SwiGLU x width mismatch");
    TORCH_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ SwiGLU workspace mismatch");
    auto gate=weight(gate_indices,gate_aux,gate_sub_scale,gate_neuron_scale,gate_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(gate_sub_bits),static_cast<int>(gate_format),static_cast<int>(gate_sign_mode),0);
    auto up=weight(up_indices,up_aux,up_sub_scale,up_neuron_scale,up_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(up_sub_bits),static_cast<int>(up_format),static_cast<int>(up_sign_mode),0);    auto output=torch::empty({1,gate.rows},x.options());
    tyloi::cuda::nvq_raw::Projection p{gate,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        output.data_ptr(),nullptr,nullptr,1,0,0};
    tyloi::cuda::nvq_raw::swiglu({p,up},false,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ SwiGLU kernel launch failed" );
    return output;
}

/** Apply the original fused NVQ projection. */
torch::Tensor nvq2_gemv_swiglu_vec4_ordered_ws_cuda(
    torch::Tensor gate_indices,
    torch::Tensor gate_aux,
    torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale,
    torch::Tensor gate_codebook,
    torch::Tensor up_indices,
    torch::Tensor up_aux,
    torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale,
    torch::Tensor up_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    torch::Tensor qx,
    torch::Tensor xscale) {
    mfq_jit_nvq::check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    mfq_jit_nvq::check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TORCH_CHECK(tyloi::cuda::nvq_raw::nvq2_format(gate_format) && tyloi::cuda::nvq_raw::nvq2_format(up_format) &&
                gate_sub_bits == 4 && up_sub_bits == 4,
                "NVQ2 ordered vec4 SwiGLU requires NVQ2 sub_bits=4 weights");
    TORCH_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ2 ordered vec4 SwiGLU gate/up output widths must match");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ2 ordered vec4 SwiGLU x must be CUDA contiguous fp16 [1,K]");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int nsign = (K + 7) / 8;
    TORCH_CHECK(nsign % 4 == 0,
                "NVQ2 ordered vec4 SwiGLU requires a multiple of four 8-value vectors");
    TORCH_CHECK(qx.numel() >= K_pad && xscale.numel() >= ng,
                "NVQ2 ordered vec4 SwiGLU workspace mismatch");
    auto gate=weight(gate_indices,gate_aux,gate_sub_scale,gate_neuron_scale,gate_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(gate_sub_bits),static_cast<int>(gate_format),static_cast<int>(gate_sign_mode),0);
    auto up=weight(up_indices,up_aux,up_sub_scale,up_neuron_scale,up_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(up_sub_bits),static_cast<int>(up_format),static_cast<int>(up_sign_mode),0);    auto output=torch::empty({1,gate.rows},x.options());
    tyloi::cuda::nvq_raw::Projection p{gate,x.data_ptr(),qx.data_ptr<int8_t>(),xscale.data_ptr<float>(),
        output.data_ptr(),nullptr,nullptr,1,0,0};
    tyloi::cuda::nvq_raw::swiglu({p,up},true,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ2 ordered vec4 SwiGLU kernel launch failed" );
    return output;
}

/** Apply the original fused NVQ projection. */
void nvq_ffn_swiglu_quant_ws_cuda(
    torch::Tensor gate_indices,
    torch::Tensor gate_aux,
    torch::Tensor gate_sub_scale,
    torch::Tensor gate_neuron_scale,
    torch::Tensor gate_codebook,
    torch::Tensor up_indices,
    torch::Tensor up_aux,
    torch::Tensor up_sub_scale,
    torch::Tensor up_neuron_scale,
    torch::Tensor up_codebook,
    torch::Tensor x,
    int64_t neuron_len,
    int64_t gs,
    int64_t gate_sub_bits,
    int64_t gate_format,
    int64_t gate_sign_mode,
    int64_t up_sub_bits,
    int64_t up_format,
    int64_t up_sign_mode,
    int64_t down_gs,
    torch::Tensor input_qx,
    torch::Tensor input_xscale,
    torch::Tensor output_qx,
    torch::Tensor output_xscale,
    torch::Tensor swiglu_scratch) {
    mfq_jit_nvq::check_common(gate_indices, gate_aux, gate_sub_scale, gate_neuron_scale, gate_codebook,
                 neuron_len, gs, gate_sub_bits, gate_format, gate_sign_mode);
    mfq_jit_nvq::check_common(up_indices, up_aux, up_sub_scale, up_neuron_scale, up_codebook,
                 neuron_len, gs, up_sub_bits, up_format, up_sign_mode);
    TORCH_CHECK(gate_format == up_format, "NVQ fused FFN requires matching gate/up formats");
    TORCH_CHECK(gate_neuron_scale.numel() == up_neuron_scale.numel(),
                "NVQ fused FFN gate/up widths must match");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kHalf && x.is_contiguous() &&
                x.dim() == 2 && x.size(0) == 1 && x.size(1) == neuron_len,
                "NVQ fused FFN x must be CUDA contiguous fp16 [1,K]");
    TORCH_CHECK(down_gs == 24 || down_gs == 28 || down_gs == 32,
                "NVQ fused FFN down gs must be 24, 28, or 32");
    const int K = static_cast<int>(neuron_len);
    const int ng = (K + static_cast<int>(gs) - 1) / static_cast<int>(gs);
    const int K_pad = ng * static_cast<int>(gs);
    const int N = static_cast<int>(gate_neuron_scale.numel());
    const int down_ng = (N + static_cast<int>(down_gs) - 1) / static_cast<int>(down_gs);
    const int down_k_pad = down_ng * static_cast<int>(down_gs);
    TORCH_CHECK(input_qx.is_cuda() && input_qx.scalar_type() == at::kChar &&
                input_qx.is_contiguous() && input_qx.numel() >= K_pad,
                "NVQ fused FFN input qx workspace mismatch");
    TORCH_CHECK(input_xscale.is_cuda() && input_xscale.scalar_type() == at::kFloat &&
                input_xscale.is_contiguous() && input_xscale.numel() >= ng,
                "NVQ fused FFN input xscale workspace mismatch");
    TORCH_CHECK(output_qx.is_cuda() && output_qx.scalar_type() == at::kChar &&
                output_qx.is_contiguous() && output_qx.numel() >= down_k_pad,
                "NVQ fused FFN output qx workspace mismatch");
    TORCH_CHECK(output_xscale.is_cuda() && output_xscale.scalar_type() == at::kFloat &&
                output_xscale.is_contiguous() && output_xscale.numel() >= down_ng,
                "NVQ fused FFN output xscale workspace mismatch");
    TORCH_CHECK(swiglu_scratch.is_cuda() && swiglu_scratch.scalar_type() == at::kFloat &&
                swiglu_scratch.is_contiguous() && swiglu_scratch.numel() >= N,
                "NVQ fused FFN SwiGLU scratch mismatch");
    auto gate=weight(gate_indices,gate_aux,gate_sub_scale,gate_neuron_scale,gate_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(gate_sub_bits),static_cast<int>(gate_format),static_cast<int>(gate_sign_mode),0);
    auto up=weight(up_indices,up_aux,up_sub_scale,up_neuron_scale,up_codebook,nullptr,K,
        static_cast<int>(gs),static_cast<int>(up_sub_bits),static_cast<int>(up_format),static_cast<int>(up_sign_mode),0);
    tyloi::cuda::nvq_raw::Projection p{gate,x.data_ptr(),input_qx.data_ptr<int8_t>(),input_xscale.data_ptr<float>(),
        nullptr,nullptr,nullptr,1,0,0};
    tyloi::cuda::nvq_raw::Ffn f{{p,up},output_qx.data_ptr<int8_t>(),output_xscale.data_ptr<float>(),
        swiglu_scratch.data_ptr<float>(),static_cast<int>(down_gs)};
    tyloi::cuda::nvq_raw::ffn(f,at::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError()==cudaSuccess,"NVQ fused FFN kernel launch failed");
}
