#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "nvq_check.cuh"
#include "nvq_raw.h"
#include <cuda_runtime_api.h>
#include <cstdint>

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
