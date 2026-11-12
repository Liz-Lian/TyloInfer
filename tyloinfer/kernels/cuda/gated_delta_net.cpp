#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include "env.h"
#include "gdn_raw.h"
#include <cstdlib>
#include <optional>
#include <vector>
namespace raw = ::tyloi::cuda::gdn_raw;
static std::vector<torch::Tensor> gdn_cuda_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> state,
    bool inplace_state, bool transposed_state=false, bool tiled_heads=false)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == at::kFloat,
                "q must be cuda contiguous f32");
    int B = q.size(0), Hq = q.size(1), T = q.size(2), D = q.size(3);
    int Hv = v.size(1);
    int kda = (g.dim() == 4) ? 1 : 0;
    auto opts = q.options();
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == at::kFloat,
                "k must be cuda contiguous f32");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == at::kFloat,
                "v must be cuda contiguous f32");
    TORCH_CHECK(k.size(0) == B && k.size(1) == Hq && k.size(2) == T && k.size(3) == D,
                "GDN CUDA: k shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(2) == T && v.size(3) == D,
                "GDN CUDA: v shape mismatch");
    TORCH_CHECK(Hv % Hq == 0, "GDN CUDA: value heads must be divisible by key heads");
    TORCH_CHECK(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == at::kFloat,
                "beta must be cuda contiguous f32");
    TORCH_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == Hv && beta.size(2) == T,
                "GDN CUDA: beta must be [B,Hv,T]");
    TORCH_CHECK(g.is_cuda() && g.is_contiguous() && g.scalar_type() == at::kFloat,
                "g must be cuda contiguous f32");
    if (kda) {
        TORCH_CHECK(g.size(0) == B && g.size(1) == Hv && g.size(2) == T && g.size(3) == D,
                    "GDN CUDA: KDA g must be [B,Hv,T,D]");
    } else {
        TORCH_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == Hv && g.size(2) == T,
                    "GDN CUDA: g must be [B,Hv,T]");
    }
    auto out = torch::empty({B, Hv, T, D}, opts);
    TORCH_CHECK(!inplace_state || (state.has_value() && state->defined() && state->numel() > 0),
                "GDN CUDA inplace: state is required");
    auto s_out = inplace_state ? state.value() : torch::empty({B, Hv, D, D}, opts);
    TORCH_CHECK(s_out.is_cuda() && s_out.is_contiguous() && s_out.scalar_type() == at::kFloat,
                "GDN CUDA: output state must be cuda contiguous f32");
    TORCH_CHECK(s_out.size(0) == B && s_out.size(1) == Hv && s_out.size(2) == D && s_out.size(3) == D,
                "GDN CUDA: state shape must be [B,Hv,D,D]");

    auto initial_state = state.has_value() && state->defined() && state->numel() > 0
        ? state->contiguous() : torch::Tensor{};
    const auto* col_env=tyloi::cuda::env_lookup("TYLOINFER_GDN_COLUMN");
    const auto* warp_env=tyloi::cuda::env_lookup("TYLOINFER_GDN_WARP");
    const bool warp=transposed_state || !(warp_env && warp_env[0]=='0');
    const auto* columns_env=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_COLUMNS");
    const int columns=columns_env?std::atoi(columns_env):(B*Hv>=16?(T==2?2:4):0);
    const auto* short_option=tyloi::cuda::env_lookup("TYLOINFER_GDN_SHORT_COLUMNS");
    const bool short_columns=!short_option || short_option[0]!='0';
    const bool column_path=warp && !kda && D==128 && (T>8 || (T>1 && short_columns)) && (columns==2 || columns==4);
    bool prepared=false,pipelined=false,staged=false;
    torch::Tensor decay;
    if(column_path) {
        const auto* option=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_PREPARED_DECAY");
        prepared=T>=128 && (!option || std::atoi(option)!=0);
        if(prepared)decay=torch::empty(g.sizes(),opts);
        const auto* pipeline=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_PIPELINED");
        pipelined=pipeline?std::atoi(pipeline)!=0:B==1 && (Hv==48 || Hv==64);
        const auto* stage=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_STAGED");
        if(columns==4 && T>8) {
            if(stage)staged=std::atoi(stage)!=0;
            else if(pipelined && T>=128 && B==1 && (Hv==48 || Hv==64)) {
                static thread_local int cached_device=-1,cached_major=0;
                int device=0; C10_CUDA_CHECK(cudaGetDevice(&device));
                if(device!=cached_device) {
                    C10_CUDA_CHECK(cudaDeviceGetAttribute(&cached_major,cudaDevAttrComputeCapabilityMajor,device));
                    cached_device=device;
                }
                staged=cached_major>=8;
            }
        }
    }
    raw::Recurrence args{q.data_ptr<float>(),k.data_ptr<float>(),v.data_ptr<float>(),g.data_ptr<float>(),beta.data_ptr<float>(),
        initial_state.defined()?initial_state.data_ptr<float>():nullptr,out.data_ptr<float>(),s_out.data_ptr<float>(),
        decay.defined()?decay.data_ptr<float>():nullptr,B,Hq,Hv,T,D,columns,bool(kda),transposed_state,tiled_heads,
        warp,col_env && col_env[0]=='1',short_columns,prepared,pipelined,staged};
    raw::recurrence(args,at::cuda::getCurrentCUDAStream().stream());
    return {out, s_out};
}

std::vector<torch::Tensor> gdn_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, c10::optional<torch::Tensor> state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, false, false);
}

std::vector<torch::Tensor> gdn_inplace_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, false);
}

std::vector<torch::Tensor> gdn_inplace_transposed_cuda(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    torch::Tensor g, torch::Tensor beta, torch::Tensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, true);
}
