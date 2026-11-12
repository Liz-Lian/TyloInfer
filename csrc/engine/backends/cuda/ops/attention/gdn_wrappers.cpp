#include "backend.h"
#include "env.h"
#include "gdn_raw.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
namespace raw = ::tyloi::cuda::gdn_raw;
static std::vector<::tyloi::DenseTensor> gdn_cuda_impl(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, MfqOptional<::tyloi::DenseTensor> state,
    bool inplace_state, bool transposed_state=false, bool tiled_heads=false)
{
    TYLOINFER_RUNTIME_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == ::tyloi::kFloat32,
                "q must be cuda contiguous f32");
    int B = q.size(0), Hq = q.size(1), T = q.size(2), D = q.size(3);
    int Hv = v.size(1);
    int kda = (g.dim() == 4) ? 1 : 0;
    auto opts = q.options();
    TYLOINFER_RUNTIME_CHECK(k.is_cuda() && k.is_contiguous() && k.scalar_type() == ::tyloi::kFloat32,
                "k must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == ::tyloi::kFloat32,
                "v must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(k.size(0) == B && k.size(1) == Hq && k.size(2) == T && k.size(3) == D,
                "GDN CUDA: k shape mismatch");
    TYLOINFER_RUNTIME_CHECK(v.size(0) == B && v.size(2) == T && v.size(3) == D,
                "GDN CUDA: v shape mismatch");
    TYLOINFER_RUNTIME_CHECK(Hv % Hq == 0, "GDN CUDA: value heads must be divisible by key heads");
    TYLOINFER_RUNTIME_CHECK(beta.is_cuda() && beta.is_contiguous() && beta.scalar_type() == ::tyloi::kFloat32,
                "beta must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == Hv && beta.size(2) == T,
                "GDN CUDA: beta must be [B,Hv,T]");
    TYLOINFER_RUNTIME_CHECK(g.is_cuda() && g.is_contiguous() && g.scalar_type() == ::tyloi::kFloat32,
                "g must be cuda contiguous f32");
    if (kda) {
        TYLOINFER_RUNTIME_CHECK(g.size(0) == B && g.size(1) == Hv && g.size(2) == T && g.size(3) == D,
                    "GDN CUDA: KDA g must be [B,Hv,T,D]");
    } else {
        TYLOINFER_RUNTIME_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == Hv && g.size(2) == T,
                    "GDN CUDA: g must be [B,Hv,T]");
    }
    auto out = ::tyloi::cuda::empty({B, Hv, T, D}, opts);
    TYLOINFER_RUNTIME_CHECK(!inplace_state || (state.has_value() && state->defined() && state->numel() > 0),
                "GDN CUDA inplace: state is required");
    auto s_out = inplace_state ? state.value() : ::tyloi::cuda::empty({B, Hv, D, D}, opts);
    TYLOINFER_RUNTIME_CHECK(s_out.is_cuda() && s_out.is_contiguous() && s_out.scalar_type() == ::tyloi::kFloat32,
                "GDN CUDA: output state must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(s_out.size(0) == B && s_out.size(1) == Hv && s_out.size(2) == D && s_out.size(3) == D,
                "GDN CUDA: state shape must be [B,Hv,D,D]");

    auto initial_state = state.has_value() && state->defined() && state->numel() > 0
        ? state->contiguous() : ::tyloi::DenseTensor{};
    const auto* col_env=tyloi::cuda::env_lookup("TYLOINFER_GDN_COLUMN");
    const auto* warp_env=tyloi::cuda::env_lookup("TYLOINFER_GDN_WARP");
    const bool warp=transposed_state || !(warp_env && warp_env[0]=='0');
    const auto* columns_env=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_COLUMNS");
    const int columns=columns_env?std::atoi(columns_env):(B*Hv>=16?(T==2?2:4):0);
    const auto* short_option=tyloi::cuda::env_lookup("TYLOINFER_GDN_SHORT_COLUMNS");
    const bool short_columns=!short_option || short_option[0]!='0';
    const bool column_path=warp && !kda && D==128 && (T>8 || (T>1 && short_columns)) && (columns==2 || columns==4);
    bool prepared=false,pipelined=false,staged=false;
    ::tyloi::DenseTensor decay;
    if(column_path) {
        const auto* option=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_PREPARED_DECAY");
        prepared=T>=128 && (!option || std::atoi(option)!=0);
        if(prepared)decay=::tyloi::cuda::empty(g.sizes(),opts);
        const auto* pipeline=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_PIPELINED");
        pipelined=pipeline?std::atoi(pipeline)!=0:B==1 && (Hv==48 || Hv==64);
        const auto* stage=tyloi::cuda::env_lookup("TYLOINFER_GDN_PREFILL_STAGED");
        if(columns==4 && T>8) {
            if(stage)staged=std::atoi(stage)!=0;
            else if(pipelined && T>=128 && B==1 && (Hv==48 || Hv==64)) {
                static thread_local int cached_device=-1,cached_major=0;
                int device=0; TYLOINFER_CUDA_CHECK(cudaGetDevice(&device));
                if(device!=cached_device) {
                    TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&cached_major,cudaDevAttrComputeCapabilityMajor,device));
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
    raw::recurrence(args,mfq_current_cuda_stream());
    return {out, s_out};
}

std::vector<::tyloi::DenseTensor> gdn_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, MfqOptional<::tyloi::DenseTensor> state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, false, false);
}

std::vector<::tyloi::DenseTensor> gdn_inplace_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, false);
}

std::vector<::tyloi::DenseTensor> gdn_transposed_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, MfqOptional<::tyloi::DenseTensor> state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, false, true);
}

std::vector<::tyloi::DenseTensor> gdn_inplace_transposed_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, true);
}

std::vector<::tyloi::DenseTensor> gdn_inplace_tiled_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, false, true);
}

std::vector<::tyloi::DenseTensor> gdn_inplace_transposed_tiled_cuda(
    ::tyloi::DenseTensor q, ::tyloi::DenseTensor k, ::tyloi::DenseTensor v,
    ::tyloi::DenseTensor g, ::tyloi::DenseTensor beta, ::tyloi::DenseTensor state)
{
    return gdn_cuda_impl(q, k, v, g, beta, state, true, true, true);
}

static std::vector<::tyloi::DenseTensor> gdn_decode_core_impl(
    ::tyloi::DenseTensor qkv,::tyloi::DenseTensor output_gate,
    ::tyloi::DenseTensor alpha,::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor convolution_state,::tyloi::DenseTensor recurrent_state,
    ::tyloi::DenseTensor convolution_weight,::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log,::tyloi::DenseTensor norm_weight,
    int64_t key_heads,int64_t value_heads,int64_t width,
    double convolution_eps,double norm_eps,bool silu_gate,bool output_half,
    bool transposed_state,::tyloi::DenseTensor workspace,bool inplace_state,
    ::tyloi::DenseTensor state_addresses,
    ::tyloi::DenseTensor checkpoint_convolution={},::tyloi::DenseTensor checkpoint_recurrent={}) {
    constexpr int threads=256;
    const int tiles=[] {
        const char* value=tyloi::cuda::env_lookup("TYLOINFER_GDN_DECODE_TILES");
        if(!value || !*value || std::strcmp(value,"4")==0)return 4;
        if(std::strcmp(value,"2")==0)return 2;
        if(std::strcmp(value,"1")==0)return 1;
        TYLOINFER_RUNTIME_CHECK(false,"TYLOINFER_GDN_DECODE_TILES must be 1, 2 or 4");
        return 4;
    }();
    TYLOINFER_RUNTIME_CHECK(width==32 || width==64 || width==128,"GDN decode core width unsupported");
    TYLOINFER_RUNTIME_CHECK(threads==256 || threads==512 || threads==1024,"GDN decode core block width unsupported");
    TYLOINFER_RUNTIME_CHECK(qkv.dim()==3 && qkv.size(0)>0 && key_heads>0 && value_heads>0 &&
        value_heads%key_heads==0 && key_heads<=std::numeric_limits<int>::max()/width &&
        value_heads<=std::numeric_limits<int>::max()/width,
        "GDN decode core heads/batch disagree");
    const int64_t batch=qkv.size(0),channels=(2*key_heads+value_heads)*width,taps=convolution_state.size(1)+1;
    TYLOINFER_RUNTIME_CHECK(channels<=std::numeric_limits<int>::max() && taps>1 &&
        taps<=std::numeric_limits<int>::max() && batch<=std::numeric_limits<int>::max()/value_heads/tiles &&
        std::isfinite(convolution_eps) && convolution_eps>0 && std::isfinite(norm_eps) && norm_eps>0,
        "GDN decode core geometry/epsilon exceeds supported bounds");
    TYLOINFER_RUNTIME_CHECK(qkv.is_cuda() && qkv.stride(2)==1 && qkv.stride(0)>=channels && qkv.scalar_type()==::tyloi::kFloat16 &&
        qkv.sizes().vec()==std::vector<int64_t>({batch,1,channels}),"GDN decode core qkv disagrees");
    const auto device=qkv.device();
    const auto real=[&](const ::tyloi::DenseTensor& t){return t.is_cuda() && t.device()==device && t.is_contiguous() && t.scalar_type()==::tyloi::kFloat32;};
    const bool independent=state_addresses.defined();
    TYLOINFER_RUNTIME_CHECK(!independent || (inplace_state && state_addresses.is_cuda() &&
        state_addresses.device()==device && state_addresses.is_contiguous() &&
        state_addresses.scalar_type()==::tyloi::kInt64 &&
        state_addresses.sizes().vec()==std::vector<int64_t>({batch,3})),
        "GDN independent state addresses require inplace CUDA int64 [B,3]");
    TYLOINFER_RUNTIME_CHECK(real(convolution_state) && real(recurrent_state) && real(convolution_weight) &&
        real(dt_bias) && real(a_log) && real(norm_weight),"GDN decode core parameters disagree");
    TYLOINFER_RUNTIME_CHECK(!checkpoint_convolution.defined() ||
        (!independent && real(checkpoint_convolution) && real(checkpoint_recurrent) &&
         checkpoint_convolution.sizes()==convolution_state.sizes() &&
         checkpoint_recurrent.sizes()==recurrent_state.sizes() &&
         checkpoint_convolution.data_ptr()!=convolution_state.data_ptr() &&
         checkpoint_recurrent.data_ptr()!=recurrent_state.data_ptr()),
        "GDN core checkpoint storage disagrees");
    const int64_t state_batch=independent?1:batch;
    TYLOINFER_RUNTIME_CHECK(convolution_state.sizes().vec()==std::vector<int64_t>({state_batch,taps-1,channels}) &&
        recurrent_state.sizes().vec()==std::vector<int64_t>({state_batch,value_heads,width,width}),"GDN decode core state shape disagrees");
    const bool channels_first=convolution_weight.sizes().vec()==std::vector<int64_t>({channels,1,taps});
    TYLOINFER_RUNTIME_CHECK(channels_first || convolution_weight.sizes().vec()==std::vector<int64_t>({taps,channels}),"GDN decode core conv weight disagrees");
    TYLOINFER_RUNTIME_CHECK(output_gate.is_cuda() && alpha.is_cuda() && beta.is_cuda() &&
        output_gate.device()==device && alpha.device()==device && beta.device()==device &&
        output_gate.dim()==3 && output_gate.stride(2)==1 && output_gate.stride(0)>=value_heads*width &&
        output_gate.sizes().vec()==std::vector<int64_t>({batch,1,value_heads*width}) &&
        alpha.sizes().vec()==std::vector<int64_t>({batch,1,value_heads}) && beta.sizes()==alpha.sizes() &&
        output_gate.scalar_type()==alpha.scalar_type() && alpha.scalar_type()==beta.scalar_type() &&
        (alpha.scalar_type()==::tyloi::kFloat16 || alpha.scalar_type()==::tyloi::kFloat32),"GDN decode core gate inputs disagree");
    TYLOINFER_RUNTIME_CHECK(dt_bias.numel()==value_heads && a_log.numel()==value_heads && norm_weight.numel()==width,
        "GDN decode core vector width disagrees");
    MfqCudaGuard guard(device);
    auto output=::tyloi::cuda::empty({batch,1,value_heads*width},qkv.options().dtype(output_half?::tyloi::kFloat16: ::tyloi::kFloat32));
    const auto* conv_option=tyloi::cuda::env_lookup("TYLOINFER_GDN_INPLACE_CONV");
    const bool conv_default=batch==1 && key_heads==16 && value_heads==48 && width==128 && taps==4 && transposed_state;
    const bool inplace_conv=(conv_option?conv_option[0]!='0':conv_default) && !independent && inplace_state &&
        tiles>1 && value_heads/key_heads<=24;
    auto conv=inplace_conv?convolution_state: ::tyloi::cuda::empty({batch,taps-1,channels},convolution_state.options());
    auto state=inplace_state?recurrent_state: ::tyloi::cuda::empty_like(recurrent_state);
    if(tiles>1 && !workspace.defined() && !independent)workspace=::tyloi::cuda::zeros({batch,value_heads,width+1},convolution_state.options());
    TYLOINFER_RUNTIME_CHECK(!workspace.defined() || (real(workspace) &&
        workspace.sizes().vec()==std::vector<int64_t>({state_batch,value_heads,width+1})),
        "GDN decode core workspace shape/device/dtype disagrees");
    const auto* broadcast=tyloi::cuda::env_lookup("TYLOINFER_GDN_BROADCAST_NORMALIZATION");
    const int mode=broadcast?std::atoi(broadcast):
        (batch<=2 && key_heads==16 && value_heads==48 && width==128 && taps==4 &&
         transposed_state && tiles==4 && threads==256?2:0);
    const auto* fixed=tyloi::cuda::env_lookup("TYLOINFER_GDN_FIXED_GEOMETRY");
    const auto* reductions=tyloi::cuda::env_lookup("TYLOINFER_GDN_WARP_REDUCTIONS");
    raw::Decode args{};
    args.qkv=qkv.data_ptr();args.gate=output_gate.data_ptr();args.alpha=alpha.data_ptr();args.beta=beta.data_ptr();
    args.conv_state=convolution_state.data_ptr<float>();args.state=recurrent_state.data_ptr<float>();
    args.weight=convolution_weight.data_ptr<float>();args.dt_bias=dt_bias.data_ptr<float>();
    args.a_log=a_log.data_ptr<float>();args.norm=norm_weight.data_ptr<float>();args.output=output.data_ptr();
    args.conv_out=conv.data_ptr<float>();args.state_out=state.data_ptr<float>();
    args.workspace=workspace.defined()?workspace.data_ptr<float>():nullptr;
    args.checkpoint_conv=checkpoint_convolution.defined()?checkpoint_convolution.data_ptr<float>():nullptr;
    args.checkpoint_state=checkpoint_recurrent.defined()?checkpoint_recurrent.data_ptr<float>():nullptr;
    args.state_addresses=independent?state_addresses.data_ptr<int64_t>():nullptr;
    args.batch=int(batch);args.key_heads=int(key_heads);args.value_heads=int(value_heads);args.width=int(width);
    args.taps=int(taps);args.tiles=tiles;args.layout=channels_first?0:1;args.mode=mode;
    args.conv_eps=float(convolution_eps);args.norm_eps=norm_eps;args.silu=silu_gate;args.output_half=output_half;
    args.gate_half=alpha.scalar_type()==::tyloi::kFloat16;args.transposed=transposed_state;args.inplace_conv=inplace_conv;
    args.fixed_geometry=!fixed || fixed[0]!='0';args.warp_reductions=!reductions || reductions[0]!='0';
    args.probe=tyloi::cuda::env_lookup("TYLOINFER_GDN_PROBE_RESOURCES")!=nullptr;
    args.alpha_stride0=alpha.stride(0);args.alpha_stride2=alpha.stride(2);
    args.beta_stride0=beta.stride(0);args.beta_stride2=beta.stride(2);
    args.qkv_stride0=qkv.stride(0);args.gate_stride0=output_gate.stride(0);
    raw::decode(args,mfq_current_cuda_stream());
    return {output,conv,state};
}

std::vector<::tyloi::DenseTensor> gdn_decode_core_cuda(
    ::tyloi::DenseTensor qkv, ::tyloi::DenseTensor output_gate,
    ::tyloi::DenseTensor alpha, ::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor convolution_state, ::tyloi::DenseTensor recurrent_state,
    ::tyloi::DenseTensor convolution_weight, ::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log, ::tyloi::DenseTensor norm_weight,
    int64_t key_heads, int64_t value_heads, int64_t width,
    double convolution_eps, double norm_eps, bool silu_gate, bool output_half,
    bool transposed_state,
    ::tyloi::DenseTensor workspace,bool inplace_state,
    ::tyloi::DenseTensor state_addresses) {
    return gdn_decode_core_impl(qkv,output_gate,alpha,beta,convolution_state,recurrent_state,convolution_weight,dt_bias,a_log,norm_weight,
        key_heads,value_heads,width,convolution_eps,norm_eps,silu_gate,output_half,transposed_state,workspace,inplace_state,state_addresses);
}

std::vector<::tyloi::DenseTensor> gdn_decode_core_checkpoint_cuda(
    ::tyloi::DenseTensor qkv, ::tyloi::DenseTensor output_gate,
    ::tyloi::DenseTensor alpha, ::tyloi::DenseTensor beta,
    ::tyloi::DenseTensor convolution_state, ::tyloi::DenseTensor recurrent_state,
    ::tyloi::DenseTensor convolution_weight, ::tyloi::DenseTensor dt_bias,
    ::tyloi::DenseTensor a_log, ::tyloi::DenseTensor norm_weight,
    int64_t key_heads, int64_t value_heads, int64_t width,
    double convolution_eps, double norm_eps, bool silu_gate, bool output_half,
    bool transposed_state,
    ::tyloi::DenseTensor workspace,bool inplace_state,
    ::tyloi::DenseTensor checkpoint_convolution,
    ::tyloi::DenseTensor checkpoint_recurrent) {
    return gdn_decode_core_impl(qkv,output_gate,alpha,beta,convolution_state,recurrent_state,convolution_weight,dt_bias,a_log,norm_weight,
        key_heads,value_heads,width,convolution_eps,norm_eps,silu_gate,output_half,transposed_state,workspace,inplace_state,{},checkpoint_convolution,checkpoint_recurrent);
}
