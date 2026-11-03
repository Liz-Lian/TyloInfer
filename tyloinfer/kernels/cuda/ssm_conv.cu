// Fused causal depthwise SSM conv + SiLU.
// conv_input [B, K - 1 + T, C], weight [C,1,K], [C,K], or [K,C], output [B,T,C].

#include <cuda_runtime.h>
#include "mfq_tensor_backend.h"
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "reduce.cuh"
#include "cpp_runtime/backends/cuda/kernels/gguf_prefill_scale.cuh"

constexpr int SSM_CONV_BD = 256;

// Unified F32 QKV projections can use the same fused decode preparation as
// split half projections. Keep the standalone convolution and L2 reduction
// orders, and update each state channel in exactly one CTA.
__global__ void ssm_conv_qkv_f32_decode_kernel(float* state,const float* input,
        const float* weight,const float* bias,float* q,float* k,float* v,
        int nk,int nv,int dk,int dv,int taps,bool channels_first,float eps) {
    const int tid=threadIdx.x,batch=blockIdx.y,qk_width=2*nk*dk,width=qk_width+nv*dv;
    const bool query_key=blockIdx.x<unsigned(2*nk);
    const int channel=query_key?int(blockIdx.x)*dk+tid:qk_width+int(blockIdx.x-2*nk)*256+tid;
    const bool valid=query_key?tid<dk:channel<width;
    float value=0.f;
    if(valid) {
        const auto base=std::int64_t(batch)*(taps-1)*width+channel;
        float sum=0.f;
        for(int j=0;j<taps-1;++j) {
            const float w=weight[channels_first?std::int64_t(channel)*taps+j:std::int64_t(j)*width+channel];
            sum+=state[base+std::int64_t(j)*width]*w;
        }
        const float current=input[std::int64_t(batch)*width+channel];
        sum+=current*weight[channels_first?std::int64_t(channel)*taps+taps-1:std::int64_t(taps-1)*width+channel];
        if(bias)sum+=bias[channel];
        value=sum/(1.f+expf(-sum));
        for(int j=0;j<taps-2;++j)state[base+std::int64_t(j)*width]=state[base+std::int64_t(j+1)*width];
        state[base+std::int64_t(taps-2)*width]=current;
    }
    if(query_key) {
        const float squared=block_sum<8>(value*value);
        const float inverse=1.f/fmaxf(sqrtf(squared),eps);
        if(valid) {
            const int which=int(blockIdx.x)/nk,head=int(blockIdx.x)%nk;
            (which?k:q)[(std::int64_t(batch)*nk+head)*dk+tid]=value*inverse;
        }
    }else if(valid)v[std::int64_t(batch)*nv*dv+channel-qk_width]=value;
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_decode_f32_cuda(
        mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor bias,
        int64_t nk,int64_t nv,int64_t dk,int64_t dv,double eps) {
    namespace tb=mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==tb::kFloat32 &&
        state.dim()==3 && state.size(0)>0 && state.size(1)>0 && nk>0 && nv>0 && nv%nk==0 && dk>0 && dk<=256 && dv>0,
        "F32 convolution decode state/geometry disagrees");
    const auto batch=state.size(0),taps=state.size(1)+1,width=2*nk*dk+nv*dv;
    TYLOINFER_RUNTIME_CHECK(batch<=65535 && width<=INT_MAX && taps<=INT_MAX &&
        2*nk+(nv*dv+255)/256<=INT_MAX,"F32 convolution decode exceeds CUDA grid bounds");
    const auto same=[&](const tb::Tensor& x){return x.is_cuda() && x.device()==state.device() &&
        x.is_contiguous() && x.scalar_type()==tb::kFloat32;};
    TYLOINFER_RUNTIME_CHECK(same(input) && input.sizes().vec()==std::vector<int64_t>({batch,1,width}) &&
        state.size(2)==width,"F32 convolution decode projection disagrees");
    TYLOINFER_RUNTIME_CHECK(same(weight) && same(bias) &&
        (bias.numel()==0 || (bias.dim()==1 && bias.numel()==width)),"F32 convolution decode parameters disagree");
    const bool channels_first=weight.sizes().vec()==std::vector<int64_t>({width,1,taps}) ||
        weight.sizes().vec()==std::vector<int64_t>({width,taps});
    TYLOINFER_RUNTIME_CHECK(channels_first || weight.sizes().vec()==std::vector<int64_t>({taps,width}),
        "F32 convolution decode weight geometry disagrees");
    MfqCudaGuard guard(state.device());
    auto q=tb::empty({batch,nk,1,dk},state.options()),k=tb::empty_like(q),v=tb::empty({batch,nv,1,dv},state.options());
    ssm_conv_qkv_f32_decode_kernel<<<dim3(unsigned(2*nk+(nv*dv+255)/256),unsigned(batch)),256,0,mfq_current_cuda_stream()>>>(
        state.data_ptr<float>(),input.data_ptr<float>(),weight.data_ptr<float>(),bias.numel()?bias.data_ptr<float>():nullptr,
        q.data_ptr<float>(),k.data_ptr<float>(),v.data_ptr<float>(),int(nk),int(nv),int(dk),int(dv),int(taps),channels_first,float(eps));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
    return {q,k,v};
}

__device__ inline float silu_f32(float x)
{
    return x / (1.0f + expf(-x));
}

__global__ void ssm_conv_silu_tiled_kernel(const float* __restrict__ x,const float* __restrict__ w,
    const float* __restrict__ bias,float* __restrict__ out,int B,int T,int C,int K,int weight_layout,int has_bias) {
    constexpr int Tile=4;
    const auto channels=(std::uint64_t(C)+SSM_CONV_BD-1)/SSM_CONV_BD;
    const auto times=(std::uint64_t(T)+Tile-1)/Tile;
    const auto total=std::uint64_t(B)*channels*times;
    for(auto tile=std::uint64_t(blockIdx.x);tile<total;tile+=gridDim.x) {
        const auto c=(tile%channels)*SSM_CONV_BD+threadIdx.x;
        const auto bt=tile/channels,b=bt/times,t=(bt%times)*Tile;
        if(c>=std::uint64_t(C))continue;
        float sums[Tile]={};
        for(int j=0;j<K;++j) {
            const float wv=weight_layout==0?w[c*K+j]:w[std::uint64_t(j)*C+c];
            #pragma unroll
            for(int r=0;r<Tile;++r)if(t+r<std::uint64_t(T))
                sums[r]+=x[(b*std::uint64_t(T+K-1)+t+r+j)*C+c]*wv;
        }
        #pragma unroll
        for(int r=0;r<Tile;++r)if(t+r<std::uint64_t(T)) {
            if(has_bias)sums[r]+=bias[c];
            out[(b*T+t+r)*C+c]=silu_f32(sums[r]);
        }
    }
}

mfq_tensor_backend::Tensor ssm_conv_silu_cuda(mfq_tensor_backend::Tensor conv_input, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias, int64_t n_tokens)
{
    TYLOINFER_RUNTIME_CHECK(conv_input.is_cuda() && conv_input.is_contiguous() && conv_input.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu: conv_input must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(conv_input.dim() == 3, "ssm_conv_silu: conv_input must be [B,K-1+T,C]");
    int B = (int)conv_input.size(0);
    int C = (int)conv_input.size(2);
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu: bias must be [C]");
    int K = 0;
    int layout = 0; // 0: channel-major [C,1,K]/[C,K], 1: [K,C]
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1, "ssm_conv_silu: [C,1,K] weight mismatch");
        K = (int)weight.size(2);
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2, "ssm_conv_silu: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C) {
            K = (int)weight.size(1);
            layout = 0;
        } else {
            TYLOINFER_RUNTIME_CHECK(weight.size(1) == C, "ssm_conv_silu: 2D weight must be [C,K] or [K,C]");
            K = (int)weight.size(0);
            layout = 1;
        }
    }
    int T = (int)n_tokens;
    TYLOINFER_RUNTIME_CHECK(T > 0 && conv_input.size(1) == T + K - 1, "ssm_conv_silu: conv_input length must be K-1+T");
    auto out = mfq_tensor_backend::empty({B, T, C}, conv_input.options());
    const auto tiles=std::uint64_t(B)*((std::uint64_t(T)+3)/4)*((std::uint64_t(C)+SSM_CONV_BD-1)/SSM_CONV_BD);
    const auto blocks=unsigned(std::min<std::uint64_t>(65535,tiles));
    ssm_conv_silu_tiled_kernel<<<blocks, SSM_CONV_BD, 0, mfq_current_cuda_stream()>>>(
        conv_input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), out.data_ptr<float>(),
        B, T, C, K, layout, has_bias);
    return out;
}

__global__ void ssm_conv_silu_decode_kernel(
    float* __restrict__ state,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ out,
    int B,
    int C,
    int K,
    int weight_layout,
    int has_bias)
{
    int b = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (b >= B || c >= C) {
        return;
    }
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = x[(size_t)b * (size_t)C + (size_t)c];
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    out[(size_t)b * (size_t)C + (size_t)c] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

mfq_tensor_backend::Tensor ssm_conv_silu_decode_cuda(mfq_tensor_backend::Tensor state, mfq_tensor_backend::Tensor x, mfq_tensor_backend::Tensor weight, mfq_tensor_backend::Tensor bias)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: state must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: x must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "ssm_conv_silu_decode: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3, "ssm_conv_silu_decode: state must be [B,K-1,C]");
    TYLOINFER_RUNTIME_CHECK(x.dim() == 3 && x.size(1) == 1, "ssm_conv_silu_decode: x must be [B,1,C]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int C = (int)state.size(2);
    TYLOINFER_RUNTIME_CHECK(x.size(0) == B && x.size(2) == C, "ssm_conv_silu_decode: x shape mismatch");
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "ssm_conv_silu_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "ssm_conv_silu_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2,
                    "ssm_conv_silu_decode: weight must be [C,1,K], [C,K], or [K,C]");
        if (weight.size(0) == C && weight.size(1) == K) {
            layout = 0;
        } else {
            TYLOINFER_RUNTIME_CHECK(weight.size(0) == K && weight.size(1) == C,
                        "ssm_conv_silu_decode: 2D weight must be [C,K] or [K,C]");
            layout = 1;
        }
    }
    auto out = mfq_tensor_backend::empty({B, 1, C}, x.options());
    dim3 blocks(B, (C + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_silu_decode_kernel<<<blocks, SSM_CONV_BD, 0, mfq_current_cuda_stream()>>>(
        state.data_ptr<float>(), x.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        out.data_ptr<float>(), B, C, K, layout, has_bias);
    return out;
}

__device__ inline float ssm_decode_cur(
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    int b,
    int c,
    int qkC,
    int vsz)
{
    if (c < qkC) {
        return __half2float(qk[(size_t)b * (size_t)qkC + (size_t)c]);
    }
    return __half2float(v[(size_t)b * (size_t)vsz + (size_t)(c - qkC)]);
}

template <int BD>
__device__ __forceinline__ void ssm_conv_qk_norm_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias,
    float eps, int row)
{
    int b = row / (2 * nk);
    int rem = row - b * 2 * nk;
    int which = rem / nk;
    int h = rem - which * nk;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int vsz = nv * dv;
    int C = qkC + nv * dv;
    int base_c = (which * nk + h) * dk;

    __shared__ float vals[BD];
    __shared__ float sums[BD];
    float val = 0.0f;
    if (tid < dk) {
        int c = base_c + tid;
        size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
        float sum = 0.0f;
        for (int j = 0; j < K - 1; ++j) {
            float xv = state[state_base + (size_t)j * (size_t)C];
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        float x_cur = ssm_decode_cur(qk, v, b, c, qkC, vsz);
        float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
        sum += x_cur * wv;
        if (has_bias) {
            sum += bias[c];
        }
        val = silu_f32(sum);
        vals[tid] = val;
        for (int j = 0; j < K - 2; ++j) {
            state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
        }
        state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
    }
    float ssq = (tid < dk) ? val * val : 0.0f;
    sums[tid] = ssq;
    __syncthreads();
    for (int stride = BD / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
        }
        __syncthreads();
    }
    float inv = 1.0f / fmaxf(sqrtf(sums[0]), eps);
    if (tid < dk) {
        float outv = vals[tid] * inv;
        float* dst = which == 0 ? q : k;
        dst[((size_t)b * nk + h) * dk + tid] = outv;
    }
}

__device__ __forceinline__ void ssm_conv_v_decode(
    float* __restrict__ state,
    const __half* __restrict__ qk,
    const __half* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B,
    int nk,
    int nv,
    int dk,
    int dv,
    int K,
    int weight_layout,
    int has_bias, int b, int idx)
{
    int vsz = nv * dv;
    if (b >= B || idx >= vsz) {
        return;
    }
    int qkC = 2 * nk * dk;
    int C = qkC + vsz;
    int c = qkC + idx;
    size_t state_base = ((size_t)b * (size_t)(K - 1)) * (size_t)C + (size_t)c;
    float sum = 0.0f;
    for (int j = 0; j < K - 1; ++j) {
        float xv = state[state_base + (size_t)j * (size_t)C];
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    float x_cur = __half2float(v_in[(size_t)b * (size_t)vsz + (size_t)idx]);
    float wv = weight_layout == 0 ? w[(size_t)c * K + (K - 1)] : w[(size_t)(K - 1) * C + c];
    sum += x_cur * wv;
    if (has_bias) {
        sum += bias[c];
    }
    v_out[(size_t)b * (size_t)vsz + (size_t)idx] = silu_f32(sum);
    for (int j = 0; j < K - 2; ++j) {
        state[state_base + (size_t)j * (size_t)C] = state[state_base + (size_t)(j + 1) * (size_t)C];
    }
    state[state_base + (size_t)(K - 2) * (size_t)C] = x_cur;
}

template<int BD>
__global__ void ssm_conv_qk_norm_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* q,float* k,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias,float eps) {
    ssm_conv_qk_norm_decode<BD>(state,qk,v,w,bias,q,k,B,nk,nv,dk,dv,K,layout,has_bias,eps,blockIdx.x);
}
__global__ void ssm_conv_v_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* w,const float* bias,float* output,int B,int nk,int nv,int dk,int dv,
    int K,int layout,int has_bias) {
    ssm_conv_v_decode(state,qk,v,w,bias,output,B,nk,nv,dk,dv,K,layout,has_bias,
        blockIdx.x,blockIdx.y*blockDim.x+threadIdx.x);
}

template<class Gate>
__global__ void ssm_conv_qkv_gate_decode_kernel(float* state,const __half* qk,const __half* v,
    const float* weight,const Gate* alpha,const Gate* beta,const float* dt_bias,const float* a_log,
    float* q,float* k,float* output,float* decay,float* beta_out,
    int B,int nk,int nv,int dk,int dv,int K,int layout,float eps,
    int64_t as0,int64_t as2,int64_t bs0,int64_t bs2) {
    const int v_blocks=(nv*dv+255)/256;
    if(blockIdx.x<unsigned(2*nk)) {
        ssm_conv_qk_norm_decode<256>(state,qk,v,weight,nullptr,q,k,B,nk,nv,dk,dv,K,layout,0,
            eps,blockIdx.y*2*nk+blockIdx.x);
    } else if(blockIdx.x<unsigned(2*nk+v_blocks)) {
        ssm_conv_v_decode(state,qk,v,weight,nullptr,output,B,nk,nv,dk,dv,K,layout,0,
            blockIdx.y,(blockIdx.x-2*nk)*256+threadIdx.x);
    } else {
        const int head=(blockIdx.x-2*nk-v_blocks)*256+threadIdx.x,b=blockIdx.y;
        if(head<nv) {
            const float a=static_cast<float>(alpha[int64_t(b)*as0+int64_t(head)*as2])+dt_bias[head];
            const float sp=fmaxf(a,0.0f)+log1pf(expf(-fabsf(a)));
            decay[int64_t(b)*nv+head]=sp*-expf(a_log[head]);
            const float bv=static_cast<float>(beta[int64_t(b)*bs0+int64_t(head)*bs2]);
            beta_out[int64_t(b)*nv+head]=1.0f/(1.0f+expf(-bv));
        }
    }
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_decode_cuda(
    mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor qk,mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor dt_bias,mfq_tensor_backend::Tensor a_log,
    int64_t nk,int64_t nv,int64_t dk,int64_t dv,double eps) {
    namespace tb=mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==tb::kFloat32 &&
        state.dim()==3 && state.size(1)>0 && nk>0 && nv>0 && nv%nk==0 && dk>0 && dk<=256 && dv>0,
        "GDN combined decode state/geometry disagrees");
    const int B=int(state.size(0)),K=int(state.size(1))+1,C=int(2*nk*dk+nv*dv);
    const auto same_device=[&](const tb::Tensor& x){return x.is_cuda() && x.device()==state.device();};
    TYLOINFER_RUNTIME_CHECK(same_device(qk) && same_device(v) && qk.is_contiguous() && v.is_contiguous() &&
        qk.scalar_type()==tb::kFloat16 && v.scalar_type()==tb::kFloat16 &&
        qk.sizes().vec()==std::vector<int64_t>({B,1,2*nk*dk}) &&
        v.sizes().vec()==std::vector<int64_t>({B,1,nv*dv}) && state.size(2)==C,
        "GDN combined decode projection shape/dtype disagrees");
    TYLOINFER_RUNTIME_CHECK(same_device(alpha) && same_device(beta) && alpha.scalar_type()==beta.scalar_type() &&
        (alpha.scalar_type()==tb::kFloat16 || alpha.scalar_type()==tb::kFloat32) &&
        alpha.sizes().vec()==std::vector<int64_t>({B,1,nv}) && beta.sizes()==alpha.sizes(),
        "GDN combined decode gates disagree");
    TYLOINFER_RUNTIME_CHECK(same_device(weight) && same_device(dt_bias) && same_device(a_log) &&
        weight.is_contiguous() && dt_bias.is_contiguous() && a_log.is_contiguous() &&
        weight.scalar_type()==tb::kFloat32 && dt_bias.scalar_type()==tb::kFloat32 && a_log.scalar_type()==tb::kFloat32 &&
        dt_bias.numel()==nv && a_log.numel()==nv,"GDN combined decode parameters disagree");
    const bool channels_first=weight.sizes().vec()==std::vector<int64_t>({C,1,K});
    TYLOINFER_RUNTIME_CHECK(channels_first || weight.sizes().vec()==std::vector<int64_t>({K,C}),
        "GDN combined decode convolution shape disagrees");
    MfqCudaGuard guard(state.device());auto options=state.options();
    auto q=tb::empty({B,nk,1,dk},options),k=tb::empty_like(q),output=tb::empty({B,nv,1,dv},options);
    auto decay=tb::empty({B,nv,1},options),beta_out=tb::empty_like(decay);
    const dim3 grid(unsigned(2*nk+(nv*dv+255)/256+(nv+255)/256),unsigned(B));
    const auto launch=[&](auto tag) {
        using Gate=decltype(tag);
        ssm_conv_qkv_gate_decode_kernel<Gate><<<grid,256,0,mfq_current_cuda_stream()>>>(
            state.data_ptr<float>(),reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
            reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()),weight.data_ptr<float>(),
            alpha.data_ptr<Gate>(),beta.data_ptr<Gate>(),dt_bias.data_ptr<float>(),a_log.data_ptr<float>(),
            q.data_ptr<float>(),k.data_ptr<float>(),output.data_ptr<float>(),decay.data_ptr<float>(),beta_out.data_ptr<float>(),
            B,int(nk),int(nv),int(dk),int(dv),K,channels_first?0:1,float(eps),
            alpha.stride(0),alpha.stride(2),beta.stride(0),beta.stride(2));
    };
    if(alpha.scalar_type()==tb::kFloat16)launch(mfq_half{});else launch(float{});
    TYLOINFER_CUDA_CHECK(cudaGetLastError());return {q,k,output,decay,beta_out};
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_decode_cuda(
    mfq_tensor_backend::Tensor state,
    mfq_tensor_backend::Tensor qk,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor weight,
    mfq_tensor_backend::Tensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_decode: state must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && qk.scalar_type() == mfq_tensor_backend::kHalf,
                "linear_conv_qkv_decode: qk must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == mfq_tensor_backend::kHalf,
                "linear_conv_qkv_decode: v must be cuda contiguous f16");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_decode: weight must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_decode: bias must be cuda contiguous f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3, "linear_conv_qkv_decode: state must be [B,K-1,C]");
    TYLOINFER_RUNTIME_CHECK(qk.dim() == 3 && qk.size(1) == 1, "linear_conv_qkv_decode: qk must be [B,1,2*nk*dk]");
    TYLOINFER_RUNTIME_CHECK(v.dim() == 3 && v.size(1) == 1, "linear_conv_qkv_decode: v must be [B,1,nv*dv]");
    int B = (int)state.size(0);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TYLOINFER_RUNTIME_CHECK(qk.size(0) == B && qk.size(2) == qkC, "linear_conv_qkv_decode: qk shape mismatch");
    TYLOINFER_RUNTIME_CHECK(v.size(0) == B && v.size(2) == vsz, "linear_conv_qkv_decode: v shape mismatch");
    TYLOINFER_RUNTIME_CHECK(state.size(2) == C, "linear_conv_qkv_decode: state width mismatch");
    TYLOINFER_RUNTIME_CHECK(nv % nk == 0, "linear_conv_qkv_decode: nv must be divisible by nk");
    TYLOINFER_RUNTIME_CHECK(dk <= 256, "linear_conv_qkv_decode: dk > 256 is unsupported");
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_decode: bias must be [C]");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_decode: [C,1,K] weight mismatch");
        layout = 0;
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_decode: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    auto opts = state.options();
    auto q = mfq_tensor_backend::empty({B, (int)nk, 1, (int)dk}, opts);
    auto k = mfq_tensor_backend::empty({B, (int)nk, 1, (int)dk}, opts);
    auto vo = mfq_tensor_backend::empty({B, (int)nv, 1, (int)dv}, opts);
    cudaStream_t stream = mfq_current_cuda_stream();
    ssm_conv_qk_norm_decode_kernel<256><<<B * 2 * (int)nk, 256, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
        reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()), weight.data_ptr<float>(),
        bias.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), B, (int)nk, (int)nv,
        (int)dk, (int)dv, K, layout, has_bias, (float)eps);
    dim3 v_blocks(B, (vsz + SSM_CONV_BD - 1) / SSM_CONV_BD);
    ssm_conv_v_decode_kernel<<<v_blocks, SSM_CONV_BD, 0, stream>>>(
        state.data_ptr<float>(), reinterpret_cast<const __half*>(qk.data_ptr<mfq_half>()),
        reinterpret_cast<const __half*>(v.data_ptr<mfq_half>()), weight.data_ptr<float>(),
        bias.data_ptr<float>(), vo.data_ptr<float>(), B, (int)nk, (int)nv, (int)dk, (int)dv,
        K, layout, has_bias);
    return {q, k, vo};
}

template <int BD, bool Packed = false, class Input = __half>
__global__ void linear_conv_qk_norm_prefill_kernel(
    const float* __restrict__ state,
    const Input* __restrict__ qk,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ q,
    float* __restrict__ k,
    int B, int T, int nk, int dk, int C, int K,
    int weight_layout, int has_bias, float eps,int64_t ps0=0,int64_t ps1=0,
    const int32_t* row_exponents=nullptr,const int32_t* column_exponents=nullptr)
{
    int row = blockIdx.x;
    int h = row % nk;
    int r = row / nk;
    int which = r & 1;
    r >>= 1;
    int t = r % T;
    int b = r / T;
    int tid = threadIdx.x;
    int qkC = 2 * nk * dk;
    int c = (which * nk + h) * dk + tid;
    float val = 0.0f;
    if (tid < dk) {
        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j >= K) break;
            int token = t + j - (K - 1);
            float xv;
            if (token < 0) {
                xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
            } else {
                xv = static_cast<float>(qk[Packed && ps1?int64_t(b)*ps0+int64_t(token)*ps1+c:
                    ((size_t)b * T + token) * (Packed ? C : qkC) + c]);
            }
            if(row_exponents && token>=0)
                xv=prefill_ldexp(xv,row_exponents[int64_t(b)*T+token]+column_exponents[c]);
            float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
            sum += xv * wv;
        }
        if (has_bias) sum += bias[c];
        val = silu_f32(sum);
    }
    float ssq = block_sum<BD / 32>(tid < dk ? val * val : 0.0f);
    float inv = 1.0f / fmaxf(sqrtf(ssq), eps);
    if (tid < dk) {
        float* dst = which == 0 ? q : k;
        dst[(((size_t)b * nk + h) * T + t) * dk + tid] = val * inv;
    }
}

template <bool Packed = false, class Gate = float, class Input = __half>
__global__ void linear_conv_v_prefill_kernel(
    const float* __restrict__ state,
    const Input* __restrict__ v_in,
    const float* __restrict__ w,
    const float* __restrict__ bias,
    float* __restrict__ v_out,
    int B, int T, int nk, int nv, int dk, int dv, int C, int K,
    int weight_layout, int has_bias,
    const Gate* alpha = nullptr, const Gate* beta = nullptr,
    const float* dt_bias = nullptr, const float* a_log = nullptr,
    float* gate_out = nullptr, float* beta_out = nullptr,
    int64_t ps0=0,int64_t ps1=0,int64_t as0=0,int64_t as1=0,int64_t bs0=0,int64_t bs1=0,
    bool stable_softplus=true,const int32_t* row_exponents=nullptr,const int32_t* column_exponents=nullptr)
{
    int bt = blockIdx.x;
    int b = bt / T;
    int t = bt - b * T;
    int vi = blockIdx.y * blockDim.x + threadIdx.x;
    int vsz = nv * dv;
    if (b >= B || vi >= vsz) return;
    int qkC = 2 * nk * dk;
    int c = qkC + vi;
    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j >= K) break;
        int token = t + j - (K - 1);
        float xv;
        if (token < 0) {
            xv = state[((size_t)b * (K - 1) + (token + K - 1)) * C + c];
        } else {
            xv = static_cast<float>(v_in[Packed && ps1?int64_t(b)*ps0+int64_t(token)*ps1+vi:
                ((size_t)b * T + token) * (Packed ? C : vsz) + vi]);
        }
        if(row_exponents && token>=0)
            xv=prefill_ldexp(xv,row_exponents[int64_t(b)*T+token]+column_exponents[c]);
        float wv = weight_layout == 0 ? w[(size_t)c * K + j] : w[(size_t)j * C + c];
        sum += xv * wv;
    }
    if (has_bias) sum += bias[c];
    int h = vi / dv;
    int d = vi - h * dv;
    v_out[(((size_t)b * nv + h) * T + t) * dv + d] = silu_f32(sum);
    if constexpr(Packed) {
        if(d == 0) {
            const float a = static_cast<float>(alpha[as1?int64_t(b)*as0+int64_t(t)*as1+h:(size_t)bt * nv + h]) + dt_bias[h];
            const float sp = stable_softplus ? fmaxf(a,0.f) + log1pf(expf(-fabsf(a)))
                                             : (a>20.f ? a : log1pf(expf(a)));
            const auto index = ((size_t)b * nv + h) * T + t;
            gate_out[index] = sp * -expf(a_log[h]);
            const float raw = static_cast<float>(beta[bs1?int64_t(b)*bs0+int64_t(t)*bs1+h:(size_t)bt * nv + h]);
            beta_out[index] = 1.f / (1.f + expf(-raw));
        }
    }
}

template <bool Packed = false, class Input = __half>
__device__ __forceinline__ float linear_prefill_input(
    const Input* qk, const Input* v, int b, int t, int c,
    int T, int qkC, int vsz,int64_t ps0=0,int64_t ps1=0)
{
    if (c < qkC) return static_cast<float>(qk[Packed && ps1?int64_t(b)*ps0+int64_t(t)*ps1+c:
        ((size_t)b * T + t) * (Packed ? qkC+vsz : qkC) + c]);
    return static_cast<float>(v[Packed && ps1?int64_t(b)*ps0+int64_t(t)*ps1+(c-qkC):
        ((size_t)b * T + t) * (Packed ? qkC+vsz : vsz) + (c - qkC)]);
}

template <bool Packed = false, class Input = __half>
__global__ void linear_conv_state_prefill_kernel(
    const float* __restrict__ old_state,
    const Input* __restrict__ qk,
    const Input* __restrict__ v,
    float* __restrict__ new_state,
    int B, int T, int qkC, int vsz, int C, int K,int64_t ps0=0,int64_t ps1=0,
    const int32_t* row_exponents=nullptr,const int32_t* column_exponents=nullptr)
{
    size_t total = (size_t)B * (K - 1) * C;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        int c = (int)(idx % C);
        size_t br = idx / C;
        int j = (int)(br % (K - 1));
        int b = (int)(br / (K - 1));
        int combined = T + j;
        if (combined < K - 1) {
            new_state[idx] = old_state[((size_t)b * (K - 1) + combined) * C + c];
        } else {
            int token = combined - (K - 1);
            const auto value=linear_prefill_input<Packed,Input>(qk, v, b, token, c, T, qkC, vsz,ps0,ps1);
            new_state[idx] = row_exponents?prefill_ldexp(value,row_exponents[int64_t(b)*T+token]+column_exponents[c]):value;
        }
    }
}

// A temporal tile restores each projected value once, then shares it across
// every convolution tap. Q/K normalization writes the recurrent head layout.
template<int RowThreads,class Input,class Gate,bool Scaled>
__global__ void packed_conv_prefill_tile(const float* state,const Input* projected,
    const float* weight,const Gate* alpha,const Gate* beta,const float* dt_bias,const float* a_log,
    const int32_t* row_exponents,const int32_t* column_exponents,
    float* q,float* k,float* v,float* next,float* decay,float* bt,
    int T,int nk,int nv,int width,int taps,int layout,float eps,bool stable_softplus,
    int64_t ps0,int64_t ps1,int64_t as0,int64_t as1,int64_t bs0,int64_t bs1) {
    constexpr int Tile=16,Segments=RowThreads/32;
    const int tiles=(T+Tile-1)/Tile,heads=2*nk+nv;
    const int tile=int(blockIdx.x)%tiles,head=(int(blockIdx.x)/tiles)%heads,b=int(blockIdx.x)/(tiles*heads);
    const int begin=tile*Tile,C=heads*width,offset=head*width;
    const bool normalize=head<2*nk;
    const int lane=threadIdx.x%32,warp=threadIdx.x/32;
    extern __shared__ float image[];
    for(int i=threadIdx.x;i<(Tile+taps-1)*width;i+=256) {
        const int column=i%width,token=begin+i/width-(taps-1);float value=0;
        if(token<0)value=state[(int64_t(b)*(taps-1)+token+taps-1)*C+offset+column];
        else if(token<T) {
            value=float(projected[int64_t(b)*ps0+int64_t(token)*ps1+offset+column]);
            if constexpr(Scaled)value=prefill_ldexp(value,row_exponents[int64_t(b)*T+token]+column_exponents[offset+column]);
        }
        image[i]=value;
    }
    __syncthreads();
    // Eight independent query rows share the same immutable temporal image.
    // Each warp keeps every 32-channel segment of its row in registers.
    for(int first=0;first<Tile;first+=8) {
        const int local=first+warp,token=begin+local;
        float value[Segments];
        #pragma unroll
        for(int segment=0;segment<Segments;++segment)value[segment]=0;
        // Interleave channel accumulators and share the tap control/address
        // calculation. The compact loop avoids four expanded tap pipelines.
        #pragma unroll 1
        for(int tap=0;tap<taps;++tap) {
            #pragma unroll
            for(int segment=0;segment<Segments;++segment) {
                const int column=segment*32+lane;
                if(token<T && column<width) {
                    const float w=weight[layout?int64_t(tap)*C+offset+column:int64_t(offset+column)*taps+tap];
                    value[segment]=fmaf(image[(local+tap)*width+column],w,value[segment]);
                }
            }
        }
        #pragma unroll
        for(int segment=0;segment<Segments;++segment)
            if(token<T && segment*32+lane<width)value[segment]=silu_f32(value[segment]);
        float inverse=1;
        if(normalize) {
            float partial=0;
            #pragma unroll
            for(int segment=0;segment<Segments;++segment) {
                const float sum=warp_sum(value[segment]*value[segment]);
                // The original kernel publishes lane zero's partial from
                // each segment. XOR reduction results can vary by lane.
                const float original=__shfl_sync(0xffffffffu,sum,0);
                if(lane==segment)partial=original;
            }
            const float sum=warp_sum(partial);
            if(!lane)inverse=1.f/fmaxf(sqrtf(sum),eps);
            inverse=__shfl_sync(0xffffffffu,inverse,0);
        }
        #pragma unroll
        for(int segment=0;segment<Segments;++segment) {
            const int column=segment*32+lane;
            if(token<T && column<width) {
                if(normalize) {
                    const int h=head<nk?head:head-nk;
                    (head<nk?q:k)[((int64_t(b)*nk+h)*T+token)*width+column]=value[segment]*inverse;
                }else {
                    const int h=head-2*nk;
                    v[((int64_t(b)*nv+h)*T+token)*width+column]=value[segment];
                }
            }
        }
        if(!normalize && token<T && !lane) {
            const int h=head-2*nk;
            const float a=float(alpha[int64_t(b)*as0+int64_t(token)*as1+h])+dt_bias[h];
            const float sp=stable_softplus?fmaxf(a,0.f)+log1pf(expf(-fabsf(a))):(a>20.f?a:log1pf(expf(a)));
            decay[(int64_t(b)*nv+h)*T+token]=sp*-expf(a_log[h]);
            const float raw=float(beta[int64_t(b)*bs0+int64_t(token)*bs1+h]);
            bt[(int64_t(b)*nv+h)*T+token]=1.f/(1.f+expf(-raw));
        }
    }
    if(tile==tiles-1)for(int i=threadIdx.x;i<(taps-1)*width;i+=256) {
        const int tap=i/width,d=i%width;
        next[(int64_t(b)*(taps-1)+tap)*C+offset+d]=image[(T+tap-begin)*width+d];
    }
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_prefill_cuda(
    mfq_tensor_backend::Tensor state,
    mfq_tensor_backend::Tensor qk,
    mfq_tensor_backend::Tensor v,
    mfq_tensor_backend::Tensor weight,
    mfq_tensor_backend::Tensor bias,
    int64_t nk,
    int64_t nv,
    int64_t dk,
    int64_t dv,
    double eps)
{
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_prefill: state must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(qk.is_cuda() && qk.is_contiguous() && (qk.scalar_type() == mfq_tensor_backend::kHalf || qk.scalar_type() == mfq_tensor_backend::kFloat32),
                "linear_conv_qkv_prefill: qk must be contiguous CUDA f16/f32");
    TYLOINFER_RUNTIME_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == qk.scalar_type(),
                "linear_conv_qkv_prefill: v must match contiguous CUDA qk dtype");
    TYLOINFER_RUNTIME_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_prefill: weight must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.scalar_type() == mfq_tensor_backend::kFloat32,
                "linear_conv_qkv_prefill: bias must be contiguous CUDA f32");
    TYLOINFER_RUNTIME_CHECK(state.dim() == 3 && qk.dim() == 3 && v.dim() == 3,
                "linear_conv_qkv_prefill: state/qk/v must be rank 3");
    int B = (int)qk.size(0);
    int T = (int)qk.size(1);
    int K = (int)state.size(1) + 1;
    int qkC = (int)(2 * nk * dk);
    int vsz = (int)(nv * dv);
    int C = qkC + vsz;
    TYLOINFER_RUNTIME_CHECK(T > 1 && qk.size(2) == qkC && v.size(0) == B && v.size(1) == T && v.size(2) == vsz,
                "linear_conv_qkv_prefill: qk/v shape mismatch");
    TYLOINFER_RUNTIME_CHECK(state.size(0) == B && state.size(2) == C, "linear_conv_qkv_prefill: state shape mismatch");
    TYLOINFER_RUNTIME_CHECK(dk <= 256 && K <= 8, "linear_conv_qkv_prefill: requires dk<=256 and K<=8");
    int layout = 0;
    if (weight.dim() == 3) {
        TYLOINFER_RUNTIME_CHECK(weight.size(0) == C && weight.size(1) == 1 && weight.size(2) == K,
                    "linear_conv_qkv_prefill: [C,1,K] weight mismatch");
    } else {
        TYLOINFER_RUNTIME_CHECK(weight.dim() == 2 && weight.size(0) == K && weight.size(1) == C,
                    "linear_conv_qkv_prefill: weight must be [C,1,K] or [K,C]");
        layout = 1;
    }
    int has_bias = (int)(bias.numel() > 0);
    TYLOINFER_RUNTIME_CHECK(!has_bias || (bias.dim() == 1 && bias.size(0) == C), "linear_conv_qkv_prefill: bias shape mismatch");

    auto opts = state.options();
    auto qo = mfq_tensor_backend::empty({B, (int)nk, T, (int)dk}, opts);
    auto ko = mfq_tensor_backend::empty_like(qo);
    auto vo = mfq_tensor_backend::empty({B, (int)nv, T, (int)dv}, opts);
    auto new_state = mfq_tensor_backend::empty_like(state);
    cudaStream_t stream = mfq_current_cuda_stream();
    const auto launch=[&]<class Input>() {
    linear_conv_qk_norm_prefill_kernel<256,false,Input><<<B * T * 2 * (int)nk, 256, 0, stream>>>(
        state.data_ptr<float>(), static_cast<const Input*>(qk.data_ptr()),
        weight.data_ptr<float>(), bias.data_ptr<float>(), qo.data_ptr<float>(), ko.data_ptr<float>(),
        B, T, (int)nk, (int)dk, C, K, layout, has_bias, (float)eps);
    linear_conv_v_prefill_kernel<false,float,Input><<<dim3(B * T, (vsz + 255) / 256), 256, 0, stream>>>(
        state.data_ptr<float>(), static_cast<const Input*>(v.data_ptr()),
        weight.data_ptr<float>(), bias.data_ptr<float>(), vo.data_ptr<float>(),
        B, T, (int)nk, (int)nv, (int)dk, (int)dv, C, K, layout, has_bias);
    int state_total = B * (K - 1) * C;
    linear_conv_state_prefill_kernel<false,Input><<<(state_total + 255) / 256, 256, 0, stream>>>(
        state.data_ptr<float>(), static_cast<const Input*>(qk.data_ptr()),
        static_cast<const Input*>(v.data_ptr()), new_state.data_ptr<float>(),
        B, T, qkC, vsz, C, K);
    };
    if(qk.scalar_type()==mfq_tensor_backend::kFloat32)launch.template operator()<float>();
    else launch.template operator()<__half>();
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    if(qk.scalar_type()==mfq_tensor_backend::kFloat32) {
        static std::atomic<bool> reported{false};
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1' && !reported.exchange(true))
            std::fprintf(stderr,"native_gdn_prefill_conv_f32 actual_match=1 tokens=%d\n",T);
    }
    return {qo, ko, vo, new_state};
}

// Keep projected Q/K/V together for prefill and short verification windows.
// The V convolution also prepares recurrent gates, preserving FP32 operands.
static std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_prefill_impl(
    mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor qkv,
    mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor alpha,
    mfq_tensor_backend::Tensor beta,mfq_tensor_backend::Tensor dt_bias,
    mfq_tensor_backend::Tensor a_log,int64_t nk,int64_t nv,int64_t width,double eps,
    bool stable_softplus,mfq_tensor_backend::Tensor row_exponents,mfq_tensor_backend::Tensor column_exponents)
{
    namespace tb=mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(qkv.is_cuda() && (qkv.scalar_type()==tb::kFloat16 || qkv.scalar_type()==tb::kFloat32) &&
        qkv.dim()==3 && qkv.size(1)>=2 && qkv.size(1)<=INT_MAX && nk>0 && nv>0 && nv%nk==0 &&
        width>0 && width<=256 && qkv.stride(2)==1 && qkv.stride(1)>=qkv.size(2) &&
        (qkv.size(0)==1 || qkv.stride(0)>=qkv.size(1)*qkv.stride(1)),
        "GDN fused preparation requires row-strided f16/f32 QKV and at least 2 tokens");
    const int B=int(qkv.size(0)),T=int(qkv.size(1)),K=int(state.size(1))+1;
    const int qkC=int(2*nk*width),vsz=int(nv*width),C=qkC+vsz;
    TYLOINFER_RUNTIME_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type()==tb::kFloat32 &&
        state.sizes().vec()==std::vector<int64_t>({B,K-1,C}) && K>=2 && K<=8 && qkv.size(2)==C,
        "short GDN convolution state geometry disagrees");
    for(const auto* tensor:{&weight,&dt_bias,&a_log})
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->is_contiguous() && tensor->scalar_type()==tb::kFloat32 &&
            tensor->device()==qkv.device(),"short GDN parameters require contiguous float on the QKV device");
    TYLOINFER_RUNTIME_CHECK(alpha.is_cuda() && beta.is_cuda() &&
        alpha.device()==qkv.device() && beta.device()==qkv.device() && state.device()==qkv.device() &&
        alpha.sizes().vec()==std::vector<int64_t>({B,T,nv}) && beta.sizes()==alpha.sizes() &&
        alpha.stride(2)==1 && beta.stride(2)==1 && alpha.stride(1)>=nv && beta.stride(1)>=nv &&
        (B==1 || (alpha.stride(0)>=T*alpha.stride(1) && beta.stride(0)>=T*beta.stride(1))) &&
        beta.scalar_type()==alpha.scalar_type() &&
        (alpha.scalar_type()==tb::kFloat16 || alpha.scalar_type()==tb::kFloat32) &&
        dt_bias.numel()==nv && a_log.numel()==nv,"short GDN gate geometry disagrees");
    const bool scaled=row_exponents.defined();
    TYLOINFER_RUNTIME_CHECK(scaled==column_exponents.defined(),"scaled convolution requires both exponent vectors");
    if(scaled)for(const auto* tensor:{&row_exponents,&column_exponents})
        TYLOINFER_RUNTIME_CHECK(tensor->is_cuda() && tensor->device()==qkv.device() && tensor->is_contiguous() &&
            tensor->scalar_type()==tb::kInt32,"scaled convolution exponents must be contiguous device int32");
    TYLOINFER_RUNTIME_CHECK(!scaled || (qkv.scalar_type()==tb::kFloat16 && row_exponents.numel()==int64_t(B)*T &&
        column_exponents.numel()==C),"scaled convolution exponent geometry disagrees");
    const auto* re=scaled?row_exponents.data_ptr<int32_t>():nullptr;
    const auto* ce=scaled?column_exponents.data_ptr<int32_t>():nullptr;
    const int layout=weight.dim()==2?1:0;
    TYLOINFER_RUNTIME_CHECK(layout?weight.sizes().vec()==std::vector<int64_t>({K,C}):
        weight.sizes().vec()==std::vector<int64_t>({C,1,K}),"short GDN convolution weight geometry disagrees");
    auto options=state.options();
    auto q=tb::empty({B,nk,T,width},options),k=tb::empty_like(q),v=tb::empty({B,nv,T,width},options);
    auto next=tb::empty_like(state),decay=tb::empty({B,nv,T},options),bt=tb::empty_like(decay);
    const auto stream=mfq_current_cuda_stream();
    const int64_t tiled_blocks=int64_t(B)*(2*nk+nv)*((int64_t(T)+15)/16);
    if(T>=256 && tiled_blocks<=INT_MAX) {
        const auto launch_input=[&]<class Input>() {
            const auto launch_gate=[&]<class Gate>() {
                const auto launch=[&]<int Threads,bool Scaled>() {
                    packed_conv_prefill_tile<Threads,Input,Gate,Scaled><<<unsigned(tiled_blocks),256,
                        size_t(16+K-1)*width*sizeof(float),stream>>>(state.data_ptr<float>(),static_cast<const Input*>(qkv.data_ptr()),
                        weight.data_ptr<float>(),static_cast<const Gate*>(alpha.data_ptr()),static_cast<const Gate*>(beta.data_ptr()),
                        dt_bias.data_ptr<float>(),a_log.data_ptr<float>(),re,ce,q.data_ptr<float>(),k.data_ptr<float>(),v.data_ptr<float>(),
                        next.data_ptr<float>(),decay.data_ptr<float>(),bt.data_ptr<float>(),T,int(nk),int(nv),int(width),K,layout,float(eps),stable_softplus,
                        qkv.stride(0),qkv.stride(1),alpha.stride(0),alpha.stride(1),beta.stride(0),beta.stride(1));
                };
                const auto choose=[&]<int Threads>() {if(scaled)launch.template operator()<Threads,true>();else launch.template operator()<Threads,false>();};
                if(width<=32)choose.template operator()<32>();else if(width<=64)choose.template operator()<64>();
                else if(width<=128)choose.template operator()<128>();else choose.template operator()<256>();
            };
            if(alpha.scalar_type()==tb::kFloat16)launch_gate.template operator()<__half>();else launch_gate.template operator()<float>();
        };
        if(qkv.scalar_type()==tb::kFloat16)launch_input.template operator()<__half>();else launch_input.template operator()<float>();
        TYLOINFER_CUDA_CHECK(cudaGetLastError());
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{};
            if(!reported.exchange(true))std::fprintf(stderr,"native_gdn_prefill_conv_tile actual_match=1 tokens=%d tile=16 scaled=%d shared_restore_once=1 warp_rows=8 inner_cta_barriers=0\n",T,int(scaled));
        }
        return {q,k,v,next,decay,bt};
    }
    const auto launch_input=[&]<class Input>() {
        const auto* projected=static_cast<const Input*>(qkv.data_ptr());
        const auto launch_qk=[&]<int Threads>() {
            linear_conv_qk_norm_prefill_kernel<Threads,true,Input><<<B*T*2*int(nk),Threads,0,stream>>>(
                state.data_ptr<float>(),projected,weight.data_ptr<float>(),nullptr,q.data_ptr<float>(),k.data_ptr<float>(),
                B,T,int(nk),int(width),C,K,layout,0,float(eps),qkv.stride(0),qkv.stride(1),re,ce);
        };
        if(width<=32)launch_qk.template operator()<32>();
        else if(width<=64)launch_qk.template operator()<64>();
        else if(width<=128)launch_qk.template operator()<128>();
        else launch_qk.template operator()<256>();
        const auto launch_gate=[&]<class Gate>() {
            linear_conv_v_prefill_kernel<true,Gate,Input><<<dim3(B*T,(vsz+255)/256),256,0,stream>>>(
                state.data_ptr<float>(),projected+qkC,weight.data_ptr<float>(),nullptr,v.data_ptr<float>(),
                B,T,int(nk),int(nv),int(width),int(width),C,K,layout,0,
                static_cast<const Gate*>(alpha.data_ptr()),static_cast<const Gate*>(beta.data_ptr()),
                dt_bias.data_ptr<float>(),a_log.data_ptr<float>(),decay.data_ptr<float>(),bt.data_ptr<float>(),
                qkv.stride(0),qkv.stride(1),alpha.stride(0),alpha.stride(1),beta.stride(0),beta.stride(1),stable_softplus,re,ce);
        };
        if(alpha.scalar_type()==tb::kFloat16)launch_gate.template operator()<__half>();else launch_gate.template operator()<float>();
        const int elements=B*(K-1)*C;
        linear_conv_state_prefill_kernel<true,Input><<<(elements+255)/256,256,0,stream>>>(
            state.data_ptr<float>(),projected,projected+qkC,next.data_ptr<float>(),B,T,qkC,vsz,C,K,qkv.stride(0),qkv.stride(1),re,ce);
    };
    if(qkv.scalar_type()==tb::kFloat16)launch_input.template operator()<__half>();else launch_input.template operator()<float>();
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    return {q,k,v,next,decay,bt};
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_prefill_cuda(
    mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor qkv,
    mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor dt_bias,mfq_tensor_backend::Tensor a_log,int64_t nk,int64_t nv,int64_t width,
    double eps,bool stable_softplus) {
    return linear_conv_qkv_gate_prefill_impl(state,qkv,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps,stable_softplus,{},{});
}
std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_scaled_prefill_cuda(
    mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor product,
    mfq_tensor_backend::Tensor row_exponents,mfq_tensor_backend::Tensor column_exponents,
    mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor dt_bias,mfq_tensor_backend::Tensor a_log,int64_t nk,int64_t nv,int64_t width,
    double eps,bool stable_softplus) {
    return linear_conv_qkv_gate_prefill_impl(state,product,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps,stable_softplus,
        row_exponents,column_exponents);
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_prefill_cuda(
    mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor qkv,
    mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor dt_bias,mfq_tensor_backend::Tensor a_log,int64_t nk,int64_t nv,int64_t width,double eps) {
    return linear_conv_qkv_gate_prefill_cuda(state,qkv,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps,true);
}

std::vector<mfq_tensor_backend::Tensor> linear_conv_qkv_gate_short_cuda(
    mfq_tensor_backend::Tensor state,mfq_tensor_backend::Tensor qkv,
    mfq_tensor_backend::Tensor weight,mfq_tensor_backend::Tensor alpha,mfq_tensor_backend::Tensor beta,
    mfq_tensor_backend::Tensor dt_bias,mfq_tensor_backend::Tensor a_log,int64_t nk,int64_t nv,int64_t width,double eps) {
    return linear_conv_qkv_gate_prefill_cuda(state,qkv,weight,alpha,beta,dt_bias,a_log,nk,nv,width,eps);
}
