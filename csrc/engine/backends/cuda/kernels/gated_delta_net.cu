#include "gdn_raw.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda/atomic>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "reduce.cuh"
static void gdn_check(cudaError_t status) { if(status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status)); }
template <int D>
__global__ void gdn_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int H, int T, int kda)
{
    int bh = blockIdx.x;
    int b = bh / H, h = bh % H;
    int j = threadIdx.x;
    if (j >= D) return;

    extern __shared__ float S[];   // S[i*D + j] = state[i][j]

    if (s_in) {
        const float* src = s_in + ((size_t)(b * H + h)) * D * D;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] = src[i * D + j];
    } else {
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] = 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    const size_t bhoff = (size_t)(b * H + h) * T;
    const float* qb = q + bhoff * D;
    const float* kb = k + bhoff * D;
    const float* vb = v + bhoff * D;
    const float* gb = kda ? g + bhoff * D : g + bhoff;
    const float* bb = beta + bhoff;
    float* ob = out + bhoff * D;

    for (int t = 0; t < T; ++t) {
        const float* qt = qb + (size_t)t * D;
        const float* kt = kb + (size_t)t * D;
        const float* vt = vb + (size_t)t * D;
        float bt = bb[t];

        // gated decay
        if (kda) {
            #pragma unroll 4
            for (int i = 0; i < D; ++i) S[i * D + j] *= expf(gb[t * D + i]);
        } else {
            float dec = expf(gb[t]);
            #pragma unroll 4
            for (int i = 0; i < D; ++i) S[i * D + j] *= dec;
        }
        __syncthreads();

        // delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
        float stk = 0.f;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) stk += S[i * D + j] * kt[i];
        float delta = (vt[j] - stk) * bt;

        // S[i][j] += k[i] * delta[j]
        #pragma unroll 4
        for (int i = 0; i < D; ++i) S[i * D + j] += kt[i] * delta;
        __syncthreads();

        // o[j] = scale * sum_i S[i][j]*q[i]
        float o = 0.f;
        #pragma unroll 4
        for (int i = 0; i < D; ++i) o += S[i * D + j] * qt[i];
        ob[(size_t)t * D + j] = o * scale;
    }

    float* dst = s_out + ((size_t)(b * H + h)) * D * D;
    #pragma unroll 4
    for (int i = 0; i < D; ++i) dst[i * D + j] = S[i * D + j];
}

template <int D, int BD>
__global__ void gdn_column_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int H, int T, int kda)
{
    constexpr int NW = BD / 32;
    int bid = blockIdx.x;
    int j = bid % D;
    int bh = bid / D;
    int b = bh / H;
    int h = bh % H;
    int tid = threadIdx.x;
    const float scale = 1.0f / sqrtf((float)D);
    const size_t state_off = ((size_t)(b * H + h)) * D * D;
    const size_t bhoff = (size_t)(b * H + h) * T;
    const float* qb = q + bhoff * D;
    const float* kb = k + bhoff * D;
    const float* vb = v + bhoff * D;
    const float* gb = kda ? g + bhoff * D : g + bhoff;
    const float* bb = beta + bhoff;
    float* ob = out + bhoff * D;
    float* so = s_out + state_off;

    for (int i = tid; i < D; i += BD) {
        so[i * D + j] = s_in ? s_in[state_off + i * D + j] : 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < T; ++t) {
        const float* qt = qb + (size_t)t * D;
        const float* kt = kb + (size_t)t * D;
        const float* vt = vb + (size_t)t * D;
        float bt = bb[t];

        float local = 0.0f;
        for (int i = tid; i < D; i += BD) {
            float sij = so[i * D + j];
            float dec = kda ? expf(gb[(size_t)t * D + i]) : expf(gb[t]);
            sij *= dec;
            so[i * D + j] = sij;
            local += sij * kt[i];
        }
        float stk = block_sum<NW>(local);
        float delta = (vt[j] - stk) * bt;
        __syncthreads();

        for (int i = tid; i < D; i += BD) {
            so[i * D + j] += kt[i] * delta;
        }
        __syncthreads();

        local = 0.0f;
        for (int i = tid; i < D; i += BD) {
            local += so[i * D + j] * qt[i];
        }
        float o = block_sum<NW>(local);
        if (tid == 0) {
            ob[(size_t)t * D + j] = o * scale;
        }
        __syncthreads();
    }
}

template <int D, bool KDA, int WARPS, bool TRANSPOSED_STATE=false, bool TILED_HEADS=false>
__global__ void gdn_warp_column_kernel(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* __restrict__ s_in,
    float* __restrict__ out, float* __restrict__ s_out,
    int Hq, int Hv, int T)
{
    constexpr int WARP = 32;
    constexpr int ROWS = (D + WARP - 1) / WARP;
    int bh = blockIdx.x;
    int b = bh / Hv;
    int hv = bh % Hv;
    int hq = TILED_HEADS ? hv % Hq : hv / (Hv / Hq);
    int lane = threadIdx.x;
    int warp = threadIdx.y;
    int col = blockIdx.y * WARPS + warp;
    if (col >= D) {
        return;
    }

    const size_t state_off = ((size_t)(b * Hv + hv)) * D * D;
    const size_t q_seq_off = ((size_t)(b * Hq + hq)) * T;
    const size_t v_seq_off = ((size_t)(b * Hv + hv)) * T;
    float s_shard[ROWS];

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int i = r * WARP + lane;
        size_t state_idx = TRANSPOSED_STATE ? (size_t)col * D + i : (size_t)i * D + col;
        s_shard[r] = (i < D && s_in) ? s_in[state_off + state_idx] : 0.0f;
    }

    const float scale = 1.0f / sqrtf((float)D);
    for (int t = 0; t < T; ++t) {
        const float* qt = q + (q_seq_off + t) * D;
        const float* kt = k + (q_seq_off + t) * D;
        const float* vt = v + (v_seq_off + t) * D;
        const float* gt = KDA ? g + (v_seq_off + t) * D : g + (v_seq_off + t);
        float bt = beta[v_seq_off + t];

        float k_reg[ROWS];
        float q_reg[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            int i = r * WARP + lane;
            k_reg[r] = (i < D) ? kt[i] : 0.0f;
            q_reg[r] = (i < D) ? qt[i] : 0.0f;
        }

        float kv = 0.0f;
        if constexpr (KDA) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int i = r * WARP + lane;
                float dec = (i < D) ? expf(gt[i]) : 0.0f;
                kv += dec * s_shard[r] * k_reg[r];
            }
        } else {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                kv += s_shard[r] * k_reg[r];
            }
        }
        kv = warp_sum(kv);

        float delta;
        if constexpr (KDA) {
            delta = (vt[col] - kv) * bt;
        } else {
            float dec = expf(*gt);
            delta = (vt[col] - dec * kv) * bt;
        }

        float o = 0.0f;
        if constexpr (KDA) {
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                int i = r * WARP + lane;
                float dec = (i < D) ? expf(gt[i]) : 0.0f;
                s_shard[r] = dec * s_shard[r] + k_reg[r] * delta;
                o += s_shard[r] * q_reg[r];
            }
        } else {
            float dec = expf(*gt);
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                s_shard[r] = dec * s_shard[r] + k_reg[r] * delta;
                o += s_shard[r] * q_reg[r];
            }
        }
        o = warp_sum(o);
        if (lane == 0) {
            out[(v_seq_off + t) * D + col] = o * scale;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        int i = r * WARP + lane;
        if (i < D) {
            size_t state_idx = TRANSPOSED_STATE ? (size_t)col * D + i : (size_t)i * D + col;
            s_out[state_off + state_idx] = s_shard[r];
        }
    }
}

#define LAUNCH_GDN(DVAL)                                                            \
    do {                                                                            \
        if (shmem > 49152)                                                          \
            cudaFuncSetAttribute((const void*)gdn_kernel<DVAL>,                     \
                cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);               \
        gdn_kernel<DVAL><<<B * H, DVAL, shmem, stream>>>(                           \
            qd, kd, vd, gd, bd, sd, od, sod, H, T, kda);                            \
    } while (0)

#define LAUNCH_GDN_COL(DVAL)                                                        \
    do {                                                                            \
        gdn_column_kernel<DVAL, 128><<<B * H * DVAL, 128, 0, stream>>>(              \
            qd, kd, vd, gd, bd, sd, od, sod, H, T, kda);                            \
    } while (0)

#define LAUNCH_GDN_WARP(DVAL, KDA_VAL, TRANSPOSED_VAL, TILED_VAL)                     \
    do {                                                                              \
        constexpr int WARPS = 4;                                                      \
        dim3 grid(B * Hv, (DVAL + WARPS - 1) / WARPS, 1);                             \
        dim3 block(32, WARPS, 1);                                                     \
        gdn_warp_column_kernel<DVAL, KDA_VAL, WARPS, TRANSPOSED_VAL, TILED_VAL><<<grid, block, 0, stream>>>( \
            qd, kd, vd, gd, bd, sd, od, sod, Hq, Hv, T);                              \
    } while (0)

#define DISPATCH_GDN_WARP(TILED_VAL)                                                   \
    do {                                                                               \
        if (transposed_state) {                                                        \
            if (kda) {                                                                 \
                if (D == 32) { LAUNCH_GDN_WARP(32, true, true, TILED_VAL); }           \
                else if (D == 64) { LAUNCH_GDN_WARP(64, true, true, TILED_VAL); }      \
                else if (D == 128) { LAUNCH_GDN_WARP(128, true, true, TILED_VAL); }    \
                else throw std::runtime_error("GDN CUDA: D must be in {32,64,128}, got "+std::to_string(D)); \
            } else {                                                                   \
                if (D == 32) { LAUNCH_GDN_WARP(32, false, true, TILED_VAL); }          \
                else if (D == 64) { LAUNCH_GDN_WARP(64, false, true, TILED_VAL); }     \
                else if (D == 128) { LAUNCH_GDN_WARP(128, false, true, TILED_VAL); }   \
                else throw std::runtime_error("GDN CUDA: D must be in {32,64,128}, got "+std::to_string(D)); \
            }                                                                          \
        } else if (kda) {                                                              \
            if (D == 32) { LAUNCH_GDN_WARP(32, true, false, TILED_VAL); }              \
            else if (D == 64) { LAUNCH_GDN_WARP(64, true, false, TILED_VAL); }         \
            else if (D == 128) { LAUNCH_GDN_WARP(128, true, false, TILED_VAL); }       \
            else throw std::runtime_error("GDN CUDA: D must be in {32,64,128}, got "+std::to_string(D));   \
        } else {                                                                       \
            if (D == 32) { LAUNCH_GDN_WARP(32, false, false, TILED_VAL); }             \
            else if (D == 64) { LAUNCH_GDN_WARP(64, false, false, TILED_VAL); }        \
            else if (D == 128) { LAUNCH_GDN_WARP(128, false, false, TILED_VAL); }      \
            else throw std::runtime_error("GDN CUDA: D must be in {32,64,128}, got "+std::to_string(D));   \
        }                                                                              \
    } while (0)

#include "gdn_prefill_columns.inc"

template<int tiles,class Gate,class Output,int FixedWidth=0,int FixedTaps=0,bool InplaceConv=false,int BroadcastNorm=0,bool Snapshot=false>
__global__ void gdn_decode_core_kernel(const __half* qkv,const Gate* output_gate,
    const Gate* alpha,const Gate* beta,const float* convolution_state,const float* recurrent_state,
    const float* convolution_weight,const float* dt_bias,const float* a_log,const float* norm_weight,
    Output* output,float* convolution_out,float* recurrent_out,int key_heads,int value_heads,
    int width,int taps,int layout,float convolution_eps,double norm_eps,bool silu_gate,
    bool transposed_state,int64_t as0,int64_t as2,int64_t bs0,int64_t bs2,
    float* workspace,const int64_t* state_addresses,int64_t qkv_stride,int64_t gate_stride,bool warp_reductions,float* checkpoint_convolution,float* checkpoint_recurrent) {
    if constexpr(FixedWidth>0)width=FixedWidth;
    if constexpr(FixedTaps>0)taps=FixedTaps;
    const int bh=blockIdx.x/tiles,tile=blockIdx.x%tiles;
    const int batch=bh/value_heads,head=bh%value_heads;
    const int state_batch=state_addresses?0:batch;
    if(state_addresses) {
        convolution_state=reinterpret_cast<const float*>(state_addresses[3*batch]);
        recurrent_state=reinterpret_cast<const float*>(state_addresses[3*batch+1]);
        recurrent_out=reinterpret_cast<float*>(state_addresses[3*batch+1]);
        workspace=reinterpret_cast<float*>(state_addresses[3*batch+2]);
    }
    const int workspace_head=state_addresses?head:bh;
    const int key_head=head/(value_heads/key_heads),tid=threadIdx.x;
    const int qk_width=2*key_heads*width,channels=qk_width+value_heads*width;
    __shared__ float query[128],key[128],value[128],attended[128];
    __shared__ float qsum[256],ksum[256],norm_sum[128],decay_value,beta_value;
    __shared__ bool final_tile,final_key_group;
    const auto store_conv=[&](size_t index,float value) {
        convolution_out[index]=value;
        if constexpr(Snapshot)checkpoint_convolution[index]=value;
    };
    float q=0,k=0,v=0;
    if(tid<width) {
        const int qc=key_head*width+tid,kc=key_heads*width+qc,vc=qk_width+head*width+tid;
        const size_t state_base=size_t(batch)*(taps-1)*channels;
        const size_t input_base=size_t(state_batch)*(taps-1)*channels;
        for(int tap=0;tap<taps-1;++tap) {
            const size_t wq=layout==0?size_t(qc)*taps+tap:size_t(tap)*channels+qc;
            const size_t wk=layout==0?size_t(kc)*taps+tap:size_t(tap)*channels+kc;
            const size_t wv=layout==0?size_t(vc)*taps+tap:size_t(tap)*channels+vc;
            q+=convolution_state[input_base+size_t(tap)*channels+qc]*convolution_weight[wq];
            k+=convolution_state[input_base+size_t(tap)*channels+kc]*convolution_weight[wk];
            v+=convolution_state[input_base+size_t(tap)*channels+vc]*convolution_weight[wv];
        }
        const float q_current=__half2float(qkv[size_t(batch)*qkv_stride+qc]);
        const float k_current=__half2float(qkv[size_t(batch)*qkv_stride+kc]);
        const float v_current=__half2float(qkv[size_t(batch)*qkv_stride+vc]);
        const size_t wq=layout==0?size_t(qc)*taps+taps-1:size_t(taps-1)*channels+qc;
        const size_t wk=layout==0?size_t(kc)*taps+taps-1:size_t(taps-1)*channels+kc;
        const size_t wv=layout==0?size_t(vc)*taps+taps-1:size_t(taps-1)*channels+vc;
        q+=q_current*convolution_weight[wq];
        k+=k_current*convolution_weight[wk];
        v+=v_current*convolution_weight[wv];
        q=q/(1.f+expf(-q));k=k/(1.f+expf(-k));v=v/(1.f+expf(-v));
        query[tid]=q;key[tid]=k;value[tid]=v;
        if constexpr(!InplaceConv) {
        for(int tap=0;tap<taps-2;++tap) {
            if(tile==0 && head%(value_heads/key_heads)==0) {
                store_conv(state_base+size_t(tap)*channels+qc,convolution_state[input_base+size_t(tap+1)*channels+qc]);
                store_conv(state_base+size_t(tap)*channels+kc,convolution_state[input_base+size_t(tap+1)*channels+kc]);
            }
            if(tile==0)store_conv(state_base+size_t(tap)*channels+vc,convolution_state[input_base+size_t(tap+1)*channels+vc]);
        }
        if(tile==0 && head%(value_heads/key_heads)==0) {
            store_conv(state_base+size_t(taps-2)*channels+qc,q_current);
            store_conv(state_base+size_t(taps-2)*channels+kc,k_current);
        }
        if(tile==0)store_conv(state_base+size_t(taps-2)*channels+vc,v_current);
        }
    }
    if(tid<256) {qsum[tid]=tid<width?q*q:0.f;ksum[tid]=tid<width?k*k:0.f;}
    if(tid==0) {
        const float a=static_cast<float>(alpha[int64_t(batch)*as0+int64_t(head)*as2])+dt_bias[head];
        const float sp=fmaxf(a,0.f)+log1pf(expf(-fabsf(a)));
        const float gate_log=sp*-expf(a_log[head]);
        decay_value=expf(gate_log);
        const float raw_beta=static_cast<float>(beta[int64_t(batch)*bs0+int64_t(head)*bs2]);
        beta_value=1.f/(1.f+expf(-raw_beta));
    }
    __syncthreads();
    if(!warp_reductions || width!=128)for(int stride=128;stride>=32;stride>>=1) {
        if(tid<stride) {qsum[tid]+=qsum[tid+stride];ksum[tid]+=ksum[tid+stride];}
        __syncthreads();
    }
    if(tid<32) {
        float qr=qsum[tid],kr=ksum[tid];
        if(warp_reductions && width==128) {
            // Reproduce the original 128,64,32 pair tree in registers.
            qr=(qr+qsum[tid+128])+(qsum[tid+64]+qsum[tid+192]);
            kr=(kr+ksum[tid+128])+(ksum[tid+64]+ksum[tid+192]);
            qr+=(qsum[tid+32]+qsum[tid+160])+(qsum[tid+96]+qsum[tid+224]);
            kr+=(ksum[tid+32]+ksum[tid+160])+(ksum[tid+96]+ksum[tid+224]);
        }
        #pragma unroll
        for(int stride=16;stride>0;stride>>=1) {
            const float qother=__shfl_down_sync(0xffffffff,qr,stride);
            const float kother=__shfl_down_sync(0xffffffff,kr,stride);
            if(tid<stride) {qr+=qother;kr+=kother;}
        }
        if(tid==0) {
            if constexpr(BroadcastNorm==1) {
                qsum[0]=1.f/fmaxf(sqrtf(qr),convolution_eps);
                ksum[0]=1.f/fmaxf(sqrtf(kr),convolution_eps);
            }else {qsum[0]=qr;ksum[0]=kr;}
        }
    }
    __syncthreads();
    if(tid<width) {
        if constexpr(BroadcastNorm==1) {query[tid]*=qsum[0];key[tid]*=ksum[0];}
        else if constexpr(BroadcastNorm==2 || BroadcastNorm==4) {
            float qi=0.f,ki=0.f;
            if((tid&31)==0) {
                qi=1.f/fmaxf(sqrtf(qsum[0]),convolution_eps);
                ki=1.f/fmaxf(sqrtf(ksum[0]),convolution_eps);
            }
            query[tid]*=__shfl_sync(0xffffffffu,qi,0);
            key[tid]*=__shfl_sync(0xffffffffu,ki,0);
        }
        else {
            query[tid]*=1.f/fmaxf(sqrtf(qsum[0]),convolution_eps);
            key[tid]*=1.f/fmaxf(sqrtf(ksum[0]),convolution_eps);
        }
    }
    __syncthreads();
    const int lane=tid&31,warp=tid>>5,warps=blockDim.x>>5;
    const int rows=(width+31)/32;
    const size_t state_head=(size_t(state_batch)*value_heads+head)*width*width;
    const float decay=decay_value,bt=beta_value,scale=1.f/sqrtf(float(width));
    const int columns_per_tile=(width+tiles-1)/tiles;
    const int column_end=min(width,(tile+1)*columns_per_tile);
    for(int column=tile*columns_per_tile+warp;column<column_end;column+=warps) {
        float states[4],keys[4],queries[4];
        #pragma unroll
        for(int row=0;row<4;++row) if(row<rows) {
            const int i=row*32+lane;
            const size_t index=transposed_state?size_t(column)*width+i:size_t(i)*width+column;
            states[row]=recurrent_state[state_head+index];
            keys[row]=key[i];queries[row]=query[i];
        }
        float projected=0.f;
        #pragma unroll
        for(int row=0;row<4;++row)if(row<rows)projected+=states[row]*keys[row];
        projected=warp_sum(projected);
        const float delta=(value[column]-decay*projected)*bt;
        float result=0.f;
        #pragma unroll
        for(int row=0;row<4;++row) if(row<rows) {
            const int i=row*32+lane;
            const size_t index=transposed_state?size_t(column)*width+i:size_t(i)*width+column;
            // Preserve the original warp-column kernel's contracted operand
            // order, including its one-row and multiple-row distinction.
            states[row]=rows==1
                ? __fmaf_rn(keys[row],delta,__fmul_rn(decay,states[row]))
                : __fmaf_rn(decay,states[row],__fmul_rn(keys[row],delta));
            result+=states[row]*queries[row];
            recurrent_out[state_head+index]=states[row];
            if constexpr(Snapshot)checkpoint_recurrent[state_head+index]=states[row];
        }
        result=warp_sum(result);
        if(lane==0) {
            const float computed=result*scale;
            if(tiles==1)attended[column]=computed;
            else workspace[size_t(workspace_head)*(width+1)+column]=computed;
        }
    }
    if(tiles>1) {
        // Only warp leaders publish attended columns into the cross-tile
        // workspace. Recurrent state is consumed after kernel completion;
        // the block barrier also completes every convolution-history read
        // before the completion counter can enable an in-place commit.
        if constexpr(BroadcastNorm==4) {
            if(lane==0)__threadfence();
        }else __threadfence();
        __syncthreads();
        if(tid==0) {
            auto* counter=reinterpret_cast<uint32_t*>(workspace+size_t(workspace_head)*(width+1)+width);
            cuda::atomic_ref<uint32_t,cuda::thread_scope_device> progress(*counter);
            const auto previous=progress.fetch_add(1u,cuda::std::memory_order_acq_rel);
            final_tile=(InplaceConv?(previous&255u):previous)==unsigned(tiles-1);
        }
        __syncthreads();
        if(!final_tile)return;
        if(tid<width)attended[tid]=workspace[size_t(workspace_head)*(width+1)+tid];
        if constexpr(InplaceConv)if(tid==0) {
            // The existing tile publication already follows every history
            // read. Publish group readiness before the output epilogue.
            const int group_size=value_heads/key_heads,first_head=key_head*group_size;
            auto* counter=reinterpret_cast<uint32_t*>(workspace+
                (size_t(batch)*value_heads+first_head)*(width+1)+width);
            cuda::atomic_ref<uint32_t,cuda::thread_scope_device> progress(*counter);
            const uint32_t bit=1u<<(8+head-first_head),mask=((1u<<group_size)-1u)<<8;
            const uint32_t previous=progress.fetch_or(bit,cuda::std::memory_order_acq_rel);
            final_key_group=((previous|bit)&mask)==mask;
        }
    }
    __syncthreads();
    if(tid<width)norm_sum[tid]=__fadd_rn(0.f,__fmul_rn(attended[tid],attended[tid]));
    __syncthreads();
    if(!warp_reductions || width!=128)for(int stride=width/2;stride>=32;stride>>=1) {
        if(tid<stride)norm_sum[tid]=__fadd_rn(norm_sum[tid],norm_sum[tid+stride]);
        __syncthreads();
    }
    if(tid<32) {
        float partial=norm_sum[tid];
        if(warp_reductions && width==128)partial=__fadd_rn(
            __fadd_rn(partial,norm_sum[tid+64]),__fadd_rn(norm_sum[tid+32],norm_sum[tid+96]));
        #pragma unroll
        for(int stride=16;stride>0;stride>>=1) {
            const float other=__shfl_down_sync(0xffffffff,partial,stride);
            if(tid<stride)partial=__fadd_rn(partial,other);
        }
        if(tid==0) {
            if constexpr(BroadcastNorm==1) {
                const float mean=__fdiv_rn(partial,float(width));
                const float variance=float(double(mean)+norm_eps);
                norm_sum[0]=__fdiv_rn(1.f,__fsqrt_rn(variance));
            }else norm_sum[0]=partial;
        }
    }
    __syncthreads();
    if constexpr(InplaceConv)if(tid>=width && tid<2*width) {
        // Threads unused by output normalization commit history in parallel.
        const int column=tid-width,vc=qk_width+head*width+column;
        const size_t base=size_t(batch)*(taps-1)*channels;
        for(int tap=0;tap<taps-2;++tap)
            store_conv(base+size_t(tap)*channels+vc,convolution_state[base+size_t(tap+1)*channels+vc]);
        store_conv(base+size_t(taps-2)*channels+vc,__half2float(qkv[size_t(batch)*qkv_stride+vc]));
        if(final_key_group) {
            const int qc=key_head*width+column,kc=key_heads*width+qc;
            for(int tap=0;tap<taps-2;++tap) {
                store_conv(base+size_t(tap)*channels+qc,convolution_state[base+size_t(tap+1)*channels+qc]);
                store_conv(base+size_t(tap)*channels+kc,convolution_state[base+size_t(tap+1)*channels+kc]);
            }
            store_conv(base+size_t(taps-2)*channels+qc,__half2float(qkv[size_t(batch)*qkv_stride+qc]));
            store_conv(base+size_t(taps-2)*channels+kc,__half2float(qkv[size_t(batch)*qkv_stride+kc]));
        }
    }
    if(tid<width) {
        float inverse;
        if constexpr(BroadcastNorm==1)inverse=norm_sum[0];
        else if constexpr(BroadcastNorm>=2) {
            inverse=0.f;
            if((tid&31)==0) {
                const float mean=__fdiv_rn(norm_sum[0],float(width));
                const float variance=float(double(mean)+norm_eps);
                inverse=__fdiv_rn(1.f,__fsqrt_rn(variance));
            }
            inverse=__shfl_sync(0xffffffffu,inverse,0);
        }
        else {
            const float mean=__fdiv_rn(norm_sum[0],float(width));
            const float variance=float(double(mean)+norm_eps);
            inverse=__fdiv_rn(1.f,__fsqrt_rn(variance));
        }
        const size_t index=(size_t(batch)*value_heads+head)*width+tid;
        const float z=static_cast<float>(output_gate[size_t(batch)*gate_stride+size_t(head)*width+tid]);
        const float sigmoid=__fdiv_rn(1.f,__fadd_rn(1.f,expf(-z)));
        const float gate=silu_gate?__fmul_rn(z,sigmoid):sigmoid;
        const float scaled=__fmul_rn(__fmul_rn(attended[tid],inverse),norm_weight[tid]);
        output[index]=static_cast<Output>(__fmul_rn(scaled,gate));
    }
    if constexpr(InplaceConv) {
        // A subsequent replay starts only after this kernel completes. All
        // group heads have published readiness, so no counters remain in use.
        const int group_size=value_heads/key_heads,first_head=key_head*group_size;
        if(final_key_group && tid<group_size) {
                auto* counter=reinterpret_cast<uint32_t*>(workspace+
                    (size_t(batch)*value_heads+first_head+tid)*(width+1)+width);
                cuda::atomic_ref<uint32_t,cuda::thread_scope_device> progress(*counter);
                progress.store(0u,cuda::std::memory_order_release);
        }
    }else if(tiles>1) {
        __syncthreads();
        if(tid==0) {
            auto* counter=reinterpret_cast<uint32_t*>(workspace+size_t(workspace_head)*(width+1)+width);
            cuda::atomic_ref<uint32_t,cuda::thread_scope_device> progress(*counter);
            progress.store(0u,cuda::std::memory_order_release);
        }
    }
}

__global__ void gdn_commit_independent_convolution(const float* values,
    const int64_t* state_addresses,int64_t row_elements) {
    const int row=blockIdx.y;
    const int64_t column=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(column<row_elements)
        reinterpret_cast<float*>(state_addresses[3*row])[column]=values[int64_t(row)*row_elements+column];
}

namespace tyloi::cuda::gdn_raw {
void recurrence(const Recurrence& a,cudaStream_t stream) {
    const int B=a.batch,Hq=a.query_heads,Hv=a.value_heads,H=Hv,T=a.tokens,D=a.width;
    const bool kda=a.kda,transposed_state=a.transposed,tiled_heads=a.tiled;
    const float *qd=a.q,*kd=a.k,*vd=a.v,*gd=a.g,*bd=a.beta,*sd=a.state;
    float *od=a.output,*sod=a.next;
    const int shmem=D*D*int(sizeof(float));
    if(a.warp && !kda && D==128 && (T>8 || (T>1 && a.short_columns)) && (a.columns==2 || a.columns==4)) {
        if(a.prepared_decay) {
            gdn_prefill_prepare_decay_kernel<<<unsigned((int64_t(B)*Hv*T+255)/256),256,0,stream>>>(gd,a.decay,int64_t(B)*Hv*T);
            gd=a.decay;
        }
        if(transposed_state) {
            if(tiled_heads)launch_gdn_prefill_columns<true,true>(a.columns,B,Hq,Hv,T,qd,kd,vd,gd,bd,sd,od,sod,stream,a.prepared_decay,a.pipelined,a.staged);
            else launch_gdn_prefill_columns<true,false>(a.columns,B,Hq,Hv,T,qd,kd,vd,gd,bd,sd,od,sod,stream,a.prepared_decay,a.pipelined,a.staged);
        } else {
            if(tiled_heads)launch_gdn_prefill_columns<false,true>(a.columns,B,Hq,Hv,T,qd,kd,vd,gd,bd,sd,od,sod,stream,a.prepared_decay,a.pipelined,a.staged);
            else launch_gdn_prefill_columns<false,false>(a.columns,B,Hq,Hv,T,qd,kd,vd,gd,bd,sd,od,sod,stream,a.prepared_decay,a.pipelined,a.staged);
        }
    } else if(a.warp) {
        if(tiled_heads){DISPATCH_GDN_WARP(true);}else{DISPATCH_GDN_WARP(false);}
    } else if(T<=4 && a.column) {
        if(Hq!=Hv)throw std::runtime_error("GDN CUDA column path requires repeated q/k heads");
        if(D==32){LAUNCH_GDN_COL(32);}else if(D==64){LAUNCH_GDN_COL(64);}else if(D==128){LAUNCH_GDN_COL(128);}
        else throw std::runtime_error("GDN CUDA: D must be in {32,64,128}, got "+std::to_string(D));
    } else {
        if(Hq!=Hv)throw std::runtime_error("GDN CUDA shared-memory path requires repeated q/k heads");
        if(D==32){LAUNCH_GDN(32);}else if(D==64){LAUNCH_GDN(64);}else if(D==128){LAUNCH_GDN(128);}
        else throw std::runtime_error("GDN CUDA: D must be in {32,64,128}, got "+std::to_string(D));
    }
}
void decode(const Decode& a,cudaStream_t stream) {
    constexpr int threads=256;
    const auto launch=[&]<int Tiles,class Gate,class Output,int FixedWidth=0,int FixedTaps=0>() {
        const auto run=[&]<bool InplaceConv>() {
            const auto normalizations=[&]<int BroadcastNorm>() {
                const auto dispatch=[&]<bool Snapshot>() {
                    if(a.probe) {
                        cudaFuncAttributes attributes{};
                        gdn_check(cudaFuncGetAttributes(&attributes,reinterpret_cast<const void*>(gdn_decode_core_kernel<Tiles,Gate,Output,FixedWidth,FixedTaps,InplaceConv,BroadcastNorm,Snapshot>)));
                        std::fprintf(stderr,"gdn_core_resources mode=%d batch=%lld threads=%d registers=%d shared=%zu local=%zu width=%lld tiles=%d inplace_conv=%d\n",BroadcastNorm,static_cast<long long>(a.batch),threads,attributes.numRegs,attributes.sharedSizeBytes,attributes.localSizeBytes,static_cast<long long>(a.width),Tiles,int(InplaceConv));
                    }
                    gdn_decode_core_kernel<Tiles,Gate,Output,FixedWidth,FixedTaps,InplaceConv,BroadcastNorm,Snapshot><<<unsigned(a.batch*a.value_heads*Tiles),threads,0,stream>>>(
                        static_cast<const __half*>(a.qkv),static_cast<const Gate*>(a.gate),static_cast<const Gate*>(a.alpha),static_cast<const Gate*>(a.beta),
                        a.conv_state,a.state,a.weight,a.dt_bias,a.a_log,a.norm,static_cast<Output*>(a.output),a.conv_out,a.state_out,
                        a.key_heads,a.value_heads,a.width,a.taps,a.layout,a.conv_eps,a.norm_eps,a.silu,a.transposed,
                        a.alpha_stride0,a.alpha_stride2,a.beta_stride0,a.beta_stride2,a.workspace,a.state_addresses,
                        a.qkv_stride0,a.gate_stride0,a.warp_reductions,a.checkpoint_conv,a.checkpoint_state);
                };
                if(a.checkpoint_conv)dispatch.template operator()<true>();else dispatch.template operator()<false>();
            };
            if constexpr(FixedWidth==128 && FixedTaps==4) {
                if(a.mode==2){normalizations.template operator()<2>();return;}
                if(a.mode==3){normalizations.template operator()<3>();return;}
                if(a.mode==4){normalizations.template operator()<4>();return;}
            }
            if(a.mode==1)normalizations.template operator()<1>();else normalizations.template operator()<0>();
        };
        if constexpr(Tiles>1)if(a.inplace_conv){run.template operator()<true>();return;}
        run.template operator()<false>();
    };
    const auto geometry=[&]<int Tiles,class Gate,class Output>() {
        if(a.fixed_geometry && a.taps==4) {
            if(a.width==32)launch.template operator()<Tiles,Gate,Output,32,4>();
            else if(a.width==64)launch.template operator()<Tiles,Gate,Output,64,4>();
            else launch.template operator()<Tiles,Gate,Output,128,4>();
        } else launch.template operator()<Tiles,Gate,Output>();
    };
    const auto types=[&]<int Tiles>() {
        if(a.gate_half) {
            if(a.output_half)geometry.template operator()<Tiles,__half,__half>();else geometry.template operator()<Tiles,__half,float>();
        } else {
            if(a.output_half)geometry.template operator()<Tiles,float,__half>();else geometry.template operator()<Tiles,float,float>();
        }
    };
    if(a.tiles==1)types.template operator()<1>();else if(a.tiles==2)types.template operator()<2>();else types.template operator()<4>();
    gdn_check(cudaGetLastError());
    if(a.state_addresses) {
        const int64_t elements=int64_t(a.taps-1)*(2*int64_t(a.key_heads)+a.value_heads)*a.width;
        gdn_commit_independent_convolution<<<dim3(unsigned((elements+255)/256),unsigned(a.batch)),256,0,stream>>>(a.conv_out,a.state_addresses,elements);
        gdn_check(cudaGetLastError());
    }
}
} // namespace tyloi::cuda::gdn_raw
