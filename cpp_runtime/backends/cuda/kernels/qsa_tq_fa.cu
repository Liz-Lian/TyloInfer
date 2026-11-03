// Sparse Flash Attention adapted from llama.cpp commit
// 51fe42c5d14bfb9adf0612ff98a9ab79e0e7275c (MIT).
// Copyright (c) 2023-2026 The ggml authors.
// Native additions preserve four-token groups, packed H4 KV, and QSA history.
// See LICENSES/llama.cpp-MIT.txt.
#include "qsa_tq_fa_mma.cuh"
#include "qsa_tq_fa.h"
#include <cstdio>
#include <cstdlib>
#include <atomic>
static int qsa_llama_launch(const float* query,const half* keys,const half* values,const half* mask,
        const int* indices,float* output,int queries,int kv_heads,int kv_rows,cudaStream_t stream,QsaFaHistory history,int requested_heads=0,float softmax_scale=1.f/16) {
    constexpr int width=256;
    const auto fn=flash_attn_ext_f16<256,256,8,8,false,false,true>;
    struct Configuration {int device=-1,warps=0,stages=0;size_t shared=0;};
    static thread_local Configuration cached;
    int device=0;auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return int(error);
    if(cached.device!=device) {
        int major=0,minor=0;
        error=cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device);
        if(error!=cudaSuccess)return int(error);
        error=cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device);
        if(error!=cudaSuccess)return int(error);
        cudaFuncAttributes image{};
        error=cudaFuncGetAttributes(&image,fn);if(error!=cudaSuccess)return int(error);
        // m16n8k16 + cp.async require SM80 device code as well as hardware.
        // A lower virtual-architecture image JITed on a newer GPU must use
        // the existing attention implementation instead of an empty body.
        if(major<8 || image.ptxVersion<80)return int(cudaErrorNotSupported);
        const int cc=major*100+minor*10;
        const int nth=ggml_cuda_fattn_mma_get_nthreads(width,width,64,cc),warps=nth/32;
        const int batch=ggml_cuda_fattn_mma_get_nbatch_fa(width,width,64,cc);
        const int bk=ggml_cuda_fattn_mma_get_nbatch_K2(width,width,64,cc);
        const int bv=ggml_cuda_fattn_mma_get_nbatch_V2(width,width,64,cc);
        const int bc=ggml_cuda_fattn_mma_get_nbatch_combine(width,width,64,cc);
        const int stages=ggml_cuda_fattn_mma_get_nstages(width,width,8,8,cc);
        const bool qreg=ggml_cuda_fattn_mma_get_Q_in_reg(width,width,64,cc);
        const bool swz=ggml_cuda_fattn_mma_get_swizzled(width,width,64,cc);
        const int sk=swz?bk:bk+4,sv=swz?bv:bv+4;
        const size_t nq=64*(width/2+4)*sizeof(half2),nk=batch*(stages>1?sk+sv:std::max(sk,sv))*sizeof(half2),nm=8*(batch/2+4)*sizeof(half2);
        const size_t combine=warps*std::min(64,get_cols_per_warp(cc))*(bc+4)*sizeof(half2);
        const size_t shared=std::max(combine,qreg?std::max(nq,nk+nm):nq+nk+nm);
        int available=0;
        error=cudaDeviceGetAttribute(&available,cudaDevAttrMaxSharedMemoryPerBlockOptin,device);
        if(error!=cudaSuccess)return int(error);
        if(shared>size_t(available))return int(cudaErrorNotSupported);
        error=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,int(shared));
        if(error!=cudaSuccess)return int(error);
        cached={device,warps,stages,shared};
    }
    // A null query performs capability/setup checks without a GPU launch.
    if(!query)return int(cudaSuccess);
    const auto warps=cached.warps;const auto shared=cached.shared;
    const int heads=requested_heads?requested_heads:kv_heads*8;
    history.gqa=heads/kv_heads;
    const int tiles=((queries+7)/8)*kv_heads*((history.gqa+7)/8);
    const bool packed=history.microblock_heads>1;
    const int row_stride=(packed?1:kv_heads)*width*2,head_stride=(packed?4:1)*width*2;
    fn<<<tiles,dim3(32,warps),shared,stream>>>(reinterpret_cast<const char*>(query),reinterpret_cast<const char*>(keys),
        reinterpret_cast<const char*>(values),reinterpret_cast<const char*>(mask),nullptr,indices,nullptr,output,nullptr,
        softmax_scale,0.f,1.f,1.f,0,0.f,
        width,init_fastdiv_values(queries),heads,1,heads*width*4,width*4,queries*heads*width*4,
        width,kv_rows,kv_heads,1,row_stride,head_stride,int64_t(kv_rows)*kv_heads*width*2,
        row_stride,head_stride,int64_t(kv_rows)*kv_heads*width*2,
        queries,1,1,kv_rows*2,queries*kv_rows*2,int64_t(queries)*kv_rows*2,history);
    const auto status=cudaGetLastError();
    if(status==cudaSuccess)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> reported{};
        if(!reported.exchange(true))std::fprintf(stderr,
            "native_qsa_sparse_pipeline actual_match=1 stages=%d separate_kv=%d shared_bytes=%zu\n",
            cached.stages,int(cached.stages>1),shared);
    }
    return int(status);
}

template <int ncols1, bool oob>
__launch_bounds__(256, 1)
static __global__ void prepare_sparse_mask_indices(
        half * mask_ptr, const int* inverse, const tyloinfer::qsa::SelectedBlock* selected_ptr,
        const int64_t* positions, int selected_count, int32_t * indices_ptr, int32_t * counts_ptr, const int ne30, const int n_queries,
        const int n_kv_max, const int64_t s31, const int64_t s33) {
    ggml_cuda_pdl_sync();

    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int values_per_lane = 8;
    using lane_mask_t = std::conditional_t<warp_size == 64, uint64_t, uint32_t>;
    constexpr uint64_t ballot_mask = ~(lane_mask_t)0;
    const auto popc = [](const lane_mask_t mask) {
        if constexpr (warp_size == 64) {
            return __popcll(mask);
        } else {
            return __popc(mask);
        }
    };
    const int tid      = threadIdx.x;
    const int warp     = tid / warp_size;
    const int lane     = tid % warp_size;
    const int sequence = blockIdx.y;
    const int group    = blockIdx.x;

    const int q0 = group*ncols1;
    const int q1 = min(q0 + ncols1, n_queries);

    // One byte carries all four causal visibility bits of a microblock.
    auto* mask=reinterpret_cast<uint8_t*>(mask_ptr)+sequence*(s33/4)+int64_t(q0)*(s31/4);
    int32_t * indices = indices_ptr + (int64_t(sequence)*gridDim.x + group)*n_kv_max;

    __shared__ int warp_offsets[256/warp_size];
    __shared__ int row_count;
    __shared__ int chunk_count;

    if (tid == 0) {
        row_count = 0;
    }
    __syncthreads();

    for (int i0 = 0; i0 < ne30/4; i0 += blockDim.x*values_per_lane) {
        lane_mask_t selected_warp[values_per_lane];
        int warp_count = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            bool selected = false;
            if (i < ne30/4) {
#pragma unroll
                for (int q = 0; q < ncols1; ++q) {
                    if (!oob || q < q1 - q0) {
                        const int token=q0+q;
                        const int slot=inverse[int64_t(token)*(ne30/4)+i];
                        unsigned visible=0;
                        if(slot>=0) {
                            const auto block=selected_ptr[int64_t(token)*selected_count+slot];
                            const int64_t left=positions[token]-int64_t(i)*4+1;
                            visible=block.token_mask&(left<=0?0u:left>=4?15u:(1u<<int(left))-1u);
                        }
                        mask[int64_t(q)*(s31/4)+i]=uint8_t(visible);
                        selected |= visible!=0;
                    }
                }
            }
            selected_warp[item] = (lane_mask_t)__ballot_sync(ballot_mask, (int) selected);
            warp_count += popc(selected_warp[item]);
        }

        if (lane == 0) {
            warp_offsets[warp] = warp_count;
        }
        __syncthreads();

        if (tid == 0) {
            int offset = 0;
#pragma unroll
            for (int iw = 0; iw < 256/warp_size; ++iw) {
                const int count = warp_offsets[iw];
                warp_offsets[iw] = offset;
                offset += count;
            }
            chunk_count = offset;
        }
        __syncthreads();

        const lane_mask_t lane_mask = lane == 0 ? 0 : ((lane_mask_t)1 << lane) - 1;
        int warp_item_offset = 0;
#pragma unroll
        for (int item = 0; item < values_per_lane; ++item) {
            const int i = i0 + (warp*values_per_lane + item)*warp_size + lane;
            const int dst = row_count + warp_offsets[warp] + warp_item_offset + popc(selected_warp[item] & lane_mask);
            if ((selected_warp[item] & ((lane_mask_t)1 << lane)) && 4*dst < n_kv_max) {
                for(int j=0;j<4;++j)indices[4*dst+j]=4*i+j;
            }
            warp_item_offset += popc(selected_warp[item]);
        }
        __syncthreads();

        if (tid == 0) {
            row_count += chunk_count;
        }
        __syncthreads();
    }

    const int count = min(4*row_count, n_kv_max);
    for (int i = count + tid; i < n_kv_max; i += blockDim.x) {
        indices[i] = -1;
    }
    if (tid == 0) {
        counts_ptr[int64_t(sequence)*gridDim.x + group] = count;
    }
    __syncthreads();

    // the dependent grid reads indices, signal once the row is complete
    ggml_cuda_pdl_lc();
}

using QsaSelectedBlock=tyloinfer::qsa::SelectedBlock;
static void qsa_llama_prepare(const void* selected,const int64_t* positions,const int* inverse,half* mask,int* indices,
        int queries,int rows,int count,cudaStream_t stream) {
    const int tiles=(queries+7)/8;
    prepare_sparse_mask_indices<8,true><<<dim3(tiles,1),256,0,stream>>>(mask,inverse,static_cast<const QsaSelectedBlock*>(selected),positions,count,
        indices,indices+int64_t(tiles)*rows,rows,queries,rows,rows,int64_t(queries)*rows);
}
__global__ void inverse_selected(const QsaSelectedBlock* selected,int* inverse,int nq,int count,int blocks) {
    for(auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<int64_t(nq)*count;i+=int64_t(gridDim.x)*blockDim.x) {
        const int id=selected[i].id;if(id>=0 && id<blocks)inverse[(i/count)*blocks+id]=i%count;
    }
}
static __global__ void initialize_inverse(int* inverse,int64_t count) {
    for(auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x)inverse[i]=-1;
}
static void qsa_llama_inverse(const void* selected,int* inverse,int nq,int count,int rows,cudaStream_t stream) {
    // This range is within the caller-checked scratch allocation.
    initialize_inverse<<<std::min(int64_t(4096),(int64_t(nq)*(rows/4)+255)/256),256,0,stream>>>(inverse,int64_t(nq)*(rows/4));
    inverse_selected<<<(int64_t(nq)*count+255)/256,256,0,stream>>>(static_cast<const QsaSelectedBlock*>(selected),inverse,nq,count,rows/4);
}

__global__ void qsa_llama_observe_mass(const QsaSelectedBlock* selected,const int64_t* positions,const float* block_log_mass,
    const float* log_norm,float* hits,float* mass,int heads,int count) {
    // Each warp reads adjacent blocks for one query head. Four warps share
    // the head reduction, keeping those reads coalesced while shortening
    // the serial head loop and exposing more independent block groups.
    constexpr int head_warps=4;
    const int lane=threadIdx.x,warp=threadIdx.y;
    const int slot=blockIdx.x*32+lane,token=blockIdx.y;
    QsaSelectedBlock block{};
    bool active=slot<count;
    if(active) {
        block=selected[int64_t(token)*count+slot];
        const auto visible=positions[token]-int64_t(block.id)*4+1;
        const auto causal_mask=visible<=0?0u:visible>=4?15u:(1u<<int(visible))-1;
        active=block.id>=0 && (block.token_mask&causal_mask);
    }
    float value=0;
    for (int h=warp;active && h<heads;h+=head_warps) {
        const auto row=int64_t(token)*heads+h;
        const float a=block_log_mass[row*count+slot],b=log_norm[row];
        if (isfinite(a) && isfinite(b))value+=__expf(a-b);
    }
    __shared__ float partial[head_warps][32];
    partial[warp][lane]=value;
    __syncthreads();
    if(!warp && active) {
        value=partial[0][lane]+partial[1][lane]+partial[2][lane]+partial[3][lane];
        atomicAdd(hits+block.id,1.f);atomicAdd(mass+block.id,value/heads);
    }
}

static void qsa_llama_observe(const void* selected,const int64_t* positions,const float* logmass,const float* norm,
        float* hits,float* mass,int queries,int heads,int count,cudaStream_t stream) {
    qsa_llama_observe_mass<<<dim3((count+31)/32,queries),dim3(32,4),0,stream>>>(static_cast<const QsaSelectedBlock*>(selected),positions,logmass,norm,hits,mass,heads,count);
}

static __device__ __forceinline__ float qsa_fa_read(const void* x,bool half_input,int64_t at) {
    return half_input?__half2float(static_cast<const half*>(x)[at]):static_cast<const float*>(x)[at];
}
__global__ void qsa_fa_rotate(tyloinfer::qsa::TqPrefillAttention a) {
    const int d=threadIdx.x,row=blockIdx.x,t=row/a.heads,h=row%a.heads;__shared__ float work[256];
    float x=d<a.width?qsa_fa_read(a.query,a.query_half,int64_t(t)*a.query_token_stride+int64_t(h)*a.query_head_stride+d)*a.key_signs[d]:0.f;
    for(int stride=1;stride<256;stride*=2) {
        if(stride<32){const float other=__shfl_xor_sync(0xffffffffu,x,stride);x=(d&stride)?other-x:x+other;}
        else{work[d]=x;__syncthreads();const float other=work[d^stride];x=(d&stride)?other-x:x+other;__syncthreads();}
    }
    a.rotated[int64_t(row)*256+d]=x*(1.f/16);
}
__global__ void qsa_fa_restore(tyloinfer::qsa::TqPrefillAttention a) {
    const int d=threadIdx.x,row=blockIdx.x,t=row/a.heads,h=row%a.heads;__shared__ float work[256];
    float x=a.temporary[int64_t(row)*256+d];
    for(int stride=1;stride<256;stride*=2) {
        if(stride<32){const float other=__shfl_xor_sync(0xffffffffu,x,stride);x=(d&stride)?other-x:x+other;}
        else{work[d]=x;__syncthreads();const float other=work[d^stride];x=(d&stride)?other-x:x+other;__syncthreads();}
    }
    if(d<a.width) {
        float v=x*a.value_signs[d]*(1.f/16);
        if(a.gate)v*=1.f/(1.f+expf(-qsa_fa_read(a.gate,a.gate_half,int64_t(t)*a.gate_token_stride+int64_t(h)*a.gate_head_stride+d)));
        if(a.output_half)static_cast<half*>(a.output)[int64_t(row)*a.width+d]=__float2half_rn(v);
        else static_cast<float*>(a.output)[int64_t(row)*a.width+d]=v;
    }
}
static int qsa_prefill_supported(const tyloinfer::qsa::TqPrefillAttention& a) {
    if(a.queries<32 || a.width<1 || a.width>256 || a.heads<1 || a.kv_heads<1 || a.heads%a.kv_heads || a.kv_rows<4 || a.kv_rows%4 || a.count<1)
        return int(cudaErrorNotSupported);
    if(tyloinfer::qsa::tq_prefill_attention_scratch(a.queries,a.heads,a.kv_rows)>a.temporary_bytes)
        return int(cudaErrorNotSupported);
    return qsa_llama_launch(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,a.queries,a.kv_heads,a.kv_rows,a.stream,{});
}
int tyloinfer::qsa::tq_sparse_attention_prefill(const TqPrefillAttention& a) {
    const auto supported=qsa_prefill_supported(a);
    if(supported!=int(cudaSuccess))return supported;
    const uint64_t blocks=a.kv_rows/4;
    const uint64_t output_bytes=uint64_t(a.queries)*a.heads*256*4,mask_bytes=(uint64_t(a.queries)*(a.kv_rows/4)+3)&~uint64_t(3);
    const uint64_t inverse_bytes=uint64_t(a.queries)*blocks*4;
    auto* bytes=reinterpret_cast<unsigned char*>(a.temporary);
    auto* mask=reinterpret_cast<half*>(bytes+output_bytes);
    auto* inverse=reinterpret_cast<int*>(bytes+output_bytes+mask_bytes);
    auto* ids=reinterpret_cast<int*>(bytes+output_bytes+mask_bytes+inverse_bytes);
    qsa_fa_rotate<<<a.queries*a.heads,256,0,a.stream>>>(a);
    qsa_llama_inverse(a.selected,inverse,a.queries,a.count,a.kv_rows,a.stream);
    qsa_llama_prepare(a.selected,a.positions,inverse,mask,ids,a.queries,a.kv_rows,a.count,a.stream);
    const auto status=qsa_llama_launch(a.rotated,reinterpret_cast<const half*>(a.keys),reinterpret_cast<const half*>(a.values),mask,ids,a.temporary,
        a.queries,a.kv_heads,a.kv_rows,a.stream,{inverse,a.logmass,a.norm,a.kv_rows/4,a.count,a.heads/a.kv_heads,a.kv_heads},
        a.heads,1.f/sqrtf(float(a.width)));
    if(status!=int(cudaSuccess))return status;
    qsa_fa_restore<<<a.queries*a.heads,256,0,a.stream>>>(a);
    if(a.hits && a.mass)qsa_llama_observe(a.selected,a.positions,a.logmass,a.norm,a.hits,a.mass,a.queries,a.heads,a.count,a.stream);
    if(const auto* flag=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");flag && flag[0]=='1') {
        static std::atomic<bool> printed{false};if(!printed.exchange(true)){std::fprintf(stderr,"native_qsa_tq_prefill_fa actual_match=1 queries=%d heads=%d kv_heads=%d width=%d kv_rows=%d scratch_reused=1 history=1 packed_mask=1 fused_selection=1\n",a.queries,a.heads,a.kv_heads,a.width,a.kv_rows);}
    }
    return int(cudaGetLastError());
}

void tyloinfer::qsa::TqPrefillGraph::reset() noexcept {
    if(!executable_)return;
    int previous=device_;const auto status=cudaGetDevice(&previous);
    if(status==cudaSuccess && previous!=device_)(void)cudaSetDevice(device_);
    (void)cudaGraphExecDestroy(executable_);executable_=nullptr;nodes_=0;
    if(status==cudaSuccess && previous!=device_)(void)cudaSetDevice(previous);
    device_=-1;
}
tyloinfer::qsa::TqPrefillGraph::~TqPrefillGraph(){reset();}
int tyloinfer::qsa::TqPrefillGraph::run(const TqPrefillAttention& a) {
    const auto available=qsa_prefill_supported(a);
    if(available!=int(cudaSuccess))return available;
    const auto fallback=[&](cudaError_t failure) {
        if(failure==cudaErrorNotSupported || failure==cudaErrorStreamCaptureUnsupported ||
           failure==cudaErrorStreamCaptureInvalidated || failure==cudaErrorMemoryAllocation) {
            reset();(void)cudaGetLastError();return tq_sparse_attention_prefill(a);
        }
        return int(failure);
    };
    cudaStreamCaptureStatus capture=cudaStreamCaptureStatusNone;
    auto error=cudaStreamIsCapturing(a.stream,&capture);
    if(error!=cudaSuccess)return int(error);
    if(!a.stream || capture!=cudaStreamCaptureStatusNone)return tq_sparse_attention_prefill(a);
    int device=0;error=cudaGetDevice(&device);if(error!=cudaSuccess)return int(error);
    if(executable_ && device_!=device)reset();
    device_=device;
    error=cudaStreamBeginCapture(a.stream,cudaStreamCaptureModeThreadLocal);
    if(error!=cudaSuccess)return fallback(error);
    const auto status=tq_sparse_attention_prefill(a);
    cudaGraph_t graph=nullptr;error=cudaStreamEndCapture(a.stream,&graph);
    if(status!=int(cudaSuccess) || error!=cudaSuccess) {
        if(graph)(void)cudaGraphDestroy(graph);
        return fallback(status!=int(cudaSuccess)?cudaError_t(status):error);
    }
    std::size_t nodes=0;error=cudaGraphGetNodes(graph,nullptr,&nodes);
    if(error!=cudaSuccess){(void)cudaGraphDestroy(graph);return fallback(error);}
    // Optional history changes the node topology, which exec-update cannot
    // modify. Rebuild directly instead of deliberately issuing a failed update.
    if(executable_ && nodes_!=nodes){reset();device_=device;}
    bool updated=false;
    if(executable_) {
#if CUDART_VERSION >= 12000
        cudaGraphExecUpdateResultInfo result{};
        error=cudaGraphExecUpdate(executable_,graph,&result);
        updated=error==cudaSuccess && result.result==cudaGraphExecUpdateSuccess;
#else
        cudaGraphNode_t failed=nullptr;cudaGraphExecUpdateResult result{};
        error=cudaGraphExecUpdate(executable_,graph,&failed,&result);
        updated=error==cudaSuccess && result==cudaGraphExecUpdateSuccess;
#endif
        if(error!=cudaSuccess && error!=cudaErrorGraphExecUpdateFailure){(void)cudaGraphDestroy(graph);return fallback(error);}
        if(!updated){(void)cudaGetLastError();reset();device_=device;}
    }
    if(!executable_)error=cudaGraphInstantiate(&executable_,graph,0);
    const auto destroyed=cudaGraphDestroy(graph);
    if(error!=cudaSuccess)return fallback(error);
    if(destroyed!=cudaSuccess)return int(destroyed);
    nodes_=nodes;
    error=cudaGraphLaunch(executable_,a.stream);
    if(error==cudaSuccess)if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::atomic<bool> first{false},update{false};
        auto& flag=updated?update:first;
        if(!flag.exchange(true))std::fprintf(stderr,"native_qsa_prefill_graph actual_match=1 updated=%d queries=%d\n",int(updated),a.queries);
    }
    return int(error);
}
