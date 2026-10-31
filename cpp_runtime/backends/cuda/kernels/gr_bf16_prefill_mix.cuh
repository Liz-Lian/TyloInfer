#pragma once

// Adapted from Strata's gr_upmix_cuda_kernel (src/prefill/kernels.cu), MIT:
// Copyright (c) 2026 Niko1221 and the Strata contributors.
// ../../../third_party/strata.LICENSE
// Keep the BF16 up projection and gated stream reduction in registers. Shape
// dimensions are runtime values; groups of four streams bound shared/register
// storage while arbitrary stream counts and matrix tails remain supported.
namespace {
constexpr int gr_mix_bm=128,gr_mix_bd=32,gr_mix_kc=32,gr_mix_ld=40,gr_mix_streams=4;
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__>=800
__device__ __forceinline__ void gr_mix_cp16(void* shared,const void* global) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
        "r"(unsigned(__cvta_generic_to_shared(shared))),"l"(global));
}
__device__ __forceinline__ void gr_mix_ldm4(unsigned (&r)[4],const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" :
        "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]):"r"(unsigned(__cvta_generic_to_shared(p))));
}
__device__ __forceinline__ void gr_mix_mma(float (&d)[4],const unsigned (&a)[4],unsigned b0,unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" :
        "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):
        "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__device__ __forceinline__ void gr_mix_stage8(uint16_t* destination,const uint16_t* source,int available) {
    if(available>=8 && !(reinterpret_cast<std::uintptr_t>(source)&15))gr_mix_cp16(destination,source);
    else {
        #pragma unroll
        for(int j=0;j<8;++j)destination[j]=j<available?source[j]:0;
    }
}
#endif
template<class Normalized,bool Reconstruct=false>
__global__ __launch_bounds__(256,2) void gr_bf16_prefill_mix_kernel(
        const uint16_t* __restrict__ input,const uint16_t* __restrict__ weight,
        const Normalized* __restrict__ normalized,float* __restrict__ output,
        int rows,int rank,int hidden,int streams,const float* norm_weight,const float* rms) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__>=800
    __shared__ __align__(16) uint16_t a_tile[2][gr_mix_bm][gr_mix_ld];
    __shared__ __align__(16) uint16_t b_tile[2][gr_mix_streams*gr_mix_bd][gr_mix_ld];
    const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,wm=warp>>1,wn=warp&1;
    const int64_t row0=int64_t(blockIdx.y)*gr_mix_bm;
    const int col0=int(blockIdx.x)*gr_mix_bd;
    float sums[2][2][4]{};
    for(int stream0=0;stream0<streams;stream0+=gr_mix_streams) {
        // As in Strata's up/mix, fetch the epilogue rows under the projection.
        // Column tiles run fastest, so adjacent blocks reuse these L2 lines.
        for(int i=tid;i<gr_mix_bm*gr_mix_streams;i+=256) {
            const auto row=row0+i/gr_mix_streams;
            const int stream=stream0+i%gr_mix_streams;
            if(row<rows && stream<streams) {
                const auto* p=normalized+(row*streams+stream)*hidden+col0;
                asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
            }
        }
        const auto stage=[&](int tile,int buffer) {
            for(int i=tid;i<gr_mix_bm*(gr_mix_kc/8);i+=256) {
                const int row=i/(gr_mix_kc/8),col=i%(gr_mix_kc/8)*8,k=tile*gr_mix_kc+col;
                const auto token=row0+row<rows?row0+row:int64_t(rows-1);
                const int available=rank-k<8?rank-k:8;
                const auto* source=available>0?input+token*rank+k:input;
                gr_mix_stage8(&a_tile[buffer][row][col],source,available>0?available:0);
            }
            for(int i=tid;i<gr_mix_streams*gr_mix_bd*(gr_mix_kc/8);i+=256) {
                const int row=i/(gr_mix_kc/8),col=i%(gr_mix_kc/8)*8,k=tile*gr_mix_kc+col;
                const int stream=stream0+row/gr_mix_bd,h=col0+row%gr_mix_bd;
                const int available=stream<streams && h<hidden?(rank-k<8?rank-k:8):0;
                const auto* source=available>0?weight+(int64_t(stream)*hidden+h)*rank+k:weight;
                gr_mix_stage8(&b_tile[buffer][row][col],source,available>0?available:0);
            }
            asm volatile("cp.async.commit_group;\n" ::);
        };
        float acc[2][gr_mix_streams][2][4]{};
        const int tiles=(rank+gr_mix_kc-1)/gr_mix_kc;
        stage(0,0);
        for(int tile=0;tile<tiles;++tile) {
            const int buffer=tile&1;
            if(tile+1<tiles) {
                stage(tile+1,buffer^1);
                asm volatile("cp.async.wait_group 1;\n" ::);
            } else asm volatile("cp.async.wait_group 0;\n" ::);
            __syncthreads();
            #pragma unroll
            for(int ks=0;ks<gr_mix_kc/16;++ks) {
                unsigned a[2][4];
                #pragma unroll
                for(int mt=0;mt<2;++mt)gr_mix_ldm4(a[mt],
                    &a_tile[buffer][wm*32+mt*16+(lane&15)][ks*16+(lane>>4)*8]);
                #pragma unroll
                for(int c=0;c<gr_mix_streams;++c) {
                    unsigned b[4];
                    gr_mix_ldm4(b,&b_tile[buffer][c*gr_mix_bd+wn*16+(lane&7)+((lane>>4)<<3)]
                        [ks*16+((lane>>3)&1)*8]);
                    #pragma unroll
                    for(int mt=0;mt<2;++mt) {
                        gr_mix_mma(acc[mt][c][0],a[mt],b[0],b[1]);
                        gr_mix_mma(acc[mt][c][1],a[mt],b[2],b[3]);
                    }
                }
            }
            __syncthreads();
        }
        #pragma unroll
        for(int mt=0;mt<2;++mt) {
            #pragma unroll
            for(int half=0;half<2;++half) {
                const int64_t row=row0+wm*32+mt*16+(lane>>2)+half*8;
                if(row>=rows)continue;
                #pragma unroll
                for(int column=0;column<2;++column) {
                    const int h=col0+wn*16+column*8+(lane&3)*2;
                    #pragma unroll
                    for(int c=0;c<gr_mix_streams;++c) {
                        const int stream=stream0+c;
                        if(stream>=streams)break;
                        #pragma unroll
                        for(int j=0;j<2;++j)if(h+j<hidden) {
                            const auto at=row*int64_t(hidden)*streams+int64_t(stream)*hidden+h+j;
                            const float gate=1.f/(1.f+expf(-acc[mt][c][column][half*2+j]));
                            float value=float(normalized[at]);
                            if constexpr(Reconstruct)value=__fmul_rn(__fmul_rn(value,rms[row*streams+stream]),
                                __fadd_rn(norm_weight[int64_t(stream)*hidden+h+j],1.f));
                            // Match the existing mix's separate rounding and
                            // canonical stream order, including stream groups.
                            sums[mt][column][half*2+j]=__fadd_rn(sums[mt][column][half*2+j],
                                __fmul_rn(gate,value));
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for(int mt=0;mt<2;++mt) {
        #pragma unroll
        for(int half=0;half<2;++half) {
            const int64_t row=row0+wm*32+mt*16+(lane>>2)+half*8;
            if(row>=rows)continue;
            #pragma unroll
            for(int column=0;column<2;++column) {
                const int h=col0+wn*16+column*8+(lane&3)*2;
                #pragma unroll
                for(int j=0;j<2;++j)if(h+j<hidden)
                    output[row*hidden+h+j]=sums[mt][column][half*2+j]/streams;
            }
        }
    }
#endif
}
mfq_tensor_backend::Tensor try_gr_bf16_prefill_mix(const mfq_tensor_backend::Tensor& input,
        const mfq_tensor_backend::Tensor& normalized,const mfq_tensor_backend::Tensor& weight,int64_t streams,
        const mfq_tensor_backend::Tensor& norm_weight={},const mfq_tensor_backend::Tensor& rms={}) {
    namespace tb=mfq_tensor_backend;
    const auto* setting=std::getenv("TYLOINFER_GR_PREFILL_BF16");
    if(setting && setting[0]=='0')return {};
    if(!input.defined() || !weight.defined() || !normalized.defined() || input.dim()<1 ||
            streams<1 || streams>INT_MAX-gr_mix_streams || !input.is_cuda() || !weight.is_cuda() ||
            !normalized.is_cuda() || input.device()!=weight.device() || input.device()!=normalized.device() ||
            (input.scalar_type()!=tb::kFloat32 && input.scalar_type()!=tb::kBFloat16) ||
            (normalized.scalar_type()!=tb::kFloat32 && normalized.scalar_type()!=tb::kFloat16) ||
            weight.scalar_type()!=tb::kBFloat16 || weight.dim()!=2 || input.size(-1)<=0 ||
            input.size(-1)>INT_MAX-gr_mix_kc || weight.size(1)!=input.size(-1) || weight.size(0)<streams || weight.size(0)<16 ||
            weight.size(0)>INT_MAX || weight.size(0)%streams)return {};
    const auto rows=input.numel()/input.size(-1);
    // Below 16 output columns the existing adapter retains FP32 operands;
    // this BF16 MMA path must preserve that precision choice.
    if(rows<=32 || rows>int64_t(65535)*gr_mix_bm)return {};
    auto expected=input.sizes().vec();expected.back()=weight.size(0);
    if(normalized.sizes().vec()!=expected)return {};
    if(norm_weight.defined() && (!rms.defined() || !norm_weight.is_cuda() || !rms.is_cuda() ||
            norm_weight.device()!=input.device() || rms.device()!=input.device() ||
            norm_weight.scalar_type()!=tb::kFloat32 || rms.scalar_type()!=tb::kFloat32 ||
            norm_weight.numel()!=weight.size(0) || rms.numel()!=rows*streams))return {};
    const MfqCudaGuard guard(input.get_device());
    int major=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,input.get_device()));
    // cp.async, ldmatrix and native BF16 MMA require compute capability 8.0.
    if(major<8)return {};
    auto x=input.to(tb::kBFloat16).contiguous(),w=weight.contiguous(),norm=normalized.contiguous();
    auto scale=norm_weight.defined()?norm_weight.contiguous():tb::Tensor{};
    auto inverse=rms.defined()?rms.contiguous():tb::Tensor{};
    auto shape=input.sizes().vec();shape.back()=weight.size(0)/streams;
    auto output=tb::empty(shape,input.options().dtype(tb::kFloat32));
    const auto stream=mfq_current_cuda_stream();
    for(const auto& tensor:{x,w,norm,output,scale,inverse})if(tensor.defined())
        tensor.record_stream(reinterpret_cast<std::uintptr_t>(stream));
    const auto launch=[&](auto value,auto reconstruct) {
        using Value=decltype(value);
        constexpr bool rebuild=decltype(reconstruct)::value;
        gr_bf16_prefill_mix_kernel<Value,rebuild><<<dim3(unsigned((shape.back()+gr_mix_bd-1)/gr_mix_bd),
            unsigned((rows+gr_mix_bm-1)/gr_mix_bm)),256,0,stream>>>(static_cast<const uint16_t*>(x.data_ptr()),
            static_cast<const uint16_t*>(w.data_ptr()),norm.data_ptr<Value>(),output.data_ptr<float>(),
            int(rows),int(weight.size(1)),int(shape.back()),int(streams),
            scale.defined()?scale.data_ptr<float>():nullptr,inverse.defined()?inverse.data_ptr<float>():nullptr);
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"gr_bf16_prefill_up_mix",
            reinterpret_cast<const void*>(gr_bf16_prefill_mix_kernel<Value,rebuild>),
            "M="+std::to_string(rows)+" hidden="+std::to_string(shape.back())+" K="+std::to_string(weight.size(1))+
            " streams="+std::to_string(streams)+" materialized_projection=0 reconstruct_norm="+std::to_string(rebuild));
    };
    const auto types=[&](auto reconstruct) {
        if(norm.scalar_type()==tb::kFloat16)launch(__half{},reconstruct);else launch(float{},reconstruct);
    };
    if(scale.defined())types(std::true_type{});else types(std::false_type{});
    TYLOINFER_CUDA_CHECK(cudaGetLastError());
    if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
        static std::once_flag reported;
        std::call_once(reported,[&] {std::fprintf(stderr,
            "native_gr_prefill_up_mix actual_match=1 M=%lld hidden=%lld K=%lld streams=%lld materialized_projection=0\n",
            static_cast<long long>(rows),static_cast<long long>(shape.back()),
            static_cast<long long>(weight.size(1)),static_cast<long long>(streams));});
    }
    return output;
}
}
