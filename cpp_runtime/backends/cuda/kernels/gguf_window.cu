#include "gguf_window.h"
#include "gguf.h"
#include "gguf_compact.h"
#include "ggml-cuda/common.cuh"
#include "gguf_decode.cuh"
#include "gguf_dot.cuh"
#include "gguf_dot_q5_group32.cuh"
#include "gguf_quantize.cuh"
#include "gguf_dense_rows.cuh"
#include "kernel_dispatch_trace.h"
#include <climits>
#include <stdexcept>

namespace mfq::cuda::gguf {
namespace {
// Retire one token's decoded activation after the output rows consume it.
// The ordinary M=1..4 prepared dots and reduction order remain unchanged.
struct Iq4WindowPrepared : PreparedDot<GGML_TYPE_IQ4_XS> {
    struct Activation { int codes[8]; float scale; };
    __device__ static Activation load_activation(const block_q8_1* input, int part) {
        Activation a;
        const auto* x = input + part / 4;
        #pragma unroll
        for (int j = 0; j < 8; ++j) a.codes[j] = get_int_b4(x->qs, j);
        a.scale = __low2float(x->ds);
        return a;
    }
    __device__ static float apply_activation(const Weight& w, const Activation& a) {
        int sum = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            sum = ggml_cuda_dp4a(w.codes[j].x, a.codes[j], sum);
            sum = ggml_cuda_dp4a(w.codes[j].y, a.codes[j + 4], sum);
        }
        sum *= w.scale_code;
        return (w.scale * a.scale) * sum;
    }
};

// Broadcast compressed words directly to the eight lanes sharing a Q5
// block. This avoids peer shuffles while preserving canonical GGUF bytes.
struct DirectQ5Group : Q5Group32Prepared {
    __device__ static Weight load(const void* data,int block,int group) {
        const auto* b=static_cast<const block_q5_K*>(data)+block;
        const auto* low=b->qs+32*(group/2);
        const auto a=__ldg(reinterpret_cast<const uint4*>(low));
        const auto c=__ldg(reinterpret_cast<const uint4*>(low+16));
        const auto h0=__ldg(reinterpret_cast<const uint4*>(b->qh));
        const auto h1=__ldg(reinterpret_cast<const uint4*>(b->qh+16));
        const unsigned lo[8]={a.x,a.y,a.z,a.w,c.x,c.y,c.z,c.w};
        const unsigned hi[8]={h0.x,h0.y,h0.z,h0.w,h1.x,h1.y,h1.z,h1.w};
        Weight w;w.group=group;
        #pragma unroll
        for(int j=0;j<8;++j)
            w.codes[j]=int((lo[j]>>((group&1)*4))&0x0f0f0f0fu)|
                int(((hi[j]>>group)<<4)&0x10101010u);
        if(group<4){w.scale=b->scales[group]&63;w.minimum=b->scales[group+4]&63;}
        else {
            w.scale=(b->scales[group+4]&15)|((b->scales[group-4]>>6)<<4);
            w.minimum=(b->scales[group+4]>>4)|((b->scales[group]>>6)<<4);
        }
        w.d=__half22float2(b->dm);
        return w;
    }
};

// Eight adjacent lanes consume one canonical 256-value Q6 block. Its
// 210-byte stride only guarantees halfword alignment. Gather the compressed
// words once per lane and exchange them within each eight-lane group.
struct CooperativeQ6Group : M1PreparedDot<GGML_TYPE_Q6_K> {
    __device__ static Weight load(const void* data,int block,int part) {
        const auto* b=static_cast<const block_q6_K*>(data)+block;
        const auto mask=__activemask();
        const auto* low=b->ql+part*16;
        const unsigned a=iq_weight_word<true>(low,0),c=iq_weight_word<true>(low,1);
        const unsigned d=iq_weight_word<true>(low,2),e=iq_weight_word<true>(low,3);
        const auto* high=b->qh+part*8;
        const unsigned h0=iq_weight_word<true>(high,0),h1=iq_weight_word<true>(high,1);
        const int low_owner=(part/4)*4+(part&1)*2,high_owner=(part/4)*4;
        Weight w;
        #pragma unroll
        for(int j=0;j<8;++j) {
            const unsigned own=j%4==0?a:j%4==1?c:j%4==2?d:e;
            const unsigned lo=__shfl_sync(mask,own,low_owner+j/4,8);
            const unsigned hi=__shfl_sync(mask,j%2?h1:h0,high_owner+j/2,8);
            const unsigned code=((lo>>(4*((part/2)&1)))&0x0f0f0f0fu)|
                (((hi>>(2*(part&3)))<<4)&0x30303030u);
            w.codes[j]=int((code+0x60606060u)^0x80808080u);
        }
        const unsigned scales=__ldg(reinterpret_cast<const unsigned short*>(b->scales)+part);
        w.scale0=int(static_cast<signed char>(scales));
        w.scale1=int(static_cast<signed char>(scales>>8));
        w.dm=make_float2(__half2float(__ldg(&b->d)),0.f);
        return w;
    }
};
template<int Format,class Base>struct SplitWindow:Base{
 __device__ static typename Base::Activation load_window_activation(const block_q8_1* input,int columns,int token,int block,int part){
  typename Base::Activation a{};
  const unsigned groups=unsigned(columns/32),group=unsigned(block)*(Base::qk/32)+(Format==GGML_TYPE_IQ4_XS?unsigned(part/4):unsigned(part));
  const unsigned packet=unsigned(token)*groups+group;
  const auto* low=static_cast<const uint8_t*>(static_cast<const void*>(input));
  const auto* high=low+uint64_t(columns)*8/2;uint4 x,y;
  asm("{ .reg .u64 p; mad.wide.u32 p,%5,16,%4; ld.global.nc.v4.u32 {%0,%1,%2,%3},[p]; }":"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(low),"r"(packet));
  asm("{ .reg .u64 p; mad.wide.u32 p,%5,16,%4; ld.global.nc.v4.u32 {%0,%1,%2,%3},[p]; }":"=r"(y.x),"=r"(y.y),"=r"(y.z),"=r"(y.w):"l"(high),"r"(packet));
  a.codes[0]=x.x;a.codes[1]=x.y;a.codes[2]=x.z;a.codes[3]=x.w;a.codes[4]=y.x;a.codes[5]=y.y;a.codes[6]=y.z;a.codes[7]=y.w;
  if constexpr(Format==GGML_TYPE_Q5_K){
   const auto helper=__ldg(reinterpret_cast<const uint2*>(low+uint64_t(groups)*8*36)+packet);
   a.d=__half2float(__ushort_as_half(helper.x&65535));a.ones=int(int16_t(helper.y))+int(int16_t(helper.y>>16));
  }else a.scale=__low2float(__ldg(reinterpret_cast<const half2*>(low+uint64_t(columns)*8)+packet));
  return a;
 }
};

}

bool project_dense_window_q8(ggml_type type, const void* packed, std::uint64_t stride,
        std::int64_t rows, std::int64_t columns, const void* input, std::int64_t tokens,
        void* output, bool half_output, cudaStream_t stream) {
    if ((type != GGML_TYPE_IQ4_XS && type != GGML_TYPE_Q5_K && type != GGML_TYPE_Q6_K && type!=GGML_TYPE_Q4_K) ||
            tokens < (type==GGML_TYPE_Q4_K?2:5) || tokens > 8 || rows < 1 || rows > INT_MAX / tokens ||
            columns < 256 || columns > INT_MAX || columns % 256 ||
            stride > INT_MAX)
        return false;
    if (!packed || !input || !output || stride < mfq::gguf::row_bytes(type, columns))
        throw std::invalid_argument("Invalid dense Q8 window storage");
    // Q5's coalesced four-word gathers require aligned source rows.
    // The existing scalar reader handles external unaligned views.
    if (type == GGML_TYPE_Q5_K && ((reinterpret_cast<std::uintptr_t>(packed) | stride) & 15u))
        return false;
    if(type==GGML_TYPE_Q4_K) {
        // Whole Q8 groups use M1's integer dot and affine correction before
        // FP32 scaling. Reuse each decoded group for the entire token window.
        // The u32 code gathers require four-byte weight/row alignment.
        if((reinterpret_cast<std::uintptr_t>(packed)|stride)&3u)return false;
        const auto count=[&]<int Tokens>() {
            const auto function=project_dense_cooperative_q8<GGML_TYPE_Q4_K,Tokens,2,1,M1PreparedDot<GGML_TYPE_Q4_K>,true>;
            function<<<unsigned((rows+1)/2),dim3(32,1),0,stream>>>(
                static_cast<const std::uint8_t*>(packed),int(stride),int(rows),int(columns),int(tokens),
                static_cast<const block_q8_1*>(input),output,half_output);
            if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_dense_affine_window",
                reinterpret_cast<const void*>(function),"format="+std::to_string(type)+
                " M="+std::to_string(tokens)+" affine_group=32 weight_reuse=1 k_warps=1");
        };
        if(tokens==2)count.template operator()<2>();else if(tokens==3)count.template operator()<3>();
        else if(tokens==4)count.template operator()<4>();else count.template operator()<8>();
        const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
        return true;
    }
    const auto launch = [&]<int Format, class Prepared, int Warps, int Rows=2, bool WarpFirst=false>() {
        const auto function = project_dense_cooperative_q8<Format,8,Rows,Warps,Prepared,WarpFirst>;
        function<<<unsigned((rows + Rows - 1) / Rows),dim3(32,Warps),0,stream>>>(
            static_cast<const std::uint8_t*>(packed), int(stride), int(rows), int(columns),
            int(tokens), static_cast<const block_q8_1*>(input), output, half_output);
        if (dispatch_trace::enabled()) dispatch_trace::record(stream, "gguf_dense_window",
            reinterpret_cast<const void*>(function), "format=" + std::to_string(type) +
            " M=" + std::to_string(tokens) + " output_rows=" + std::to_string(Rows) +
            " activation_reuse=1 warp_first=" + std::to_string(WarpFirst) +
            " cooperative_weight_gather=" + std::to_string(Format==GGML_TYPE_Q6_K) +
            " warps=" + std::to_string(Warps));
    };
    if (type == GGML_TYPE_IQ4_XS) {
        const int warps=canonical_iq4_k_warps(columns);
        if(warps==3)launch.template operator()<GGML_TYPE_IQ4_XS,Iq4WindowPrepared,3,4,true>();
        else if(warps==4)launch.template operator()<GGML_TYPE_IQ4_XS,Iq4WindowPrepared,4,4,true>();
        else launch.template operator()<GGML_TYPE_IQ4_XS,Iq4WindowPrepared,1,4,true>();
    }
    // Cover a whole Q8 group before applying scales, reusing the same
    // contiguous decoder as M=1. Eight-value fragments repeat metadata and
    // normalization four times across a full verification window.
    else if (type == GGML_TYPE_Q6_K) launch.template operator()<GGML_TYPE_Q6_K,CooperativeQ6Group,4>();
    else {
        const auto parts = columns / 32;
        int warps = parts <= 32 ? 1 : 2;
        auto work = ((parts + 32 * warps - 1) / (32 * warps)) * warps;
        for (int candidate = warps + 1; candidate <= 4; ++candidate) {
            const auto cost = ((parts + 32 * candidate - 1) / (32 * candidate)) * candidate;
            if (cost <= work) { work = cost; warps = candidate; }
        }
        const auto q5 = [&]<int Warps>() {
            if (tokens == 8) launch.template operator()<GGML_TYPE_Q5_K,DirectQ5Group,Warps,4,true>();
            else launch.template operator()<GGML_TYPE_Q5_K,Q5Group32Prepared,Warps>();
        };
        if (warps == 1) q5.template operator()<1>();
        else if (warps == 2) q5.template operator()<2>();
        else if (warps == 3) q5.template operator()<3>();
        else q5.template operator()<4>();
    }
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    return true;
}
bool supports_dense_split_window_q8(ggml_type type,const void* packed,std::uint64_t stride,
        std::int64_t rows,std::int64_t columns,std::int64_t tokens) noexcept {
    if((type!=GGML_TYPE_Q5_K && type!=GGML_TYPE_Q6_K) || tokens!=8 || !packed ||
            rows<1 || rows>INT_MAX/8 || columns<256 || columns>INT_MAX || columns%256 ||
            stride>INT_MAX || stride<mfq::gguf::row_bytes(type,columns))return false;
    return type!=GGML_TYPE_Q5_K || !((reinterpret_cast<std::uintptr_t>(packed)|stride)&15u);
}

bool project_dense_split_window_q8(ggml_type type,const void* packed,std::uint64_t stride,
        std::int64_t rows,std::int64_t columns,const void* input,std::int64_t tokens,
        void* output,bool half_output,cudaStream_t stream) {
    if(!supports_dense_split_window_q8(type,packed,stride,rows,columns,tokens))return false;
    if(!input || !output)throw std::invalid_argument("Dense split-window activation/output missing");
    const auto launch=[&]<int Format,class Base,int Warps,int Rows,bool WarpFirst>() {
        const auto function=project_dense_cooperative_q8<Format,8,Rows,Warps,SplitWindow<Format,Base>,WarpFirst>;
        function<<<unsigned((rows+Rows-1)/Rows),dim3(32,Warps),0,stream>>>(
            static_cast<const uint8_t*>(packed),int(stride),int(rows),int(columns),int(tokens),
            static_cast<const block_q8_1*>(input),output,half_output);
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_dense_split_window",reinterpret_cast<const void*>(function),
            "format="+std::to_string(type)+" M=8 split_q8_planes=1 cached_integer_sum="+std::to_string(Format==GGML_TYPE_Q5_K)+" warps="+std::to_string(Warps));
    };
    if(type==GGML_TYPE_Q6_K)launch.template operator()<GGML_TYPE_Q6_K,CooperativeQ6Group,4,2,false>();
    else {
        const auto parts=columns/32;int warps=parts<=32?1:2;
        auto work=((parts+32*warps-1)/(32*warps))*warps;
        for(int candidate=warps+1;candidate<=4;++candidate){const auto cost=((parts+32*candidate-1)/(32*candidate))*candidate;if(cost<=work){work=cost;warps=candidate;}}
        const auto q5=[&]<int Warps>(){launch.template operator()<GGML_TYPE_Q5_K,DirectQ5Group,Warps,4,true>();};
        if(warps==1)q5.template operator()<1>();else if(warps==2)q5.template operator()<2>();else if(warps==3)q5.template operator()<3>();else q5.template operator()<4>();
    }
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));return true;
}
}
