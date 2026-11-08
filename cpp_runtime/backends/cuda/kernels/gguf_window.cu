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
template<bool Streaming=false,bool Aligned8=false>
struct Iq4WindowPrepared : M1PreparedDot<GGML_TYPE_IQ4_XS> {
    using Base=M1PreparedDot<GGML_TYPE_IQ4_XS>;
    __device__ static Weight load(const void* data,int block,int part) {
        if constexpr(Aligned8)return Base::template load_aligned_window<Streaming>(data,block,part);
        else if constexpr(Streaming)return Base::load_streaming(data,block,part);
        else return Base::load(data,block,part);
    }
};

bool stream_iq4_window(std::int64_t columns) {
    // Two complete four-warp K tiles produce a wider input
    // working set benefits from retaining activations instead of weight
    // fragments. Two output rows reduce the long window's register load.
    constexpr int groups_per_tile=4*32;
    return columns/32>=2*groups_per_tile;
}

// Q6 uses the M=1 aligned-word decoder for every window. Reconstructing
// each lane's words avoids exchanging compressed payloads between lanes.
template<int Format,class Base,int InputTokens=8,bool ReuseRowActivations=false>struct SplitWindow:Base{
 static constexpr int exact_window_tokens=InputTokens;
 __device__ static typename Base::Activation load_window_activation(const block_q8_1* input,int columns,int token,int block,int part){
  typename Base::Activation a{};
  const unsigned groups=unsigned(columns/32),group=unsigned(block)*(Base::qk/32)+(Format==GGML_TYPE_IQ4_XS?unsigned(part/4):unsigned(part));
  const unsigned packet=unsigned(token)*groups+group;
  const auto* low=static_cast<const uint8_t*>(static_cast<const void*>(input));
  const auto* high=low+uint64_t(columns)*InputTokens/2;uint4 x,y;
  if constexpr(ReuseRowActivations) {
   // Expose the read-only packets to common-subexpression elimination
   // across output rows. Opaque address assembly repeats these loads.
   x=__ldg(reinterpret_cast<const uint4*>(low)+packet);
   y=__ldg(reinterpret_cast<const uint4*>(high)+packet);
  }else {
  asm("{ .reg .u64 p; mad.wide.u32 p,%5,16,%4; ld.global.nc.v4.u32 {%0,%1,%2,%3},[p]; }":"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(low),"r"(packet));
  asm("{ .reg .u64 p; mad.wide.u32 p,%5,16,%4; ld.global.nc.v4.u32 {%0,%1,%2,%3},[p]; }":"=r"(y.x),"=r"(y.y),"=r"(y.z),"=r"(y.w):"l"(high),"r"(packet));
  }
  a.codes[0]=x.x;a.codes[1]=x.y;a.codes[2]=x.z;a.codes[3]=x.w;a.codes[4]=y.x;a.codes[5]=y.y;a.codes[6]=y.z;a.codes[7]=y.w;
  if constexpr(Format==GGML_TYPE_Q5_K || Format==GGML_TYPE_Q4_K){
   const auto helper=__ldg(reinterpret_cast<const uint2*>(low+uint64_t(groups)*InputTokens*36)+packet);
   if constexpr(Format==GGML_TYPE_Q5_K) {
    a.d=__half2float(__ushort_as_half(helper.x&65535));a.ones=int(int16_t(helper.y))+int(int16_t(helper.y>>16));
   }else {
    a.scale=__half2float(__ushort_as_half(helper.x&65535));
    a.lo_sum=int(int16_t(helper.y));a.hi_sum=int(int16_t(helper.y>>16));
   }
  }else a.scale=__low2float(__ldg(reinterpret_cast<const half2*>(low+uint64_t(columns)*InputTokens)+packet));
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
            " weight_first=" + std::to_string(Format==GGML_TYPE_Q5_K && Rows>2 && WarpFirst) +
            " streaming_weight_loads=" + std::to_string(std::is_same_v<Prepared,DirectQ5Group> ||
                std::is_same_v<Prepared,Iq4WindowPrepared<true>> ||
                std::is_same_v<Prepared,Iq4WindowPrepared<true,true>>) +
            " aligned_weight_vector=" + std::to_string(std::is_same_v<Prepared,Iq4WindowPrepared<true,true>> ||
                std::is_same_v<Prepared,Iq4WindowPrepared<false,true>>) +
            " direct_weight_gather=" + std::to_string(Format==GGML_TYPE_Q6_K) +
            " warps=" + std::to_string(Warps));
    };
    if (type == GGML_TYPE_IQ4_XS) {
        const int warps=canonical_iq4_k_warps(columns);
        const auto iq4=[&]<int Warps,bool Aligned>() {
            if(stream_iq4_window(columns))
                launch.template operator()<GGML_TYPE_IQ4_XS,Iq4WindowPrepared<true,Aligned>,Warps,2,true>();
            else launch.template operator()<GGML_TYPE_IQ4_XS,Iq4WindowPrepared<false,Aligned>,Warps,4,true>();
        };
        const auto alignment=[&]<bool Aligned>() {
            if(warps==3)iq4.template operator()<3,Aligned>();
            else if(warps==4)iq4.template operator()<4,Aligned>();
            else iq4.template operator()<1,Aligned>();
        };
        if((reinterpret_cast<std::uintptr_t>(packed)|stride)&7u)alignment.template operator()<false>();
        else alignment.template operator()<true>();
    }
    // Cover a whole Q8 group before applying scales, reusing the same
    // contiguous decoder as M=1. Eight-value fragments repeat metadata and
    // normalization four times across a full verification window.
    else if (type == GGML_TYPE_Q6_K) launch.template operator()<GGML_TYPE_Q6_K,M1PreparedDot<GGML_TYPE_Q6_K>,4,2,true>();
    else {
        const int warps=whole_q8_k_warps(columns);
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
    const auto block_columns=type==GGML_TYPE_Q8_0 || type==GGML_TYPE_IQ4_NL?32:256;
    if((type!=GGML_TYPE_Q5_K && type!=GGML_TYPE_Q6_K && type!=GGML_TYPE_Q4_K && type!=GGML_TYPE_IQ4_XS &&
            type!=GGML_TYPE_Q3_K && type!=GGML_TYPE_IQ4_NL && type!=GGML_TYPE_Q8_0) ||
            tokens<5 || tokens>8 || !packed ||
            rows<1 || rows>INT_MAX/tokens || columns<block_columns || columns>INT_MAX || columns%block_columns ||
            stride>INT_MAX || stride<mfq::gguf::row_bytes(type,columns))return false;
    // The shared producer writes a uint2 helper per Q8 block. An odd
    // number of 32-value packets puts that plane at four-byte alignment;
    // those small external matrices retain the native Q8 packet instead.
    if((std::uint64_t(columns)*std::uint64_t(tokens))&63u)return false;
    const auto alignment=type==GGML_TYPE_Q5_K?15u:(type==GGML_TYPE_Q4_K || type==GGML_TYPE_IQ4_XS)?3u:1u;
    return !((reinterpret_cast<std::uintptr_t>(packed)|stride)&alignment);
}

bool project_dense_split_window_q8(ggml_type type,const void* packed,std::uint64_t stride,
        std::int64_t rows,std::int64_t columns,const void* input,std::int64_t tokens,
        void* output,bool half_output,cudaStream_t stream) {
    if(!supports_dense_split_window_q8(type,packed,stride,rows,columns,tokens))return false;
    if(!input || !output)throw std::invalid_argument("Dense split-window activation/output missing");
    const auto launch=[&]<int Count,int Format,class Base,int Warps,int Rows,bool WarpFirst>() {
        const auto function=project_dense_cooperative_q8<Format,Count,Rows,Warps,
            SplitWindow<Format,Base,Count,(Format==GGML_TYPE_Q5_K && Rows>2 && WarpFirst)>,WarpFirst>;
        function<<<unsigned((rows+Rows-1)/Rows),dim3(32,Warps),0,stream>>>(
            static_cast<const uint8_t*>(packed),int(stride),int(rows),int(columns),int(tokens),
            static_cast<const block_q8_1*>(input),output,half_output);
        if(dispatch_trace::enabled())dispatch_trace::record(stream,"gguf_dense_split_window",reinterpret_cast<const void*>(function),
            "format="+std::to_string(type)+" M="+std::to_string(tokens)+" output_rows="+std::to_string(Rows)+
            " weight_first="+std::to_string(Format==GGML_TYPE_Q5_K && Rows>2 && WarpFirst)+
            " streaming_weight_loads="+std::to_string(Format==GGML_TYPE_Q5_K ||
                std::is_same_v<Base,Iq4WindowPrepared<true>> || std::is_same_v<Base,Iq4WindowPrepared<true,true>>)+
            " aligned_weight_vector="+std::to_string(std::is_same_v<Base,Iq4WindowPrepared<true,true>> ||
                std::is_same_v<Base,Iq4WindowPrepared<false,true>>)+" split_q8_planes=1 exact_token_tile=1 cached_integer_sum="+
            std::to_string(Format==GGML_TYPE_Q5_K || Format==GGML_TYPE_Q4_K)+" warps="+std::to_string(Warps));
    };
    const auto count=[&]<int Count>() {
    if(type==GGML_TYPE_Q8_0 || type==GGML_TYPE_Q3_K || type==GGML_TYPE_IQ4_NL) {
        const auto groups=columns/32,needed=(groups+31)/32;
        const int warps=groups<64?1:needed<4?int(needed):4;
        const auto format=[&]<int Format>() {
            const auto run=[&]<int Warps>() {
                launch.template operator()<Count,Format,WholeWindowDot<Format>,Warps,2,true>();
            };
            if(warps==1)run.template operator()<1>();else if(warps==2)run.template operator()<2>();
            else if(warps==3)run.template operator()<3>();else run.template operator()<4>();
        };
        if(type==GGML_TYPE_Q8_0)format.template operator()<GGML_TYPE_Q8_0>();
        else if(type==GGML_TYPE_Q3_K)format.template operator()<GGML_TYPE_Q3_K>();
        else format.template operator()<GGML_TYPE_IQ4_NL>();
    }
    else if(type==GGML_TYPE_Q6_K)launch.template operator()<Count,GGML_TYPE_Q6_K,M1PreparedDot<GGML_TYPE_Q6_K>,4,2,true>();
    // Four output rows amortize activation reads across the wider window.
    // Count<=8 keeps the warp reduction within its 32-output capacity.
    else if(type==GGML_TYPE_Q4_K)launch.template operator()<Count,GGML_TYPE_Q4_K,M1PreparedDot<GGML_TYPE_Q4_K>,1,4,true>();
    else if(type==GGML_TYPE_IQ4_XS) {
        const int warps=canonical_iq4_k_warps(columns);
        const auto iq4=[&]<int Warps,bool Aligned>() {
            if(stream_iq4_window(columns))
                launch.template operator()<Count,GGML_TYPE_IQ4_XS,Iq4WindowPrepared<true,Aligned>,Warps,2,true>();
            else launch.template operator()<Count,GGML_TYPE_IQ4_XS,Iq4WindowPrepared<false,Aligned>,Warps,4,true>();
        };
        const auto alignment=[&]<bool Aligned>() {
            if(warps==3)iq4.template operator()<3,Aligned>();
            else if(warps==4)iq4.template operator()<4,Aligned>();
            else iq4.template operator()<1,Aligned>();
        };
        if((reinterpret_cast<std::uintptr_t>(packed)|stride)&7u)alignment.template operator()<false>();
        else alignment.template operator()<true>();
    }else {
        const int warps=whole_q8_k_warps(columns);
        const auto q5=[&]<int Warps>(){
            if constexpr(Count==8)launch.template operator()<Count,GGML_TYPE_Q5_K,DirectQ5Group,Warps,4,true>();
            else launch.template operator()<Count,GGML_TYPE_Q5_K,DirectQ5Group,Warps,2,false>();
        };
        if(warps==1)q5.template operator()<1>();else if(warps==2)q5.template operator()<2>();else if(warps==3)q5.template operator()<3>();else q5.template operator()<4>();
    }
    };
    if(tokens==5)count.template operator()<5>();else if(tokens==6)count.template operator()<6>();
    else if(tokens==7)count.template operator()<7>();else count.template operator()<8>();
    const auto status=cudaGetLastError();if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));return true;
}
}
