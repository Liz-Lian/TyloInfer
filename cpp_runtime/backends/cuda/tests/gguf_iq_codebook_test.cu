#include "kernels/gguf.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-cuda/common.cuh"
#include "kernels/gguf_decode.cuh"
#include "kernels/gguf_dot.cuh"
#include <cstdio>
#include <stdexcept>
namespace mfq::cuda::gguf {namespace {
template<int Format,int Count,int Signs,bool Raw=false>
__global__ void selector_tables_test(unsigned* errors) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=Count*Signs)return;
    const int code=i/Signs;unsigned sign=i%Signs;
    unsigned words[2];uint64_t reference;
    if constexpr(Format==GGML_TYPE_IQ2_XXS) {
        const auto value=PreparedIq2<Format,Raw>::decode_selectors(iq2xxs_grid_selectors[code],sign);
        words[0]=unsigned(value.x);words[1]=unsigned(value.y);reference=iq2xxs_grid[code];
    }else if constexpr(Format==GGML_TYPE_IQ2_XS) {
        const auto value=PreparedIq2<Format,Raw>::decode_selectors(iq2xs_grid_selectors[code],sign);
        words[0]=unsigned(value.x);words[1]=unsigned(value.y);reference=iq2xs_grid[code];
    }else if constexpr(Format==GGML_TYPE_IQ2_S) {
        const auto value=PreparedIq2<Format,Raw>::decode_selectors(iq2s_grid_selectors[code],sign);
        words[0]=unsigned(value.x);words[1]=unsigned(value.y);reference=iq2s_grid[code];
    }else {
        const unsigned magnitude=PreparedIq3<Format,Raw>::magnitude_at(code);
        if constexpr(Signs==128)sign|=(__popc(sign)&1u)<<7;
        for(int half=0;half<2;++half) {
            const unsigned mask=iq_nibble_sign_mask((sign>>(4*half))&15);
            if constexpr(Format==GGML_TYPE_IQ3_S)words[half]=magnitude^(mask&0xfefefefeu);
            else words[half]=(magnitude^mask)+(mask&0x01010101u);
        }
        const unsigned original=Format==GGML_TYPE_IQ3_S?iq3s_grid[code]:iq3xxs_grid[code];
        reference=uint64_t(original)|(uint64_t(original)<<32);
    }
    if constexpr(Signs==128)sign|=(__popc(i%Signs)&1u)<<7;
    bool mismatch=false;
    for(int j=0;j<8;++j) {
        int expected=int((reference>>(8*j))&255);if((sign>>j)&1)expected=-expected;
        const int actual=int(int8_t(words[j/4]>>(8*(j%4))));
        mismatch|=actual!=expected;
    }
    if(mismatch)atomicAdd(errors,1u);
}
template<bool Medium,bool Shared=false>
__global__ void iq1_selector_tables_test(unsigned* errors) {
    using F=PreparedIq1M1<Medium,Shared>;
    if constexpr(Shared)F::stage(threadIdx.x,blockDim.x);
    constexpr int Signs=Medium?2:1;
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2048*Signs)return;
    const unsigned code=i/Signs;const bool negative=i%Signs;
    const auto decoded=F::decode_grid(code,negative);
    const unsigned words[]={unsigned(decoded.x),unsigned(decoded.y)};
    const auto grid=iq1s_grid_gpu[code];bool mismatch=false;
    for(int j=0;j<8;++j) {
        int expected=(grid>>(8*(j%4)+4*(j/4)))&15;
        if constexpr(Medium)expected=8*expected-(negative?9:7);
        mismatch|=int(int8_t(words[j/4]>>(8*(j%4))))!=expected;
    }
    if(mismatch)atomicAdd(errors,1u);
}

template<bool Shared=false>
__global__ void iq1s_prepared_dot_test(unsigned* errors) {
    using F=PreparedIq1M1<false,Shared>;
    if constexpr(Shared)F::stage(threadIdx.x,blockDim.x);
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=2048*32*2)return;
    const unsigned code=i/64,pattern=(i/2)%32;const bool negative=i&1;
    typename F::Weight w{};typename F::Activation a{};
    w.scale=.125f;w.high=negative?0x8000u:0u;
    int expected_dot=0,activation_sum=0;
    #pragma unroll
    for(int group=0;group<4;++group) {
        const unsigned index=(code+37*group)&2047;
        w.codes|=(index&255)<<(8*group);w.high|=(index>>8)<<(3*group);
        const auto grid=iq1s_grid_gpu[index];unsigned words[2]{};
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int value=int((pattern*41+j*29+group*17)&255)-128;
            const int coefficient=(grid>>(8*(j%4)+4*(j/4)))&15;
            words[j/4]|=unsigned(uint8_t(value))<<(8*(j%4));
            expected_dot+=coefficient*value;activation_sum+=value;
        }
        a.codes[2*group]=int(words[0]);a.codes[2*group+1]=int(words[1]);
    }
    a.ds=make_float2(.03125f,float(activation_sum)*.03125f+.125f);
    const float delta=-1.f+IQ1S_DELTA-(negative?2.f*IQ1S_DELTA:0.f);
    const float expected=w.scale*(a.ds.x*expected_dot+a.ds.y*delta);
    F::prepare_activation(a);
    if(F::apply_activation(w,a)!=expected)atomicAdd(errors,1u);
}
// All codewords, four independent grid signs, and both 3-bit local scales.
// Build reference integer dots directly from the canonical nibble grids.
template<bool Shared>
__global__ void iq1m_prepared_dot_test(unsigned* errors) {
    using F=PreparedIq1M1<true,Shared>;
    if constexpr(Shared)F::stage(threadIdx.x,blockDim.x);
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=2048*64*16)return;
    const unsigned code=i%2048,pattern=i/2048,signs=pattern>>6;
    typename F::Weight w{};typename F::Activation a{};
    w.scale=.125f;w.scales=pattern&63;
    int expected_dot[2]{};
    #pragma unroll
    for(int group=0;group<4;++group) {
        const unsigned index=(code+37*group)&2047;const bool negative=(signs>>group)&1;
        w.codes|=(index&255)<<(8*group);
        w.high|=((index>>8)|(negative?8u:0u))<<(4*group);
        const auto grid=iq1s_grid_gpu[index];unsigned words[2]{};
        #pragma unroll
        for(int j=0;j<8;++j) {
            const int value=int((code*13+pattern*41+j*29+group*17)&255)-128;
            const int rank=(grid>>(8*(j%4)+4*(j/4)))&15;
            words[j/4]|=unsigned(uint8_t(value))<<(8*(j%4));
            expected_dot[group/2]+=(8*rank-(negative?9:7))*value;
        }
        a.codes[2*group]=int(words[0]);a.codes[2*group+1]=int(words[1]);
    }
    a.ds=make_float2(.03125f,123.f);
    const int sum=expected_dot[0]*(2*(w.scales&7)+1)+expected_dot[1]*(2*((w.scales>>3)&7)+1);
    const float expected=(w.scale*a.ds.x)*(float(sum)*.125f);
    F::prepare_activation(a);
    if(F::apply_activation(w,a)!=expected)atomicAdd(errors,1u);
}

}}
void run_gguf_iq_codebooks() {
    const auto check=[](cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));};
    unsigned* errors=nullptr;check(cudaMalloc(&errors,sizeof(unsigned)));check(cudaMemset(errors,0,sizeof(unsigned)));
    using namespace mfq::cuda::gguf;
    selector_tables_test<GGML_TYPE_IQ2_XXS,256,128><<<256*128/256,256>>>(errors);
    selector_tables_test<GGML_TYPE_IQ2_XS,512,128><<<512*128/256,256>>>(errors);
    selector_tables_test<GGML_TYPE_IQ2_S,1024,256><<<1024*256/256,256>>>(errors);
    selector_tables_test<GGML_TYPE_IQ3_XXS,256,128><<<256*128/256,256>>>(errors);
    selector_tables_test<GGML_TYPE_IQ3_S,512,256><<<512*256/256,256>>>(errors);
    iq1_selector_tables_test<false><<<2048/256,256>>>(errors);
    iq1_selector_tables_test<true><<<4096/256,256>>>(errors);
    iq1s_prepared_dot_test<><<<2048*32*2/256,256>>>(errors);
    iq1_selector_tables_test<false,true><<<2048/256,256,4096>>>(errors);
    iq1_selector_tables_test<true,true><<<4096/256,256,8192>>>(errors);
    iq1s_prepared_dot_test<true><<<2048*32*2/256,256,4096>>>(errors);
    iq1m_prepared_dot_test<false><<<2048*64*16/256,256>>>(errors);
    iq1m_prepared_dot_test<true><<<2048*64*16/256,256,8192>>>(errors);
    selector_tables_test<GGML_TYPE_IQ2_XXS,256,128,true><<<256*128/256,256>>>(errors);
    selector_tables_test<GGML_TYPE_IQ3_S,512,256,true><<<512*256/256,256>>>(errors);
    check(cudaGetLastError());unsigned count=0;check(cudaMemcpy(&count,errors,sizeof(count),cudaMemcpyDeviceToHost));check(cudaFree(errors));
    if(count)throw std::runtime_error("Compressed IQ selector tables differ from canonical byte coefficients");
    std::puts("All IQ1/2/3 codewords, signs, M=1 raw permutations and canonical IQ1_S/M dots (global/shared): 5156864 exact CUDA cases PASS");
}
