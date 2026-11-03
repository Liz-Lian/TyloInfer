#pragma once
// M=1 aligned activation planes retain native Q8 codes/scales and GGUF weights.

// Retire IQ1_S activation addresses after each read, keeping one base pointer
// for the code and scale planes. Validated M=1 columns fit INT_MAX, so code
// offsets and columns + 4*(columns/32) fit unsigned 32-bit arithmetic.
__device__ __forceinline__ uint4 iq1s_aligned_codes(const uint4* base,unsigned index) {
    uint4 value;
    asm volatile("{ .reg .u64 p; mad.wide.u32 p,%5,16,%4; ld.global.nc.v4.u32 {%0,%1,%2,%3},[p]; }"
        :"=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w):"l"(base),"r"(index));
    return value;
}
__device__ __forceinline__ unsigned iq1s_aligned_scale(const uint4* base,unsigned offset) {
    unsigned value;
    asm volatile("{ .reg .u64 p; mad.wide.u32 p,%2,1,%1; ld.global.nc.u32 %0,[p]; }"
        :"=r"(value):"l"(base),"r"(offset));
    return value;
}
template<int Format,int Rows,int Warps=1,int BlockWarps=4,class Store=M1ProjectionStore,
    class Prepared=M1PreparedDot<Format>>
__device__ __forceinline__ void project_aligned_m1_body(
        const std::uint8_t* packed,std::uint64_t stride,int rows,int columns,
        const uint4* input,const unsigned* scales,void* output,bool half_output,unsigned tile,float* scratch=nullptr,Store store={}) {
    using F=Prepared;
    static_assert(F::qk==256 && F::parts==8);
    const int warp=int(threadIdx.y),lane=int(threadIdx.x);
    static_assert(Warps>0 && Warps<=BlockWarps);
    constexpr int ActiveWarps=BlockWarps/Warps*Warps;
    const bool active=BlockWarps%Warps==0 || warp<ActiveWarps;
    const unsigned row0=(tile*(BlockWarps/Warps)+unsigned(warp/Warps))*Rows;
    if constexpr(Warps==1)if(row0>=unsigned(rows))return;
    float sums[Rows]{};const int count=columns/F::qk*F::parts;
    for(int i=(warp%Warps)*32+lane;active && row0<unsigned(rows) && i<count;i+=32*Warps) {
        const int block=i/F::parts,part=F::step*(i%F::parts);
        const int at=block*(F::qk/32)+part/F::step;
        uint4 packed_a,packed_b;unsigned scale_bits;
        if constexpr(Format==GGML_TYPE_IQ1_S) {
            packed_a=iq1s_aligned_codes(input,2*at);
            packed_b=iq1s_aligned_codes(input,2*at+1);
            scale_bits=iq1s_aligned_scale(input,unsigned(columns)+unsigned(at)*4);
        }else {
            packed_a=__ldg(input+2*at);packed_b=__ldg(input+2*at+1);
            scale_bits=__ldg(scales+at);
        }
        int codes[8]={int(packed_a.x),int(packed_a.y),int(packed_a.z),int(packed_a.w),int(packed_b.x),int(packed_b.y),int(packed_b.z),int(packed_b.w)};
        const float2 ds=make_float2(__half2float(__ushort_as_half(scale_bits&65535u)),__half2float(__ushort_as_half(scale_bits>>16)));
        if constexpr(requires{typename F::Activation;}){
            typename F::Activation activation;
            #pragma unroll
            for(int j=0;j<8;++j)activation.codes[j]=codes[j];
            if constexpr(requires{activation.ds;})activation.ds=ds;else activation.scale=ds.x;
            if constexpr(requires{activation.lo_sum;}){
                activation.lo_sum=0;activation.hi_sum=0;
                #pragma unroll
                for(int j=0;j<4;++j){activation.lo_sum=ggml_cuda_dp4a(0x01010101,codes[j],activation.lo_sum);activation.hi_sum=ggml_cuda_dp4a(0x01010101,codes[j+4],activation.hi_sum);}
            }
            if constexpr(Format==GGML_TYPE_IQ1_S)F::prepare_activation(activation);
            #pragma unroll
            for(int r=0;r<Rows;++r)if(Rows>2||row0+r<rows){const auto* row=packed+uint64_t(min(row0+unsigned(r),unsigned(rows-1)))*stride;const auto w=load_dense_m1_weight<F>(row,block,part);sums[r]+=F::apply_activation(w,activation);}
        }else{
            #pragma unroll
            for(int r=0;r<Rows;++r)if(Rows>2||row0+r<rows){const auto* row=packed+uint64_t(min(row0+unsigned(r),unsigned(rows-1)))*stride;const auto w=load_dense_m1_weight<F>(row,block,part);sums[r]+=F::apply_u(w,codes,ds.x);}
        }
    }
    constexpr int Capacity=warp_sums::capacity(Rows);
    float values[Capacity]{};
    #pragma unroll
    for(int r=0;r<Rows;++r)values[r]=sums[r];
    float sum=warp_sums::reduce<Capacity>(values,lane);
    const int index=warp_sums::index<Capacity>(lane);
    const bool owner=lane==warp_sums::owner<Capacity>(index);
    if constexpr(Warps==1) {
        if(owner && index<Rows && row0+index<unsigned(rows))store(output,row0+index,sum,half_output);
    } else {
        if(active && owner && index<Rows)scratch[warp*Rows+index]=sum;
        __syncthreads();
        if(active && owner && index<Rows && warp%Warps==0 && row0+index<unsigned(rows)) {
            sum=scratch[warp*Rows+index];
            #pragma unroll
            for(int w=1;w<Warps;++w)sum+=scratch[(warp+w)*Rows+index];
            store(output,row0+index,sum,half_output);
        }
    }
}
