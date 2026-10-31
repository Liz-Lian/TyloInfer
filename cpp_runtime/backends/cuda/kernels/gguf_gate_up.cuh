#pragma once
#include "warp_sums.cuh"
// Strata's expert decoder stages Q8 words in two aligned planes and reuses
// them across several decoded weight rows (MIT, third_party/strata.LICENSE).
// Here the two rows are Gate and Up, with independent expert pointer tables.
namespace mfq::cuda::gguf {
namespace {
template<int Format,int Lanes>
__device__ __forceinline__ void gate_up_direct_dot(const uint8_t* gate_weight,const uint8_t* up_weight,
        const block_q8_1* input,int columns,int lane,float& g,float& u,const uint32_t* grid) {
    using F=StagedPreparedDot<Format>;
    constexpr int Parts=32/Lanes;
    float sg[Parts]{},su[Parts]{};
    const int items=columns/F::qk*F::parts;
    #pragma unroll
    for(int p=0;p<Parts;++p)for(int i=lane+p*Lanes;(gate_weight || up_weight) && i<items;i+=32) {
        const int block=i/F::parts,part=F::step*(i%F::parts);
        typename F::Weight wg{},wu{};
        if(gate_weight)wg=F::load(gate_weight,block,part,grid);
        if(up_weight)wu=F::load(up_weight,block,part,grid);
        const auto& x=input[block*(F::qk/32)+(gate_weight?F::input_block(wg):F::input_block(wu))];
        int codes[8];
        #pragma unroll
        for(int j=0;j<8;++j)codes[j]=get_int_b4(x.qs,j);
        const float scale=__low2float(x.ds);
        sg[p]+=F::apply_u(wg,codes,scale);su[p]+=F::apply_u(wu,codes,scale);
    }
    #pragma unroll
    for(int step=Parts/2;step;step>>=1) {
        #pragma unroll
        for(int p=0;p<step;++p){sg[p]+=sg[p+step];su[p]+=su[p+step];}
    }
    g=sg[0];u=su[0];
    #pragma unroll
    for(int step=Lanes/2;step;step>>=1) {
        g+=__shfl_down_sync(0xffffffffu,g,step,Lanes);u+=__shfl_down_sync(0xffffffffu,u,step,Lanes);
    }
}
template<int Format,int Lanes>
__global__ void __launch_bounds__(128) project_gate_up_direct_q8(const uint64_t* gate_addresses,const uint64_t* up_addresses,
        uint64_t gate_stride,uint64_t up_stride,int experts,int rows,int columns,int slots,
        const int64_t* ids,int routes,const block_q8_1* input,float* gate,float* up,bool preserve_missing) {
    using F=PreparedDot<Format>;
    const auto index=blockIdx.x*(128/Lanes)+threadIdx.x/Lanes;
    if(index>=unsigned(slots)*unsigned(rows))return;
    const int row=int(index)%rows,slot=int(index)/rows,lane=threadIdx.x%Lanes;
    const auto expert=ids[slot];
    const auto ga=expert>=0 && expert<experts?gate_addresses[expert]:0;
    const auto ua=expert>=0 && expert<experts?up_addresses[expert]:0;
    float g,u;
    gate_up_direct_dot<Format,Lanes>(ga?reinterpret_cast<const uint8_t*>(ga)+uint64_t(row)*gate_stride:nullptr,
        ua?reinterpret_cast<const uint8_t*>(ua)+uint64_t(row)*up_stride:nullptr,
        input+(slot/routes)*(columns/32),columns,lane,g,u,nullptr);
    if(!lane) {
        if(ga || !preserve_missing)gate[index]=g;
        if(ua || !preserve_missing)up[index]=u;
    }
}
// One CTA belongs to a single expert. Complete misses return uniformly before
// staging its exact lookup tables; live rows amortize those tables over a larger
// tile. The zero-table formats retain four warps without shared staging.
template<int Format,int Lanes>
__global__ void __launch_bounds__(StagedPreparedDot<Format>::words?256:128)
project_gate_up_table_q8(const uint64_t* gate_addresses,const uint64_t* up_addresses,
        uint64_t gate_stride,uint64_t up_stride,int experts,int rows,int columns,int slots,
        const int64_t* ids,int routes,const block_q8_1* input,float* gate,float* up,bool preserve_missing) {
    using F=StagedPreparedDot<Format>;
    constexpr int Threads=F::words?256:128;
    const int slot=blockIdx.y,row=int(blockIdx.x)*(Threads/Lanes)+threadIdx.x/Lanes;
    const int lane=threadIdx.x%Lanes;
    const auto expert=ids[slot];
    const auto ga=expert>=0 && expert<experts?gate_addresses[expert]:0;
    const auto ua=expert>=0 && expert<experts?up_addresses[expert]:0;
    const auto index=int64_t(slot)*rows+row;
    if(!ga && !ua) {
        if(row<rows && !lane && !preserve_missing){gate[index]=0.f;up[index]=0.f;}
        return;
    }
    __shared__ __align__(16) uint32_t grid[F::words?F::words:1];
    F::stage(grid,threadIdx.x,Threads);
    if(row>=rows)return;
    float g,u;
    gate_up_direct_dot<Format,Lanes>(ga?reinterpret_cast<const uint8_t*>(ga)+uint64_t(row)*gate_stride:nullptr,
        ua?reinterpret_cast<const uint8_t*>(ua)+uint64_t(row)*up_stride:nullptr,
        input+(slot/routes)*(columns/32),columns,lane,g,u,grid);
    if(!lane) {
        if(ga || !preserve_missing)gate[index]=g;
        if(ua || !preserve_missing)up[index]=u;
    }
}

template<int Format,int Capacity,int Lanes,bool Paired,bool Quantized>
__global__ void __launch_bounds__(Quantized?32*Lanes:256,Quantized?1024/(32*Lanes):4) project_gate_up_planar_q8(const uint64_t* gate_addresses,const uint64_t* up_addresses,
        uint64_t gate_stride,uint64_t up_stride,int experts,int rows,int columns,int slots,
        const int64_t* ids,const RouteGroup* groups,int routes,const block_q8_1* input,
        float* gate,float* up,bool preserve_missing,bool routed_input,
        block_q8_1* hidden,bool half_output,float limit) {
    using F=StagedPreparedDot<Format>;
    const bool source_is_routed=!Paired && routed_input;
    constexpr int Threads=Quantized?32*Lanes:256,Parts=32/Lanes,Rows=Threads/Lanes;
    static_assert(!Quantized || Paired,"hidden quantization needs both projections");
    extern __shared__ __align__(16) uint32_t activation_words[];
    __shared__ __align__(16) uint32_t grid[F::words?F::words:1];
    const int tid=threadIdx.y*32+threadIdx.x,lane=tid%Lanes,row=int(blockIdx.x)*Rows+tid/Lanes;
    const int blocks=columns/32,items=columns/F::qk*F::parts;
    bool grid_ready=false;
    const int group_count=groups?groups[0].count:slots;
    for(int index=blockIdx.y;index<group_count;index+=gridDim.y) {
        const auto* group=groups?groups+1+index:nullptr;
        const int count=group?group->count:1;
        const auto expert=group?group->expert:ids[index];
        const auto ga=expert>=0 && expert<experts?gate_addresses[expert]:0;
        const auto ua=Paired && expert>=0 && expert<experts?up_addresses[expert]:0;
        if(!ga && !ua) {
            if constexpr(Quantized) {
                const int t=tid/32,q_lane=tid%32;
                if(t<count)store_q8(0.f,int64_t(group?group->slots[t]:index)*rows+int64_t(blockIdx.x)*32+q_lane,hidden);
            }else if(row<rows && !lane && !preserve_missing)for(int t=0;t<count;++t) {
                const int slot=group?group->slots[t]:index;
                gate[int64_t(slot)*rows+row]=0.f;
                if constexpr(Paired)up[int64_t(slot)*rows+row]=0.f;
            }
            continue;
        }
        // Residency partitions leave some CTAs without any weights. They
        // must not stage lookup tables or activations for absent experts.
        if(!grid_ready){F::stage(grid,tid,Threads);grid_ready=true;}
        const auto compute=[&]<int Count>() {
            auto* hidden_gate=reinterpret_cast<float*>(activation_words+Capacity*blocks*9);
            auto* hidden_up=hidden_gate+Capacity*32;
            auto* words=reinterpret_cast<int4*>(activation_words);
            auto* scales=reinterpret_cast<float*>(activation_words+Count*blocks*8);
            if constexpr(Count>1)for(int t=0;t<Count;++t) {
                const int slot=group?group->slots[t]:index;
                const auto* source=reinterpret_cast<const int*>(input+(source_is_routed?slot:slot/routes)*blocks);
                auto* planes=reinterpret_cast<int*>(words);
                for(int word=tid;word<blocks*9;word+=Threads) {
                    const int k=word/9,j=word%9,v=source[word];
                    if(!j)scales[t*blocks+k]=__low2float(*reinterpret_cast<const half2*>(&v));
                    else {const int q=j-1;planes[((2*t+q/4)*blocks+k)*4+q%4]=v;}
                }
            }
            if constexpr(Count>1)__syncthreads();
            // A singleton keeps the original IQ3_S projection order. Larger
            // token groups share the word-major integer tile below.
            if constexpr(Paired && Format==GGML_TYPE_IQ3_S && Count==1) {
                // A complete projection retires its accumulators before the next
                // one starts. Both reuse the staged activation and lookup table;
                // the reduction order within each projection stays unchanged.
                const auto project=[&]<bool IsUp>() {
                    const auto address=IsUp?ua:ga;
                    const auto stride=IsUp?up_stride:gate_stride;
                    float sums[Count][Parts]{};
                    #pragma unroll
                    for(int p=0;p<Parts;++p)for(int i=lane+p*Lanes;address && row<rows && i<items;i+=32) {
                        const int block=i/F::parts,part=F::step*(i%F::parts);
                        const auto weight=F::load(reinterpret_cast<const uint8_t*>(address)+uint64_t(row)*stride,block,part,grid);
                        const int k=block*(F::qk/32)+F::input_block(weight);
                        #pragma unroll
                        for(int t=0;t<Count;++t) {
                            int x[8];float scale;
                            if constexpr(Count==1) {
                                const int slot=group?group->slots[0]:index;
                                const auto& a=input[(source_is_routed?slot:slot/routes)*blocks+k];
                                #pragma unroll
                                for(int j=0;j<8;++j)x[j]=get_int_b4(a.qs,j);
                                scale=__low2float(a.ds);
                            }else {
                                const int4 a=words[(2*t)*blocks+k],b=words[(2*t+1)*blocks+k];
                                x[0]=a.x;x[1]=a.y;x[2]=a.z;x[3]=a.w;x[4]=b.x;x[5]=b.y;x[6]=b.z;x[7]=b.w;
                                scale=scales[t*blocks+k];
                            }
                            sums[t][p]+=F::apply_u(weight,x,scale);
                        }
                    }
                    constexpr int P=warp_sums::capacity(Count);
                    float values[P]{};
                    #pragma unroll
                    for(int t=0;t<Count;++t) {
                        #pragma unroll
                        for(int step=Parts/2;step;step>>=1) {
                            #pragma unroll
                            for(int p=0;p<step;++p)sums[t][p]+=sums[t][p+step];
                        }
                        values[t]=sums[t][0];
                    }
                    const float value=warp_sums::reduce<P,Lanes>(values,lane);
                    const int t=warp_sums::index<P,Lanes>(lane);
                    if(t<Count && row<rows && lane==warp_sums::owner<P,Lanes>(t)) {
                        const int slot=group?group->slots[t]:index;
                        if constexpr(Quantized) {
                            auto* target=IsUp?hidden_up:hidden_gate;
                            target[t*32+tid/Lanes]=value;
                        }else if(address || !preserve_missing) {
                            auto* target=IsUp?up:gate;
                            target[int64_t(slot)*rows+row]=value;
                        }
                    }
                };
                project.template operator()<false>();
                if constexpr(Paired)project.template operator()<true>();
            } else {
                float sg[Count][Parts]{},su[Count][Parts]{};
                #pragma unroll
                for(int p=0;p<Parts;++p)for(int i=lane+p*Lanes;row<rows && i<items;i+=32) {
                    const int block=i/F::parts,part=F::step*(i%F::parts);
                    typename F::Weight wg{},wu{};
                    if(ga)wg=F::load(reinterpret_cast<const uint8_t*>(ga)+uint64_t(row)*gate_stride,block,part,grid);
                    if constexpr(Paired)if(ua)wu=F::load(reinterpret_cast<const uint8_t*>(ua)+uint64_t(row)*up_stride,block,part,grid);
                    const int k=block*(F::qk/32)+(ga?F::input_block(wg):F::input_block(wu));
                    if constexpr(Paired && Count>1 && Format==GGML_TYPE_IQ3_S) {
                        // Issue independent token dots a packed word at a time.
                        // Decode each weight once, and share each activation
                        // word between Gate/Up without retaining eight words
                        // for every token until an entire serial dot finishes.
                        int gate_sums[Count]{},up_sums[Count]{};
                        const auto* planes=reinterpret_cast<const int*>(words);
                        #pragma unroll
                        for(int j=0;j<8;++j) {
                            #pragma unroll
                            for(int t=0;t<Count;++t) {
                                const int x=planes[((2*t+j/4)*blocks+k)*4+j%4];
                                gate_sums[t]=ggml_cuda_dp4a(wg.codes[j],x,gate_sums[t]);
                                up_sums[t]=ggml_cuda_dp4a(wu.codes[j],x,up_sums[t]);
                            }
                        }
                        #pragma unroll
                        for(int t=0;t<Count;++t) {
                            const float scale=scales[t*blocks+k];
                            const auto scaled=[](const auto& weight,int sum){return sum*weight.scale_code;};
                            sg[t][p]+=(wg.scale*scale)*scaled(wg,gate_sums[t]);
                            su[t][p]+=(wu.scale*scale)*scaled(wu,up_sums[t]);
                        }
                    } else {
                        #pragma unroll
                        for(int t=0;t<Count;++t) {
                            int x[8];float scale;
                            if constexpr(Count==1) {
                                const int slot=group?group->slots[0]:index;
                                const auto& a=input[(source_is_routed?slot:slot/routes)*blocks+k];
                                #pragma unroll
                                for(int j=0;j<8;++j)x[j]=get_int_b4(a.qs,j);
                                scale=__low2float(a.ds);
                            }else {
                                const int4 a=words[(2*t)*blocks+k],b=words[(2*t+1)*blocks+k];
                                x[0]=a.x;x[1]=a.y;x[2]=a.z;x[3]=a.w;x[4]=b.x;x[5]=b.y;x[6]=b.z;x[7]=b.w;
                                scale=scales[t*blocks+k];
                            }
                            sg[t][p]+=F::apply_u(wg,x,scale);
                            if constexpr(Paired)su[t][p]+=F::apply_u(wu,x,scale);
                        }
                    }
                }
                constexpr int Outputs=Count*(Paired?2:1),P=warp_sums::capacity(Outputs);
                float values[P]{};
                #pragma unroll
                for(int t=0;t<Count;++t) {
                    #pragma unroll
                    for(int step=Parts/2;step;step>>=1) {
                        #pragma unroll
                        for(int p=0;p<step;++p){sg[t][p]+=sg[t][p+step];if constexpr(Paired)su[t][p]+=su[t][p+step];}
                    }
                    values[t]=sg[t][0];
                    if constexpr(Paired)values[Count+t]=su[t][0];
                }
                const float value=warp_sums::reduce<P,Lanes>(values,lane);
                const int result=warp_sums::index<P,Lanes>(lane);
                if(result<Outputs && row<rows && lane==warp_sums::owner<P,Lanes>(result)) {
                    const bool is_up=Paired && result>=Count;
                    const int t=is_up?result-Count:result,slot=group?group->slots[t]:index;
                    if constexpr(Quantized) {
                        auto* target=is_up?hidden_up:hidden_gate;
                        target[t*32+tid/Lanes]=value;
                    }else {
                        const auto address=is_up?ua:ga;
                        if(address || !preserve_missing)(is_up?up:gate)[int64_t(slot)*rows+row]=value;
                    }
                }
            }
            if constexpr(Quantized) {
                __syncthreads();
                const int t=tid/32,q_lane=tid%32;
                if(t<Count) {
                    const int slot=group?group->slots[t]:index;
                    const float g=hidden_gate[t*32+q_lane],u=hidden_up[t*32+q_lane];
                    const float value=half_output?swiglu_q8_value<true>(g,u,limit):swiglu_q8_value<false>(g,u,limit);
                    store_q8(value,int64_t(slot)*rows+int64_t(blockIdx.x)*32+q_lane,hidden);
                }
                // Even a singleton's shared epilogue is reused by the next group.
                if(index+int(gridDim.y)<group_count)__syncthreads();
            }
        };
        if constexpr(Capacity==1)compute.template operator()<1>();
        else if(count==1)compute.template operator()<1>();
        else if constexpr(Capacity==2)compute.template operator()<2>();
        else if constexpr(Capacity==3){if(count==2)compute.template operator()<2>();else compute.template operator()<3>();}
        else {if(count==2)compute.template operator()<2>();else if(count==3)compute.template operator()<3>();else compute.template operator()<4>();}
        // A later group may overwrite the same activation arena.
        if(!Quantized && count>1 && index+int(gridDim.y)<group_count)__syncthreads();
    }
}
}
}
