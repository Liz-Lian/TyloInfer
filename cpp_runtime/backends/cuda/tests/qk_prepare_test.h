#pragma once
#include "mfq_cuda_attention_ops.h"
#include "mfq_cuda_norm_ops.h"
#include "mfq_cuda_cache_ops.h"
#include "../kernels/gguf.h"
#include "../kernels/gguf_head_gate.h"
#include "core/rope.h"
#include "runtime/decode_window.h"
#include <cmath>
#include <cstring>
#include <iostream>

inline void qk_prepare_case() {
    namespace tb=mfq_tensor_backend;
    const auto data=[](std::vector<int64_t> shape,int salt) {
        auto host=tb::empty(shape,tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32));
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=std::sin(float(i)*.173f+salt)*.4f;
        return host.to(tb::kCUDA);
    };
    const auto equal=[](const tb::Tensor& a,const tb::Tensor& b,const char* message) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
            std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes())) {
            if(x.scalar_type()==tb::kFloat32) {
                int64_t differences=0;double error=0.;
                for(int64_t i=0;i<x.numel();++i)if(x.data_ptr<float>()[i]!=y.data_ptr<float>()[i]) {
                    if(differences<8)std::cerr<<"diff "<<i<<" actual="<<x.data_ptr<float>()[i]<<" expected="<<y.data_ptr<float>()[i]<<'\n';
                    ++differences;error=std::max(error,std::abs(double(x.data_ptr<float>()[i])-y.data_ptr<float>()[i]));
                }
                std::cerr<<"differing="<<differences<<" max_abs="<<error<<'\n';
            }
            throw std::runtime_error(message);
        }
    };
    int checked=0;
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int width:{31,64,128,255,256,320,513})for(int batch:{1,2})for(int query_heads:{5,6}) {
        const int key_heads=query_heads==6?2:1,capacity=13,rotary=2*(width/4);
        auto q=data({batch,query_heads,1,2*width+3},1).to(dtype).narrow(3,0,width);
        auto k=data({batch,key_heads,1,width+5},2).to(dtype).narrow(3,0,width);
        auto v=data({batch,key_heads,1,width+7},3).to(dtype).narrow(3,0,width);
        auto qw=data({width},4),kw=data({width},5);
        RopeCache rope(17,rotary,10000.);
        const auto integer=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kInt64);
        auto position=tb::full({1},0,integer),slot=tb::full({1},0,integer);
        auto initial=data({batch,key_heads,capacity,width},6).to(tb::kFloat16);
        auto kc=initial.clone(),vc=initial.clone(),reference_k=initial.clone(),reference_v=initial.clone();
        tb::Tensor actual;
        const auto reset=[&]{kc.copy_(initial);vc.copy_(initial);};
        mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
        const double offset=batch==1?0.:1.;
        graph.capture([&]{actual=qk_norm_rope_cache_cuda(q,k,v,qw,kw,position,slot,
            rope.cos,rope.sin,kc,vc,rotary,1e-6,offset);},reset,[&]{actual={};});
        for(int repeat=0;repeat<3;++repeat) {
            const int target=repeat==0?2:repeat==1?capacity-1:0;
            position.fill_(repeat==0?-3:repeat==1?23:7);slot.fill_(target);
            q.copy_(data(q.sizes().vec(),7+repeat).to(dtype));
            k.copy_(data(k.sizes().vec(),11+repeat).to(dtype));
            v.copy_(data(v.sizes().vec(),15+repeat).to(dtype));
            auto normalized=dtype==tb::kFloat16?
                rms_norm_pair_f16_f32_offset_cuda(q.contiguous(),k.contiguous(),qw,kw,1e-6,offset):
                std::vector<tb::Tensor>{rms_norm_offset_cuda(q.contiguous(),qw,1e-6,offset),
                                      rms_norm_offset_cuda(k.contiguous(),kw,1e-6,offset)};
            auto expected=rope.apply(normalized[0],position);
            auto expected_key=rope.apply(normalized[1],position).to(tb::kFloat16).contiguous();
            reference_k.copy_(initial);reference_v.copy_(initial);reset();
            kv_cache_write_cuda(reference_k,reference_v,expected_key,v.to(tb::kFloat16).contiguous(),slot);
            graph.run();
            equal(actual,expected,"Fused QK preparation changed FP32 query");
            equal(kc,reference_k,"Fused QK preparation changed KV key bytes or untouched slots");
            equal(vc,reference_v,"Fused QK preparation changed KV value bytes or untouched slots");
            ++checked;
        }
    }
    std::cout<<"QK fused normalization/rotary/cache changing-graph bitwise cases="<<checked<<" PASS\n";
    int gate_checked=0;
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int batch:{1,2})for(int tokens:{1,3})
    for(int heads:{2,4})for(int width:{32,80,96})for(bool silu:{false,true}) {
        auto input=data({batch,tokens,heads*width},19).to(dtype);
        auto gate=data({batch,tokens,heads,2*width+5},23).to(dtype).narrow(3,0,width);
        auto flat=gate.contiguous().reshape_as(input);
        const auto bytes=batch*tokens*int64_t(mfq::cuda::gguf::q8_row_bytes(heads*width));
        auto expected=tb::empty({bytes},input.options().dtype(tb::kUInt8)),actual=tb::empty_like(expected);
        for(bool aligned:{false,true}) {
            if(aligned && batch*tokens!=1)continue;
            const auto quantize=aligned?mfq::cuda::gguf::quantize_gated_input_m1_aligned:
                mfq::cuda::gguf::quantize_gated_input;
            quantize(input.data_ptr(),flat.data_ptr(),dtype==tb::kFloat16,batch*tokens,
                heads*width,silu,expected.data_ptr(),mfq_current_cuda_stream());
            mfq::cuda::gguf::quantize_gated_head_input(input.data_ptr(),gate.data_ptr(),dtype==tb::kFloat16,
                batch,tokens,heads,width,gate.stride(0),gate.stride(1),gate.stride(2),silu,aligned,
                actual.data_ptr(),mfq_current_cuda_stream());
            equal(actual,expected,"Head-gated preparation changed Q8 codes/scales");++gate_checked;
        }
    }
    std::cout<<"Strided head-gate native/aligned Q8 bitwise cases="<<gate_checked<<" PASS\n";
}
