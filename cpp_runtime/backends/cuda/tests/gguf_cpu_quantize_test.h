#pragma once
#include "gguf_cpu.h"
#include "gguf_cpu_q2.h"
#include "ggml-quants.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace gguf_cpu_quantize_test {
inline void reference(const float* input,void* packed,int columns,int type) {
    auto* q=static_cast<block_q8_0*>(packed);quantize_row_q8_0_ref(input,q,columns);
    if(type!=42)return;
    namespace simd=mfq::cpu::gguf::simd;
    auto* prepared=reinterpret_cast<simd::Q2Activation*>(static_cast<std::uint8_t*>(packed)+simd::q2_activation_offset(columns));
    for(int block=0;block<(columns+127)/128;++block) {
        auto& a=prepared[block];
        for(int half=0;half<4;++half) {
            const int index=block*4+half;
            a.scale[half]=index<columns/32?ggml_fp16_to_fp32(q[index].d):0.f;
            int sum=0;if(index<columns/32)for(int i=0;i<32;++i)sum+=q[index].qs[i];
            a.sum[half]=float(sum)*a.scale[half];
        }
        for(int plane=0;plane<4;++plane)for(int i=0;i<32;++i) {
            const int at=block*128+(i/16)*64+(i%16)*4+plane;
            a.planes[plane*32+i]=at<columns?q[at/32].qs[at%32]:0;
        }
    }
}
inline void check() {
    std::vector<float> input(2561);std::uint32_t random=1387;
    for(int pattern=0;pattern<6;++pattern) {
        for(std::size_t i=0;i<input.size();++i) {
            random=random*1664525u+1013904223u;
            if(pattern==0)input[i]=(int(random%800000)-400000)*.001f;
            if(pattern==1)input[i]=float(int(i%127)-63)+.5f;
            if(pattern==2)input[i]=std::nextafter(float(int(i%127)-63)+.5f,0.f);
            if(pattern==3)input[i]=std::nextafter(float(int(i%127)-63)+.5f,1000.f);
            if(pattern==4)input[i]=i%32==31?127.f:float(int(i%31)-15)*.5f;
            if(pattern==5)input[i]=i%2?-0.f:0.f;
        }
        for(int type:{20,42})for(int n:{32,64,128,160,192,256,640,768,2560})for(int offset:{0,1}) {
            if(type==42 && n%64)continue;
            const auto bytes=mfq::cpu::gguf::hidden_bytes(n,type);
            std::vector<std::uint8_t> expected(bytes+8,0x5a),actual(expected);
            // Deliberately avoid 32-byte alignment of both the input and output.
            reference(input.data()+offset,expected.data()+4,n,type);
            mfq::cpu::gguf::quantize_hidden(input.data()+offset,actual.data()+4,n,type);
            if(actual!=expected) {
                std::size_t at=0;while(at<actual.size() && actual[at]==expected[at])++at;
                std::fprintf(stderr,"CPU hidden quantization differs pattern=%d type=%d n=%d offset=%d byte=%zu got=%u ref=%u\n",
                    pattern,type,n,offset,at,unsigned(actual[at]),unsigned(expected[at]));
                throw std::runtime_error("CPU hidden quantization changed Q8 bytes or Q2 planes/scales/sums");
            }
        }
    }
    std::puts("CPU hidden quantization Q8 rounding/ties/Q2 planes/scales/sums/tails/unaligned byte checks PASS");
}
inline void bench() {
    std::vector<float> input(640);
    for(int i=0;i<640;++i)input[i]=std::sin(i*.17f)*3.f;
    for(int type:{20,42}) {
        std::vector<std::uint8_t> output(mfq::cpu::gguf::hidden_bytes(640,type));
        double times[2]{};volatile unsigned checksum=0;
        for(int mode=0;mode<2;++mode) {
            const auto begin=std::chrono::steady_clock::now();
            for(int repeat=0;repeat<10000;++repeat) {
                if(mode)mfq::cpu::gguf::quantize_hidden(input.data(),output.data(),640,type);
                else reference(input.data(),output.data(),640,type);
                checksum=checksum+output[repeat%output.size()];
            }
            times[mode]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/10000;
        }
        std::printf("cpu_hidden_quant count=640 type=%d scalar_us=%.6f vector_us=%.6f checksum=%u\n",type,times[0],times[1],unsigned(checksum));
    }
}
}
