#pragma once
#include "gguf_cpu_activation.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gguf_cpu_activation_test {
// The original CPU expert expression, retained as a bitwise reference.
inline float reference(float gate,float up,bool half,float limit) {
    const auto round=[half](float v){return half?ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)):v;};
    float g=round(gate),u=round(up);
    if(limit>0) {
        g=std::min(g,limit);u=std::clamp(u,-limit,limit);
        return round((g*(1.f/(1.f+std::exp(-g))))*u);
    }
    const float sigmoid=round(1.f/(1.f+std::exp(-g)));
    return round(round(g*sigmoid)*u);
}
inline std::uint32_t bits(float value) {
    std::uint32_t result;std::memcpy(&result,&value,sizeof(result));return result;
}
inline bool same_result(float a,float b) {
    // IEEE operations do not prescribe a NaN's sign or payload. All numeric
    // results, including infinities and signed zero, must match bit for bit.
    return bits(a)==bits(b) || (std::isnan(a) && std::isnan(b));
}
inline void check() {
    std::vector<float> gate(8193),up(gate.size()),expected(gate.size()),actual(gate.size());
    std::uint32_t random=177;
    for(std::size_t i=0;i<gate.size();++i) {
        random=random*1664525u+1013904223u;gate[i]=(int(random%400000)-200000)*.0001f;
        random=random*1664525u+1013904223u;up[i]=(int(random%200000)-100000)*.0001f;
    }
    const auto compare=[&](const float* a,const float* b,std::size_t count,bool half,float limit) {
        for(std::size_t i=0;i<count;++i)expected[i]=reference(a[i],b[i],half,limit);
        mfq::cpu::gguf::swiglu(a,b,actual.data(),count,half,limit);
        for(std::size_t i=0;i<count;++i)if(!same_result(actual[i],expected[i])) {
            std::fprintf(stderr,"CPU activation differs i=%zu count=%zu half=%d limit=%g gate=%a up=%a got=%08x ref=%08x\n",
                i,count,int(half),limit,a[i],b[i],bits(actual[i]),bits(expected[i]));
            throw std::runtime_error("CPU activation changed result bits");
        }
        auto in_place=std::vector<float>(a,a+count);
        mfq::cpu::gguf::swiglu(in_place.data(),b,in_place.data(),count,half,limit);
        for(std::size_t i=0;i<count;++i)if(!same_result(in_place[i],expected[i]))throw std::runtime_error("CPU activation gate alias");
        in_place.assign(b,b+count);
        mfq::cpu::gguf::swiglu(a,in_place.data(),in_place.data(),count,half,limit);
        for(std::size_t i=0;i<count;++i)if(!same_result(in_place[i],expected[i]))throw std::runtime_error("CPU activation up alias");
    };
    for(bool half:{false,true})for(float limit:{0.f,.01f,1.5f})for(int offset:{0,1})
        for(std::size_t count:{0,1,7,8,17,32,257,8192})
            compare(gate.data()+offset,up.data()+offset,count,half,limit);
    const float special[]={0.f,-0.f,1e-8f,-1e-8f,0x1p-24f,-0x1p-24f,0x1p-14f,
        1.f,-1.f,1.f+0x1p-11f,1.f+3.f*0x1p-11f,65504.f,65520.f,-65520.f,
        1e38f,-1e38f,INFINITY,-INFINITY,std::numeric_limits<float>::quiet_NaN()};
    std::size_t at=0;
    for(float g:special)for(float u:special){gate[at]=g;up[at++]=u;}
    for(bool half:{false,true})for(float limit:{0.f,.01f,1.5f,INFINITY})
        compare(gate.data(),up.data(),at,half,limit);
    // Include half subnormals and conversion ties in finite
    // vector blocks, rather than testing them only beside exceptional lanes.
    for(std::size_t i=0;i<gate.size();++i) {
        const auto h=ggml_fp16_t((i*97)%0x7c00);
        gate[i]=ggml_fp16_to_fp32(h)*(i%2?1.f:-1.f);up[i]=float(int(i%31)-15)*.125f;
    }
    for(bool half:{false,true})for(float limit:{0.f,1.5f})
        compare(gate.data(),up.data(),gate.size(),half,limit);
    std::puts("CPU activation FP16/FP32/clipping/tails/alias/special-value bit checks PASS");
}
inline void bench() {
    std::vector<float> gate(640),up(640),output(640);
    for(int i=0;i<640;++i){gate[i]=std::sin(i*.17f)*4.f;up[i]=std::cos(i*.31f)*2.f;}
    for(bool half:{false,true})for(float limit:{0.f,1.5f}) {
        double times[2]{};volatile float checksum=0;
        for(int mode=0;mode<2;++mode) {
            const auto begin=std::chrono::steady_clock::now();
            for(int repeat=0;repeat<3000;++repeat) {
                for(int at=0;at<640;at+=32) {
                    if(mode)mfq::cpu::gguf::swiglu(gate.data()+at,up.data()+at,output.data()+at,32,half,limit);
                    else for(int i=at;i<at+32;++i)output[i]=reference(gate[i],up[i],half,limit);
                }
                checksum=checksum+output[repeat%640];
            }
            times[mode]=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/3000;
        }
        std::printf("cpu_activation count=640 tile=32 half=%d limit=%.1f scalar_us=%.6f vector_us=%.6f checksum=%f\n",
            int(half),limit,times[0],times[1],float(checksum));
    }
}
}
