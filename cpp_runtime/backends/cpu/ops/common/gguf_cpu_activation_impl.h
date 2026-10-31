#pragma once
#include "ggml.h"
#include <algorithm>
#include <cmath>

namespace mfq::cpu::gguf::detail {
inline float swiglu_scalar(float gate,float up,bool half,float limit) {
    const auto round=[half](float value) {
        return half?ggml_fp16_to_fp32(ggml_fp32_to_fp16(value)):value;
    };
    float g=round(gate),u=round(up);
    if(limit>0) {
        g=std::min(g,limit);u=std::clamp(u,-limit,limit);
        return round((g*(1.f/(1.f+std::exp(-g))))*u);
    }
    const float sigmoid=round(1.f/(1.f+std::exp(-g)));
    return round(round(g*sigmoid)*u);
}
}
