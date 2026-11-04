#include "quant_dot.h"
#include "engine/backends/cpu/native/features.h"

namespace tyloi::cpu {
float scaled_i8_dot_scalar(const std::int8_t* weights, const float* input, int count, float scale) {
    float result=0;
    for (int i=0; i<count; ++i) {
        const float decoded=scale*weights[i];
        result += decoded*input[i];
    }
    return result;
}

ScaledI8Dot scaled_i8_dot_kernel() {
#ifdef TYLOINFER_CPU_PACKED_AVX2
    if (has_avx2()) return detail::scaled_i8_dot_avx2;
#endif
    return scaled_i8_dot_scalar;
}
} // namespace tyloi::cpu
