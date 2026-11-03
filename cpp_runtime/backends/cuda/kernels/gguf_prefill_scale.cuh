#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// Exact for normal powers of two; keep ldexpf for FP32 subnormal boundaries.
static __device__ __forceinline__ float prefill_ldexp(float value, int exponent) {
    if (exponent >= -126 && exponent <= 126)
        return __fmul_rn(value, __int_as_float((exponent + 127) << 23));
    return ldexpf(value, exponent);
}

// Bound a complete FP16 dot product below 2^14 even when all terms agree.
static __device__ __forceinline__ int prefill_unit_exponent(float maximum, std::int64_t columns) {
    const int bits = columns > 1 ? 64 - __clzll(static_cast<unsigned long long>(columns - 1)) : 0;
    const int shift = max(0, (bits - 14 + 1) / 2);
    return maximum > 0 && isfinite(maximum) ? max(-149, min(140, ilogbf(maximum) + 1 + shift)) : 0;
}
