#pragma once
#include <cuda_fp16.h>
#include <cstdint>

__device__ __forceinline__ float gdn_qkv_value(const void* input,std::int64_t index,bool fp32) {
    return fp32?static_cast<const float*>(input)[index]:__half2float(static_cast<const __half*>(input)[index]);
}

// The caller validates every grid/head index below INT_MAX. For d>=2,
// ceil(2^32/d) produces either floor(n/d) or one larger; one exact product
// comparison corrects it. Since n,d<2^31, that comparison cannot overflow.
// Compute reciprocals on the host instead of dividing independently in each CTA.
struct GdnHeadDivisors {
    std::uint32_t value_heads,key_heads,groups,group_size;
};
inline GdnHeadDivisors gdn_head_divisors(unsigned value_heads,unsigned key_heads) {
    const auto reciprocal=[](unsigned d) {
        return d==1?0u:unsigned(((std::uint64_t(1)<<32)+d-1)/d);
    };
    const unsigned groups=value_heads/key_heads;
    return {reciprocal(value_heads),reciprocal(key_heads),reciprocal(groups),groups};
}
__device__ __forceinline__ unsigned gdn_head_quotient(unsigned n,unsigned d,unsigned reciprocal) {
    const unsigned quotient=__umulhi(n,reciprocal);
    return d==1?n:quotient-unsigned(quotient*d>n);
}
