#pragma once

#include <cuda_runtime.h>

// Decode, batched verification and prefix replay must contract the same
// product. A different contraction changes the committed recurrent state
// when the speculative window length changes.
__device__ __forceinline__ float mfq_gdn_update(
        float previous,float decay,float key,float delta) {
    return __fmaf_rn(decay,previous,__fmul_rn(key,delta));
}
