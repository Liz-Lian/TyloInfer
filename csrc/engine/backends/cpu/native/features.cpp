#include "features.h"

#if defined(TYLOINFER_CPU_PACKED_AVX2) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace mfq::cpu {

bool has_avx2() {
    static const bool supported = [] {
#if defined(TYLOINFER_CPU_PACKED_AVX2) && defined(_MSC_VER)
        int leaf[4];
        __cpuid(leaf, 0);
        if (leaf[0] < 7) return false;
        __cpuidex(leaf, 1, 0);
        if ((leaf[2] & ((1<<27) | (1<<28))) != ((1<<27) | (1<<28))) return false;
        if ((_xgetbv(0) & 6) != 6) return false;
        __cpuidex(leaf, 7, 0);
        return (leaf[1] & (1<<5)) != 0;
#elif defined(TYLOINFER_CPU_PACKED_AVX2) && (defined(__GNUC__) || defined(__clang__))
        __builtin_cpu_init();
        return static_cast<bool>(__builtin_cpu_supports("avx2"));
#else
        return false;
#endif
    }();
    return supported;
}

} // namespace mfq::cpu
