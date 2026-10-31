#pragma once
#include <atomic>
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
#include <intrin.h>
#endif

namespace mfq::cpu {
// As in Strata's worker waits, let a spinning thread release execution
// resources to its sibling. Atomic loads still provide synchronization.
inline void spin_pause() noexcept {
#if defined(_M_X64) || defined(_M_IX86) || defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#elif defined(_MSC_VER) && (defined(_M_ARM64) || defined(_M_ARM))
    __yield();
#elif defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}
} // namespace mfq::cpu
