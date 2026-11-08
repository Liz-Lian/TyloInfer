#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace mfq::cuda::detail {
__device__ __forceinline__ std::uint32_t load_system_acquire(const std::uint32_t* address) {
#if __CUDA_ARCH__ >= 700
    // Scoped acquire/release loads and stores require PTX target SM 7.0.
    std::uint32_t value;
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];":"=r"(value):"l"(address):"memory");
    return value;
#else
    const auto value=*reinterpret_cast<const volatile std::uint32_t*>(address);
    __threadfence_system();
    return value;
#endif
}
__device__ __forceinline__ void store_system_release(std::uint32_t* address,std::uint32_t value) {
#if __CUDA_ARCH__ >= 700
    asm volatile("st.release.sys.global.u32 [%0], %1;"::"l"(address),"r"(value):"memory");
#else
    __threadfence_system();
    *reinterpret_cast<volatile std::uint32_t*>(address)=value;
#endif
}
__device__ __forceinline__ void system_wait_pause() {
#if __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
}
__device__ __forceinline__ void wait_system_ready(const std::uint32_t* address) {
    if(!address)return;
#if __CUDA_ARCH__ >= 700
    while(!load_system_acquire(address))system_wait_pause();
#else
    while(!*reinterpret_cast<const volatile std::uint32_t*>(address)){}
    __threadfence_system();
#endif
}
}
