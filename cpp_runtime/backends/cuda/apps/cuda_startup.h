#pragma once
#include <cstdlib>
#include <stdexcept>

namespace mfq::cuda {
inline void configure_cuda_startup() {
    // Account for module code before expert/KV cache budgeting, as Strata
    // does, and avoid first-use loader synchronization between streams.
    // Preserve an explicit caller choice, including LAZY.
    if(std::getenv("CUDA_MODULE_LOADING"))return;
#if defined(_WIN32)
    const auto error=_putenv_s("CUDA_MODULE_LOADING","EAGER");
#else
    const auto error=setenv("CUDA_MODULE_LOADING","EAGER",0);
#endif
    if(error)throw std::runtime_error("could not configure CUDA module loading");
}
}
