#pragma once
#include <cstdlib>
#include <stdexcept>

namespace mfq::cuda {
inline void configure_cuda_startup() {
    // Load the kernels needed by this model. Eager loading also retains code
    // and driver metadata for unused formats and model architectures. Backend
    // warmup prepares the active paths before serving requests. Honor overrides.
    if(std::getenv("CUDA_MODULE_LOADING"))return;
#if defined(_WIN32)
    const auto error=_putenv_s("CUDA_MODULE_LOADING","LAZY");
#else
    const auto error=setenv("CUDA_MODULE_LOADING","LAZY",0);
#endif
    if(error)throw std::runtime_error("could not configure CUDA module loading");
}
}
