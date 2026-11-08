#pragma once

#include <memory>

namespace tyloi::models { struct Mtp; }
namespace tyloi::weights { struct Loader; class Cache; }
#ifndef TYLOINFER_NATIVE_CUDA
struct CudaExecutionContext;
#endif

namespace tyloi::cuda::deepseek_v41 {

struct Model;

std::unique_ptr<tyloi::models::Mtp> load_components(
    Model& model, const tyloi::weights::Loader& weights, bool load_optional_components,
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::weights::Cache& cache);
#else
    CudaExecutionContext& execution);
#endif

} // namespace tyloi::cuda::deepseek_v41
