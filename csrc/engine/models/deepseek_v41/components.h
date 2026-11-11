#pragma once

#include <memory>

namespace tyloi::models { struct Mtp; }
namespace tyloi::weights { struct Loader; }
#ifndef TYLOINFER_NATIVE_CUDA
struct CudaExecutionContext;
#endif

namespace tyloi::cuda::deepseek_v41 {

struct Model;

#ifdef TYLOINFER_NATIVE_CUDA
/** Load the model's optional vision and predictor components. */
std::unique_ptr<tyloi::models::Mtp> load_components(
    Model& model, const tyloi::weights::Loader& weights, bool load_optional_components);
#else
/** Load the model's optional vision and predictor components. */
std::unique_ptr<tyloi::models::Mtp> load_components(
    Model& model, const tyloi::weights::Loader& weights, bool load_optional_components,
    CudaExecutionContext& execution);
#endif

} // namespace tyloi::cuda::deepseek_v41
