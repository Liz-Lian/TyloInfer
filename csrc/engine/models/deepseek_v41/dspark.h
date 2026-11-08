#pragma once

#include <memory>

namespace tyloi {
class ModelSource;
class Tensor;
namespace models { struct Mtp; namespace deepseek_v41 { struct Config; } }
}
#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda { class Pipelines; }
#else
struct CudaExecutionContext;
#endif
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::deepseek_v41_runtime {

std::unique_ptr<tyloi::models::Mtp> load_dspark_if_present(
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::cuda::Pipelines& pipelines,
#else
    CudaExecutionContext& execution,
#endif
    const tyloi::weights::Loader& weights,
    const tyloi::models::deepseek_v41::Config& config,
    std::shared_ptr<const ::tyloi::Tensor> embedding, std::shared_ptr<const ::tyloi::Tensor> output, double softcap);

} // namespace tyloi::cuda::deepseek_v41_runtime
