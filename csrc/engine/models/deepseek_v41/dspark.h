#pragma once

#include <memory>

namespace tyloi {
class ModelSource;
class Tensor;
namespace models { struct Mtp; namespace deepseek_v41 { struct Config; } }
}
struct CudaExecutionContext;
namespace tyloi::weights { struct Loader; }

namespace tyloi::cuda::deepseek_v41_runtime {

std::unique_ptr<tyloi::models::Mtp> load_dspark_if_present(
    CudaExecutionContext& execution,
    const tyloi::weights::Loader& weights,
    const tyloi::models::deepseek_v41::Config& config,
    std::shared_ptr<const ::tyloi::Tensor> embedding, std::shared_ptr<const ::tyloi::Tensor> output, double softcap);
void run_dspark_self_check();

} // namespace tyloi::cuda::deepseek_v41_runtime
