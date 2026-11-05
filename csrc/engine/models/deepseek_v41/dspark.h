#pragma once

#include <memory>

namespace tyloi {
class ModelSource;
class Tensor;
namespace models::deepseek_v41 { struct Config; }
}
struct CudaExecutionContext;
struct MtpModule;
namespace tyloi::cuda::weight_loader { struct Loader; }

namespace tyloi::cuda::deepseek_v41_runtime {

std::unique_ptr<::MtpModule> load_dspark_if_present(
    CudaExecutionContext& execution,
    const weight_loader::Loader& weights,
    const tyloi::models::deepseek_v41::Config& config,
    std::shared_ptr<const ::tyloi::Tensor> embedding, std::shared_ptr<const ::tyloi::Tensor> output, double softcap);
void run_dspark_self_check();

} // namespace tyloi::cuda::deepseek_v41_runtime
