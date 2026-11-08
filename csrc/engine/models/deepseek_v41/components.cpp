#include "engine/models/deepseek_v41/components.h"

#include "engine/models/deepseek_v41/dspark.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/deepseek_v41/vision.h"
#include "engine/models/common/mtp.h"

#include <stdexcept>

namespace tyloi::cuda::deepseek_v41 {

std::unique_ptr<tyloi::models::Mtp> load_components(
        Model& model, const tyloi::weights::Loader& weights, bool load_optional_components,
#ifdef TYLOINFER_NATIVE_CUDA
        tyloi::weights::Cache& cache) {
#else
        CudaExecutionContext& execution) {
    auto& cache = execution.weights;
#endif
    if (!load_optional_components) return {};
    if ((model.plan.vision != CudaVisionAdapter::none &&
         model.plan.vision != CudaVisionAdapter::deepseek_v41) ||
            (model.plan.predictor != CudaPredictorAdapter::none &&
             model.plan.predictor != CudaPredictorAdapter::deepseek_v41_dspark)) {
        throw std::runtime_error("unsupported DeepSeek-V4.1 CUDA component adapter");
    }
    if (model.plan.vision == CudaVisionAdapter::deepseek_v41) {
        model.vision = std::make_unique<deepseek_v41_runtime::Vision>(
            deepseek_v41_runtime::Vision::load(model.source, weights, cache, model.config));
    }
    if (model.plan.predictor == CudaPredictorAdapter::none) return {};
    TYLOINFER_RUNTIME_CHECK(model.supports_suffix_speculation() && model.tables.engram_parameters,
        "invalid DeepSeek-V4.1 DSpark model");
    return deepseek_v41_runtime::load_dspark_if_present(
#ifndef TYLOINFER_NATIVE_CUDA
        execution,
#endif
        weights, model.config, model.embed, model.lm_head,
        model.metadata.final_logit_softcapping);
}

} // namespace tyloi::cuda::deepseek_v41
