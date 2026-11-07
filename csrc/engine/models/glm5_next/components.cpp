#include "engine/models/glm5_next/components.h"

#include "engine/models/glm5_next/forward.h"
#include "engine/models/glm5_next/model.h"
#include "engine/models/glm5_next/mtp.h"
#include "engine/weights/loader.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::glm5_next {

std::unique_ptr<tyloi::models::Mtp> load_components(
        tyloi::cuda::glm5_next::Model& model,
        const tyloi::weights::Loader& weights,
        bool load_optional_components, tyloi::cuda::Projector projector,
        tyloi::cuda::Collectives& copies, CudaProfiler& profiler) {
    std::unique_ptr<tyloi::models::Mtp> result;
    if (!load_optional_components) return result;
    if (model.plan.vision != tyloi::cuda::CudaVisionAdapter::none ||
            (model.plan.predictor != tyloi::cuda::CudaPredictorAdapter::none &&
             model.plan.predictor !=
                 tyloi::cuda::CudaPredictorAdapter::flash_next)) {
        throw std::runtime_error(
            "unsupported GLM5 CUDA component adapter");
    }
    if (model.plan.predictor ==
            tyloi::cuda::CudaPredictorAdapter::none) return result;
    TYLOINFER_RUNTIME_CHECK(
        tyloi::cuda::glm5_next::supports_speculation(model),
        "invalid Flash-Next predictor model");
    auto predictor = tyloi::cuda::glm5_next::Mtp::load_if_present(
        weights, std::move(projector), copies, profiler, model.embed, model.config);
    if (predictor) {
        result =
            std::make_unique<tyloi::cuda::glm5_next::Mtp>(
                std::move(*predictor));
    }
    return result;
}

} // namespace tyloi::cuda::glm5_next
