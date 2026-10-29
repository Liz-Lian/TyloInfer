#include "models/load.h"
#include "models/batch.h"
#include "models/qwen35/forward.h"
#include "models/minicpmo45/forward.h"
#include "engine/executor/backend.h"
#include "engine/executor/instance.h"

#include <memory>
#include <utility>
#include <variant>

namespace mfq::cuda::engine_detail {

using State = std::variant<
    std::unique_ptr<Text<Loaded<qwen35::Model,qwen35::State,qwen35::Components>>>,
    std::unique_ptr<Text<Loaded<MiniCPMO45CausalLm>>>,
    std::unique_ptr<Text<Loaded<minicpmo45::Model, minicpmo45::State>>>,
    std::unique_ptr<Text<Loaded<Gemma4CausalLm>>>,
    std::unique_ptr<Text<Loaded<GlmDsaCausalLm>>>,
    std::unique_ptr<Text<Loaded<Glm5CausalLm>>>,
    std::unique_ptr<Text<Loaded<Qwen4CausalLm>>>,
    std::unique_ptr<Text<Loaded<DeepseekV4CausalLm>>>,
    std::unique_ptr<Text<Loaded<DeepseekV41CausalLm>>>,
    std::unique_ptr<Text<Loaded<NemotronHCausalLm>>> >;

} // namespace mfq::cuda::engine_detail

namespace mfq::cuda {

std::unique_ptr<engine::Engine> load_cuda_engine(CudaEngineOptions options) {
    return std::make_unique<engine::EngineInstance<
        engine_detail::Backend<engine_detail::State, with_model>>>(std::move(options));
}

} // namespace mfq::cuda
