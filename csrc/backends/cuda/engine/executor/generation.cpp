#include "models/deepseek_v4/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/minicpmo45/forward.h"
#include "models/nemotron_h/ops.h"

#include "backends/cuda/engine/executor/generation_impl.h"

namespace mfq::cuda::internal {
#define TYLOINFER_INSTANTIATE_FLOW(MODEL, STATE)                                                                \
    template Generation generate(MODEL &, STATE &, mfq::cuda::Executor*, mfq::cuda::Collectives&, \
        mfq::cuda::Executor::Mode, CudaProfiler&, mfq::cuda::RoutedCache&, bool&, bool, bool,       \
        std::function<void()>,                                                                      \
        const mfq::models::CausalLmMetadata &,                                                       \
        DecodeGraphCache &,                                                                        \
        TextSessionCache &,                                                                        \
        const CudaRuntimeConfig &,                                                                 \
        mfq::engine::InferenceRequest &,                                                           \
        mfq::engine::InferenceOutput &,                                                            \
        MtpModule *,                                                                               \
        std::optional<PreparedPrompt>,                                                         \
        mfq::engine::Batch*, std::string,                    \
        std::function<void()>);
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::minicpmo45::Model, mfq::cuda::minicpmo45::State)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::MiniCPMO45CausalLm, mfq::cuda::MiniCPMO45CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::Gemma4CausalLm, mfq::cuda::Gemma4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::GlmDsaCausalLm, mfq::cuda::GlmDsaCausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::Glm5CausalLm, mfq::cuda::Glm5CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::Qwen4CausalLm, mfq::cuda::Qwen4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::DeepseekV4CausalLm, mfq::cuda::DeepseekV4CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::DeepseekV41CausalLm, mfq::cuda::DeepseekV41CausalLm)
TYLOINFER_INSTANTIATE_FLOW(mfq::cuda::NemotronHCausalLm, mfq::cuda::NemotronHCausalLm)
#undef TYLOINFER_INSTANTIATE_FLOW
} // namespace mfq::cuda::internal
