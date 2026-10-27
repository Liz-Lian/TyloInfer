#pragma once

#include "engine/executor/mtp.h"
#include "models/common/causal_model_ops.h"

#include <variant>

struct RopeCache;

namespace mfq::cuda::qwen35 {
class Run;
}

struct MtpTarget {
    std::variant<mfq::cuda::qwen35::Run*, mfq::cuda::MiniCPMO45CausalLm*,
        mfq::cuda::MiniCPMOTtsCausalLm*, mfq::cuda::Gemma4CausalLm*,
        mfq::cuda::GlmDsaCausalLm*, mfq::cuda::Glm5CausalLm*, mfq::cuda::Qwen4CausalLm*,
        mfq::cuda::DeepseekV4CausalLm*, mfq::cuda::DeepseekV41CausalLm*,
        mfq::cuda::NemotronHCausalLm*> model;
    const RopeCache* rope = nullptr;
    template <class Model> explicit MtpTarget(Model& target) : model(&target), rope(&target.rope) {}
    explicit MtpTarget(mfq::cuda::qwen35::Run& target);
    mfq_tensor_backend::Tensor embed(mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const;
};
