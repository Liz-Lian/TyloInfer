#pragma once

#include "engine/executor/mtp.h"
#include "models/common/causal_model_ops.h"

#include <type_traits>
#include <variant>

struct RopeCache;

namespace mfq::cuda {
class Collectives;
class Executor;
}

namespace mfq::cuda::qwen35 {
struct Model;
}

struct QwenTarget {
    const mfq::cuda::qwen35::Model* model = nullptr;
    mfq::cuda::Executor* executor = nullptr;
    mfq::cuda::Collectives* collectives = nullptr;
    int primary = 0;
};

struct MtpTarget {
    std::variant<QwenTarget, mfq::cuda::MiniCPMO45CausalLm*,
        mfq::cuda::MiniCPMOTtsCausalLm*, mfq::cuda::Gemma4CausalLm*,
        mfq::cuda::GlmDsaCausalLm*, mfq::cuda::Glm5CausalLm*, mfq::cuda::Qwen4CausalLm*,
        mfq::cuda::DeepseekV4CausalLm*, mfq::cuda::DeepseekV41CausalLm*,
        mfq::cuda::NemotronHCausalLm*> model;
    const RopeCache* rope = nullptr;
    template <class Model>
        requires std::is_constructible_v<decltype(model), Model*>
    explicit MtpTarget(Model& target) : model(&target), rope(&target.rope) {}
    template <class Model>
        requires std::is_constructible_v<decltype(model), Model*>
    MtpTarget(Model& target, mfq::cuda::Executor*, mfq::cuda::Collectives&, int)
        : MtpTarget(target) {}
    explicit MtpTarget(const mfq::cuda::qwen35::Model& target,
                       mfq::cuda::Executor* executor,
                       mfq::cuda::Collectives& collectives, int primary);
    mfq_tensor_backend::Tensor embed(mfq_tensor_backend::Tensor ids) const;
    mfq_tensor_backend::Tensor logits(mfq_tensor_backend::Tensor hidden) const;
};
