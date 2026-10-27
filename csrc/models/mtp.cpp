#include "models/mtp.h"

#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/ops.h"
#include "models/qwen35/forward.h"
#include "models/qwen4_exp/ops.h"

#include <stdexcept>
#include <type_traits>
#include <utility>

MtpTarget::MtpTarget(const mfq::cuda::qwen35::Model& target,
                     mfq::cuda::Executor* executor,
                     mfq::cuda::Collectives& collectives, int primary)
    : model(QwenTarget{&target, executor, &collectives, primary}), rope(&target.rope) {
    if (executor == nullptr) throw std::invalid_argument("Qwen MTP target requires executor");
}

mfq_tensor_backend::Tensor MtpTarget::embed(mfq_tensor_backend::Tensor ids) const {
    return std::visit(
        [&](const auto& target) -> mfq_tensor_backend::Tensor {
            using Target = std::decay_t<decltype(target)>;
            if constexpr (std::is_same_v<Target, QwenTarget>) {
                return mfq::cuda::qwen35::embed(*target.model, *target.executor,
                                                *target.collectives, target.primary,
                                                std::move(ids));
            } else {
                return target->embed_forward(std::move(ids));
            }
        },
        model);
}

mfq_tensor_backend::Tensor MtpTarget::logits(mfq_tensor_backend::Tensor hidden) const {
    return std::visit(
        [&](const auto& target) -> mfq_tensor_backend::Tensor {
            using Target = std::decay_t<decltype(target)>;
            if constexpr (std::is_same_v<Target, QwenTarget>) {
                return mfq::cuda::qwen35::logits(*target.model, *target.executor,
                                                 std::move(hidden));
            } else {
                return target->logits_from_hidden(std::move(hidden));
            }
        },
        model);
}
