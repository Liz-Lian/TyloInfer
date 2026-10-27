#include "models/mtp.h"

#include "engine/executor/qwen35.h"
#include "models/deepseek_v4/ops.h"
#include "models/deepseek_v41/ops.h"
#include "models/gemma4/ops.h"
#include "models/glm5_next/ops.h"
#include "models/glm_dsa/ops.h"
#include "models/minicpmo45/ops.h"
#include "models/nemotron_h/ops.h"
#include "models/qwen4_exp/ops.h"

MtpTarget::MtpTarget(mfq::cuda::qwen35::Run& target)
    : model(&target), rope(&target.weights->rope) {}

mfq_tensor_backend::Tensor MtpTarget::embed(mfq_tensor_backend::Tensor ids) const {
    return std::visit([&](auto* target) { return target->embed_forward(std::move(ids)); }, model);
}

mfq_tensor_backend::Tensor MtpTarget::logits(mfq_tensor_backend::Tensor hidden) const {
    return std::visit([&](auto* target) { return target->logits_from_hidden(std::move(hidden)); }, model);
}
