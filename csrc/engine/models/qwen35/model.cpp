#include "engine/models/qwen35/model.h"
#include "engine/ops.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

#ifdef TYLOINFER_NATIVE_CUDA
std::shared_ptr<tyloi::Tensor> Model::embedding(const tyloi::Tensor& ids, int primary) const {
    if (ids.dim() != 1 && ids.dim() != 2)
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::ops::unsqueeze(ids, 0) : nullptr;
    const auto& input = shaped ? *shaped : ids;
    if (input.size(0) <= 0 || input.size(1) <= 0)
        throw std::runtime_error("token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(*tyloi::ops::copy(
        *tyloi::ops::cast(input, tyloi::ScalarType::int64),
        tyloi::Device{tyloi::DeviceType::cuda, primary}));
    auto result = tyloi::ops::measure("model.embed", normalized->device(), [&] {
        return tyloi::ops::embedding(*normalized, *embed);
    });
    return tyloi::ops::contiguous(*tyloi::ops::cast(*result, tyloi::ScalarType::float16));
}
#endif

std::array<tyloi::cuda::Task, 2> Layer::tasks() const {
    if (const auto* full = std::get_if<Full>(&weights)) {
        return {{
            {task::Attention{0, 1, 2, full->attention, 0}, full->attention.layer},
#ifdef TYLOINFER_NATIVE_CUDA
            {task::FullFfn{1, 2, 0, full->ffn, *full->ffn_norm, full->attention},
#else
            {task::FullFfn{1, 2, 0, full->ffn, full->ffn_norm, full->attention},
#endif
             full->attention.layer},
        }};
    }
    const auto& linear = std::get<Linear>(weights);
    return {{
        {task::Linear{0, 1, 2, linear.attention, 0}},
#ifdef TYLOINFER_NATIVE_CUDA
        {task::LinearFfn{1, 2, 0, linear.ffn, *linear.ffn_norm, linear.attention}},
#else
        {task::LinearFfn{1, 2, 0, linear.ffn, linear.ffn_norm, linear.attention}},
#endif
    }};
}

Task Linear::replay(std::int64_t tokens) const {
    return {task::Replay{attention, 0, tokens}};
}

} // namespace tyloi::cuda::qwen35
