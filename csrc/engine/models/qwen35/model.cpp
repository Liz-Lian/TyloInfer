#include "engine/models/qwen35/model.h"
#include "engine/ops.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
tyloi::DenseTensor Model::embedding(tyloi::DenseTensor ids, int primary) const {
    if (ids.dim() != 1 && ids.dim() != 2)
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    if (ids.dim() == 1) ids = DenseTensor(std::move(ids)).unsqueeze(0);
    if (ids.size(0) <= 0 || ids.size(1) <= 0)
        throw std::runtime_error("token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(tyloi::ops::copy(
        tyloi::ops::cast(std::move(ids), tyloi::ScalarType::int64),
        tyloi::Device{tyloi::DeviceType::cuda, primary}));
    auto result = tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(std::move(normalized), *embed);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(std::move(result), tyloi::ScalarType::float16));
}
#endif

std::array<tyloi::cuda::Task, 2> Layer::tasks() const {
    if (const auto* full = std::get_if<Full>(&weights)) {
        return {{
            {task::Attention{0, 1, 2, full->attention, 0}, full->attention.layer},
            {task::FullFfn{1, 2, 0, full->ffn, full->ffn_norm, full->attention},
             full->attention.layer},
        }};
    }
    const auto& linear = std::get<Linear>(weights);
    return {{
        {task::Linear{0, 1, 2, linear.attention, 0}},
        {task::LinearFfn{1, 2, 0, linear.ffn, linear.ffn_norm, linear.attention}},
    }};
}

Task Linear::replay(std::int64_t tokens) const {
    return {task::Replay{attention, 0, tokens}};
}

} // namespace tyloi::cuda::qwen35
