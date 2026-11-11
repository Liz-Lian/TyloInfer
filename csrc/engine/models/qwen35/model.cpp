#include "engine/models/qwen35/model.h"
#include "engine/ops.h"

#include <stdexcept>
#include <utility>

namespace tyloi::cuda::qwen35 {

#ifdef TYLOINFER_NATIVE_CUDA
::tyloi::Tensor Model::embedding(const tyloi::Tensor& ids, int primary) const {
    if (ids.dim() != 1 && ids.dim() != 2)
        throw std::runtime_error("token IDs must have shape [tokens] or [batch,tokens]");
    auto shaped = ids.dim() == 1 ? tyloi::Tensor(tyloi::ops::unsqueeze(ids, 0)) : tyloi::Tensor{};
    const auto& input = shaped ? shaped : ids;
    if (input.size(0) <= 0 || input.size(1) <= 0)
        throw std::runtime_error("token IDs must not be empty");
    auto normalized = tyloi::ops::contiguous(tyloi::ops::copy(
        tyloi::ops::cast(input, tyloi::ScalarType::int64),
        tyloi::Device{tyloi::DeviceType::cuda, primary}));
    auto result = tyloi::ops::measure("model.embed", normalized.device(), [&] {
        return tyloi::ops::embedding(normalized, embed);
    });
    return tyloi::ops::contiguous(tyloi::ops::cast(result, tyloi::ScalarType::float16));
}
#endif

} // namespace tyloi::cuda::qwen35
