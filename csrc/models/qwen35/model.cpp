#include "models/qwen35/model.h"

namespace mfq::cuda::qwen35 {

std::array<mfq::cuda::Task, 2> Layer::tasks() const {
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

} // namespace mfq::cuda::qwen35
