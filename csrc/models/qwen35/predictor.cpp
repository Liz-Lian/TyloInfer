#include "models/qwen35/predictor.h"

#include <utility>

namespace mfq::cuda::qwen35 {

std::vector<Task> Predictor::tasks() const {
    const auto eps = config.rms_norm_eps;
    const auto offset = config.legacy_tensor_layout.norm_weight_offset;
    std::vector<Task> result;
    result.reserve(5 + 2 * layers.size());
    result.push_back({task::Norm{1, 1, embedding_norm, eps, offset}, 0, "mtp.embedding_norm"});
    result.push_back({task::Norm{0, 0, hidden_norm, eps, offset}, 0, "mtp.hidden_norm"});
    result.push_back({task::Concat{1, 0, 0}});
    result.push_back({task::Project{0, 0, fusion}, 0, "mtp.fusion"});
    const auto dense = [](const Ffn& ffn) -> Mlp {
        TYLOINFER_RUNTIME_CHECK(!ffn.is_moe, "Qwen MTP requires dense predictor projections");
        return {ffn.gate_up, ffn.down, ffn.activation, ffn.geglu, ffn.swiglu_limit};
    };
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        const auto& attention = layer.attention;
        result.push_back({task::Attention{0, 1, 2, attention, index}, attention.layer});
        task::Dense ffn{1, 2, 0, {dense(layer.ffn), {}, layer.ffn_norm,
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate}};
        if (layer.ffn.important_neurons)
            ffn.weights.important.emplace(dense(*layer.ffn.important_neurons));
        result.push_back({std::move(ffn), attention.layer});
    }
    result.push_back({task::Norm{0, 0, output_norm, eps, offset}, 0, "mtp.output_norm"});
    return result;
}

} // namespace mfq::cuda::qwen35
