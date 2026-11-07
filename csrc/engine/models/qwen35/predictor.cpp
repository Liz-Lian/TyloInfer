#include "engine/models/qwen35/predictor.h"

#include <utility>

namespace tyloi::cuda::qwen35 {

std::vector<Task> Predictor::tasks() const {
    const auto eps = config.rms_norm_eps;
    const auto offset = config.legacy_tensor_layout.norm_weight_offset;
    std::vector<Task> result;
    result.reserve(5 + 2 * layers.size());
#ifdef TYLOINFER_NATIVE_CUDA
    result.push_back({task::Norm{1, 1, *embedding_norm, eps, offset}, 0, "mtp.embedding_norm"});
    result.push_back({task::Norm{0, 0, *hidden_norm, eps, offset}, 0, "mtp.hidden_norm"});
#else
    result.push_back({task::Norm{1, 1, embedding_norm, eps, offset}, 0, "mtp.embedding_norm"});
    result.push_back({task::Norm{0, 0, hidden_norm, eps, offset}, 0, "mtp.hidden_norm"});
#endif
    result.push_back({task::Concat{1, 0, 0}});
    result.push_back({task::Project{0, 0, *fusion}, 0, "mtp.fusion"});
    for (size_t index = 0; index < layers.size(); ++index) {
        const auto& layer = layers[index];
        const auto& attention = layer.attention;
        TYLOINFER_RUNTIME_CHECK(!layer.ffn.is_moe &&
            (!layer.ffn.important_neurons || !layer.ffn.important_neurons->is_moe),
            "Qwen MTP requires dense predictor projections");
        result.push_back({task::Attention{0, 1, 2, attention, index}, attention.layer});
#ifdef TYLOINFER_NATIVE_CUDA
        task::Dense ffn{1, 2, 0, layer.ffn, *layer.ffn_norm,
#else
        task::Dense ffn{1, 2, 0, layer.ffn, layer.ffn_norm,
#endif
            attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate};
        result.push_back({std::move(ffn), attention.layer});
    }
#ifdef TYLOINFER_NATIVE_CUDA
    result.push_back({task::Norm{0, 0, *output_norm, eps, offset}, 0, "mtp.output_norm"});
#else
    result.push_back({task::Norm{0, 0, output_norm, eps, offset}, 0, "mtp.output_norm"});
#endif
    return result;
}

} // namespace tyloi::cuda::qwen35
