#include "engine/models/qwen35/predictor.h"
#include "engine/models/load.h"
#include "engine/models/qwen35/weights.h"

namespace mfq::cuda::qwen35 {

std::optional<Predictor> Predictor::load(TransformerWeightLoader& loader,
        const mfq::models::qwen35::Config& config, KvQuantization quantization) {
    const bool fusion_present = loader.has("predictor.fusion.weight");
    const bool any = fusion_present || loader.has("predictor.hidden_norm.weight") ||
        loader.has("predictor.embedding_norm.weight") ||
        loader.has("predictor.output_norm.weight") ||
        loader.has("predictor.block.0.attention.query.weight");
    if (!fusion_present) {
        TYLOINFER_RUNTIME_CHECK(!any, "Qwen model source contains an incomplete MTP head");
        return {};
    }
    TYLOINFER_RUNTIME_CHECK(config.mtp_num_hidden_layers > 0 && !config.mtp_use_dedicated_embeddings,
        "Qwen MTP requires declared predictor layers and shared embeddings");
    Predictor model;
    model.config = config;
    model.hidden_norm = loader.dense("predictor.hidden_norm.weight");
    model.embedding_norm = loader.dense("predictor.embedding_norm.weight");
    model.output_norm = loader.dense("predictor.output_norm.weight");
    model.fusion = loader.linear("predictor.fusion.weight");
    TYLOINFER_RUNTIME_CHECK(model.fusion.neuron_len() == 2 * config.hidden_size &&
        model.fusion.out() == config.hidden_size &&
        model.hidden_norm.numel() == config.hidden_size &&
        model.embedding_norm.numel() == config.hidden_size &&
        model.output_norm.numel() == config.hidden_size,
        "Qwen MTP component dimensions disagree with the backbone");
    model.layers.reserve(config.mtp_num_hidden_layers);
    for (int index = 0; index < config.mtp_num_hidden_layers; ++index) {
        const auto prefix = "predictor.block." + std::to_string(index) + ".";
        Predictor::Layer layer;
        layer.attention.kv_quantization = quantization;
        mfq::models::load_attention_mixer(layer.attention, loader, config, index, prefix,
            config.legacy_tensor_layout.norm_weight_offset, config.attention_output_gate);
        layer.ffn_norm = loader.dense(prefix + "mlp.norm.weight");
        layer.ffn = mfq::models::qwen35::load_ffn<Ffn>(loader, config, index, "predictor");
        model.layers.push_back(std::move(layer));
    }
    return model;
}

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
