#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tyloi { struct ModelGraph; }

struct MfqModelCapabilities {
    std::string family = "unknown";
    bool text = true;
    bool image_input = false;
    bool video_input = false;
    bool audio_input = false;
    bool audio_output = false;
    bool full_duplex = false;
    bool mtp = false;
    std::string source;
};

namespace tyloi::models {

/** Identify a supported model backbone implementation. */
enum class Backbone { generic_qwen, minicpmo45, minicpmo_tts, gemma4, glm_dsa,
    glm5_next, qwen4_exp, deepseek_v4, deepseek_v41, nemotron_h, unsupported };
/** Identify a supported multimodal input implementation. */
enum class Vision { none, grid_vit, minicpmo45, deepseek_v41, radio };
/** Identify a supported next-token prediction implementation. */
enum class Predictor { none, qwen35, flash_next, deepseek_v41_dspark, nemotron_h };
/** Describe the implementations selected for a model graph. */
struct Plan {
    Backbone backbone = Backbone::unsupported;
    Vision vision = Vision::none;
    Predictor predictor = Predictor::none;
};
/** Describe input, output and prediction capabilities. */
struct Capabilities {
    bool text = false, image_input = false, video_input = false;
    bool audio_input = false, audio_output = false, full_duplex = false, mtp = false;
};
/** Describe declared, supported, available and enabled components. */
struct ComponentState {
    bool vision_declared = false, vision_supported = false, vision_available = false, vision_enabled = false;
    bool mtp_declared = false, mtp_supported = false, mtp_available = false, mtp_enabled = false;
};
/** Return the capabilities implied by the selected implementations. */
Capabilities capability_matrix(const Plan& plan) noexcept;
/** Describe components using actual loaded and enabled state. */
ComponentState component_state(const ModelGraph& graph, const Plan& plan, bool vision_loaded,
    bool mtp_loaded, bool enable_vision = true, bool enable_mtp = true) noexcept;
/** Resolve capabilities against the graph and loaded components. */
Capabilities runtime_capabilities(const ModelGraph& graph, const Plan& plan,
    const ComponentState& components, bool multimodal_available, bool duplex_available) noexcept;

/** State whether decode capture needs resident weights or can capture managed weight access. */
enum class GraphWeights { captured, resident };
/** Select ordinary chunks or a merged logical span with model-owned internal tiles. */
enum class PrefillPolicy { chunked, merged };
/** Identify who organises model-specific decode graph execution. */
enum class GraphOwner { executor, model };

struct Info {
    std::int64_t vocab_size;
    std::int64_t max_context;
    std::string model_type;
    std::size_t prefill_row_bytes;
    bool decode_graph;
    bool double_warmup;
    GraphWeights graph_weights = GraphWeights::captured;
    PrefillPolicy prefill = PrefillPolicy::chunked;
    /** Whether the model supports early host-token row prefetch. */
    bool row_prefetch = false;
    GraphOwner graph = GraphOwner::executor;
    /** Whether attention state supports the configured streamed KV storage policy. */
    bool streamed_kv = false;
};

struct CausalLmMetadata {
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t num_hidden_layers = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t max_position_embeddings = 0;
    int64_t rotary_dim = 0;
    int64_t num_experts = 0;
    int64_t hc_mult = 1;
    double rope_base = 0.0;
    double rms_norm_eps = 1e-6;
    double norm_weight_offset = 0.0;
    double hc_eps = 1e-6;
    double final_logit_softcapping = 0.0;
    double embedding_scale = 1.0;
    bool tie_word_embeddings = false;
    bool rope_interleaved = false;
    // NoPE models have no rotary table; adapters leave rotary_dim at zero.
    bool no_rope = false;
    bool multi_axis_positions = false;
    bool flash_next = false;
    bool gemma4 = false;
    bool decode_graph_double_warmup = false;
    std::string model_type;
    std::vector<std::string> layer_types;
};

inline Info model_info(const CausalLmMetadata& metadata) {
    return {metadata.vocab_size, metadata.max_position_embeddings, metadata.model_type,
        std::size_t(metadata.hidden_size) * metadata.hc_mult * sizeof(float) * 8,
        !metadata.flash_next, metadata.decode_graph_double_warmup};
}

template <class Model>
Info model_info(const Model& model) {
    if constexpr (requires { model.info(); }) return model.info();
    else return model_info(model.metadata);
}

} // namespace tyloi::models
