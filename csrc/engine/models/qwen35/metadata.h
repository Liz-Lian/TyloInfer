#pragma once

#include "models/qwen35/config.h"
#include "../common/causal_metadata.h"

namespace mfq::models::qwen35 {

inline models::CausalLmMetadata metadata_from_config(const Config& config) {
    models::CausalLmMetadata metadata;
    metadata.vocab_size = config.vocab_size;
    metadata.hidden_size = config.hidden_size;
    metadata.num_hidden_layers = config.num_hidden_layers;
    metadata.num_attention_heads = config.num_attention_heads;
    metadata.num_key_value_heads = config.num_key_value_heads;
    metadata.head_dim = config.head_dim;
    metadata.max_position_embeddings = config.max_position_embeddings;
    metadata.rotary_dim = config.rotary_dim;
    metadata.num_experts = config.num_experts;
    metadata.rope_base = config.rope_base;
    metadata.rms_norm_eps = config.rms_norm_eps;
    metadata.norm_weight_offset = config.legacy_tensor_layout.norm_weight_offset;
    metadata.tie_word_embeddings = config.tie_word_embeddings;
    metadata.model_type = config.model_type;
    metadata.layer_types = config.layer_types;
    return metadata;
}

} // namespace mfq::models::qwen35
