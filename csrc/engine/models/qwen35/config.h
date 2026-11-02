#pragma once

#include "engine/models/common/vision/grid_vision.h"
#include "engine/models/common/legacy_names.h"
#include "engine/models/common/model_graph.h"
#include "engine/models/common/model_config.h"
#include "engine/backends/cuda/ops/ffn/math.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mfq::models::qwen35 {

struct Config : ModelConfig {
    TensorLayout legacy_tensor_layout;
    bool attention_output_gate = false;
    std::int64_t num_experts = 0;
    std::int64_t num_experts_per_tok = 0;
    std::int64_t moe_intermediate_size = 0;
    std::int64_t shared_expert_intermediate_size = 0;
    double routed_scaling_factor = 1.0;
    bool norm_topk_prob = false;
    std::string expert_gating_func = "softmax";
    std::int64_t mtp_num_hidden_layers = 0;
    bool mtp_use_dedicated_embeddings = false;
    std::int64_t linear_conv_kernel_dim = 4;
    std::int64_t linear_key_head_dim = 128;
    std::int64_t linear_value_head_dim = 128;
    std::int64_t linear_num_key_heads = 0;
    std::int64_t linear_num_value_heads = 0;
    std::vector<std::int64_t> mrope_sections;
    bool mrope_interleaved = false;
    std::optional<GridVisionConfig> grid_vision;
    std::optional<std::int64_t> image_token_id;
    std::optional<std::int64_t> video_token_id;

    enum class AttentionKind { full, linear };
    static AttentionKind attention_kind(std::string_view type) {
        if (type == "full_attention")
            return AttentionKind::full;
        if (type == "linear_attention")
            return AttentionKind::linear;
        throw std::runtime_error("unsupported Qwen attention type: " + std::string(type));
    }
    mfq::cuda::MoeRouting routing() const {
        if (expert_gating_func == "sqrtsoftplus")
            return {mfq::cuda::RouterActivation::sqrt_softplus, norm_topk_prob, false, routed_scaling_factor};
        // Full softmax -> top-k -> renormalize equals softmax on selected logits.
        return {mfq::cuda::RouterActivation::softmax, false, true, 1.0};
    }
    static bool valid_positions(int64_t rank, int64_t rows, int64_t columns, int64_t batch,
                                int64_t tokens, bool mrope) {
        return (rank == 1 && columns == tokens) ||
               (rank == 2 && columns == tokens && (rows == batch || (mrope && rows == 3)));
    }

    std::int64_t linear_k_size() const noexcept {
        return linear_num_key_heads * linear_key_head_dim;
    }
    std::int64_t linear_v_size() const noexcept {
        return linear_num_value_heads * linear_value_head_dim;
    }

    static Config from_json(std::string_view payload, const ModelGraph &graph);

};

} // namespace mfq::models::qwen35
