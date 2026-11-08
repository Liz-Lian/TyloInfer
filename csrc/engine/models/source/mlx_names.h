#pragma once

#include <string>
#include <string_view>
#include <cstdint>
#include <vector>

namespace mfq {

inline std::string mlx_hf_tensor_name(std::string_view name) {
    std::string result(name);
    constexpr std::string_view language = "language_model.model.";
    constexpr std::string_view head = "language_model.lm_head.";
    if (name.substr(0, language.size()) == language)
        result = "model.language_model." + std::string(name.substr(language.size()));
    else if (name.substr(0, head.size()) == head)
        result = "lm_head." + std::string(name.substr(head.size()));
    const auto replace = [&](std::string_view from, std::string_view to) {
        const auto offset = result.find(from);
        if (offset != std::string::npos) result.replace(offset, from.size(), to);
    };
    replace(".mlp.switch_mlp.", ".mlp.experts.");
    replace(".ngram_embedding.shards.", ".ngram_embedding.shard_");
    return result;
}

inline std::vector<std::int64_t> mlx_hf_tensor_shape(std::string_view canonical,
    std::vector<std::int64_t> shape) {
    const auto matches = [&](std::string_view suffix) {
        return canonical.size() >= suffix.size() && canonical.substr(canonical.size() - suffix.size()) == suffix;
    };
    if ((matches(".linear_attention.conv.weight") || matches(".position_embedding.conv.weight")) &&
        shape.size() == 3 && shape[2] == 1 && shape[1] > 1) {
        shape[2] = shape[1]; shape[1] = 1;
    }
    return shape;
}

} // namespace mfq
