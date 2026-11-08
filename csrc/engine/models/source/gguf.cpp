#include "gguf.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::models {

using json = nlohmann::json;

std::string gguf_name(const std::string& name) {
    if (name == "token_embd.weight") return "model.token_embedding.weight";
    if (name == "output.weight") return "model.output.weight";
    if (name == "output_norm.weight") return "model.output_norm.weight";
    if (name == "output_hc_norm.weight") return "model.mhc.pre.norm.weight";
    if (name == "output_hc_down.weight") return "model.mhc.pre.down.weight";
    if (name == "output_hc_up.weight") return "model.mhc.pre.up.weight";
    if (name.rfind("blk.", 0) != 0) return name;
    const auto dot = name.find('.', 4);
    if (dot == std::string::npos) return name;
    static const std::map<std::string, std::string> names{
        {"attn_norm.weight", "attention.norm.weight"},
        {"attn_q.weight", "attention.query.weight"},
        {"attn_k.weight", "attention.key.weight"},
        {"attn_v.weight", "attention.value.weight"},
        {"attn_output.weight", "attention.output.weight"},
        {"attn_q_norm.weight", "attention.query_norm.weight"},
        {"attn_k_norm.weight", "attention.key_norm.weight"},
        {"post_attention_norm.weight", "mlp.norm.weight"},
        {"ffn_norm.weight", "mlp.norm.weight"},
        {"ffn_gate.weight", "mlp.gate.weight"},
        {"ffn_up.weight", "mlp.up.weight"},
        {"ffn_down.weight", "mlp.down.weight"},
        {"attn_qkv.weight", "linear_attention.qkv.weight"},
        {"attn_gate.weight", "linear_attention.gate.weight"},
        {"ssm_alpha.weight", "linear_attention.alpha.weight"},
        {"ssm_beta.weight", "linear_attention.beta.weight"},
        {"ssm_conv1d.weight", "linear_attention.conv.weight"},
        {"ssm_dt.bias", "linear_attention.dt_bias"},
        {"ssm_a", "linear_attention.a"},
        {"ssm_norm.weight", "linear_attention.norm.weight"},
        {"ssm_out.weight", "linear_attention.output.weight"},
        {"ffn_gate_inp.weight", "mlp.router.weight"},
        {"ffn_gate_exps.weight", "mlp.experts.gate.weight"},
        {"ffn_up_exps.weight", "mlp.experts.up.weight"},
        {"ffn_down_exps.weight", "mlp.experts.down.weight"},
        {"ffn_gate_shexp.weight", "mlp.shared_expert.gate.weight"},
        {"ffn_up_shexp.weight", "mlp.shared_expert.up.weight"},
        {"ffn_down_shexp.weight", "mlp.shared_expert.down.weight"},
        {"ffn_gate_inp_shexp.weight", "mlp.shared_expert.router.weight"},
        {"hc_attn_norm.weight", "attention.mhc.pre.norm.weight"},
        {"hc_attn_down.weight", "attention.mhc.pre.down.weight"},
        {"hc_attn_up.weight", "attention.mhc.pre.up.weight"},
        {"hc_attn_inject.weight", "attention.mhc.post.inject.weight"},
        {"hc_ffn_norm.weight", "mlp.mhc.pre.norm.weight"},
        {"hc_ffn_down.weight", "mlp.mhc.pre.down.weight"},
        {"hc_ffn_up.weight", "mlp.mhc.pre.up.weight"},
        {"hc_ffn_inject.weight", "mlp.mhc.post.inject.weight"},
        {"indexer.q_proj.weight", "attention.indexer.query.weight"},
        {"indexer.k_proj.weight", "attention.indexer.key.weight"},
        {"indexer.q_norm.weight", "attention.indexer.query_norm.weight"},
        {"indexer.k_norm.weight", "attention.indexer.key_norm.weight"},
        {"ple_key.weight", "position_embedding.key.weight"},
        {"ple_value.weight", "position_embedding.value.weight"},
        {"ple_norm_key.weight", "position_embedding.key_norm.weight"},
        {"ple_norm_query.weight", "position_embedding.query_norm.weight"},
        {"ple_norm_conv.weight", "position_embedding.conv_norm.weight"},
        {"ple_conv1d.weight", "position_embedding.conv.weight"},
    };
    const auto found = names.find(name.substr(dot + 1));
    return found == names.end() ? name : "model.block." + name.substr(4, dot - 4) + "." + found->second;
}

json gguf_config(const json& meta, const json& shapes, const std::string& arch) {
    const auto key = [&](const char* suffix) { return arch + "." + suffix; };
    const auto number = [&](const char* suffix, json fallback = nullptr) -> json {
        const auto it = meta.find(key(suffix));
        return it == meta.end() ? fallback : *it;
    };
    json config;
    config["model_type"] = arch == "qwen35" ? "qwen3_5_text" : arch;
    for (const auto& [dst, src] : std::map<std::string, std::string>{
             {"hidden_size", "embedding_length"}, {"intermediate_size", "feed_forward_length"},
             {"num_hidden_layers", "block_count"}, {"num_attention_heads", "attention.head_count"},
             {"num_key_value_heads", "attention.head_count_kv"},
             {"max_position_embeddings", "context_length"},
             {"rms_norm_eps", "attention.layer_norm_rms_epsilon"},
             {"rope_theta", "rope.freq_base"}}) {
        if (meta.contains(key(src.c_str()))) config[dst] = meta[key(src.c_str())];
    }
    if (shapes.contains("model.token_embedding.weight"))
        config["vocab_size"] = shapes["model.token_embedding.weight"][0];
    config["tie_word_embeddings"] = !shapes.contains("model.output.weight");
    if (arch == "qwen35" || arch == "qwen4exp") {
        const auto heads = config.value("num_attention_heads", int64_t{0});
        const auto hidden = config.value("hidden_size", int64_t{0});
        if (heads <= 0 || hidden <= 0) throw std::runtime_error("GGUF is missing attention geometry");
        const auto dim = number("attention.key_length", hidden / heads).get<int64_t>();
        if (dim <= 0 || dim > std::numeric_limits<int32_t>::max())
            throw std::runtime_error("invalid GGUF attention head dimension");
        config["head_dim"] = dim;
        config["partial_rotary_factor"] = number("rope.dimension_count", dim).get<double>() / dim;
        config["norm_weight_offset"] = 0.0;
        config["linear_a_is_log"] = false;
        config["attn_output_gate"] = true;
        config["full_attention_interval"] = number("full_attention_interval", 4);
        config["linear_conv_kernel_dim"] = number("ssm.conv_kernel", 4);
        config["linear_key_head_dim"] = number("ssm.state_size", 128);
        config["linear_value_head_dim"] = number("ssm.state_size", 128);
        config["linear_num_key_heads"] = number("ssm.group_count", 0);
        config["linear_num_value_heads"] = number("ssm.time_step_rank", 0);
        json layers = json::array();
        const auto count = config.value("num_hidden_layers", int64_t{0});
        for (int64_t i = 0; i < count; ++i) {
            const auto prefix = "model.block." + std::to_string(i) + ".";
            layers.push_back(shapes.contains(prefix + "linear_attention.qkv.weight") ? "linear_attention" : "full_attention");
        }
        config["layer_types"] = std::move(layers);
        if (arch == "qwen4exp") {
            config["model_type"] = "qwen4_exp_text";
            config["linear_tiled_heads"] = true;
            for (const auto& [dst, src] : std::map<std::string, std::string>{
                     {"num_experts", "expert_count"}, {"num_experts_per_tok", "expert_used_count"},
                     {"moe_intermediate_size", "expert_feed_forward_length"},
                     {"shared_expert_intermediate_size", "expert_shared_feed_forward_length"},
                     {"hc_count", "hyper_connection.count"}, {"hc_lowrank", "hyper_connection.low_rank"},
                     {"indexer_n_heads", "attention.indexer.head_count"},
                     {"indexer_head_dim", "attention.indexer.key_length"},
                     {"indexer_budget", "attention.indexer.top_k"},
                     {"ngram_size", "ple.ngram_size"}, {"heads_per_ngram", "ple.heads_per_ngram"},
                     {"ple_conv_kernel_size", "ple.conv_kernel"}, {"eos_token_id", "ple.eos_token_id"}}) {
                if (!meta.contains(key(src.c_str()))) throw std::runtime_error("missing GGUF metadata: " + key(src.c_str()));
                config[dst] = meta[key(src.c_str())];
            }
            config["rope_parameters"] = {{"rope_theta", config.at("rope_theta")},
                {"partial_rotary_factor", config.at("partial_rotary_factor")}};
            if (const auto sections = number("rope.dimension_sections"); !sections.is_null()) {
                std::vector<int64_t> dimensions;
                for (const auto& section : sections) if (section.get<int64_t>() > 0) dimensions.push_back(section.get<int64_t>());
                config["rope_parameters"]["mrope_section"] = dimensions;
                config["rope_parameters"]["mrope_interleaved"] = true;
            }
            int64_t ratio = 0;
            for (const auto& value : number("attention.compress_ratios", json::array())) {
                const auto current = value.get<int64_t>();
                if (current <= 0) continue;
                if (ratio && ratio != current) throw std::runtime_error("mixed GGUF QSA compression ratios are unsupported");
                ratio = current;
            }
            if (!ratio) throw std::runtime_error("GGUF QSA compression ratio is missing");
            config["indexer_compress_ratio"] = ratio;
            config["indexer_kv_heads"] = 1;
            config["ple_embed_dim"] = hidden;
            config["split_ngram_parts"] = 1;
            config["ple_layer_ids"] = json::array();
            for (const auto& layer : number("ple.layers", json::array())) config["ple_layer_ids"].push_back(layer.get<int64_t>() + 1);
            const auto vocab = number("ple.head_vocab_sizes");
            if (!vocab.is_array() || vocab.empty()) throw std::runtime_error("GGUF PLE vocabulary sizes are missing");
            config["ngram_vocab_size_base"] = *std::min_element(vocab.begin(), vocab.end());
            config["hidden_act"] = "silu";
            config["output_gate_type"] = "sigmoid";
            config["norm_topk_prob"] = true;
            const auto predictor = shapes.find("predictor.embedding_norm.weight");
            bool has_predictor = false;
            for (auto item = shapes.begin(); item != shapes.end(); ++item)
                has_predictor = has_predictor || item.key().rfind("predictor.", 0) == 0;
            if (has_predictor) {
                const auto matches = [&](const std::string& name, const std::vector<int64_t>& shape) {
                    return shapes.contains(name) && shapes.at(name) == shape;
                };
                if (predictor == shapes.end() || !matches("predictor.embedding_norm.weight", {hidden}) ||
                    !matches("predictor.hidden_norm.weight", {hidden * config.at("hc_count").get<int64_t>()}) ||
                    !matches("predictor.fusion.embedding.weight", {hidden, hidden}) ||
                    !matches("predictor.fusion.hidden.weight", {hidden, hidden}))
                    throw std::runtime_error("GGUF predictor input geometry differs from the backbone");
                int64_t layers = 0;
                while (shapes.contains("predictor.block." + std::to_string(layers) + ".attention.query.weight")) {
                    const auto root = "predictor.block." + std::to_string(layers) + ".mlp.";
                    const auto experts = config.at("num_experts").get<int64_t>();
                    const auto width = config.at("moe_intermediate_size").get<int64_t>();
                    if (!matches(root + "router.weight", {experts, hidden}) ||
                        !matches(root + "experts.gate_up.weight", {experts, 2 * width, hidden}) ||
                        !matches(root + "experts.down.weight", {experts, hidden, width}))
                        throw std::runtime_error("GGUF predictor expert geometry differs from the backbone");
                    ++layers;
                }
                if (!layers) throw std::runtime_error("GGUF predictor has no attention layers");
                config["mtp_num_hidden_layers"] = layers;
                config["mtp_use_dedicated_embeddings"] = false;
            }
        }
    }
    return config;
}

} // namespace tyloi::models
