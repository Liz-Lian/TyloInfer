#include "tyloinfer/gguf_model_source.h"
#include "tyloinfer/hf_model_source.h"

#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <regex>
#include <sstream>

namespace mfq {
namespace {
using json = nlohmann::json;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Ggml = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

std::vector<std::filesystem::path> raw_parts(const std::filesystem::path& path) {
    const auto name = path.filename().string();
    if (name.find(".gguf.part-") == std::string::npos) return {std::filesystem::absolute(path)};
    static const std::regex pattern(R"((.+\.gguf)\.part-([0-9]{1,5})-of-([0-9]{1,5}))");
    std::smatch match;
    if (!std::regex_match(name, match, pattern)) throw std::runtime_error("invalid GGUF raw part filename");
    const auto index = std::stoul(match[2]), count = std::stoul(match[3]);
    if (!index || index > count || !count) throw std::runtime_error("invalid GGUF raw part number");
    int width = int(match[2].length());
    if (width > 1 && match[2].str().front() != '0') {
        std::ostringstream first;
        first << match[1] << ".part-" << std::setfill('0') << std::setw(width) << 1 << "-of-" << match[3];
        const auto plain = match[1].str() + ".part-1-of-" + match[3].str();
        if (!std::filesystem::is_regular_file(path.parent_path() / first.str()) &&
            std::filesystem::is_regular_file(path.parent_path() / plain)) width = 1;
    }
    std::vector<std::filesystem::path> result;
    for (unsigned i = 1; i <= count; ++i) {
        std::ostringstream part;
        part << match[1] << ".part-" << std::setfill('0') << std::setw(width) << i << "-of-" << match[3];
        result.push_back(std::filesystem::absolute(path.parent_path() / part.str()));
    }
    return result;
}

class GgufFile {
    struct Part {
        uint64_t begin, size;
        std::shared_ptr<FileRangeReader> file;
    };
public:
    explicit GgufFile(const std::filesystem::path& path) : paths_(raw_parts(path)) {
        for (const auto& part : paths_) {
            if (!std::filesystem::is_regular_file(part)) throw std::runtime_error("missing GGUF part: " + part.filename().string());
            const auto bytes = std::filesystem::file_size(part);
            if (!bytes || bytes > UINT64_MAX - size_) throw std::runtime_error("invalid GGUF part size");
            parts_.push_back({size_, bytes, std::make_shared<FileRangeReader>(part, model_file_read_mode())});
            size_ += bytes;
        }
    }
    const std::vector<std::filesystem::path>& paths() const noexcept { return paths_; }
    uint64_t size() const noexcept { return size_; }
    void read(uint64_t offset, std::byte* destination, std::size_t bytes) const {
        if (parts_.size() == 1) { parts_.front().file->read(offset, destination, bytes); return; }
        visit(offset, destination, bytes, [](const Part& part, const ReadSpan& span) {
            part.file->read(span.offset, span.destination, span.size);
        });
    }
    void read_contiguous(const std::vector<ReadSpan>& spans) const {
        if (parts_.size() == 1) { parts_.front().file->read_contiguous(spans); return; }
        std::vector<ReadSpan> group;
        const Part* previous = nullptr;
        for (const auto& span : spans) visit(span.offset, span.destination, span.size, [&](const Part& part, const ReadSpan& slice) {
            if (previous != &part && !group.empty()) { previous->file->read_contiguous(group); group.clear(); }
            previous = &part;
            group.push_back(slice);
        });
        if (!group.empty()) previous->file->read_contiguous(group);
    }
    Gguf load(ggml_context** tensors) const {
        if (parts_.size() == 1) {
            Gguf ctx(gguf_init_from_file(paths_.front().string().c_str(), {true, tensors}), gguf_free);
            if (!ctx) throw std::runtime_error("invalid GGUF file: " + paths_.front().filename().string());
            return ctx;
        }
        struct Reader {
            const GgufFile* file;
            std::exception_ptr error;
            uint64_t begin = 0;
            std::vector<std::byte> cache;
        } reader{this, {}, 0, {}};
        const auto callback = [](void* state, void* output, uint64_t offset, std::size_t bytes) -> std::size_t {
            auto& reader = *static_cast<Reader*>(state);
            try {
                if (offset > reader.file->size() || bytes > reader.file->size() - offset) return 0;
                if (bytes > 65536) reader.file->read(offset, static_cast<std::byte*>(output), bytes);
                else {
                    if (offset < reader.begin || offset - reader.begin > reader.cache.size() ||
                        bytes > reader.cache.size() - (offset - reader.begin)) {
                        reader.begin = offset;
                        reader.cache.resize(std::min<uint64_t>(65536, reader.file->size() - offset));
                        reader.file->read(offset, reader.cache.data(), reader.cache.size());
                    }
                    std::memcpy(output, reader.cache.data() + (offset - reader.begin), bytes);
                }
                return bytes;
            } catch (...) { reader.error = std::current_exception(); return 0; }
        };
        Gguf ctx(gguf_init_from_callback(callback, &reader, 1 << 20, size_, {true, tensors}), gguf_free);
        if (reader.error) std::rethrow_exception(reader.error);
        if (!ctx) throw std::runtime_error("invalid multipart GGUF: " + paths_.front().filename().string());
        return ctx;
    }
private:
    template<typename F> void visit(uint64_t offset, std::byte* destination, std::size_t bytes, F&& apply) const {
        if (offset > size_ || bytes > size_ - offset) throw std::out_of_range("GGUF part read exceeds file size");
        if (!bytes) return;
        if (!destination) throw std::invalid_argument("GGUF part destination is null");
        auto part = std::upper_bound(parts_.begin(), parts_.end(), offset,
            [](uint64_t at, const Part& part) { return at < part.begin; }) - 1;
        while (bytes) {
            const auto within = offset - part->begin;
            const auto count = std::min<uint64_t>(bytes, part->size - within);
            apply(*part, ReadSpan{within, destination, std::size_t(count)});
            offset += count; destination += count; bytes -= count;
            ++part;
        }
    }
    std::vector<std::filesystem::path> paths_;
    std::vector<Part> parts_;
    uint64_t size_ = 0;
};

json scalar(const gguf_context* ctx, int64_t id) {
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8: return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8: return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16: return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64: return gguf_get_val_i64(ctx, id);
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, id);
        case GGUF_TYPE_BOOL: return gguf_get_val_bool(ctx, id);
        case GGUF_TYPE_STRING: return gguf_get_val_str(ctx, id);
        default: return nullptr;
    }
}

json metadata_value(const gguf_context* ctx, int64_t id) {
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) return scalar(ctx, id);
    const auto count = gguf_get_arr_n(ctx, id);
    const auto* data = gguf_get_arr_type(ctx, id) == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, id);
    json result = json::array();
    for (std::size_t i = 0; i < count; ++i) {
        switch (gguf_get_arr_type(ctx, id)) {
            case GGUF_TYPE_UINT8: result.push_back(static_cast<const uint8_t*>(data)[i]); break;
            case GGUF_TYPE_INT8: result.push_back(static_cast<const int8_t*>(data)[i]); break;
            case GGUF_TYPE_UINT16: result.push_back(static_cast<const uint16_t*>(data)[i]); break;
            case GGUF_TYPE_INT16: result.push_back(static_cast<const int16_t*>(data)[i]); break;
            case GGUF_TYPE_UINT32: result.push_back(static_cast<const uint32_t*>(data)[i]); break;
            case GGUF_TYPE_INT32: result.push_back(static_cast<const int32_t*>(data)[i]); break;
            case GGUF_TYPE_UINT64: result.push_back(static_cast<const uint64_t*>(data)[i]); break;
            case GGUF_TYPE_INT64: result.push_back(static_cast<const int64_t*>(data)[i]); break;
            case GGUF_TYPE_FLOAT32: result.push_back(static_cast<const float*>(data)[i]); break;
            case GGUF_TYPE_FLOAT64: result.push_back(static_cast<const double*>(data)[i]); break;
            case GGUF_TYPE_BOOL: result.push_back(static_cast<const uint8_t*>(data)[i] != 0); break;
            case GGUF_TYPE_STRING: result.push_back(gguf_get_arr_str(ctx, id, i)); break;
            default: throw std::runtime_error("unsupported GGUF metadata array");
        }
    }
    return result;
}

std::string canonical_name(const std::string& name) {
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
        {"attn_q_a.weight", "attention.query_a.weight"},
        {"attn_q_b.weight", "attention.query_b.weight"},
        {"attn_q_a_norm.weight", "attention.query_a_norm.weight"},
        {"attn_kv.weight", "attention.key_value.weight"},
        {"attn_kv_a_norm.weight", "attention.key_value_norm.weight"},
        {"attn_sinks.weight", "attention.sink"},
        {"attn_output_a.weight", "attention.output_a.weight"},
        {"attn_output_b.weight", "attention.output_b.weight"},
        {"attn_compressor_kv.weight", "attention.compressor.key_value.weight"},
        {"attn_compressor_gate.weight", "attention.compressor.gate.weight"},
        {"attn_compressor_norm.weight", "attention.compressor.norm.weight"},
        {"indexer.attn_q_b.weight", "attention.indexer.query.weight"},
        {"indexer.attn_k.weight", "attention.indexer.key.weight"},
        {"indexer.proj.weight", "attention.indexer.score.weight"},
        {"hc_attn_fn.weight", "attention.mhc.pre.function"},
        {"hc_attn_base.weight", "attention.mhc.pre.base"},
        {"hc_attn_scale.weight", "attention.mhc.pre.scale"},
        {"hc_ffn_fn.weight", "mlp.mhc.pre.function"},
        {"hc_ffn_base.weight", "mlp.mhc.pre.base"},
        {"hc_ffn_scale.weight", "mlp.mhc.pre.scale"},
        {"exp_probs_b.bias", "mlp.router.bias"},
        {"exp_probs_b_vl.bias", "mlp.router.vision_bias"},
        {"engram_embd.weight", "associative_memory.embedding.weight"},
        {"engram_q_norm.weight", "associative_memory.query.weight"},
        {"engram_k_norm.weight", "associative_memory.key.weight"},
        {"engram_kv.weight", "associative_memory.projection.weight"},
    };
    const auto found = names.find(name.substr(dot + 1));
    return found == names.end() ? name : "model.block." + name.substr(4, dot - 4) + "." + found->second;
}

std::vector<std::byte> dense_prefix(const std::vector<int64_t>& shape) {
    std::vector<std::byte> bytes(4 + shape.size() * 8);
    const auto rank = static_cast<uint32_t>(shape.size());
    for (unsigned i = 0; i < 4; ++i) bytes[i] = std::byte(rank >> (i * 8));
    for (std::size_t d = 0; d < shape.size(); ++d)
        for (unsigned i = 0; i < 8; ++i) bytes[4 + d * 8 + i] = std::byte(uint64_t(shape[d]) >> (i * 8));
    return bytes;
}

json model_config(const json& meta, const json& shapes, const std::string& arch) {
    if (arch == "deepseek41") {
        const auto& embedded = meta.at("deepseek41.config");
        if (!embedded.is_string()) throw std::runtime_error("DeepSeek GGUF config must be a JSON string");
        auto config = json::parse(embedded.get_ref<const std::string&>());
        if (!config.is_object() || config.value("model_type", std::string{}) != "deepseek_v41" ||
            !config.contains("text_config") || !config.at("text_config").is_object() ||
            config.at("text_config").value("model_type", std::string{}) != "deepseek_v41_text")
            throw std::runtime_error("DeepSeek GGUF embedded architecture is inconsistent");
        const auto& text = config.at("text_config");
        if (text.value("hidden_size", int64_t{0}) <= 0 || text.value("vocab_size", int64_t{0}) <= 0 ||
            text.value("num_hidden_layers", int64_t{0}) <= 0)
            throw std::runtime_error("DeepSeek GGUF embedded dimensions are invalid");
        if (shapes.contains("model.token_embedding.weight") && shapes.at("model.token_embedding.weight") !=
                json::array({text.at("vocab_size"), text.at("hidden_size")}))
            throw std::runtime_error("DeepSeek GGUF embedding geometry disagrees with config");
        return config;
    }
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

std::vector<std::byte> deepseek_engram_asset(const json& meta, const json& config) {
    const auto& text = config.at("text_config");
    const auto layers = text.at("engram_layer_ids").get<std::vector<int64_t>>();
    if (layers.empty()) return {};
    const auto rows = text.at("engram_num_embeddings").get<std::vector<int64_t>>();
    const auto vocab = text.at("vocab_size").get<int64_t>();
    const auto compressed = text.at("engram_compressed_vocab_size").get<int64_t>();
    const auto ngram = text.at("engram_max_ngram_size").get<int64_t>();
    const auto heads = text.at("engram_n_heads").get<int64_t>();
    const auto max_int = std::numeric_limits<int32_t>::max();
    if (vocab <= 0 || vocab > max_int || compressed <= 0 || compressed > max_int ||
        ngram < 2 || ngram > max_int || heads <= 0 || heads > max_int ||
        layers.size() != rows.size() || layers.size() > std::size_t(max_int) ||
        meta.at("deepseek41.engram.layer_ids") != layers || meta.at("deepseek41.engram.rows") != rows ||
        meta.at("deepseek41.engram.compressed_vocab_size") != compressed ||
        uint64_t(ngram - 1) * uint64_t(heads) > SIZE_MAX / layers.size())
        throw std::runtime_error("DeepSeek GGUF Engram dimensions disagree");
    const auto primes = meta.at("deepseek41.engram.primes").get<std::vector<int64_t>>();
    const auto multipliers = meta.at("deepseek41.engram.multipliers").get<std::vector<uint64_t>>();
    const auto tokens = meta.at("deepseek41.engram.token_map").get<std::vector<int64_t>>();
    const auto pad = meta.at("deepseek41.engram.pad_id").get<int64_t>();
    const auto buckets = std::size_t(ngram - 1) * heads;
    if (primes.size() != layers.size() * buckets ||
        uint64_t(ngram) > SIZE_MAX / layers.size() || multipliers.size() != layers.size() * ngram ||
        tokens.size() != std::size_t(vocab) || pad < 0 || pad >= compressed ||
        std::any_of(tokens.begin(), tokens.end(), [=](auto t) { return t < 0 || t >= compressed; }) ||
        std::any_of(multipliers.begin(), multipliers.end(), [](auto m) { return m == 0 || !(m & 1) || m > INT64_MAX; }))
        throw std::runtime_error("DeepSeek GGUF Engram hash metadata is invalid");
    const auto pad_token = text.at("engram_pad_token_id").get<int64_t>();
    if (pad_token < 0 || pad_token >= vocab || tokens[pad_token] != pad)
        throw std::runtime_error("DeepSeek GGUF Engram pad token differs");
    std::vector<int64_t> offsets(primes.size());
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        if (layers[layer] < 0 || layers[layer] >= text.at("num_hidden_layers").get<int64_t>() ||
            std::find(layers.begin(), layers.begin() + layer, layers[layer]) != layers.begin() + layer)
            throw std::runtime_error("DeepSeek GGUF Engram layer is invalid");
        int64_t offset = 0;
        for (std::size_t i = layer * buckets; i < (layer + 1) * buckets; ++i) {
            if (primes[i] <= 0 || primes[i] > INT64_MAX - offset)
                throw std::runtime_error("DeepSeek GGUF Engram bucket overflow");
            offsets[i] = offset; offset += primes[i];
        }
        if (offset != rows[layer]) throw std::runtime_error("DeepSeek GGUF Engram bucket extent differs");
    }
    std::vector<std::byte> result;
    for (const char c : std::string_view("MFQENGR1")) result.push_back(std::byte(c));
    const auto append = [&](uint64_t bits, unsigned width) {
        for (unsigned i = 0; i < width; ++i) result.push_back(std::byte(bits >> (i * 8)));
    };
    for (const auto v : {vocab, compressed, int64_t(layers.size()), ngram, heads, pad}) append(v, 4);
    for (const auto v : layers) append(v, 4);
    for (const auto v : rows) append(v, 8);
    for (const auto v : primes) append(v, 8);
    for (const auto v : offsets) append(v, 8);
    for (const auto v : multipliers) append(v, 8);
    for (const auto v : tokens) append(v, 4);
    return result;
}
} // namespace

struct GgufModelSource::Impl {
    struct Record {
        struct Piece {
            uint64_t begin, size, file_offset;
            std::shared_ptr<GgufFile> file;
        };
        TensorMetadata tensor;
        std::shared_ptr<GgufFile> file;
        std::filesystem::path path;
        uint64_t offset = 0;
        std::vector<std::byte> prefix;
        std::vector<std::byte> inline_bytes;
        std::vector<Piece> pieces;
        std::shared_ptr<HfModelSource> external;
        std::string external_name;
    };
    std::vector<std::filesystem::path> paths;
    std::string arch;
    std::unordered_map<std::string, std::string> metadata;
    std::vector<TensorMetadata> tensors;
    std::unordered_map<std::string, Record> records;
    std::unordered_map<std::string, std::vector<std::byte>> asset_data;
    std::vector<std::string> assets;
};

bool is_gguf_file(const std::filesystem::path& path) {
    try {
        const auto parts = raw_parts(path);
        std::ifstream file(parts.front(), std::ios::binary);
        char magic[4]{};
        if (!file.read(magic, 4)) {
            if (parts.size() == 1) return false;
            GgufFile(path).read(0, reinterpret_cast<std::byte*>(magic), sizeof(magic));
        }
        return std::memcmp(magic, "GGUF", 4) == 0;
    } catch (const std::exception&) { return false; }
}

GgufModelSource::GgufModelSource(const std::filesystem::path& path,
        std::vector<std::filesystem::path> companions) : impl_(std::make_shared<Impl>()) {
    std::map<std::filesystem::path, std::shared_ptr<GgufFile>> files;
    const auto open = [&](const std::filesystem::path& path) {
        const auto key = std::filesystem::absolute(path);
        auto& file = files[key];
        if (!file) file = std::make_shared<GgufFile>(key);
        return file;
    };
    auto first = open(path)->load(nullptr);
    const auto split_key = gguf_find_key(first.get(), "split.count");
    const auto splits = split_key < 0 ? 1 : scalar(first.get(), split_key).get<int>();
    if (splits < 1 || splits > 99999) throw std::runtime_error("invalid GGUF split count");
    if (splits > 1) {
        const auto filename = path.filename().string();
        const auto marker = filename.rfind("-of-");
        if (marker == std::string::npos || marker < 6 || filename.substr(filename.size() - 5) != ".gguf")
            throw std::runtime_error("split GGUF requires a -00001-of-000NN.gguf filename");
        const auto stem = filename.substr(0, marker - 6);
        for (int i = 1; i <= splits; ++i) {
            std::ostringstream name;
            name << stem << '-' << std::setfill('0') << std::setw(5) << i << "-of-" << std::setw(5) << splits << ".gguf";
            const auto shard = path.parent_path() / name.str();
            if (!std::filesystem::is_regular_file(shard)) throw std::runtime_error("missing GGUF shard: " + name.str());
            impl_->paths.push_back(std::filesystem::absolute(shard));
        }
    } else impl_->paths.push_back(std::filesystem::absolute(path));
    const auto model_shards = impl_->paths.size();
    std::vector<std::filesystem::path> table_directories;
    for (const auto& companion : companions) {
        const auto absolute = std::filesystem::absolute(companion);
        if (std::filesystem::is_directory(absolute)) {
            table_directories.push_back(absolute);
            continue;
        }
        if (std::find(impl_->paths.begin(), impl_->paths.end(), absolute) != impl_->paths.end())
            throw std::runtime_error("GGUF companion duplicates a model shard");
        impl_->paths.push_back(absolute);
    }

    json meta = json::object(), shapes = json::object();
    for (std::size_t shard = 0; shard < impl_->paths.size(); ++shard) {
        ggml_context* raw = nullptr;
        auto file = open(impl_->paths[shard]);
        auto ctx = file->load(&raw);
        Ggml tensors(raw, ggml_free);
        if (gguf_get_n_tensors(ctx.get()) > (1 << 20))
            throw std::runtime_error("GGUF has too many tensor records");
        if (splits > 1 && shard < model_shards) {
            const auto split_no = gguf_find_key(ctx.get(), "split.no");
            const auto split_count = gguf_find_key(ctx.get(), "split.count");
            if (split_no < 0 || split_count < 0 || scalar(ctx.get(), split_no) != shard || scalar(ctx.get(), split_count) != splits)
                throw std::runtime_error("inconsistent GGUF shard metadata");
        }
        const auto file_size = file->size();
        if (shard == 0) {
            for (int64_t i = 0; i < gguf_get_n_kv(ctx.get()); ++i) {
                const std::string name = gguf_get_key(ctx.get(), i);
                if (name.rfind("tokenizer.", 0) == 0 && gguf_get_kv_type(ctx.get(), i) == GGUF_TYPE_ARRAY) continue;
                const auto value = metadata_value(ctx.get(), i);
                if (!value.is_null()) meta[name] = value;
            }
            if (!meta.contains("general.architecture")) throw std::runtime_error("GGUF has no architecture");
            impl_->arch = meta["general.architecture"].get<std::string>();
            Gguf tokenizer(gguf_init_empty(), gguf_free);
            gguf_set_kv(tokenizer.get(), ctx.get());
            auto& asset = impl_->asset_data[kTokenizerGgufAsset];
            asset.resize(gguf_get_meta_size(tokenizer.get()));
            gguf_get_meta_data(tokenizer.get(), asset.data());
        }
        std::unordered_map<std::string, std::string> predictor_names;
        const auto architecture_key = gguf_find_key(ctx.get(), "general.architecture");
        const bool raw_predictor = shard >= model_shards && architecture_key >= 0 &&
            scalar(ctx.get(), architecture_key) == "qwen4exp-mtp";
        if (raw_predictor) {
            if (impl_->arch != "qwen4exp")
                throw std::runtime_error("Qwen4 MTP companion requires a Qwen4 backbone");
            std::vector<std::string> names;
            for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i)
                names.emplace_back(gguf_get_tensor_name(ctx.get(), i));
            const auto aliases = make_legacy_tensor_aliases("qwen4_exp_text",
                json{{"model_type", "qwen4_exp_text"}, {"num_hidden_layers", meta.at("qwen4exp.block_count")}}.dump(), names);
            for (const auto& [canonical, stored] : aliases.canonical_to_stored)
                if (canonical.rfind("predictor.", 0) == 0) predictor_names.emplace(stored, canonical);
            if (predictor_names.size() != names.size())
                throw std::runtime_error("GGUF MTP companion contains unsupported predictor tensors");
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
            const auto* original = gguf_get_tensor_name(ctx.get(), i);
            const auto* tensor = ggml_get_tensor(tensors.get(), original);
            if (!tensor) throw std::runtime_error("GGUF tensor shape is missing");
            const auto type = gguf_get_tensor_type(ctx.get(), i);
            if (type >= GGML_TYPE_TQ2_T && type <= GGML_TYPE_TQK4 &&
                meta.value("agention.trellis.state_bits", int64_t{-1}) != 15)
                throw std::runtime_error("Gyro GGUF requires agention.trellis.state_bits=15");
            std::vector<int64_t> shape;
            for (int d = ggml_n_dims(tensor) - 1; d >= 0; --d) shape.push_back(tensor->ne[d]);
            auto name = raw_predictor ? predictor_names.at(original) : canonical_name(original);
            if (impl_->arch == "qwen4exp") {
                if (name == "per_layer_token_embd.weight") {
                    const auto layers = meta.at("qwen4exp.ple.layers").get<std::vector<int64_t>>();
                    if (layers.size() != 1) throw std::runtime_error("GGUF PLE table requires one declared layer");
                    name = "model.block." + std::to_string(layers.front()) + ".position_embedding.ngram.shard.0.weight";
                }
                if (name.find(".mlp.shared_expert.router.weight") != std::string::npos && shape.size() == 1)
                    shape.insert(shape.begin(), 1);
            }
            if (impl_->records.count(name)) throw std::runtime_error("duplicate GGUF tensor: " + name);
            Impl::Record record;
            record.file = file;
            record.path = impl_->paths[shard];
            const auto data_offset = gguf_get_data_offset(ctx.get());
            const auto tensor_offset = gguf_get_tensor_offset(ctx.get(), i);
            if (tensor_offset > std::numeric_limits<uint64_t>::max() - data_offset)
                throw std::runtime_error("GGUF tensor offset overflow: " + name);
            record.offset = data_offset + tensor_offset;
            auto size = gguf_get_tensor_size(ctx.get(), i);
            if (record.offset > file_size || size > file_size - record.offset)
                throw std::runtime_error("truncated GGUF tensor: " + name);
            const bool engram = impl_->arch == "deepseek41" && name.find(".associative_memory.embedding.weight") != std::string::npos;
            std::string dtype;
            if (engram) {
                if (type != GGML_TYPE_I8 || shape.size() != 2 || shape.back() != 264 ||
                    meta.value("deepseek41.engram.encoding", std::string{}) != "e4m3_e8m0_32_row264")
                    throw std::runtime_error("unsupported DeepSeek GGUF Engram row encoding");
                shape.back() = 256;
                dtype = "GGUF_ENGRAM_MXFP8";
            } else if (type == GGML_TYPE_F32) dtype = "F32";
            else if (type == GGML_TYPE_F16) dtype = "F16";
            else if (type == GGML_TYPE_BF16) dtype = "BF16";
            else if (type == GGML_TYPE_I32) dtype = "I32";
            else if (type == GGML_TYPE_I64) dtype = "I64";
            else if (ggml_is_quantized(type)) dtype = "GGUF_" + std::string(ggml_type_name(type));
            else throw std::runtime_error("unsupported GGUF tensor type: " + std::string(ggml_type_name(type)));
            std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char c) { return std::toupper(c); });
            if (!ggml_is_quantized(type) && !engram) record.prefix = dense_prefix(shape);
            record.tensor = {name, dtype, dtype, shape, size + record.prefix.size()};
            const bool norm_delta = impl_->arch == "qwen4exp" && !raw_predictor && name.find("norm.weight") != std::string::npos &&
                name.find(".linear_attention.norm.weight") == std::string::npos;
            const bool decay_log = impl_->arch == "qwen4exp" && name.find(".linear_attention.a") != std::string::npos &&
                name.size() >= 2 && name.substr(name.size() - 2) == ".a";
            if (norm_delta || decay_log) {
                if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16)
                    throw std::runtime_error("GGUF normalization requires floating-point tensors: " + name);
                std::vector<std::byte> raw(size);
                file->read(record.offset, raw.data(), raw.size());
                const auto elements = ggml_nelements(tensor);
                record.inline_bytes = dense_prefix(shape);
                const auto begin = record.inline_bytes.size();
                record.inline_bytes.resize(begin + elements * sizeof(float));
                for (int64_t j = 0; j < elements; ++j) {
                    float value = 0;
                    if (type == GGML_TYPE_F32) std::memcpy(&value, raw.data() + j * 4, 4);
                    else {
                        uint16_t bits; std::memcpy(&bits, raw.data() + j * 2, 2);
                        if (type == GGML_TYPE_F16) value = ggml_fp16_to_fp32(bits);
                        else { const uint32_t expanded = uint32_t(bits) << 16; std::memcpy(&value, &expanded, 4); }
                    }
                    if (decay_log) {
                        if (!std::isfinite(value) || value >= 0) throw std::runtime_error("invalid GGUF GDN decay: " + name);
                        value = std::log(-value);
                    } else value -= 1.f;
                    std::memcpy(record.inline_bytes.data() + begin + j * 4, &value, 4);
                }
                record.tensor.dtype = record.tensor.stored_dtype = "F32";
                record.tensor.nbytes = record.inline_bytes.size();
            }
            shapes[name] = shape;
            impl_->tensors.push_back(record.tensor);
            impl_->records.emplace(name, std::move(record));
        }
    }
    std::vector<std::filesystem::path> physical_paths;
    for (const auto& path : impl_->paths) for (const auto& part : open(path)->paths()) {
        if (std::find(physical_paths.begin(), physical_paths.end(), part) != physical_paths.end())
            throw std::runtime_error("GGUF companion duplicates a model part");
        physical_paths.push_back(part);
    }
    impl_->paths = std::move(physical_paths);
    json mixed_experts = json::object();
    std::vector<std::string> type_maps;
    const std::string map_suffix = ".ffn_exps_gu_type";
    const std::string down_map_suffix = ".ffn_exps_down_type";
    for (const auto& [name, record] : impl_->records) {
        if (name.rfind("blk.", 0) == 0 && name.size() > map_suffix.size() &&
            name.compare(name.size() - map_suffix.size(), map_suffix.size(), map_suffix) == 0) type_maps.push_back(name);
        else if (name.rfind("blk.", 0) == 0 && name.size() > down_map_suffix.size() &&
            name.compare(name.size() - down_map_suffix.size(), down_map_suffix.size(), down_map_suffix) == 0)
            type_maps.push_back(name.substr(0, name.size() - down_map_suffix.size()) + map_suffix);
    }
    std::sort(type_maps.begin(), type_maps.end());
    type_maps.erase(std::unique(type_maps.begin(), type_maps.end()), type_maps.end());
    for (const auto& map_name : type_maps) {
        const auto root = map_name.substr(0, map_name.size() - std::string("ffn_exps_gu_type").size());
        const auto dot = root.find('.', 4);
        if (dot == std::string::npos) throw std::runtime_error("invalid GGUF mixed expert layer name");
        for (const auto* role : {"gate", "up", "down"}) {
            const auto name = canonical_name(root + "ffn_" + role + "_exps.weight");
            if (impl_->records.count(name)) continue;
            const auto role_map = root + (std::string_view(role) == "down" ? "ffn_exps_down_type" : "ffn_exps_gu_type");
            const auto map_it = impl_->records.find(role_map);
            if (map_it == impl_->records.end() || map_it->second.tensor.dtype != "I32" ||
                !map_it->second.tensor.shape || map_it->second.tensor.shape->size() != 1)
                throw std::runtime_error("GGUF mixed expert type map is missing or invalid");
            const auto count = map_it->second.tensor.shape->front();
            if (count <= 0 || count > (1 << 20)) throw std::runtime_error("invalid GGUF mixed expert count");
            const auto encoded = read(role_map);
            if (encoded.size() != map_it->second.prefix.size() + std::size_t(count) * 4)
                throw std::runtime_error("invalid GGUF mixed expert type map size");
            Impl::Record joined;
            json layout = json::array();
            std::unordered_map<int, uint64_t> slots;
            std::unordered_map<int, std::string> banks;
            std::vector<int64_t> geometry;
            uint64_t total = 0;
            for (int64_t e = 0; e < count; ++e) {
                int32_t type;
                std::memcpy(&type, encoded.data() + map_it->second.prefix.size() + e * 4, 4);
                if (type < 0 || type >= GGML_TYPE_COUNT || ggml_blck_size(static_cast<ggml_type>(type)) == 0 ||
                    !ggml_is_quantized(static_cast<ggml_type>(type)))
                    throw std::runtime_error("unsupported GGUF mixed expert type");
                auto suffix = std::string(ggml_type_name(static_cast<ggml_type>(type)));
                std::transform(suffix.begin(), suffix.end(), suffix.begin(), [](unsigned char c) { return std::tolower(c); });
                const auto bank_name = root + "ffn_" + role + "_exps." + suffix + ".weight";
                const auto bank_it = impl_->records.find(bank_name);
                if (bank_it == impl_->records.end()) throw std::runtime_error("GGUF mixed expert format bank is missing");
                const auto& bank = bank_it->second;
                auto dtype = "GGUF_" + std::string(ggml_type_name(static_cast<ggml_type>(type)));
                std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char c) { return std::toupper(c); });
                if (!bank.tensor.shape || (bank.tensor.shape->size() != 2 && bank.tensor.shape->size() != 3) ||
                    !bank.file || !bank.prefix.empty() || bank.tensor.dtype != dtype)
                    throw std::runtime_error("invalid GGUF mixed expert bank");
                const auto& shape = *bank.tensor.shape;
                const int64_t n = shape[shape.size() - 2], k = shape.back();
                if (geometry.empty()) geometry = {count, n, k};
                if (geometry != std::vector<int64_t>{count, n, k}) throw std::runtime_error("GGUF mixed expert bank dimensions differ");
                const auto bytes = uint64_t(ggml_row_size(static_cast<ggml_type>(type), k)) * n;
                const auto slot = slots[type]++;
                const auto experts = shape.size() == 3 ? shape.front() : 1;
                if (slot >= uint64_t(experts) || bytes > UINT64_MAX - total || bytes > UINT64_MAX / uint64_t(experts) ||
                    bank.tensor.nbytes != bytes * uint64_t(experts))
                    throw std::runtime_error("GGUF mixed expert bank size differs from its type map");
                joined.pieces.push_back({total, bytes, bank.offset + slot * bytes, bank.file});
                layout.push_back({{"dtype", bank.tensor.dtype}, {"offset", total}, {"nbytes", bytes}});
                total += bytes; banks[type] = bank_name;
            }
            for (const auto& [type, bank_name] : banks) {
                const auto& shape = *impl_->records.at(bank_name).tensor.shape;
                if (slots[type] != uint64_t(shape.size() == 3 ? shape.front() : 1))
                    throw std::runtime_error("GGUF mixed expert bank has unmapped experts");
                impl_->records.erase(bank_name); shapes.erase(bank_name);
                impl_->tensors.erase(std::remove_if(impl_->tensors.begin(), impl_->tensors.end(),
                    [bank_name = bank_name](const TensorMetadata& tensor) { return tensor.name == bank_name; }), impl_->tensors.end());
            }
            joined.tensor = {name, "GGUF_MIXED", "GGUF_MIXED", geometry, total};
            shapes[name] = geometry;
            mixed_experts[name] = std::move(layout);
            impl_->tensors.push_back(joined.tensor);
            impl_->records.emplace(name, std::move(joined));
        }
    }
    if (!mixed_experts.empty()) impl_->metadata["gguf_mixed_experts"] = mixed_experts.dump();
    if (meta.contains("prism.hadamard.version")) {
        const auto version = meta.at("prism.hadamard.version").get<int>();
        const auto block = meta.value("prism.hadamard.block_size", int64_t{0});
        const auto mode = meta.value("prism.hadamard.sign_mode", std::string{});
        if ((version != 1 && version != 2) || block < 1 || (block & (block - 1)) || block > INT32_MAX ||
            meta.value("prism.hadamard.transform", std::string{}) != "normalized-sylvester-walsh-hadamard" ||
            meta.value("prism.hadamard.axis", std::string{}) != "input-last-dimension" ||
            (mode != "explicit" && mode != "identity"))
            throw std::runtime_error("unsupported GGUF Hadamard transform metadata");
        if (!meta.value("prism.hadamard.inverse_weight_names", json::array()).empty() ||
            meta.value("prism.hadamard.tied_output", false) || meta.value("prism.hadamard.gdn_v_grouped", false))
            throw std::runtime_error("GGUF inverse embedding or grouped GDN Hadamard layout is unsupported");
        std::map<int64_t, std::vector<int8_t>> signs;
        if (mode == "explicit") {
            const auto widths = meta.at("prism.hadamard.sign_widths").get<std::vector<int64_t>>();
            const auto values = meta.at("prism.hadamard.sign_values").get<std::vector<int>>();
            std::size_t offset = 0;
            for (const auto width : widths) {
                if (width < 1 || width % block || uint64_t(width) > values.size() - offset || signs.count(width))
                    throw std::runtime_error("invalid GGUF Hadamard sign widths");
                std::vector<int8_t> pattern;
                for (int64_t i = 0; i < width; ++i) {
                    const int value = values[offset++];
                    if (value != 1 && value != -1) throw std::runtime_error("invalid GGUF Hadamard sign value");
                    pattern.push_back(int8_t(value));
                }
                signs.emplace(width, std::move(pattern));
            }
            if (widths.empty() || offset != values.size()) throw std::runtime_error("GGUF Hadamard sign vector lengths differ");
        }
        json transforms = json::object();
        const auto weights = meta.at("prism.hadamard.weight_names").get<std::vector<std::string>>();
        if (weights.empty()) throw std::runtime_error("GGUF Hadamard weight list is empty");
        for (const auto& original : weights) {
            const auto name = canonical_name(original);
            const auto record = impl_->records.find(name);
            if (record == impl_->records.end() || !record->second.tensor.shape || record->second.tensor.shape->size() < 2 ||
                record->second.tensor.dtype.rfind("GGUF_", 0) != 0 || transforms.contains(name))
                throw std::runtime_error("invalid or unsupported GGUF Hadamard weight");
            const auto width = record->second.tensor.shape->back();
            if (width % block || (mode == "explicit" && !signs.count(width)))
                throw std::runtime_error("GGUF Hadamard weight has no compatible sign vector");
            transforms[name] = {{"block_size", block}, {"signs", mode == "explicit" ? json(signs.at(width)) : json::array()}};
        }
        impl_->metadata["gguf_hadamard"] = transforms.dump();
    }
    if (impl_->arch == "qwen4exp") {
        const auto blocks = meta.at("qwen4exp.block_count").get<int64_t>();
        for (int64_t block = 0; block < blocks; ++block) {
            const auto root = "model.block." + std::to_string(block) + ".attention.indexer.";
            const auto q = impl_->records.find(root + "query.weight");
            const auto k = impl_->records.find(root + "key.weight");
            if (q == impl_->records.end() && k == impl_->records.end()) continue;
            if (q == impl_->records.end() || k == impl_->records.end()) throw std::runtime_error("GGUF indexer query/key pair is incomplete");
            const auto& qt = q->second.tensor;
            const auto& kt = k->second.tensor;
            if (!qt.shape || !kt.shape || qt.shape->size() != 2 || kt.shape->size() != 2 ||
                qt.shape->back() != kt.shape->back() || qt.dtype != kt.dtype ||
                (qt.dtype != "F32" && qt.dtype != "F16" && qt.dtype != "BF16"))
                throw std::runtime_error("GGUF indexer query/key layouts cannot be joined without changing precision");
            const std::vector<int64_t> shape{qt.shape->front() + kt.shape->front(), qt.shape->back()};
            const auto name = root + "query_key.weight";
            Impl::Record joined;
            joined.inline_bytes = dense_prefix(shape);
            for (const auto* record : {&q->second, &k->second}) {
                const auto bytes = read(record->tensor.name);
                joined.inline_bytes.insert(joined.inline_bytes.end(), bytes.begin() + record->prefix.size(), bytes.end());
            }
            joined.tensor = {name, qt.dtype, qt.stored_dtype, shape, joined.inline_bytes.size()};
            impl_->records.erase(root + "query.weight");
            impl_->records.erase(root + "key.weight");
            impl_->tensors.erase(std::remove_if(impl_->tensors.begin(), impl_->tensors.end(), [&](const TensorMetadata& tensor) {
                return tensor.name == root + "query.weight" || tensor.name == root + "key.weight";
            }), impl_->tensors.end());
            shapes.erase(root + "query.weight"); shapes.erase(root + "key.weight");
            shapes[name] = shape;
            impl_->tensors.push_back(joined.tensor);
            impl_->records.emplace(name, std::move(joined));
        }
        const auto layers = meta.at("qwen4exp.ple.layers").get<std::vector<int64_t>>();
        for (const auto layer : layers) {
            const auto root = "model.block." + std::to_string(layer) + ".position_embedding.ngram.";
            for (const auto* key : {"layer_multipliers", "head_offsets", "head_vocab_sizes"}) {
                const auto values = meta.at(std::string("qwen4exp.ple.") + key).get<std::vector<int64_t>>();
                Impl::Record record;
                const auto name = root + key;
                std::vector<int64_t> shape{static_cast<int64_t>(values.size())};
                record.inline_bytes = dense_prefix(shape);
                const auto start = record.inline_bytes.size();
                record.inline_bytes.resize(start + values.size() * sizeof(int64_t));
                std::memcpy(record.inline_bytes.data() + start, values.data(), values.size() * sizeof(int64_t));
                record.tensor = {name, "I64", "I64", shape, record.inline_bytes.size()};
                shapes[name] = shape;
                impl_->tensors.push_back(record.tensor);
                impl_->records.emplace(name, std::move(record));
            }
        }
    }
    const auto config_json = model_config(meta, shapes, impl_->arch);
    if (!table_directories.empty()) {
        if (impl_->arch != "deepseek41") throw std::runtime_error("GGUF Safetensors companions require DeepSeek Engram tables");
        const auto& text = config_json.at("text_config");
        const auto layers = text.at("engram_layer_ids").get<std::vector<int64_t>>();
        const auto rows = text.at("engram_num_embeddings").get<std::vector<int64_t>>();
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            const auto name = "model.block." + std::to_string(layers[layer]) + ".associative_memory.embedding.weight";
            const auto original = "layers." + std::to_string(layers[layer]) + ".engram.embed.weight";
            if (impl_->records.count(name)) throw std::runtime_error("GGUF Engram companion duplicates an embedded table");
            std::shared_ptr<HfModelSource> source;
            const TensorMetadata* table = nullptr;
            for (const auto& directory : table_directories) {
                const auto index_path = directory / "model.safetensors.index.json";
                if (std::filesystem::is_regular_file(index_path)) {
                    if (std::filesystem::file_size(index_path) > (uint64_t{256} << 20))
                        throw std::runtime_error("GGUF Engram Safetensors index is too large");
                    std::ifstream stream(index_path);
                    json index; stream >> index;
                    const auto& map = index.at("weight_map");
                    if (!map.is_object()) throw std::runtime_error("invalid GGUF Engram Safetensors index");
                    if (!map.contains(original)) continue;
                    const auto shard = std::filesystem::path(map.at(original).get<std::string>());
                    if (shard.is_absolute() || std::find(shard.begin(), shard.end(), "..") != shard.end())
                        throw std::runtime_error("invalid GGUF Engram Safetensors shard path");
                    if (!std::filesystem::is_regular_file(directory / shard)) continue;
                }
                const auto candidate = std::make_shared<HfModelSource>(directory,
                    original.substr(0, original.size() - std::string("weight").size()));
                const auto* found = candidate->find_tensor(name);
                if (!found) found = candidate->find_tensor(original);
                if (!found) continue;
                if (source) throw std::runtime_error("GGUF Engram table exists in multiple companions");
                source = candidate; table = found;
            }
            if (!table || table->dtype != "MXFP8" || !table->shape || layer >= rows.size() ||
                *table->shape != std::vector<int64_t>{rows[layer], 256})
                throw std::runtime_error("GGUF Engram companion is missing or has incompatible FP8 rows");
            Impl::Record record;
            record.tensor = *table; record.tensor.name = name;
            record.external_name = table->name; record.external = source;
            shapes[name] = *table->shape;
            impl_->tensors.push_back(record.tensor);
            impl_->records.emplace(name, std::move(record));
            for (const auto& file : source->source_paths())
                if (std::find(impl_->paths.begin(), impl_->paths.end(), file) == impl_->paths.end()) impl_->paths.push_back(file);
        }
    }
    if (impl_->arch == "deepseek41") {
        auto asset = deepseek_engram_asset(meta, config_json);
        if (!asset.empty()) impl_->asset_data["__mfq_asset__/deepseek-v41-engram-v1.bin"] = std::move(asset);
    }
    const auto config = config_json.dump();
    impl_->asset_data[std::string(kModelConfigAsset)] = std::vector<std::byte>(
        reinterpret_cast<const std::byte*>(config.data()), reinterpret_cast<const std::byte*>(config.data() + config.size()));
    for (const auto& [name, data] : impl_->asset_data) impl_->assets.push_back(name);
    impl_->metadata["source_format"] = "\"gguf\"";
    impl_->metadata["tensor_shapes"] = shapes.dump();
    impl_->metadata["gguf_metadata"] = meta.dump();
}

GgufModelSource::~GgufModelSource() = default;
const std::vector<std::filesystem::path>& GgufModelSource::source_paths() const noexcept { return impl_->paths; }
std::string_view GgufModelSource::architecture() const noexcept { return impl_->arch; }
const std::unordered_map<std::string, std::string>& GgufModelSource::metadata() const noexcept { return impl_->metadata; }
const std::vector<TensorMetadata>& GgufModelSource::tensors() const noexcept { return impl_->tensors; }
const TensorMetadata* GgufModelSource::find_tensor(std::string_view name) const noexcept {
    const auto it = impl_->records.find(std::string(name));
    return it == impl_->records.end() ? nullptr : &it->second.tensor;
}
ModelSource::TensorReader GgufModelSource::tensor_reader(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    return [saved = std::shared_ptr<const Impl::Record>(impl_, &it->second)](uint64_t offset, std::byte* destination, std::size_t size) {
        const auto& record = *saved;
        if (offset > record.tensor.nbytes || size > record.tensor.nbytes - offset)
            throw std::out_of_range("GGUF tensor read exceeds its byte range");
        if (!size) return;
        if (!destination) throw std::invalid_argument("GGUF tensor destination is null");
        if (record.external) {
            record.external->read_range_into(record.external_name, offset, destination, size);
            return;
        }
        if (!record.inline_bytes.empty()) {
            if (size) std::memcpy(destination, record.inline_bytes.data() + offset, size);
            return;
        }
        if (!record.pieces.empty()) {
            auto piece = std::upper_bound(record.pieces.begin(), record.pieces.end(), offset,
                [](uint64_t at, const Impl::Record::Piece& value) { return at < value.begin; }) - 1;
            while (size) {
                const auto within = offset - piece->begin;
                const auto count = std::min<uint64_t>(size, piece->size - within);
                piece->file->read(piece->file_offset + within, destination, count);
                offset += count; destination += count; size -= count;
                ++piece;
            }
            return;
        }
        if (offset < record.prefix.size()) {
            const auto count = std::min<std::size_t>(size, record.prefix.size() - offset);
            std::memcpy(destination, record.prefix.data() + offset, count);
            offset += count; destination += count; size -= count;
        }
        if (size) record.file->read(record.offset + offset - record.prefix.size(), destination, size);
    };
}
std::optional<TensorFileRange> GgufModelSource::tensor_file_range(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end() || !it->second.prefix.empty() || !it->second.inline_bytes.empty() || !it->second.pieces.empty() || it->second.external || it->second.file->paths().size() != 1) return std::nullopt;
    return TensorFileRange{it->second.path, it->second.offset, it->second.tensor.nbytes};
}
ModelSource::TensorBatchReader GgufModelSource::tensor_batch_reader(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    const auto fallback = tensor_reader(name);
    return [saved = std::shared_ptr<const Impl::Record>(impl_, &it->second), fallback](const std::vector<ReadSpan>& spans) {
        const auto& record = *saved;
        for (const auto& span : spans) {
            if (span.offset > record.tensor.nbytes || span.size > record.tensor.nbytes - span.offset)
                throw std::out_of_range("GGUF tensor read exceeds its byte range");
            if (span.size && !span.destination) throw std::invalid_argument("GGUF tensor destination is null");
        }
        if (!record.prefix.empty() || !record.inline_bytes.empty() || record.external) {
            for (const auto& span : spans) fallback(span.offset, span.destination, span.size);
            return;
        }
        if (!record.pieces.empty()) {
            std::vector<std::pair<std::uintptr_t, std::uintptr_t>> targets;
            targets.reserve(spans.size());
            for (const auto& span : spans) if (span.size) {
                const auto first = reinterpret_cast<std::uintptr_t>(span.destination);
                if (span.size > UINTPTR_MAX - first) throw std::invalid_argument("GGUF tensor destination range overflow");
                targets.emplace_back(first, first + span.size);
            }
            std::sort(targets.begin(), targets.end());
            for (std::size_t i = 1; i < targets.size(); ++i) if (targets[i].first < targets[i - 1].second) {
                for (const auto& span : spans) fallback(span.offset, span.destination, span.size);
                return;
            }
            struct Group {
                std::shared_ptr<GgufFile> file;
                std::vector<ReadSpan> spans;
            };
            std::vector<Group> groups;
            for (const auto& span : spans) {
                if (!span.size) continue;
                auto offset = span.offset;
                auto size = span.size;
                auto* destination = span.destination;
                auto piece = std::upper_bound(record.pieces.begin(), record.pieces.end(), offset,
                    [](uint64_t at, const Impl::Record::Piece& value) { return at < value.begin; }) - 1;
                while (size) {
                    const auto within = offset - piece->begin;
                    const auto count = std::min<uint64_t>(size, piece->size - within);
                    auto group = std::find_if(groups.begin(), groups.end(),
                        [&](const Group& value) { return value.file == piece->file; });
                    if (group == groups.end()) {
                        groups.push_back({piece->file, {}});
                        group = groups.end() - 1;
                    }
                    group->spans.push_back({piece->file_offset + within, destination, std::size_t(count)});
                    offset += count; destination += count; size -= count;
                    ++piece;
                }
            }
            std::vector<ReadSpan> destinations;
            for (auto& group : groups) {
                std::sort(group.spans.begin(), group.spans.end(), [](const auto& a, const auto& b) { return a.offset < b.offset; });
                for (std::size_t begin = 0; begin < group.spans.size();) {
                    destinations.clear();
                    auto end = begin;
                    auto next = group.spans[begin].offset;
                    while (end < group.spans.size() && group.spans[end].offset == next) {
                        destinations.push_back(group.spans[end]);
                        next += group.spans[end++].size;
                    }
                    group.file->read_contiguous(destinations);
                    begin = end;
                }
            }
            return;
        }
        std::vector<ReadSpan> destinations;
        for (std::size_t begin = 0; begin < spans.size();) {
            destinations.clear();
            auto end = begin;
            auto next = spans[begin].offset;
            while (end < spans.size() && spans[end].offset == next) {
                destinations.push_back({record.offset + spans[end].offset, spans[end].destination, spans[end].size});
                next += spans[end++].size;
            }
            record.file->read_contiguous(destinations);
            begin = end;
        }
    };
}
void GgufModelSource::read_range_into(std::string_view name, uint64_t offset, std::byte* destination, std::size_t size) const {
    tensor_reader(name)(offset, destination, size);
}
const std::vector<std::string>& GgufModelSource::assets() const noexcept { return impl_->assets; }
bool GgufModelSource::has_asset(std::string_view name) const noexcept { return impl_->asset_data.count(std::string(name)) != 0; }
std::vector<std::byte> GgufModelSource::read_asset(std::string_view name) const { return impl_->asset_data.at(std::string(name)); }
std::optional<ModelGraph> GgufModelSource::model_graph() const { return std::nullopt; }

} // namespace mfq
