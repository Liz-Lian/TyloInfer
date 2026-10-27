#include "tyloinfer/gguf_model_source.h"

#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>

namespace mfq {
namespace {
using json = nlohmann::json;
using Gguf = std::unique_ptr<gguf_context, decltype(&gguf_free)>;
using Ggml = std::unique_ptr<ggml_context, decltype(&ggml_free)>;

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
        }
    }
    return config;
}
} // namespace

struct GgufModelSource::Impl {
    struct Record {
        TensorMetadata tensor;
        std::shared_ptr<FileRangeReader> file;
        std::filesystem::path path;
        uint64_t offset = 0;
        std::vector<std::byte> prefix;
        std::vector<std::byte> inline_bytes;
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
    std::ifstream file(path, std::ios::binary);
    char magic[4]{};
    return file.read(magic, 4) && std::memcmp(magic, "GGUF", 4) == 0;
}

GgufModelSource::GgufModelSource(const std::filesystem::path& path,
        std::vector<std::filesystem::path> companions) : impl_(std::make_shared<Impl>()) {
    const auto load = [](const std::filesystem::path& file, ggml_context** tensors) {
        Gguf ctx(gguf_init_from_file(file.string().c_str(), {true, tensors}), gguf_free);
        if (!ctx) throw std::runtime_error("invalid GGUF file: " + file.filename().string());
        return ctx;
    };
    auto first = load(path, nullptr);
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
    for (const auto& companion : companions) {
        const auto absolute = std::filesystem::absolute(companion);
        if (std::find(impl_->paths.begin(), impl_->paths.end(), absolute) != impl_->paths.end())
            throw std::runtime_error("GGUF companion duplicates a model shard");
        impl_->paths.push_back(absolute);
    }

    json meta = json::object(), shapes = json::object();
    for (std::size_t shard = 0; shard < impl_->paths.size(); ++shard) {
        ggml_context* raw = nullptr;
        auto ctx = load(impl_->paths[shard], &raw);
        Ggml tensors(raw, ggml_free);
        if (gguf_get_n_tensors(ctx.get()) > (1 << 20))
            throw std::runtime_error("GGUF has too many tensor records");
        if (splits > 1 && shard < model_shards) {
            const auto split_no = gguf_find_key(ctx.get(), "split.no");
            const auto split_count = gguf_find_key(ctx.get(), "split.count");
            if (split_no < 0 || split_count < 0 || scalar(ctx.get(), split_no) != shard || scalar(ctx.get(), split_count) != splits)
                throw std::runtime_error("inconsistent GGUF shard metadata");
        }
        const auto file_size = std::filesystem::file_size(impl_->paths[shard]);
        auto file = std::make_shared<FileRangeReader>(impl_->paths[shard], model_file_read_mode());
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
        for (int64_t i = 0; i < gguf_get_n_tensors(ctx.get()); ++i) {
            const auto* original = gguf_get_tensor_name(ctx.get(), i);
            const auto* tensor = ggml_get_tensor(tensors.get(), original);
            if (!tensor) throw std::runtime_error("GGUF tensor shape is missing");
            const auto type = gguf_get_tensor_type(ctx.get(), i);
            std::vector<int64_t> shape;
            for (int d = ggml_n_dims(tensor) - 1; d >= 0; --d) shape.push_back(tensor->ne[d]);
            auto name = canonical_name(original);
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
            std::string dtype;
            if (type == GGML_TYPE_F32) dtype = "F32";
            else if (type == GGML_TYPE_F16) dtype = "F16";
            else if (type == GGML_TYPE_BF16) dtype = "BF16";
            else if (type == GGML_TYPE_I32) dtype = "I32";
            else if (type == GGML_TYPE_I64) dtype = "I64";
            else if (ggml_is_quantized(type)) dtype = "GGUF_" + std::string(ggml_type_name(type));
            else throw std::runtime_error("unsupported GGUF tensor type: " + std::string(ggml_type_name(type)));
            std::transform(dtype.begin(), dtype.end(), dtype.begin(), [](unsigned char c) { return std::toupper(c); });
            if (!ggml_is_quantized(type)) record.prefix = dense_prefix(shape);
            record.tensor = {name, dtype, dtype, shape, size + record.prefix.size()};
            const bool norm_delta = impl_->arch == "qwen4exp" && name.find("norm.weight") != std::string::npos &&
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
    const auto config = model_config(meta, shapes, impl_->arch).dump();
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
    return [record = it->second](uint64_t offset, std::byte* destination, std::size_t size) {
        if (offset > record.tensor.nbytes || size > record.tensor.nbytes - offset)
            throw std::out_of_range("GGUF tensor read exceeds its byte range");
        if (!record.inline_bytes.empty()) {
            if (size) std::memcpy(destination, record.inline_bytes.data() + offset, size);
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
    if (it == impl_->records.end() || !it->second.prefix.empty() || !it->second.inline_bytes.empty()) return std::nullopt;
    return TensorFileRange{it->second.path, it->second.offset, it->second.tensor.nbytes};
}
ModelSource::TensorBatchReader GgufModelSource::tensor_batch_reader(std::string_view name) const {
    const auto it = impl_->records.find(std::string(name));
    if (it == impl_->records.end()) throw std::out_of_range("GGUF tensor not found: " + std::string(name));
    const auto fallback = tensor_reader(name);
    return [record = it->second, fallback](const std::vector<ReadSpan>& spans) {
        for (const auto& span : spans) {
            if (span.offset > record.tensor.nbytes || span.size > record.tensor.nbytes - span.offset)
                throw std::out_of_range("GGUF tensor read exceeds its byte range");
            if (span.size && !span.destination) throw std::invalid_argument("GGUF tensor destination is null");
        }
        if (!record.prefix.empty() || !record.inline_bytes.empty()) {
            for (const auto& span : spans) fallback(span.offset, span.destination, span.size);
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
