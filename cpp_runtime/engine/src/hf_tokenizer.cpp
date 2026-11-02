#include "hf_tokenizer.h"
#include "tyloinfer/model_source.h"
#include "gguf.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

namespace mfq::engine {
namespace {
using json = nlohmann::json;
constexpr std::size_t max_vocabulary = 2'000'000;

json asset_json(const ModelSource& source, const char* name, bool required = true) {
    if (!source.has_asset(name)) {
        if (required) throw std::runtime_error(std::string("missing native tokenizer asset: ") + name);
        return json::object();
    }
    const auto bytes = source.read_asset(name);
    if (bytes.size() > 64 * 1024 * 1024) throw std::runtime_error("tokenizer asset exceeds 64 MiB");
    auto value = json::parse(reinterpret_cast<const char*>(bytes.data()),
        reinterpret_cast<const char*>(bytes.data()) + bytes.size());
    if (!value.is_object()) throw std::runtime_error("tokenizer asset must contain an object");
    return value;
}

std::optional<std::uint32_t> token_id(const json& value) {
    if (!value.is_number_integer() || value < 0 || value >= max_vocabulary) return {};
    return value.get<std::uint32_t>();
}

std::string tokenizer_pre(const std::string& type) {
    for (const auto* prefix : {"qwen3_5", "qwen3_6", "qwen3_8", "qwen4_exp"})
        if (type.rfind(prefix, 0) == 0) return "qwen35";
    if (type.rfind("glm5_next", 0) == 0 || type.rfind("glm_moe_dsa", 0) == 0) return "glm4";
    if (type.rfind("gemma4", 0) == 0) return "gemma4";
    if (type.rfind("minicpmo", 0) == 0) return "qwen2";
    if (type.rfind("deepseek_v4", 0) == 0) return "joyai-llm";
    throw std::runtime_error("unsupported native HF tokenizer family: " + type);
}

void string_array(gguf_context* metadata, const char* key, const std::vector<std::string>& strings) {
    std::vector<const char*> pointers;
    pointers.reserve(strings.size());
    for (const auto& value : strings) pointers.push_back(value.c_str());
    gguf_set_arr_str(metadata, key, pointers.data(), pointers.size());
}
}

std::vector<std::uint8_t> hf_tokenizer_gguf(const ModelSource& source) {
    const auto tokenizer = asset_json(source, "__mfq_asset__/hf/tokenizer.json");
    const auto settings = asset_json(source, "__mfq_asset__/hf/tokenizer_config.json", false);
    const auto generation = asset_json(source, "__mfq_asset__/hf/generation_config.json", false);
    const auto config = json::parse(source.model_config_json());
    const auto text = config.value("text_config", json::object());
    if (!config.is_object() || !text.is_object()) throw std::runtime_error("invalid tokenizer model configuration");
    const auto& model = tokenizer.at("model");
    if (!model.is_object() || model.value("type", "") != "BPE" ||
            !model.at("vocab").is_object() || !model.at("merges").is_array())
        throw std::runtime_error("native HF loading requires a BPE tokenizer vocabulary and merges");
    const auto added = tokenizer.value("added_tokens", json::array());
    if (!added.is_array()) throw std::runtime_error("invalid HF added tokens");
    std::unordered_map<std::uint32_t, std::pair<std::string, std::int32_t>> assigned;
    std::unordered_map<std::string, std::uint32_t> token_ids;
    std::size_t vocabulary = 0;
    const auto assign = [&](const std::string& token, const json& raw, std::int32_t type, bool replace) {
        const auto id = token_id(raw);
        if (!id || token.empty() || token.find('\0') != std::string::npos)
            throw std::runtime_error("invalid HF vocabulary entry");
        const auto previous = assigned.find(*id);
        if (previous != assigned.end() && (!replace || previous->second.first != token))
            throw std::runtime_error("conflicting HF vocabulary token IDs");
        assigned[*id] = {token, type};
        token_ids[token] = *id;
        vocabulary = std::max(vocabulary, std::size_t(*id) + 1);
    };
    for (const auto& item : model["vocab"].items()) assign(item.key(), item.value(), 1, false);
    for (const auto& item : added) {
        if (!item.is_object() || !item.at("content").is_string()) throw std::runtime_error("invalid HF added token");
        const auto token = item["content"].get<std::string>();
        const bool control = item.value("special", false) || token.rfind("<|fim_", 0) == 0 ||
            token == "<|repo_name|>" || token == "<|file_sep|>" || token == "</s>";
        assign(token, item.at("id"), control ? 3 : 4, true);
    }
    for (const auto* object : {&config, &text}) {
        if (object->contains("vocab_size")) {
            const auto count = token_id(object->at("vocab_size"));
            if (!count || !*count) throw std::runtime_error("invalid model vocabulary size");
            vocabulary = std::max(vocabulary, std::size_t(*count));
        }
    }
    if (!vocabulary) throw std::runtime_error("HF tokenizer vocabulary is empty");
    std::vector<std::string> tokens(vocabulary);
    std::vector<std::int32_t> types(vocabulary, 5);
    for (std::size_t id = 0; id < vocabulary; ++id) {
        const auto found = assigned.find(id);
        if (found == assigned.end()) tokens[id] = "[PAD" + std::to_string(id) + "]";
        else { tokens[id] = found->second.first; types[id] = found->second.second; }
    }
    std::vector<std::string> merges;
    for (const auto& merge : model["merges"]) {
        if (merge.is_string()) merges.push_back(merge.get<std::string>());
        else if (merge.is_array() && merge.size() == 2 && merge[0].is_string() && merge[1].is_string())
            merges.push_back(merge[0].get<std::string>() + " " + merge[1].get<std::string>());
        else throw std::runtime_error("invalid HF BPE merge");
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), gguf_free);
    gguf_set_val_str(metadata.get(), "general.architecture", "llama");
    gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(metadata.get(), "tokenizer.ggml.pre", tokenizer_pre(config.at("model_type").get<std::string>()).c_str());
    string_array(metadata.get(), "tokenizer.ggml.tokens", tokens);
    string_array(metadata.get(), "tokenizer.ggml.merges", merges);
    gguf_set_arr_data(metadata.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
    for (const auto* name : {"bos", "eos", "unk", "pad"}) {
        std::optional<std::uint32_t> id;
        const auto key = std::string(name) + "_token";
        auto content = settings.value(key, json());
        if (content.is_object()) content = content.value("content", json());
        if (content.is_string()) {
            const auto found = token_ids.find(content.get<std::string>());
            if (found != token_ids.end()) id = found->second;
        }
        for (const auto* object : {&config, &text, &generation}) {
            if (id || !object->contains(key + "_id")) continue;
            const auto& value = object->at(key + "_id");
            id = token_id(value.is_array() && !value.empty() ? value[0] : value);
        }
        if (id) {
            if (*id >= vocabulary) throw std::runtime_error("special token ID exceeds the vocabulary");
            gguf_set_val_u32(metadata.get(), ("tokenizer.ggml." + key + "_id").c_str(), *id);
        }
    }
    std::set<std::uint32_t> eog;
    for (const auto* object : {&generation, &config, &text, &settings}) {
        if (!object->contains("eos_token_id")) continue;
        auto values = object->at("eos_token_id");
        if (!values.is_array()) values = json::array({values});
        for (const auto& value : values) {
            const auto id = token_id(value);
            if (!id || *id >= vocabulary) throw std::runtime_error("eos_token_id exceeds the vocabulary");
            eog.insert(*id);
        }
    }
    if (!eog.empty()) {
        const std::vector<std::uint32_t> ids(eog.begin(), eog.end());
        gguf_set_arr_data(metadata.get(), "tokenizer.ggml.eog_token_ids", GGUF_TYPE_UINT32, ids.data(), ids.size());
    }
    for (const auto* name : {"add_bos_token", "add_eos_token"})
        if (settings.contains(name)) {
            if (!settings[name].is_boolean()) throw std::runtime_error("invalid tokenizer special token flag");
            gguf_set_val_bool(metadata.get(), (std::string("tokenizer.ggml.") + name).c_str(), settings[name].get<bool>());
        }
    auto chat = settings.value("chat_template", json());
    if (chat.is_array()) {
        json selected;
        for (const auto& item : chat)
            if (item.is_object() && item.value("name", "") == "default") selected = item.at("template");
        chat = std::move(selected);
    }
    if ((!chat.is_string() || chat.get_ref<const std::string&>().empty()) && source.has_asset("__mfq_asset__/hf/chat_template.jinja")) {
        const auto bytes = source.read_asset("__mfq_asset__/hf/chat_template.jinja");
        if (bytes.size() > 1024 * 1024) throw std::runtime_error("chat template exceeds 1 MiB");
        chat = std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    if (chat.is_string()) gguf_set_val_str(metadata.get(), "tokenizer.chat_template", chat.get_ref<const std::string&>().c_str());
    std::vector<std::uint8_t> bytes(gguf_get_meta_size(metadata.get()));
    gguf_get_meta_data(metadata.get(), bytes.data());
    return bytes;
}
} // namespace mfq::engine
