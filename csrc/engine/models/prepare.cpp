#include "prepare.h"
#include "prepare_internal.h"
#include "deepseek_v4/input.h"
#include "minicpmo45/input.h"
#include "engine/requests/request.h"
#include "engine/requests/tokenizer.h"
#include "nlohmann/json.hpp"

#include <stdexcept>

namespace tyloi::models {

std::unordered_set<std::int64_t> stop_token_ids(std::string_view config, std::int64_t vocabulary) {
    std::unordered_set<std::int64_t> result;
    if (vocabulary <= 0) return result;
    const auto root = nlohmann::json::parse(config.begin(), config.end(), nullptr, false);
    if (!root.is_object()) return result;
    const auto* text = &root;
    if (root.contains("text_config") && root["text_config"].is_object()) text = &root["text_config"];
    auto found = text->find("eos_token_id");
    if (found == text->end() || found->is_null()) {
        text = &root;
        found = text->find("eos_token_id");
    }
    if (found == text->end()) return result;
    const auto add = [&](const auto& value) {
        if (!value.is_number_integer()) return;
        if (value.is_number_unsigned() && value.template get<std::uint64_t>() >= std::uint64_t(vocabulary)) return;
        const auto id = value.template get<std::int64_t>();
        if (id >= 0 && id < vocabulary) result.insert(id);
    };
    if (found->is_array()) for (const auto& value : *found) add(value);
    else add(*found);
    return result;
}

std::int64_t special_token(
        const engine::MfqTokenizer& tokenizer,
        const std::string& marker) {
    const auto tokens = tokenizer.tokenize(marker, true, false);
    if (tokens.size() != 1) {
        throw std::runtime_error(
            "model tokenizer does not encode " + marker +
            " as one special token");
    }
    return tokens.front();
}

void prepare_prefix(const std::string& model_type, const engine::MfqTokenizer& tokenizer,
        const engine::InferenceInput& input, engine::InferenceRequest& work) {
    deepseek_v4::prepare_prefix(model_type, tokenizer, input, work);
}

void prepare_media(
        const engine::MfqTokenizer& tokenizer,
        std::int32_t vocabulary,
        std::vector<std::int64_t>& prompt,
        MfqMultimodalInput& media) {
    switch (media.processor) {
        case MfqMultimodalProcessor::minicpmo:
            minicpmo45::prepare_media(tokenizer, prompt, media);
            return;
        case MfqMultimodalProcessor::deepseek_v4:
            deepseek_v4::prepare_media(
                tokenizer, vocabulary, prompt, media, false);
            return;
        case MfqMultimodalProcessor::deepseek_v41:
            deepseek_v4::prepare_media(
                tokenizer, vocabulary, prompt, media, true);
            return;
        case MfqMultimodalProcessor::grid_vision:
            return;
    }
}

void prepare_duplex_session(const engine::MfqTokenizer& tokenizer,
        const std::string& system_prompt, MfqDuplexSessionParams& parameters) {
    minicpmo45::prepare_duplex_session(tokenizer, system_prompt, parameters);
}

void prepare_duplex_step(const engine::MfqTokenizer& tokenizer,
        const std::string& text, MfqDuplexStepInput& step) {
    minicpmo45::prepare_duplex_step(tokenizer, text, step);
}

} // namespace tyloi::models
