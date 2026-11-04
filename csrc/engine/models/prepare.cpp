#include "prepare.h"

#include "engine/requests/request.h"
#include "engine/requests/tokenizer.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <utility>

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

using engine::InferenceInputError;
using engine::InferenceInputErrorCode;
using engine::MfqTokenizer;

namespace {

std::string normalized_identity(std::string value) {
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return std::isalnum(c)
                ? static_cast<char>(std::tolower(c))
                : '_';
        });
    return value;
}

std::int64_t special_token(
        const MfqTokenizer& tokenizer,
        const std::string& marker) {
    const auto tokens = tokenizer.tokenize(marker, true, false);
    if (tokens.size() != 1) {
        throw std::runtime_error(
            "model tokenizer does not encode " + marker +
            " as one special token");
    }
    return tokens.front();
}

struct DeepseekImageBlock {
    std::vector<std::int64_t> types;
    std::vector<std::int64_t> permutation;
};

DeepseekImageBlock deepseek_v4_image_block(
        std::int64_t height,
        std::int64_t width,
        std::size_t start_position) {
    constexpr std::int64_t image_start = 0;
    constexpr std::int64_t image_pad = 1;
    constexpr std::int64_t image = 2;
    constexpr std::int64_t image_newline = 3;
    constexpr std::int64_t image_end = 4;
    const std::int64_t padded_height = height + height % 2;
    const std::int64_t row_length = width + 1;
    const std::int64_t compress_pad =
        3 - static_cast<std::int64_t>(start_position % 4);
    const std::int64_t trailing_pad =
        ((padded_height / 2 * row_length) % 2) * 2;
    DeepseekImageBlock result;
    result.types.insert(
        result.types.end(), static_cast<std::size_t>(compress_pad), image_pad);
    result.types.push_back(image_start);
    for (std::int64_t pair = 0; pair < padded_height / 2; ++pair) {
        for (std::int64_t column = 0; column < row_length; ++column) {
            for (std::int64_t within = 0; within < 2; ++within) {
                const std::int64_t row = pair * 2 + within;
                if (row < height && column < width) {
                    result.types.push_back(image);
                    result.permutation.push_back(row * width + column);
                } else if (row < height) {
                    result.types.push_back(image_newline);
                } else {
                    result.types.push_back(image_pad);
                }
            }
        }
    }
    result.types.insert(
        result.types.end(), static_cast<std::size_t>(trailing_pad), image_pad);
    result.types.push_back(image_end);
    return result;
}

DeepseekImageBlock deepseek_v41_image_block(
        std::int64_t height,
        std::int64_t width) {
    constexpr std::int64_t image_start = 0;
    constexpr std::int64_t image = 1;
    constexpr std::int64_t image_newline = 2;
    constexpr std::int64_t image_end = 3;
    DeepseekImageBlock result;
    result.types.push_back(image_start);
    for (std::int64_t row = 0; row < height; ++row) {
        result.types.insert(
            result.types.end(), static_cast<std::size_t>(width), image);
        result.types.push_back(image_newline);
    }
    result.types.push_back(image_end);
    return result;
}

void prepare_deepseek_media(
        const MfqTokenizer& tokenizer,
        std::int32_t vocabulary,
        std::vector<std::int64_t>& prompt,
        MfqMultimodalInput& media,
        bool v41) {
    const std::int64_t placeholder = special_token(
        tokenizer, "<｜deepseek_image｜>");
    const std::int64_t images = media.pixel_shape.at(0);
    std::vector<std::int64_t> expanded;
    expanded.reserve(
        prompt.size() + static_cast<std::size_t>(images) *
            (v41 ? 1024 : 384));
    media.image_bounds.clear();
    media.image_permutation.clear();
    media.image_permutation_offsets = {0};
    std::int64_t source = 0;
    for (const std::int64_t token : prompt) {
        if (token != placeholder) {
            expanded.push_back(token);
            continue;
        }
        if (source >= images) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "image placeholders exceed processed images", "messages");
        }
        const std::int64_t height = media.vision_grid[4 * source + 2];
        const std::int64_t width = media.vision_grid[4 * source + 3];
        auto block = v41
            ? deepseek_v41_image_block(height, width)
            : deepseek_v4_image_block(height, width, expanded.size());
        const std::int64_t begin =
            static_cast<std::int64_t>(expanded.size());
        for (const std::int64_t type : block.types) {
            expanded.push_back(v41 ? placeholder : vocabulary + type);
        }
        media.image_bounds.insert(
            media.image_bounds.end(),
            {0, source, begin, static_cast<std::int64_t>(expanded.size())});
        media.image_permutation.insert(
            media.image_permutation.end(),
            block.permutation.begin(), block.permutation.end());
        media.image_permutation_offsets.push_back(
            static_cast<std::int64_t>(media.image_permutation.size()));
        ++source;
    }
    if (source != images) {
        throw InferenceInputError(
            InferenceInputErrorCode::Invalid,
            "image placeholders do not match processed images", "messages");
    }
    prompt = std::move(expanded);
}

void prepare_minicpmo_media(
        const MfqTokenizer& tokenizer,
        const std::vector<std::int64_t>& prompt,
        MfqMultimodalInput& media) {
    media.image_bounds.clear();
    if (!media.pixel_shape.empty()) {
        const auto image_start = special_token(tokenizer, "<image>");
        const auto image_end = special_token(tokenizer, "</image>");
        const auto slice_start = special_token(tokenizer, "<slice>");
        const auto slice_end = special_token(tokenizer, "</slice>");
        for (std::size_t index = 0; index < prompt.size(); ++index) {
            const auto token = prompt[index];
            if (token != image_start && token != slice_start) continue;
            const auto end_token = token == image_start ? image_end : slice_end;
            const auto found = std::find(
                prompt.begin() + static_cast<std::ptrdiff_t>(index + 1),
                prompt.end(), end_token);
            if (found == prompt.end()) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o image placeholder is missing its end token",
                    "messages");
            }
            const auto end = static_cast<std::size_t>(found - prompt.begin());
            if (end - index - 1 != 64) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o image placeholder must contain 64 query tokens",
                    "messages");
            }
            const auto source = static_cast<std::int64_t>(
                media.image_bounds.size() / 4);
            media.image_bounds.insert(
                media.image_bounds.end(),
                {0, source, static_cast<std::int64_t>(index + 1),
                 static_cast<std::int64_t>(end)});
            index = end;
        }
        if (media.image_bounds.size() / 4 !=
                static_cast<std::size_t>(media.pixel_shape.at(0))) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "MiniCPM-o image placeholders do not match processed image slices",
                "messages");
        }
    }

    media.audio_bounds.clear();
    if (!media.audio_features_shape.empty()) {
        std::vector<std::int64_t> pooled_lengths;
        pooled_lengths.reserve(media.audio_lengths.size());
        for (const auto length : media.audio_lengths) {
            const auto after_convolution = (length - 1) / 2 + 1;
            pooled_lengths.push_back((after_convolution - 5) / 5 + 1);
        }
        const auto audio_start = special_token(tokenizer, "<|audio_start|>");
        const auto audio_end = special_token(tokenizer, "<|audio_end|>");
        for (std::size_t index = 0; index < prompt.size(); ++index) {
            if (prompt[index] != audio_start) continue;
            const auto found = std::find(
                prompt.begin() + static_cast<std::ptrdiff_t>(index + 1),
                prompt.end(), audio_end);
            if (found == prompt.end()) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o audio placeholder is missing its end token",
                    "messages");
            }
            const auto end = static_cast<std::size_t>(found - prompt.begin());
            const auto source = media.audio_bounds.size() / 4;
            if (source >= pooled_lengths.size() ||
                    static_cast<std::int64_t>(end - index - 1) !=
                        pooled_lengths[source]) {
                throw InferenceInputError(
                    InferenceInputErrorCode::Invalid,
                    "MiniCPM-o audio placeholder does not match pooled audio length",
                    "messages");
            }
            media.audio_bounds.insert(
                media.audio_bounds.end(),
                {0, static_cast<std::int64_t>(source),
                 static_cast<std::int64_t>(index + 1),
                 static_cast<std::int64_t>(end)});
            index = end;
        }
        if (media.audio_bounds.size() / 4 != pooled_lengths.size()) {
            throw InferenceInputError(
                InferenceInputErrorCode::Invalid,
                "MiniCPM-o audio placeholders do not match processed audio chunks",
                "messages");
        }
    }
}

} // namespace

void prepare_prefix(
        const std::string& model_type,
        const MfqTokenizer& tokenizer,
        const engine::InferenceInput& input,
        engine::InferenceRequest& work) {
    if (input.chat && input.chat_input &&
            normalized_identity(model_type).rfind(
                "deepseek_v4", 0) == 0 &&
            input.chat_input->template_inputs.add_generation_prompt) {
        const std::string marker = work.sampling.enable_thinking
            ? "<think>" : "</think>";
        const auto tokens = tokenizer.tokenize(marker, true);
        if (!tokens.empty() && tokens.size() < work.prompt.size() &&
                std::equal(
                    tokens.rbegin(), tokens.rend(), work.prompt.rbegin())) {
            work.cache_plan.stable_prefix_tokens =
                work.prompt.size() - tokens.size();
        }
    }
}

void prepare_media(
        const MfqTokenizer& tokenizer,
        std::int32_t vocabulary,
        std::vector<std::int64_t>& prompt,
        MfqMultimodalInput& media) {
    switch (media.processor) {
        case MfqMultimodalProcessor::minicpmo:
            prepare_minicpmo_media(tokenizer, prompt, media);
            return;
        case MfqMultimodalProcessor::deepseek_v4:
            prepare_deepseek_media(
                tokenizer, vocabulary, prompt, media, false);
            return;
        case MfqMultimodalProcessor::deepseek_v41:
            prepare_deepseek_media(
                tokenizer, vocabulary, prompt, media, true);
            return;
        case MfqMultimodalProcessor::grid_vision:
            return;
    }
}

void prepare_duplex_session(
        const MfqTokenizer& tokenizer,
        const std::string& system_prompt,
        MfqDuplexSessionParams& parameters) {
    parameters.system_prefix = tokenizer.tokenize(
        "<|im_start|>system\n" + system_prompt + "\n<|audio_start|>",
        true, false);
    parameters.system_suffix = tokenizer.tokenize(
        "<|audio_end|><|im_end|>", true, false);
    parameters.special_ids = {
        tokenizer.special_token_id("<unit>"),
        tokenizer.special_token_id("</unit>"),
        tokenizer.special_token_id("<image>"),
        tokenizer.special_token_id("</image>"),
        tokenizer.special_token_id("<slice>"),
        tokenizer.special_token_id("</slice>"),
        tokenizer.special_token_id("<|listen|>"),
        tokenizer.special_token_id("<|speak|>"),
        tokenizer.special_token_id("<|tts_bos|>"),
        tokenizer.special_token_id("<|tts_eos|>"),
        tokenizer.special_token_id("<|chunk_eos|>"),
        tokenizer.special_token_id("<|chunk_tts_eos|>"),
        tokenizer.special_token_id("<|turn_eos|>"),
        tokenizer.special_token_id("<|tts_pad|>"),
        151687,
    };
    parameters.forbidden_ids = {
        tokenizer.special_token_id("<|tts_pad|>"),
    };
}

void prepare_duplex_step(
        const MfqTokenizer& tokenizer,
        const std::string& text,
        MfqDuplexStepInput& step) {
    step.text_tokens = tokenizer.tokenize(text, false, false);
    if (step.text_tokens.empty()) {
        throw InferenceInputError(
            InferenceInputErrorCode::Invalid,
            "input.text produced no tokens", "text");
    }
}

} // namespace tyloi::models
