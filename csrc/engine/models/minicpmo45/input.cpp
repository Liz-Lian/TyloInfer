#include "input.h"
#include "engine/models/prepare_internal.h"
#include "engine/requests/request.h"
#include "engine/requests/tokenizer.h"

#include <algorithm>

namespace tyloi::models::minicpmo45 {
using engine::InferenceInputError;
using engine::InferenceInputErrorCode;
using engine::MfqTokenizer;

void prepare_media(
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

} // namespace tyloi::models::minicpmo45
