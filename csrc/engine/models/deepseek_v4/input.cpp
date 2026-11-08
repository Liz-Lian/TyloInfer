#include "input.h"
#include "engine/models/prepare_internal.h"
#include "engine/requests/request.h"
#include "engine/requests/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace tyloi::models::deepseek_v4 {
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

} // namespace

void prepare_media(
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

} // namespace tyloi::models::deepseek_v4
