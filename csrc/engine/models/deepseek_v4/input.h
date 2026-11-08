#pragma once

#include "engine/models/prepare.h"

namespace tyloi::models::deepseek_v4 {
void prepare_prefix(const std::string& model_type, const engine::MfqTokenizer& tokenizer,
    const engine::InferenceInput& input, engine::InferenceRequest& work);
void prepare_media(const engine::MfqTokenizer& tokenizer, std::int32_t vocabulary,
    std::vector<std::int64_t>& prompt, MfqMultimodalInput& media, bool v41);
}
