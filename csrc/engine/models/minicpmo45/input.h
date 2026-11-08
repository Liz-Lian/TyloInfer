#pragma once

#include "engine/models/prepare.h"

namespace tyloi::models::minicpmo45 {
void prepare_media(const engine::MfqTokenizer& tokenizer,
    const std::vector<std::int64_t>& prompt, MfqMultimodalInput& media);
void prepare_duplex_session(const engine::MfqTokenizer& tokenizer,
    const std::string& system_prompt, MfqDuplexSessionParams& parameters);
void prepare_duplex_step(const engine::MfqTokenizer& tokenizer,
    const std::string& text, MfqDuplexStepInput& step);
}
