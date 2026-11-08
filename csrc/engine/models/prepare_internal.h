#pragma once

#include <cstdint>
#include <string>

namespace tyloi::engine { class MfqTokenizer; }

namespace tyloi::models {
std::int64_t special_token(const engine::MfqTokenizer& tokenizer, const std::string& marker);
}
