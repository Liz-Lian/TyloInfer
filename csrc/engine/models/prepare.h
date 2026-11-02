#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

struct MfqMultimodalInput;
struct MfqDuplexSessionParams;
struct MfqDuplexStepInput;
namespace mfq::engine {
class MfqTokenizer;
struct InferenceInput;
struct InferenceRequest;
}

namespace mfq::models {

std::unordered_set<std::int64_t> stop_token_ids(std::string_view config, std::int64_t vocabulary);

void prepare_prefix(const std::string& model_type, const engine::MfqTokenizer& tokenizer,
    const engine::InferenceInput& input, engine::InferenceRequest& work);
void prepare_media(const engine::MfqTokenizer& tokenizer, std::int32_t vocabulary,
    std::vector<std::int64_t>& prompt, MfqMultimodalInput& media);
void prepare_duplex_session(const engine::MfqTokenizer& tokenizer,
    const std::string& system_prompt, MfqDuplexSessionParams& parameters);
void prepare_duplex_step(const engine::MfqTokenizer& tokenizer,
    const std::string& text, MfqDuplexStepInput& step);

} // namespace mfq::models
