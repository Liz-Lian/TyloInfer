#pragma once

#include "chat.h"
#include "media.h"
#include "sampling.h"
#include "constraint.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

// Describes the portion of a rendered prompt whose KV state is stable across
// requests. A runtime may retain this exact token prefix after generation and
// reuse it only when the next request starts with the same token sequence.
struct MfqPromptCachePlan {
    std::string session_id;
    size_t stable_prefix_tokens = 0;
    bool cache_output_tokens = false;
};

namespace mfq::engine {

enum class InferenceInputErrorCode {
    Invalid,
    Unsupported,
    ContextLength,
};

class InferenceInputError final : public std::runtime_error {
public:
    InferenceInputError(
        InferenceInputErrorCode code,
        std::string message,
        std::string field = {});

    InferenceInputErrorCode code;
    std::string field;
};

struct ChatInput {
    common_chat_templates_inputs template_inputs;
    std::optional<std::string> preformatted_prompt;
    std::optional<bool> parallel_tool_calls;
    common_reasoning_format reasoning_format = COMMON_REASONING_FORMAT_AUTO;
};

struct InferenceInput {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    std::string prompt;
    std::optional<ChatInput> chat_input;
    std::vector<std::string> stops;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    std::optional<MfqMultimodalInput> media;
    bool prefix_cache_enabled = true;
    std::optional<std::size_t> benchmark_prompt_tokens;
};

struct ChatTemplateCapabilities {
    bool thinking = false;
    std::vector<std::string> reasoning_effort_values;
};

struct InferenceRequest {
    bool chat = true;
    bool stream = false;
    bool include_usage = false;
    common_chat_parser_params chat_parser;
    std::unordered_set<std::int64_t> preserved_tokens;
    std::vector<std::int64_t> prompt;
    std::vector<std::string> stops;
    std::unordered_set<std::int64_t> stop_token_ids;
    MfqSamplingParams sampling;
    MfqPromptCachePlan cache_plan;
    MfqTokenConstraintPtr token_constraint;
    std::optional<MfqMultimodalInput> vision;
};

using RequestId = std::string;
using Clock = std::chrono::steady_clock;

struct EngineRequest {
    RequestId id;
    InferenceInput input;
    std::optional<Clock::time_point> deadline;
    int priority = 0;
    // Diagnostics only. Production submits owned text/media and is tokenized
    // on admission, never against a tokenizer from before a queued reload.
    std::vector<std::int64_t> token_ids;
};

struct SessionCommand {
    enum class Kind { fork, close, clear, trim, budget, refresh, memory_budget, metrics } kind;
    std::string source, target;
    std::uint64_t bytes = 0;
    std::optional<std::uint64_t> disk_bytes;
    std::optional<std::uint64_t> weight_bytes;
    std::optional<std::uint64_t> ram_bytes;
};
struct DecodeTokens {
    std::vector<std::int64_t> tokens;
    std::unordered_set<std::int64_t> excluded;
};
struct PrepareDuplex { std::string prompt; MfqDuplexSessionParams parameters; };
struct PrepareDuplexStep { std::string text; MfqDuplexStepInput input; };
struct StopDuplex {};
struct RuntimeMetrics {};
struct CountTokens { std::string prompt; };
struct ScoreText {
    std::string prompt;
    std::vector<std::string> continuations;
    bool next_token = false;
};
struct ScoreTokens {
    std::vector<std::vector<std::int64_t>> sequences;
    std::vector<std::vector<std::int64_t>> targets;
    std::size_t prompt_tokens = 0;
    bool next_token = false;
};
using ControlRequest = std::variant<DecodeTokens, PrepareDuplex,
    PrepareDuplexStep, MfqDuplexSessionParams, MfqDuplexStepInput, StopDuplex,
    RuntimeMetrics, CountTokens, ScoreText, ScoreTokens>;

} // namespace mfq::engine
