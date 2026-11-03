#pragma once

#include "request.h"
#include "../metrics.h"
#include "../generation/mtp/metrics.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <variant>
#include <vector>

struct MfqDuplexStepResult {
    std::vector<int64_t> generated_tokens;
    std::vector<int32_t> audio_tokens;
    bool is_listen = false;
    bool end_of_turn = false;
    bool tts_force_flush = false;
    int64_t audio_chunk_index = 0;
    int64_t language_cache_position = 0;
    int64_t audio_cache_position = 0;
    int64_t tts_cache_position = 0;
    double inference_ms = 0.0;
};

struct MfqPrefillTiming {
    size_t prompt_tokens = 0;
    // Language-model prompt evaluation only.
    double llm_ms = 0.0;
    // Vision/audio encoder and multimodal projector work preceding the LLM.
    double multimodal_ms = 0.0;
    // Complete model-side prefill wall time.
    double model_ms = 0.0;
};

namespace mfq::engine {

struct InferenceMetrics {
    using Clock = std::chrono::steady_clock;

    Clock::time_point started = Clock::now();
    Clock::time_point first_token;
    Clock::time_point last_token;
    std::size_t prefill_tokens = 0;
    double prefill_ms = 0.0;
    double multimodal_ms = 0.0;
    double model_prefill_ms = 0.0;
    bool saw_token = false;
    bool saw_prefill = false;
    mtp::GenerationStats mtp;

    void mark_prefill(const MfqPrefillTiming& timing);
    void mark_token();
};

struct InferenceResult {
    std::string text;
    std::string reasoning_text;
    std::vector<common_chat_tool_call> tool_calls;
    std::string finish_reason = "length";
    std::int32_t completion_tokens = 0;
    bool client_connected = true;
    bool cancelled = false;
};

struct TokenOutput {
    std::vector<std::int64_t> token_ids;
    std::vector<common_chat_msg_diff> diffs;
};

struct PrefillProgress {
    MfqPrefillTiming timing;
    // Prompt state is ready for caching before decode timing starts.
    bool session_boundary = false;
};
using OutputDelta = TokenOutput;
struct UsageUpdate { std::size_t prompt_tokens = 0, completion_tokens = 0; };
// Text/tool output travels only in deltas, so terminals have a bounded size.
struct Completed { std::string finish_reason; UsageUpdate usage; InferenceMetrics metrics; };
struct Cancelled { UsageUpdate usage; InferenceMetrics metrics; };
struct Failed { std::string code, message; bool retryable = false; };
using EventData = std::variant<PrefillProgress, OutputDelta, UsageUpdate,
                               Completed, Cancelled, Failed>;
struct EngineEvent { RequestId id; EventData data; };

class Scheduler;
class Output {
public:
    std::vector<EngineEvent> wait();
    bool done() const;

private:
    friend class Scheduler;
    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<EngineEvent> events_;
    std::size_t bytes_ = 0;
    bool terminal_ = false;
    std::shared_ptr<std::condition_variable> wake_;
};

inline bool terminal(const EventData& event) {
    return std::holds_alternative<Completed>(event) ||
        std::holds_alternative<Cancelled>(event) ||
        std::holds_alternative<Failed>(event);
}

struct SessionResult { std::uint64_t count = 0; Metrics metrics; };
struct TokenCount {
    std::size_t prompt_tokens = 0;
    std::int64_t context_size = 0;
};
struct TokenLikelihood {
    std::vector<std::int64_t> token_ids;
    std::vector<double> token_logprobs;
    double log_likelihood = 0;
};
struct LikelihoodResult {
    std::size_t prompt_tokens = 0;
    std::vector<TokenLikelihood> scores;
};
using ControlResult = std::variant<std::monostate,
    std::string, MfqDuplexSessionParams, MfqDuplexStepInput, MfqDuplexStepResult,
    Metrics, TokenCount, LikelihoodResult>;

} // namespace mfq::engine
