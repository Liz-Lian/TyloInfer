#pragma once

#include "request.h"
#include "result.h"
#include "emitter.h"
#include "tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace mfq::engine {

// Engine-owned parser state. append/finish only return values.
class InferenceOutput {
public:
    InferenceOutput(const InferenceRequest& request,
                    const MfqTokenizer* tokenizer, std::string request_id);
    TokenOutput append(const std::vector<std::int64_t>& tokens);
    TokenOutput finish();
    bool stopped() const noexcept;
    InferenceResult result;
    InferenceMetrics metrics;
    // Internal teardown result, checked before the terminal event is built.
    std::exception_ptr cleanup_failure;
private:
    void parse(std::string piece, bool partial, TokenOutput& output);
    const InferenceRequest& request_;
    const MfqTokenizer* tokenizer_;
    std::string id_, generated_;
    TextEmitter emitter_;
    common_chat_msg message_;
    std::vector<std::string> tool_ids_;
    bool stopped_ = false, finished_ = false;
};

} // namespace mfq::engine
