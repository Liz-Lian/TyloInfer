#pragma once

#include "engine/generation/sequence.h"
#include "engine/tensor.h"

#include <cstdint>
#include <memory>
#include <random>

namespace tyloi::models {

/** Select the official speech generation policy. */
struct SpeechOptions {
    std::int64_t steps = 0, eos_token = 6561, minimum_steps = 50, top_k = 25, minimum_keep = 0;
    double temperature = 0.8, top_p = 0.85, repetition_penalty = 1.05, min_p = 0.0;
};

/** Own the conditioning, generated codes, and optional per-step logits. */
struct SpeechResult {
    Tensor condition, codes, logits;
};

/** Generate speech using model-owned weights in the caller's execution scope. */
class Speech {
public:
    /** Own speech request state until the caller drains and retires it. */
    class Request {
    public:
        virtual ~Request() = default;
        /** Advance generation or report completion. */
        virtual StepResult<SpeechResult> next() = 0;
    };
    virtual ~Speech() = default;
    /** Build conditioning from aligned text IDs and normalized language hidden states. */
    virtual Tensor condition(const Tensor& ids, const Tensor& hidden) const = 0;
    /** Run speech with optional logits capture and a caller-owned evaluator RNG. */
    virtual std::unique_ptr<Request> generate(Tensor condition, SpeechOptions options,
        bool capture_logits = false, std::mt19937* evaluator_rng = nullptr) const = 0;
};

} // namespace tyloi::models
