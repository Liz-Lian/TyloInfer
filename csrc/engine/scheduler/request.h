#pragma once

#include "../engine.h"
#include "../generation/step.h"
#include "../models/state.h"

#include <memory>

namespace mfq::engine {

class TextProcessor;

// Scheduler retires generation before publishing the terminal event.
struct Request {
    EngineRequest submitted;
    std::shared_ptr<Output> outbox;
    InferenceRequest input;
    std::optional<InferenceOutput> output;
    std::unique_ptr<mfq::models::State> state;
    Generation generation;
    std::exception_ptr error;
    bool started = false, batched = false, done = false;
    bool cancelling = false, finished = false;
    std::optional<Failed> failure;
    // At most one engine quantum waits here; it prevents further execution.
    std::deque<EngineEvent> pending;

    Request(EngineRequest request, std::shared_ptr<Output> box)
        : submitted(std::move(request)), outbox(std::move(box)) {}
    Request(const Request&) = delete;
    Request& operator=(const Request&) = delete;
    Request(Request&&) = delete;
    Request& operator=(Request&&) = delete;

    void complete(std::exception_ptr failure = {}) {
        done = true;
        error = failure;
    }
};

void prepare_request(Request &current, const TextProcessor *text, const EngineInfo &info, bool batched, bool mtp_available);
void finish_request(const RequestId &id, Request &request, bool &healthy, std::vector<EngineEvent> &events);

} // namespace mfq::engine
