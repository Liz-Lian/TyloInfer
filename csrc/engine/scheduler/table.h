#pragma once

#include "../engine.h"
#include "../generation/step.h"
#include "../models/state.h"
#include "../requests/processor.h"
#include "components/error.h"

#include <memory>

namespace mfq::engine {

struct EngineStepResult {
    std::vector<RequestId> advanced;
    std::vector<EngineEvent> events;
    EngineStatus status;
    bool has_work = false;
    std::optional<Clock::time_point> wake_at;
};

template <class Cache> SessionResult control_session(Cache &cache, const SessionCommand &command) {
    switch (command.kind) {
    case SessionCommand::Kind::fork:
        return {cache.fork_session(command.source, command.target), {}};
    case SessionCommand::Kind::close:
        return {cache.close_session(command.source), {}};
    case SessionCommand::Kind::clear:
        return {cache.clear(), {}};
    case SessionCommand::Kind::trim:
        return {cache.trim_hot(command.bytes), {}};
    case SessionCommand::Kind::budget:
        if constexpr (requires { cache.set_hot_limit(command.bytes); cache.set_disk_limit(command.bytes); }) {
            if (command.disk_bytes) cache.set_disk_limit(*command.disk_bytes);
            auto released = cache.set_hot_limit(command.bytes);
            if constexpr (requires { cache.set_ram_limit(command.bytes); }) {
                if (command.ram_bytes) released += cache.set_ram_limit(*command.ram_bytes);
            }
            return {released, {}};
        }
        else throw std::invalid_argument("prefix cache budgets are unavailable");
    case SessionCommand::Kind::metrics:
        return {0, cache.metrics()};
    case SessionCommand::Kind::refresh:
        if constexpr (requires { cache.refresh_disk_index(); }) return {cache.refresh_disk_index(), {}};
        else throw std::invalid_argument("prefix cache refresh is unavailable");
    case SessionCommand::Kind::memory_budget:
        throw std::invalid_argument("online resident memory budgets are unavailable for this backend");
    }
    throw std::invalid_argument("unknown session command");
}

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
void finish_request(const RequestId &id, Request &request, bool &healthy, EngineStepResult &result);

} // namespace mfq::engine
