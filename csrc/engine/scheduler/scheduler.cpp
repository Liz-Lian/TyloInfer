#include "scheduler.h"
#include "request.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace mfq::engine {

namespace {
constexpr std::size_t terminal_reserve = sizeof(EngineEvent) + 128 + 64 + 512;
std::size_t event_bytes(const EventData& data) {
    return std::visit([](const auto& value) -> std::size_t {
        using T = std::decay_t<decltype(value)>;
        std::size_t bytes = sizeof(EngineEvent);
        if constexpr (std::is_same_v<T, OutputDelta>) {
            bytes += value.token_ids.size() * sizeof(std::int64_t) + value.diffs.size() * sizeof(common_chat_msg_diff);
            for (const auto& diff : value.diffs) {
                bytes += diff.content_delta.size() + diff.reasoning_content_delta.size();
                bytes += diff.tool_call_delta.name.size() + diff.tool_call_delta.arguments.size() + diff.tool_call_delta.id.size();
            }
        } else if constexpr (std::is_same_v<T, Completed>) {
            bytes += value.finish_reason.size();
        } else if constexpr (std::is_same_v<T, Failed>) {
            bytes += value.code.size() + value.message.size();
        }
        return bytes;
    }, data);
}
}

void Engine::reset_states(std::int64_t batch) { scheduler_->reset_states(batch); }
void Engine::start() {
    try { scheduler_ = std::make_unique<Scheduler>(*this); }
    catch (...) {
        try { unload_model(); }
        catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "runtime_unload_cleanup error=unknown\n"; }
        throw;
    }
}
std::shared_ptr<Output> Engine::submit(EngineRequest request) const { return scheduler_->submit(std::move(request)); }
bool Engine::cancel(const std::string& id) const { return scheduler_->cancel_request(id); }
bool Engine::cancel_session(const std::string& id) const { return scheduler_->cancel_session(id); }
void Engine::cancel_all() const { scheduler_->cancel_all(); }
EngineInfo Engine::info() const { return scheduler_->info(); }
EngineStatus Engine::status() const { return scheduler_->status(); }
Engine::Snapshot Engine::snapshot() const { return scheduler_->snapshot(); }
ControlResult Engine::control(ControlRequest request) const { return scheduler_->control(std::move(request)); }
SessionResult Engine::session(SessionCommand command) const { return scheduler_->session(std::move(command)); }
std::int64_t Engine::reload(std::int64_t context) const { return scheduler_->reload(context); }
void Engine::enable_reload() { scheduler_->enable_reload(); }
std::optional<Engine::Reload> Engine::next_reload() { return scheduler_->next_reload(); }
void Engine::shutdown() { if (scheduler_) scheduler_->shutdown(); else unload_model(); }
void Engine::prepare_duplex_session(const std::string& prompt, MfqDuplexSessionParams& params) const {
    params = std::get<MfqDuplexSessionParams>(control(PrepareDuplex{prompt, params}));
}
void Engine::prepare_duplex_step(const std::string& text, MfqDuplexStepInput& input) const {
    input = std::get<MfqDuplexStepInput>(control(PrepareDuplexStep{text, input}));
}
std::string Engine::decode_tokens(const std::vector<std::int64_t>& tokens,
        const std::unordered_set<std::int64_t>& excluded) const {
    return std::get<std::string>(control(DecodeTokens{tokens, excluded}));
}

std::vector<EngineEvent> Output::wait() {
    std::unique_lock lock(mutex_);
    ready_.wait(lock, [&] { return terminal_ || !events_.empty(); });
    std::vector<EngineEvent> result;
    while (!events_.empty()) {
        result.push_back(std::move(events_.front()));
        events_.pop_front();
    }
    bytes_ = 0;
    lock.unlock();
    wake_->notify_one();
    return result;
}

bool Output::done() const {
    std::lock_guard lock(mutex_);
    return terminal_ && events_.empty();
}

Scheduler::Scheduler(Engine& engine) : Scheduler(engine, Limits{}) {}
Scheduler::Scheduler(Engine& engine, Limits limits)
    : engine_(engine), limits_(limits), info_(engine.describe()) {
    if (limits.events < 4 || limits.bytes < 4096)
        throw std::invalid_argument("outbox requires at least 4 events and 4096 bytes");
    std::promise<void> ready;
    auto initialized = ready.get_future();
    worker_ = std::thread([this, &ready] {
        refresh_snapshot(true, false, true);
        ready.set_value();
        loop();
    });
    initialized.get();
}
Scheduler::~Scheduler() { shutdown(); }
void Scheduler::shutdown() {
    { std::lock_guard lock(mutex_); stopping_ = true; snapshot_.status.available = 0; }
    wake_->notify_one();
    reload_wake_.notify_all();
    if (worker_.joinable()) worker_.join();
}
void Scheduler::enqueue(Command command) const {
    std::lock_guard lock(mutex_);
    if (stopping_) throw std::runtime_error("scheduler is stopping");
    mailbox_.push_back(std::move(command));
    wake_->notify_one();
}
EngineInfo Scheduler::info() const {
    std::lock_guard lock(mutex_);
    auto info = info_;
    info.reload &= reload_enabled_ && !stopping_;
    return info;
}
void Scheduler::enable_reload() {
    std::lock_guard lock(mutex_);
    if (stopping_) throw std::runtime_error("scheduler is stopping");
    if (reload_enabled_) throw std::runtime_error("reload owner already enabled");
    reload_enabled_ = true;
}
std::optional<Engine::Reload> Scheduler::next_reload() {
    std::unique_lock lock(mutex_);
    reload_wake_.wait(lock, [&] { return reload_job_.has_value() || stopping_; });
    if (!reload_job_) return std::nullopt;
    auto job = std::move(reload_job_);
    reload_job_.reset();
    return job;
}
EngineStatus Scheduler::status() const {
    Status command;
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
Engine::Snapshot Scheduler::snapshot() const {
    std::lock_guard lock(mutex_);
    if (!stopping_ && Clock::now() - snapshot_.captured >= std::chrono::milliseconds(250)) {
        snapshot_requested_ = true;
        wake_->notify_one();
    }
    return snapshot_;
}
void Scheduler::refresh_snapshot(bool healthy, bool blocked, bool metrics) {
    auto status = execution_status();
    status.healthy &= healthy;
    if (blocked || !status.healthy) status.available = 0;
    std::optional<Metrics> runtime, session;
    if (metrics) {
        try { runtime = std::get<Metrics>(engine_.control_command(RuntimeMetrics{})); } catch (...) {}
        try { session = engine_.session_command({SessionCommand::Kind::metrics}).metrics; } catch (...) {}
    }
    std::lock_guard lock(mutex_);
    if (stopping_) status.available = 0;
    snapshot_.status = status;
    const auto merge = [](Metrics& target, const Metrics& updates) {
        for (const auto& value : updates) {
            auto found = std::find_if(target.begin(), target.end(), [&](const auto& existing) {
                return existing.first == value.first;
            });
            if (found == target.end()) target.push_back(value);
            else found->second = value.second;
        }
    };
    if (runtime) merge(snapshot_.runtime, *runtime);
    if (session) merge(snapshot_.session, *session);
    if (runtime && session) snapshot_.captured = Clock::now();
}
std::shared_ptr<Output> Scheduler::submit(EngineRequest input) const {
    if (input.id.empty() || input.id.size() > 128)
        throw std::invalid_argument("request ID must contain 1-128 bytes");
    auto outbox = std::make_shared<Output>();
    outbox->wake_ = wake_;
    Submit command{std::move(input), outbox, {}};
    auto result = command.reply.get_future();
    enqueue(std::move(command));
    result.get();
    return outbox;
}
bool Scheduler::cancel_request(const std::string& id) const {
    Cancel command{id, false, false, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
bool Scheduler::cancel_session(const std::string& id) const {
    Cancel command{id, true, false, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
void Scheduler::cancel_all() const {
    Cancel command{{}, false, true, {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); result.get();
}
ControlResult Scheduler::control(ControlRequest request) const {
    Control command{std::move(request), {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
SessionResult Scheduler::session(SessionCommand request) const {
    Session command{std::move(request), {}};
    auto result = command.reply.get_future(); enqueue(std::move(command)); return result.get();
}
std::int64_t Scheduler::reload(std::int64_t context) const {
    Reload command{context, {}};
    auto result = command.reply.get_future();
    {
        std::lock_guard lock(mutex_);
        if (stopping_) throw std::runtime_error("scheduler is stopping");
        if (!reload_enabled_ || !info_.reload) throw std::runtime_error("reload is unavailable");
        mailbox_.push_back(std::move(command));
    }
    wake_->notify_one();
    return result.get();
}
void Scheduler::publish(Request& request, EngineEvent event) {
    auto& box = *request.outbox;
    std::unique_lock lock(box.mutex_);
    if (box.terminal_) return;
    const bool end = terminal(event.data);
    request.finished |= end;
    if (request.failure) {
        if (!end) return;
        event.data = *request.failure;
    }
    if (auto* failed = std::get_if<Failed>(&event.data)) {
        failed->code.resize(std::min<std::size_t>(failed->code.size(), 64));
        failed->message.resize(std::min<std::size_t>(failed->message.size(), 512));
    }
    if (auto* completed = std::get_if<Completed>(&event.data))
        completed->finish_reason.resize(std::min<std::size_t>(completed->finish_reason.size(), 64));
    const auto bytes = event_bytes(event.data) + event.id.size();
    if (!end && bytes > limits_.bytes - terminal_reserve) {
        request.failure = Failed{"output_limit", "engine output exceeds the request outbox budget"};
        lock.unlock();
        cancel(request);
        return;
    }
    if (!request.pending.empty() || (!end &&
            (box.events_.size() >= limits_.events - 1 || box.bytes_ + bytes > limits_.bytes - terminal_reserve))) {
        request.pending.push_back(std::move(event));
        return;
    }
    box.bytes_ += bytes;
    box.terminal_ = end;
    box.events_.push_back(std::move(event));
    box.ready_.notify_all();
}
void Scheduler::flush(Request& request) {
    auto& box = *request.outbox;
    std::lock_guard lock(box.mutex_);
    while (!request.pending.empty()) {
        auto& event = request.pending.front();
        const bool end = terminal(event.data);
        const auto bytes = event_bytes(event.data) + event.id.size();
        if (!end && (box.events_.size() >= limits_.events - 1 ||
                box.bytes_ + bytes > limits_.bytes - terminal_reserve)) break;
        box.bytes_ += bytes;
        box.terminal_ = end;
        box.events_.push_back(std::move(event));
        request.pending.pop_front();
        box.ready_.notify_all();
    }
}
void Scheduler::cancel(Request& request) {
    if (request.finished) {
        if (request.pending.empty()) return;
        auto terminal = std::move(request.pending.back());
        request.pending.clear();
        publish(request, std::move(terminal));
        return;
    }
    if (request.cancelling) return;
    request.cancelling = true;
    request.pending.clear();
    if (request.output) request.output->result.cancelled = true;
    else {
        publish(request, {request.submitted.id, Cancelled{}});
    }
}

bool Scheduler::has_active() const {
    return std::any_of(requests_.begin(), requests_.end(),
        [](const auto& entry) { return entry.second.output.has_value(); });
}

EngineStatus Scheduler::execution_status() const {
    if (!healthy_ || !engine_.ready()) return {0, false};
    if (engine_.exclusive()) return {0, true};
    std::size_t active = 0;
    for (const auto& [id, request] : requests_) {
        if (!request.output) continue;
        if (!request.batched) return {0, true};
        ++active;
    }
    return {info_.max_requests - active, true};
}

void Scheduler::reset_states(std::int64_t batch) {
    for (auto& [id, request] : requests_)
        if (request.state) request.state->reset(batch);
}

std::pair<bool, std::vector<EngineEvent>> Scheduler::advance(const std::vector<RequestId>& eligible) {
    auto scope = engine_.enter();
    try {
        engine_.execute(eligible);
    } catch (...) {
        healthy_ = false;
        for (auto& [id, request] : requests_)
            if (request.output) request.complete(std::current_exception());
    }
    bool advanced = false;
    std::vector<EngineEvent> events;
    for (auto& [id, request] : requests_) {
        if (!request.output) continue;
        if (!request.done && (request.output->result.cancelled ||
            std::find(eligible.begin(), eligible.end(), id) != eligible.end())) {
            try {
                if (!request.started) {
                    request.started = true;
                    request.generation = engine_.generate(id, request);
                }
                auto step = request.generation.next();
                if (step.state == StepState::advanced) advanced = true;
                if (step.value) {
                    if (auto* progress = std::get_if<PrefillProgress>(&*step.value))
                        request.output->metrics.mark_prefill(progress->timing);
                    events.push_back({id, std::move(*step.value)});
                }
                if (!step) request.complete();
            } catch (...) { request.complete(std::current_exception()); }
        }
        if (!request.done) continue;
        engine_.retire_request(request);
        if (request.output->cleanup_failure) {
            healthy_ = false;
            request.error = request.output->cleanup_failure;
        }
        finish_request(id, request, healthy_, events);
        request.error = {};
        request.output.reset();
        request.input = {};
    }
    return {advanced, std::move(events)};
}

void Scheduler::retire_failed() noexcept {
    std::exception_ptr failure;
    const auto remember = [&](std::exception_ptr error) { if (!failure) failure = std::move(error); };
    for (auto& [id, request] : requests_)
        if (request.output) request.output->result.cancelled = true;
    if (has_active() && engine_.ready())
        try { (void)advance({}); } catch (...) { remember(std::current_exception()); }
    const auto clear = [&] {
        for (auto& [id, request] : requests_) {
            if (!request.output) continue;
            engine_.retire_request(request);
            if (request.output->cleanup_failure) {
                healthy_ = false;
                remember(request.output->cleanup_failure);
            }
            request.error = {};
            request.output.reset();
            request.input = {};
        }
    };
    if (has_active()) {
        if (engine_.ready()) {
            try { auto scope = engine_.enter(); clear(); }
            catch (...) { healthy_ = false; remember(std::current_exception()); clear(); }
        } else clear();
    }
    if (failure) {
        try { std::rethrow_exception(failure); }
        catch (const std::exception& error) { std::cerr << "request_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "request_cleanup error=unknown\n"; }
    }
}

void Scheduler::loop() noexcept {
    std::optional<Reload> reload;
    bool duplex_active = false;
    for (;;) {
        std::deque<Command> commands;
        bool stopping;
        { std::lock_guard lock(mutex_); commands.swap(mailbox_); stopping = stopping_; }
        for (auto& command : commands) std::visit([&](auto& value) {
            using T = std::decay_t<decltype(value)>;
            try {
                if constexpr (std::is_same_v<T, Submit>) {
                    auto& input = value.input;
                    const auto id = input.id;
                    if (stopping || !healthy_ || reload || duplex_active)
                        throw std::runtime_error("scheduler is not accepting requests");
                    if (requests_.contains(id)) throw std::invalid_argument("request ID is already active");
                    const auto& session = input.input.cache_plan.session_id;
                    if (!session.empty()) for (const auto& [other_id, other] : requests_)
                        if (other.submitted.input.cache_plan.session_id == session)
                            throw std::invalid_argument("session already has an active request");
                    requests_.try_emplace(id, std::move(input), std::move(value.outbox));
                    try { order_.push_back(id); }
                    catch (...) { requests_.erase(id); throw; }
                    value.reply.set_value();
                } else if constexpr (std::is_same_v<T, Cancel>) {
                    bool found = false;
                    for (auto& [id, request] : requests_) {
                        if (value.all || (value.session ? request.submitted.input.cache_plan.session_id == value.id : id == value.id)) {
                            cancel(request); found = true;
                        }
                    }
                    value.reply.set_value(found);
                } else if constexpr (std::is_same_v<T, Control>) {
                    if ((std::holds_alternative<ScoreText>(value.request) || std::holds_alternative<ScoreTokens>(value.request)) &&
                        (duplex_active || !requests_.empty()))
                        throw std::runtime_error("probability scoring requires an idle runtime");
                    const bool start_duplex = std::holds_alternative<MfqDuplexSessionParams>(value.request);
                    const bool stop_duplex = std::holds_alternative<StopDuplex>(value.request);
                    if (reload) throw std::runtime_error("reload is pending");
                    if ((std::holds_alternative<MfqDuplexSessionParams>(value.request) ||
                         std::holds_alternative<MfqDuplexStepInput>(value.request) ||
                         std::holds_alternative<StopDuplex>(value.request)) && !requests_.empty())
                        throw std::runtime_error("duplex conflicts with active generation");
                    auto result = engine_.control_command(std::move(value.request));
                    if (start_duplex) duplex_active = true;
                    if (stop_duplex) duplex_active = false;
                    refresh_snapshot(healthy_, duplex_active || stopping, true);
                    value.reply.set_value(std::move(result));
                } else if constexpr (std::is_same_v<T, Session>) {
                    if (value.request.kind != SessionCommand::Kind::metrics) {
                        if (duplex_active || reload) throw std::runtime_error("session operation conflicts with runtime control");
                        for (const auto& [id, request] : requests_) {
                            const auto& session = request.submitted.input.cache_plan.session_id;
                            if (value.request.kind == SessionCommand::Kind::clear ||
                                value.request.kind == SessionCommand::Kind::memory_budget ||
                                value.request.kind == SessionCommand::Kind::refresh ||
                                ((value.request.kind == SessionCommand::Kind::fork ||
                                  value.request.kind == SessionCommand::Kind::close) &&
                                 (session == value.request.source || session == value.request.target)))
                                throw std::runtime_error("session operation conflicts with active generation");
                        }
                    }
                    auto result = engine_.session_command(value.request);
                    refresh_snapshot(healthy_, reload.has_value() || duplex_active || stopping, true);
                    value.reply.set_value(std::move(result));
                } else if constexpr (std::is_same_v<T, Status>) {
                    auto status = execution_status();
                    if (!status.healthy || reload || duplex_active || stopping) status.available = 0;
                    value.reply.set_value(status);
                } else {
                    if (reload) throw std::runtime_error("reload already pending");
                    if (duplex_active) {
                        (void)engine_.control_command(StopDuplex{});
                        duplex_active = false;
                    }
                    for (auto& [id, request] : requests_) cancel(request);
                    reload.emplace(std::move(value));
                }
            } catch (...) { value.reply.set_exception(std::current_exception()); }
        }, command);
        bool executable = false;
        auto wake_at = Clock::time_point::max();
        try {
            std::vector<RequestId> eligible;
            std::vector<RequestId> ordered(order_.begin(), order_.end());
            std::stable_sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) {
                return requests_.at(a).submitted.priority > requests_.at(b).submitted.priority;
            });
            bool admission_blocked = false;
            for (const auto& id : ordered) {
                auto& request = requests_.at(id);
                flush(request);
                if (stopping || (request.submitted.deadline && *request.submitted.deadline <= Clock::now())) cancel(request);
                if (request.finished) continue;
                if (request.submitted.deadline && !request.cancelling) wake_at = std::min(wake_at, *request.submitted.deadline);
                if (request.cancelling) { executable |= bool(request.output); continue; }
                if (!request.output && !reload && !admission_blocked && execution_status().available > 0) {
                    try {
                        auto scope = engine_.enter();
                        const bool batched = engine_.can_batch(request.submitted);
                        const bool active = has_active();
                        if (!batched && active) admission_blocked = true;
                        else {
                            engine_.prepare_request(request, batched);
                        }
                    }
                    catch (const InferenceInputError& error) {
                        publish(request, {id, Failed{error.code == InferenceInputErrorCode::Unsupported
                            ? "unsupported_input" : "invalid_request", error.what()}}); continue;
                    } catch (const std::invalid_argument& error) {
                        publish(request, {id, Failed{"invalid_request", error.what()}}); continue;
                    }
                }
                if (!request.output) continue;
                if (!request.pending.empty()) continue;
                auto& box = *request.outbox;
                std::lock_guard lock(box.mutex_);
                // Reserve one quantum and a separate terminal slot. No wait or
                // consumer acknowledgement occurs on the execution thread.
                if (!box.terminal_ && box.events_.size() + 3 < limits_.events &&
                        box.bytes_ < (limits_.bytes - terminal_reserve) / 2)
                    eligible.push_back(id);
            }
            executable |= !eligible.empty();
            if (executable) {
                auto [advanced, events] = advance(eligible);
                auto status = execution_status();
                // Teardown has completed when the engine becomes idle. Publish
                // its final resource snapshot before the terminal response;
                // otherwise a short request can leave prefill usage visible.
                if (!has_active() && std::any_of(events.begin(), events.end(),
                        [](const auto& event) { return terminal(event.data); }))
                    refresh_snapshot(healthy_ && status.healthy,
                        reload.has_value() || duplex_active || stopping, true);
                for (auto& event : events) {
                    auto found = requests_.find(event.id);
                    if (found != requests_.end()) publish(found->second, std::move(event));
                }
                if (!status.healthy) throw std::runtime_error("engine is unhealthy");
                executable = advanced || !events.empty();
            }
        } catch (const std::exception& error) {
            healthy_ = false;
            retire_failed();
            for (auto& [id, request] : requests_)
                if (!request.finished)
                    try { publish(request, {id, Failed{"backend_failure", error.what()}}); } catch (...) {}
        } catch (...) {
            healthy_ = false;
            retire_failed();
            for (auto& [id, request] : requests_)
                if (!request.finished)
                    try { publish(request, {id, Failed{"backend_failure", "unknown backend error"}}); } catch (...) {}
        }
        for (auto it = order_.begin(); it != order_.end();) {
            auto box = requests_.at(*it).outbox;
            std::lock_guard lock(box->mutex_);
            if (box->terminal_) { requests_.erase(*it); it = order_.erase(it); }
            else ++it;
        }
        if (reload && requests_.empty()) {
            bool unloading = false;
            try {
                if (reload->context < 1) throw std::invalid_argument("reload context must be positive");
                unloading = true;
                engine_.unload_model();
                Engine::Reload job{reload->context, {}};
                auto done = job.reply.get_future();
                {
                    std::lock_guard lock(mutex_);
                    if (stopping_ || !reload_enabled_) throw std::runtime_error("reload owner is unavailable");
                    snapshot_.status = {0, false};
                    reload_job_.emplace(std::move(job));
                }
                reload_wake_.notify_one();
                done.get();
                if (!engine_.ready()) throw std::runtime_error("reload did not load the engine");
                auto info = engine_.describe();
                const auto context = info.max_context;
                { std::lock_guard lock(mutex_); info_ = std::move(info); }
                healthy_ = true;
                refresh_snapshot(healthy_, stopping, true);
                reload->reply.set_value(context);
            } catch (...) {
                const auto failure = std::current_exception();
                healthy_ = false;
                if (unloading) {
                    try { engine_.unload_model(); }
                    catch (const std::exception& error) { std::cerr << "runtime_unload_cleanup error=" << error.what() << '\n'; }
                    catch (...) { std::cerr << "runtime_unload_cleanup error=unknown\n"; }
                }
                try { refresh_snapshot(healthy_, stopping, true); } catch (...) {}
                reload->reply.set_exception(failure);
            }
            reload.reset();
        }
        if (!order_.empty()) { order_.push_back(order_.front()); order_.pop_front(); }
        if (stopping && requests_.empty()) {
            try { engine_.unload_model(); } catch (...) {}
            return;
        }
        bool refresh;
        { std::lock_guard lock(mutex_); refresh = std::exchange(snapshot_requested_, false); }
        refresh_snapshot(healthy_, reload.has_value() || duplex_active || stopping, !commands.empty() || refresh);
        if (!executable) {
            std::unique_lock lock(mutex_);
            if (mailbox_.empty() && !stopping_) {
                // A timed recheck also covers a drain racing the wait setup.
                wake_->wait_until(lock, std::min(wake_at, Clock::now() + std::chrono::milliseconds(10)));
            }
        }
    }
}

} // namespace mfq::engine
