#pragma once

#include "engine.h"
#include "../requests/processor.h"
#include "../requests/table.h"

#include <memory>

namespace mfq::engine {

// Backend owns native resources and guards; this composition owns the common
// Engine lifecycle. Destroy request coroutines before their text/model resources.
template <class Backend> class EngineInstance final : public Engine {
  public:
    Backend backend;

    explicit EngineInstance(typename Backend::Options options) : backend{std::move(options)} {
        load();
    }
    EngineInfo info() const override { return info_; }
    EngineStatus status() const override {
        return loaded_ ? requests_.status(backend.exclusive()) : EngineStatus{0, false};
    }
    Admission admit(EngineRequest request) override {
        return backend.visit([&](auto& ops) {
            return requests_.admit(std::move(request), text_.get(), info_, ops);
        });
    }
    void cancel(const RequestId& id) override { requests_.cancel(id); }
    EngineStepResult step(const std::vector<RequestId>& eligible) override {
        return backend.visit([&](auto& ops) { return requests_.step(eligible, ops); });
    }
    SessionResult session(const SessionCommand& command) override {
        if(command.kind == SessionCommand::Kind::memory_budget && !requests_.empty())
            throw std::runtime_error("memory budgeting requires a quiescent engine");
        return backend.visit([&](auto& ops) {
            if constexpr(requires { ops.session(command); })return ops.session(command);
            else return control_session(ops.cache, command);
        });
    }
    ControlResult control(ControlRequest request) override {
        if (!loaded_) throw std::runtime_error("engine is unloaded");
        return std::visit([&](auto&& value) -> ControlResult {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, DecodeTokens> || std::is_same_v<T, CountTokens> ||
                          std::is_same_v<T, PrepareDuplex> || std::is_same_v<T, PrepareDuplexStep>) {
                if (!text_) throw std::invalid_argument("text control requires a tokenizer");
                if constexpr (std::is_same_v<T, DecodeTokens>)
                    return text_->decode_tokens(value.tokens, value.excluded);
                else if constexpr (std::is_same_v<T, CountTokens>)
                    return TokenCount{text_->count_tokens(value.prompt), info_.max_context};
                else if constexpr (std::is_same_v<T, PrepareDuplex>) {
                    text_->prepare_duplex_session(value.prompt, value.parameters);
                    return std::move(value.parameters);
                } else {
                    text_->prepare_duplex_step(value.text, value.input);
                    return std::move(value.input);
                }
            } else
                return backend.visit([&](auto& ops) { return ops.control(std::move(value)); });
        }, std::move(request));
    }
    std::int64_t reload(std::int64_t context) override {
        if (context < 1) throw std::invalid_argument("reload context must be positive");
        if (!requests_.empty()) throw std::runtime_error("reload requires a quiescent engine");
        shutdown();
        backend.options.context_size = context;
        load();
        return info_.max_context;
    }
    void shutdown() override {
        loaded_ = false;
        auto failure = requests_.clear();
        text_.reset();
        backend.unload();
        if (failure) std::rethrow_exception(failure);
    }

  private:
    std::unique_ptr<TextProcessor> text_;
    EngineInfo info_;
    RequestExecutor requests_;
    bool loaded_ = false;

    void load() {
        try {
            auto [info, text] = backend.load();
            info_ = std::move(info);
            text_ = std::move(text);
            if (text_) info_.chat = text_->chat_template_capabilities();
            requests_ = RequestExecutor(info_.max_requests);
            loaded_ = true;
        } catch (...) {
            shutdown();
            throw;
        }
    }
};

} // namespace mfq::engine
