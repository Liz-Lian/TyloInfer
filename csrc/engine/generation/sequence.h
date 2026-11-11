#pragma once

#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <stdexcept>
#include <utility>

namespace tyloi {

class Tensor;

enum class StepState { waiting, advanced, complete };

/** Request a workspace loan or detach its final output before the next visible quantum. */
struct Workspace {
    /** Describe a numerical tile whose admitted row count is written before resuming. */
    struct Tile {
        /** Total rows available in this numerical span. */
        std::int64_t tokens = 0;
        /** Temporary workspace required per admitted row. */
        std::size_t row_bytes = 0;
        /** Coroutine-owned output slot, valid until the next resume. */
        std::int64_t* compute = nullptr;
    };
    /** Positive bytes begin a loan; zero bytes finish the current loan. */
    std::size_t bytes = 0;
    /** Owning output slot to detach on finish; valid until the sequence resumes. */
    Tensor* output = nullptr;
    /** Optional tiled allocation geometry; bytes then describes fixed live storage. */
    std::optional<Tile> tile = std::nullopt;
    /** Additional output that shares the loan and must detach before its return. */
    Tensor* extra = nullptr;
};

template <class T> struct StepResult {
    StepState state = StepState::complete;
    std::optional<T> value;
    std::optional<Workspace> workspace;
    explicit operator bool() const { return state != StepState::complete; }
};

// A resume performs one bounded quantum. Values and progress are independent:
// waiting publishes nothing, while preparation can advance without a value.
template <class T> class StepSequence {
  public:
    struct promise_type {
        StepResult<T> step;
        std::exception_ptr error;
        StepSequence get_return_object() {
            return StepSequence{std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        std::suspend_always yield_value(T value) {
            step = {StepState::advanced, std::move(value)};
            return {};
        }
        std::suspend_always yield_value(StepState state) {
            step = {state, {}};
            return {};
        }
        std::suspend_always yield_value(Workspace workspace) {
            step = {StepState::advanced, {}, workspace};
            return {};
        }
        std::suspend_always yield_value(StepResult<T> value) {
            step = std::move(value);
            return {};
        }
        void return_void() noexcept {}
        void unhandled_exception() noexcept { error = std::current_exception(); }
    };
    StepSequence() = default;
    explicit StepSequence(std::coroutine_handle<promise_type> handle) : handle_(handle) {}
    StepSequence(StepSequence&& other) noexcept : handle_(std::exchange(other.handle_, {})) {}
    StepSequence& operator=(StepSequence&& other) noexcept {
        if (this != &other) {
            if (handle_) handle_.destroy();
            handle_ = std::exchange(other.handle_, {});
        }
        return *this;
    }
    ~StepSequence() { if (handle_) handle_.destroy(); }
    StepResult<T> next() {
        if (!handle_ || handle_.done()) return {};
        auto& promise = handle_.promise();
        promise.step = {};
        handle_.resume();
        if (promise.error) std::rethrow_exception(promise.error);
        return std::move(promise.step);
    }

  private:
    std::coroutine_handle<promise_type> handle_;
};

// Offline callers consume the same model steps synchronously.
template <class T> T finish_steps(StepSequence<T> sequence) {
    std::optional<T> result;
    while (auto step = sequence.next())
        if (step.workspace) throw std::logic_error("workspace requests require an execution owner");
        else if (step.value) result = std::move(step.value);
    return std::move(result.value());
}

} // namespace tyloi
