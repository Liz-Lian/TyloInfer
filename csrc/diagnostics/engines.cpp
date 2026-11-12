#include "diagnostics/engines.h"
#include "engine/engine.h"

#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <latch>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
/** Submit fixed token IDs and collect output until the real terminal event. */
std::vector<std::int64_t> tokens(tyloi::engine::Engine& engine,
    const std::vector<std::int64_t>& prompt, const MfqSamplingParams& sampling) {
    tyloi::engine::EngineRequest request;
    request.id = "check";
    request.token_ids = prompt;
    request.input.sampling = sampling;
    auto output = engine.submit(std::move(request));
    std::vector<std::int64_t> result;
    bool done = false;
    while (!done) {
        for (const auto& event : output->wait()) {
            if (const auto* delta = std::get_if<tyloi::engine::OutputDelta>(&event.data))
                result.insert(result.end(), delta->token_ids.begin(), delta->token_ids.end());
            if (const auto* failed = std::get_if<tyloi::engine::Failed>(&event.data))
                throw std::runtime_error("engine isolation request failed: " + failed->message);
            if (std::holds_alternative<tyloi::engine::Cancelled>(event.data))
                throw std::runtime_error("engine isolation request cancelled");
            if (std::holds_alternative<tyloi::engine::Completed>(event.data)) done = true;
        }
    }
    return result;
}
}

int run_cuda_engine_isolation_check(CudaEngineOptions options) {
    if (options.model_path.empty()) throw std::runtime_error("engine isolation check requires --model");
    if (options.context_size == 0) options.context_size = 64;
    options.continuous_batching = 0;

    std::unique_ptr<tyloi::engine::Engine> first, second;
    std::exception_ptr first_error, second_error;
    {
        std::latch ready(2);
        std::latch start(1);
        std::atomic<bool> cancelled{false};
        auto load = [&](std::unique_ptr<tyloi::engine::Engine>& engine, std::exception_ptr& error) {
            ready.count_down();
            start.wait();
            if (cancelled.load()) return;
            try { engine = load_cuda_engine(options); }
            catch (...) { error = std::current_exception(); }
        };
        std::jthread first_worker, second_worker;
        try {
            first_worker = std::jthread(load, std::ref(first), std::ref(first_error));
            second_worker = std::jthread(load, std::ref(second), std::ref(second_error));
        } catch (...) {
            cancelled.store(true);
            start.count_down();
            throw;
        }
        ready.wait();
        start.count_down();
        first_worker.join();
        second_worker.join();
    }
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);

    MfqSamplingParams sampling;
    sampling.max_tokens = 2;
    sampling.temperature = 0.0;
    sampling.top_k = 1;
    sampling.top_p = 1.0;
    sampling.enable_mtp = false;
    const std::vector<std::int64_t> first_prompt{101, 138, 175, 212, 249};
    const std::vector<std::int64_t> second_prompt{113, 166, 219, 272, 325, 378};
    const auto first_reference = tokens(*first, first_prompt, sampling);
    if (first_reference.size() != static_cast<std::size_t>(sampling.max_tokens))
        throw std::runtime_error("CUDA engine isolation oracle failed");
    const auto second_reference = tokens(*second, second_prompt, sampling);
    if (second_reference.size() != static_cast<std::size_t>(sampling.max_tokens))
        throw std::runtime_error("CUDA engine isolation oracle failed");

    std::vector<std::int64_t> first_output, second_output;
    first_error = nullptr;
    second_error = nullptr;
    {
        std::latch ready(2);
        std::latch start(1);
        std::atomic<bool> cancelled{false};
        auto generate = [&](tyloi::engine::Engine& engine, const std::vector<std::int64_t>& prompt,
                            std::vector<std::int64_t>& result, std::exception_ptr& error) {
            ready.count_down();
            start.wait();
            if (cancelled.load()) return;
            try { result = tokens(engine, prompt, sampling); }
            catch (...) { error = std::current_exception(); }
        };
        std::jthread first_worker, second_worker;
        try {
            first_worker = std::jthread(generate, std::ref(*first), std::cref(first_prompt),
                std::ref(first_output), std::ref(first_error));
            second_worker = std::jthread(generate, std::ref(*second), std::cref(second_prompt),
                std::ref(second_output), std::ref(second_error));
        } catch (...) {
            cancelled.store(true);
            start.count_down();
            throw;
        }
        ready.wait();
        start.count_down();
        first_worker.join();
        second_worker.join();
    }
    if (first_error) std::rethrow_exception(first_error);
    if (second_error) std::rethrow_exception(second_error);

    if (first_output.size() != static_cast<std::size_t>(sampling.max_tokens) ||
        second_output.size() != static_cast<std::size_t>(sampling.max_tokens) ||
        first_output != first_reference || second_output != second_reference ||
        first->info().model_type != second->info().model_type ||
        first->info().max_context != options.context_size || second->info().max_context != options.context_size ||
        std::get<tyloi::engine::Metrics>(first->control(tyloi::engine::GetMetrics{})).empty() ||
        std::get<tyloi::engine::Metrics>(second->control(tyloi::engine::GetMetrics{})).empty())
        throw std::runtime_error("concurrent CUDA engines did not remain isolated");
    std::cout << "cuda_engine_isolation_check PASS engines=2 generated="
              << first_output.size() + second_output.size() << " context=" << options.context_size << '\n';
    return 0;
}
} // namespace tyloi::cuda::diagnostics
