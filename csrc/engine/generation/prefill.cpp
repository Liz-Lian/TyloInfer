#include "engine/generation/prefill.h"

#include "engine/budget.h"
#include "engine/executor.h"
#include "engine/generation/prefill_budget.h"
#include "engine/models/causal_model.h"
#include "engine/requests/output.h"
#include "engine/weights/cache.h"
#include "engine/weights/workspace.h"

#include <array>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {

models::Prefill prefill(models::CausalModel& model, models::State& state, Executor& executor,
    weights::Cache& weights, Budget& budget, const models::ForwardOptions& options,
    models::Input input, std::int64_t compute, InferenceOutput& output) {
    std::unique_ptr<weights::Loan> loan;
    tyloi::StepSequence<models::Prefill> job;
    std::optional<weights::Cache::Priority> priority;
    models::Prefill result;
    bool observed = false;
    bool computed = false;
    auto timer = executor.timer();
    const auto drain = [&] {
        std::exception_ptr failure;
        try { weights.drain(); } catch (...) { failure = std::current_exception(); }
        try { executor.drain(); } catch (...) { if (!failure) failure = std::current_exception(); }
        if (failure) std::rethrow_exception(failure);
    };
    try {
        for (const auto& [device, bytes] : state.reset_growth(input.ids.size(0)))
            if (bytes) budget.reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        for (const auto& [device, bytes] : state.growth(input.ids.size(1), options, input.confirmed))
            if (bytes) budget.reserve_kv(device, prefill_budget_add(bytes, std::size_t(64) << 20));
        const auto tokens = input.ids.size(1);
        const bool transformed = bool(input.embeddings) || bool(input.positions) || bool(input.lengths) ||
            bool(input.cache_positions) || bool(input.image_mask) || input.suffix || input.confirmed != 0;
        const auto mode = weights.prefill_mode(tokens, state.position() > 0);
        models::PrefillOptions policy{tokens, compute, 0, 0, transformed,
            mode == ::tyloi::weights::Cache::Prefill::retained,
            weights.decode() == ::tyloi::weights::Cache::Decode::routed,
            weights.transfer() == ::tyloi::weights::Cache::Transfer::coordinated,
            weights.split_prefill(tokens)};
        job = model.prefill(input, state, options, policy);
        while (!output.stopped()) {
            auto step = job.next();
            if (step.workspace) {
                if (step.state != StepState::advanced || step.value)
                    throw std::logic_error("invalid prefill workspace event");
                observed = true;
                const auto request = *step.workspace;
                if (request.bytes) {
                    if (loan || request.output || request.extra)
                        throw std::logic_error("invalid prefill workspace acquisition");
                    loan = std::make_unique<weights::Loan>(weights);
                    if (request.tile) {
                        const auto tile = *request.tile;
                        if (!tile.compute || *tile.compute <= 0 || tile.tokens <= 0 || !tile.row_bytes)
                            throw std::invalid_argument("invalid prefill workspace geometry");
                        const auto memory = [&] {
                            const std::array<Device, 1> devices{executor.device()};
                            const auto snapshot = executor.memory(devices).at(0);
                            return prefill_budget_add(snapshot.usable, snapshot.workspace);
                        };
                        const auto available = weights.prefill_budget(true);
                        const auto fixed = prefill_budget_add(request.bytes, available.transfer_bytes);
                        const auto before = prefill_budget_add(memory(), available.lendable_bytes);
                        auto selected = select_prefill_chunk(tile.tokens, *tile.compute, before, fixed, tile.row_bytes);
                        if (selected <= 0) throw std::runtime_error("insufficient CUDA prefill workspace before expert loan");
                        const auto rows = std::size_t(selected) > std::numeric_limits<std::size_t>::max() / tile.row_bytes
                            ? std::numeric_limits<std::size_t>::max() : std::size_t(selected) * tile.row_bytes;
                        loan->begin(prefill_budget_add(fixed, rows));
                        const auto after = memory();
                        const auto after_fixed = prefill_budget_add(request.bytes, loan->transfer_bytes());
                        const auto requested = selected;
                        selected = select_prefill_chunk(tile.tokens, selected, after, after_fixed, tile.row_bytes);
                        if (selected <= 0) throw std::runtime_error("insufficient CUDA prefill workspace after expert loan");
                        if (selected != requested)
                            std::cerr << "runtime_prefill_chunk_budget requested=" << requested
                                      << " compute=" << selected << " available_bytes=" << after << '\n';
                        *tile.compute = selected;
                    } else loan->begin(request.bytes);
                } else {
                    if (!loan || !request.output || request.tile)
                        throw std::logic_error("invalid prefill workspace completion");
                    std::array<Tensor*, 2> outputs{request.output, request.extra};
                    loan->finish(std::span<Tensor* const>{outputs.data(), request.extra ? 2U : 1U});
                    loan.reset();
                }
                continue;
            }
            if (!step) {
                if (loan) throw std::logic_error("prefill job ended while workspace is borrowed");
                if (observed) throw std::runtime_error("prefill job ended without a result");
                auto* raw = input.raw_hidden;
                result.hidden = model.forward(std::move(input), state, options);
                if (raw) result.raw = *raw;
                return result;
            }
            observed = true;
            if (!step.value) continue;
            result = std::move(*step.value);
            if (result.phase == models::Prefill::Phase::computed) {
                timer->stop();
                computed = true;
            } else if (result.phase == models::Prefill::Phase::encode) {
                if (result.first < result.last) {
                    if (priority) { priority->finish(); priority.reset(); }
                    priority.emplace(weights, result.first, result.last);
                }
            } else {
                if (priority) { priority->finish(); priority.reset(); }
                if (result.phase == models::Prefill::Phase::complete) {
                    if (loan) throw std::logic_error("prefill completed while workspace is borrowed");
                    if (computed) result.elapsed_ms = timer->elapsed_ms();
                    return result;
                }
            }
        }
        drain();
        job = {};
        loan.reset();
        return {};
    } catch (...) {
        auto failure = std::current_exception();
        try { drain(); } catch (...) {}
        job = {};
        loan.reset();
        std::rethrow_exception(failure);
    }
}

} // namespace tyloi::engine
