#include "diagnostics/batch.h"

#include "engine/program.h"
#include "engine/config.h"
#include "engine/engine.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/generation/batch.h"
#include "engine/models/causal_model.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"
#include "engine/requests/output.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
using namespace tyloi::engine;

void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }

struct Row {
    InferenceRequest input;
    InferenceOutput output;
    Generation sequence;
    std::vector<std::int64_t> tokens;
    int prefills = 0;
    bool done = false;
    static InferenceRequest request(const std::vector<std::int64_t>& prompt, const MfqSamplingParams& sampling) {
        InferenceRequest value; value.prompt = prompt; value.sampling = sampling; return value;
    }
    Row(std::string id, const std::vector<std::int64_t>& prompt, const MfqSamplingParams& sampling)
        : input(request(prompt, sampling)), output(input, nullptr, std::move(id)) {}
};

/** Keep every sequence and input alive through drain and numerical retirement. */
struct BatchWork {
    Program& program;
    std::unique_ptr<Program::BatchLease> lease;
    std::map<std::string, std::unique_ptr<Row>> rows;
    ~BatchWork() noexcept {
        try { program.drain(); } catch (...) {}
        rows.clear();
        try { program.drain(); } catch (...) {}
        lease.reset();
    }
    void add(const std::string& id, const std::vector<std::int64_t>& prompt, const MfqSamplingParams& sampling) {
        auto row = std::make_unique<Row>(id, prompt, sampling);
        row->sequence = lease->batch().generate(id, row->input, row->output);
        auto [it, added] = rows.emplace(id, std::move(row));
        check(added, "duplicate diagnostic batch request");
        advance(*it->second);
    }
    void advance(Row& row) {
        if (row.done) return;
        auto step = row.sequence.next();
        if (!step) {
            row.done = true;
            row.sequence = {};
            if (row.output.cleanup_failure) std::rethrow_exception(row.output.cleanup_failure);
            return;
        }
        check(!step.workspace && !step.graph && !step.layer, "unexpected batch resource event");
        if (!step.value) return;
        if (const auto* delta = std::get_if<OutputDelta>(&*step.value))
            row.tokens.insert(row.tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        if (std::holds_alternative<PrefillProgress>(*step.value)) ++row.prefills;
        if (const auto* failed = std::get_if<Failed>(&*step.value)) throw std::runtime_error(failed->message);
    }
    void tick(const std::vector<std::string>& eligible) {
        lease->batch().step(eligible);
        for (const auto& id : eligible) advance(*rows.at(id));
        for (auto& [id, row] : rows)
            if (row->output.result.cancelled && !row->done) advance(*row);
    }
    bool finished(const std::vector<std::string>& ids) const {
        return std::all_of(ids.begin(), ids.end(), [&](const auto& id) { return rows.at(id)->done; });
    }
};

std::vector<std::int64_t> serial(Program& program, const std::vector<std::int64_t>& prompt,
        const MfqSamplingParams& sampling, MfqPromptCachePlan plan = {}) {
    Row row("serial", prompt, sampling);
    row.input.cache_plan = std::move(plan);
    row.sequence = program.generate(row.input, row.output, false, "serial");
    try {
        while (auto step = row.sequence.next()) {
            check(!step.workspace && !step.graph && !step.layer, "unexpected serial resource event");
            if (step.value) if (const auto* delta = std::get_if<OutputDelta>(&*step.value))
                row.tokens.insert(row.tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
        }
        program.drain();
        row.sequence = {};
        if (row.output.cleanup_failure) std::rethrow_exception(row.output.cleanup_failure);
    } catch (...) {
        try { program.drain(); } catch (...) {}
        row.sequence = {};
        try { program.drain(); } catch (...) {}
        throw;
    }
    return row.tokens;
}

struct SchedulerResult { std::vector<std::int64_t> tokens; int completed = 0, cancelled = 0, terminal = 0; };
SchedulerResult collect(const std::shared_ptr<Output>& output) {
    SchedulerResult result;
    do {
        for (const auto& event : output->wait()) {
            if (const auto* delta = std::get_if<OutputDelta>(&event.data))
                result.tokens.insert(result.tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
            if (const auto* failed = std::get_if<Failed>(&event.data)) throw std::runtime_error(failed->message);
            result.completed += std::holds_alternative<Completed>(event.data);
            result.cancelled += std::holds_alternative<Cancelled>(event.data);
            result.terminal += terminal(event.data);
        }
    } while (!output->done());
    return result;
}
std::shared_ptr<Output> submit(Engine& engine, std::string id, const std::vector<std::int64_t>& prompt,
        const MfqSamplingParams& sampling, MfqPromptCachePlan plan = {}) {
    EngineRequest request;
    request.id = std::move(id); request.token_ids = prompt;
    request.input.sampling = sampling; request.input.cache_plan = std::move(plan);
    return engine.submit(std::move(request));
}
} // namespace

int run_continuous_batching_check(CudaLoadOptions options, int requests) {
    check(requests >= 2 && requests <= 32, "batching check supports 2..32 requests");
    if (!options.context_size) options.context_size = 512;
    check(options.context_size >= (requests > 2 ? 273 : 213), "batching check requires a larger context");
    MfqSamplingParams sampling;
    sampling.max_tokens = 20; sampling.temperature = 0; sampling.top_k = 1; sampling.enable_mtp = false;
    std::vector<std::int64_t> first(193), second(17), growth(253);
    std::vector<std::int64_t> first_ref, second_ref, growth_ref;
    std::string backbone;
    {
        auto program = Program::open(options);
        program->load(false);
        auto scope = program->executor().enter(program->config().layer_placement.primary_device());
        auto loader = program->loader();
        const auto* source = dynamic_cast<const ModelSource*>(&loader.source);
        check(source != nullptr, "batch check requires a model source");
        backbone = models::graph(*source).backbone;
        check(backbone == "qwen3_5" || backbone == "qwen4_exp", "batch check requires Qwen3.5 or Qwen4");
        const auto vocab = program->model().info().vocab_size;
        check(vocab > 0, "batch check requires a positive vocabulary");
        for (std::size_t i=0; i<first.size(); ++i) first[i] = (101 + (i*37)%900)%vocab;
        for (std::size_t i=0; i<second.size(); ++i) second[i] = (113 + (i*53)%880)%vocab;
        for (std::size_t i=0; i<growth.size(); ++i) growth[i] = (137 + i*41)%vocab;
        auto config = resolve_config();
        config.generation.prefill_chunk_size = 64;
        config.prefill_chunk_auto = false;
        config.continuous_batch.max_sequences = std::max(4, requests);
        config.workspace_reserve = cuda::options::workspace_reserve_bytes();
        config.continuous_batch.packed_prefill = cuda::options::batch_packed_prefill();
        config.continuous_batch.packed_tokens = cuda::options::batch_packed_prefill_token_budget();
        config.continuous_batch.preallocate = cuda::options::kv_vram_preallocated();
        config.continuous_batch.cpu_offload = !program->config().dsv4_cpu_offload_layers.empty();
        config.continuous_batch.cached_slots = program->config().weights.min_slots;
        program->configure(config);
        first_ref = serial(*program, first, sampling);
        second_ref = serial(*program, second, sampling);
        if (requests > 2) growth_ref = serial(*program, growth, sampling);
        {
            BatchWork work{*program, program->make_batch()};
            work.add("first", first, sampling); work.add("second", second, sampling);
            const std::vector<std::string> pair{"first", "second"};
            int paused = 0;
            for (int ticks=0; ticks<500 && !work.finished(pair); ++ticks) {
                auto& a = *work.rows.at("first"); auto& b = *work.rows.at("second");
                const bool pause = !a.tokens.empty() && paused < 4 && !b.done;
                const auto prior = a.tokens.size();
                work.tick(pause ? std::vector<std::string>{"second"} : pair);
                if (pause) { ++paused; check(a.tokens.size() == prior, "paused row advanced"); }
            }
            for (const auto& [id, ref] : std::vector<std::pair<std::string, std::vector<std::int64_t>>>{{"first",first_ref},{"second",second_ref}}) {
                const auto& actual = work.rows.at(id)->tokens;
                if (actual != ref) {
                    std::size_t i=0; while (i<std::min(actual.size(),ref.size()) && actual[i]==ref[i]) ++i;
                    std::cerr << "continuous_batching_mismatch request=" << id << " index=" << i
                        << " actual_count=" << actual.size() << " reference_count=" << ref.size();
                    if (i<actual.size()) std::cerr << " actual=" << actual[i];
                    if (i<ref.size()) std::cerr << " reference=" << ref[i];
                    std::cerr << '\n';
                }
                check(work.rows.at(id)->done && actual == ref, "batched output differs from serial oracle");
            }
            check(paused == 4, "batch pause was not exercised four times");
            work.add("cancel", second, sampling);
            auto& cancelled = *work.rows.at("cancel");
            for (int ticks=0; ticks<100 && cancelled.tokens.empty(); ++ticks) work.tick({"cancel"});
            check(!cancelled.tokens.empty(), "cancel test never decoded");
            const auto count = cancelled.tokens.size(); cancelled.output.result.cancelled = true;
            work.tick({}); work.tick({});
            check(cancelled.done && cancelled.tokens.size() == count, "cancelled numerical request advanced");
            work.add("survivor", second, sampling);
            auto& survivor = *work.rows.at("survivor");
            for (int ticks=0; ticks<100 && survivor.tokens.empty(); ++ticks) work.tick({"survivor"});
            work.add("partial", first, sampling);
            auto& partial = *work.rows.at("partial");
            for (int ticks=0; ticks<100 && !partial.prefills; ++ticks) work.tick({"partial"});
            check(partial.prefills == 1 && partial.tokens.empty(), "partial prefill did not yield");
            partial.output.result.cancelled = true;
            for (int ticks=0; ticks<500 && (!survivor.done || !partial.done); ++ticks) work.tick({"survivor"});
            check(partial.done && partial.tokens.empty() && survivor.done && survivor.tokens == second_ref,
                "prefill cancellation corrupted a live slot");
            if (requests > 2) {
                std::vector<std::string> ids;
                for (int row=0; row<requests; ++row) {
                    ids.push_back("parallel-" + std::to_string(row));
                    work.add(ids.back(), row%2 ? second : growth, sampling);
                }
                for (const auto& id : ids) {
                    for (int ticks=0; ticks<100 && work.rows.at(id)->tokens.empty(); ++ticks) work.tick({id});
                    check(work.rows.at(id)->tokens.size()==1, "stress request never completed prefill");
                }
                for (int ticks=0; ticks<500 && !work.finished(ids); ++ticks) work.tick(ids);
                check(work.finished(ids), "parallel context growth did not finish");
                for (int row=0; row<requests; ++row)
                    check(work.rows.at(ids[row])->tokens == (row%2 ? second_ref : growth_ref), "parallel context growth differs from serial oracle");
                std::cout << "continuous_batching_growth_check requests=" << requests << " boundary=256 PASS\n";
            }
            if (backbone == "qwen4_exp") {
                std::vector<std::string> ids;
                for (int row=0; row<requests; ++row) {
                    ids.push_back("packed" + std::to_string(row));
                    work.add(ids.back(), row%2 ? second : first, sampling);
                }
                for (int ticks=0; ticks<500 && !work.finished(ids); ++ticks) work.tick(ids);
                check(work.finished(ids), "packed prefill requests did not finish");
                for (int row=0; row<requests; ++row)
                    check(work.rows.at(ids[row])->tokens == (row%2 ? second_ref : first_ref), "packed prefill differs from serial oracle");
            }
            double captures=0, replays=0, packed=0, eligible=0;
            for (const auto& [key, value] : work.lease->batch().metrics()) {
                if (key == "paged_kv_live_pages") check(value==0, "paged KV leaked");
                if (key == "continuous_batching_active" || key == "continuous_batching_prefilling") check(value==0, "batch retained finished requests");
                if (key == "continuous_batching_cuda_graph_captures") captures=value;
                if (key == "continuous_batching_cuda_graph_replays") replays=value;
                if (key == "continuous_batching_packed_prefill_batches") packed=value;
                if (key == "continuous_batching_cuda_graph_eligible") eligible=value;
            }
            if (backbone == "qwen3_5" && eligible) check(captures>0 && replays>captures, "batch graph was not replayed");
            std::cout << "continuous_batching_state_check paged=" << config.continuous_batch.paged_kv
                << " captures=" << captures << " replays=" << replays << " packed_prefills=" << packed
                << " prefill_cancel=1 slot_reuse=1\n";
            program->drain();
        }
        program->configure(config);
        MfqPromptCachePlan plan{"batch-check", first.size()-1};
        const auto fresh = serial(*program, first, sampling, plan);
        const auto restored = serial(*program, first, sampling, plan);
        check(fresh == restored, "session restore differs from serial oracle");
    }
    {
        CudaEngineOptions engine_options;
        static_cast<CudaLoadOptions&>(engine_options) = options;
        engine_options.continuous_batching = std::max(4, requests);
        engine_options.prefill_chunk_size = 64;
        auto engine = load_cuda_engine(engine_options);
        auto serial_first = collect(submit(*engine, "scheduler-serial-first", first, sampling));
        auto serial_second = collect(submit(*engine, "scheduler-serial-second", second, sampling));
        check(serial_first.terminal == 1 && serial_first.completed == 1 &&
            serial_second.terminal == 1 && serial_second.completed == 1, "scheduler serial terminal mismatch");
        auto a = submit(*engine, "scheduler-first", first, sampling);
        auto b = submit(*engine, "scheduler-second", second, sampling);
        auto result_a = collect(a), result_b = collect(b);
        check(result_a.terminal == 1 && result_a.completed == 1 && result_a.tokens == serial_first.tokens &&
            result_b.terminal == 1 && result_b.completed == 1 && result_b.tokens == serial_second.tokens,
            "scheduler batch output or terminal differs from serial");
        auto cancelled = submit(*engine, "scheduler-cancel", first, sampling);
        check(engine->cancel("scheduler-cancel"), "scheduler request finished before cancellation");
        auto cancelled_result = collect(cancelled);
        check(cancelled_result.terminal == 1 && cancelled_result.cancelled == 1,
            "scheduler cancellation terminal mismatch");
        engine->session({SessionCommand::Kind::clear});
        MfqPromptCachePlan plan{"batch-check", first.size()-1};
        auto fresh = collect(submit(*engine, "scheduler-session-fresh", first, sampling, plan));
        auto restored = collect(submit(*engine, "scheduler-session-restored", first, sampling, plan));
        check(fresh.terminal == 1 && fresh.completed == 1 && restored.terminal == 1 && restored.completed == 1 &&
            fresh.tokens == restored.tokens, "scheduler session restore differs from fresh request");
    }
    std::cout << "continuous_batching_check PASS requests=2 paused_ticks=4 cancellation=1 session_reuse=1\n";
    return 0;
}
} // namespace tyloi::cuda::diagnostics
