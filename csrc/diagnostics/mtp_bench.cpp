#include "diagnostics/mtp_bench.h"

#include "engine/program.h"
#include "engine/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/models/causal_model.h"
#include "engine/models/source.h"
#include "engine/models/source/model_graph.h"
#include "engine/ops.h"
#include "engine/requests/output.h"
#include "engine/weights/loader.h"
#include "env.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::diagnostics {
namespace {
using Clock = std::chrono::steady_clock;
namespace ops = ::tyloi::ops;

/** Retain request and tensor owners until outstanding work has drained. */
struct Work {
    Program& program;
    std::vector<Tensor> tensors;
    std::optional<engine::InferenceRequest> request;
    std::optional<engine::InferenceOutput> output;
    StepSequence<engine::EventData> sequence;

    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "mtp_bench_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "mtp_bench_cleanup error=unknown\n"; }
    }
    Tensor keep(Tensor value) {
        tensors.push_back(value);
        return value;
    }
    void retire() {
        program.drain();
        sequence = {};
        output.reset();
        request.reset();
        tensors.clear();
    }
};

double milliseconds(Clock::time_point first, Clock::time_point last) {
    return std::chrono::duration<double, std::milli>(last - first).count();
}
}  // namespace

int run_qwen35_mtp_bench(Program& program, bool enable_mtp,
        int generated_tokens, int repetitions) {
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work work{program};
    auto& model = program.model();
    auto& state = program.state();
    const auto info = model.info();
    auto loader = program.loader();
    const auto* source = dynamic_cast<const ModelSource*>(&loader.source);
    if (!source || models::graph(*source).backbone != "qwen3_5" || !model.mtp())
        throw std::runtime_error("MTP benchmark requires a Qwen3.5 model with MTP");
    if (generated_tokens < 2 || repetitions <= 0 || repetitions > 100 ||
        info.max_context < static_cast<std::int64_t>(generated_tokens) + 17)
        throw std::runtime_error("MTP benchmark requires gen>=2, reps1-100 and context>=gen+17");

    auto config = engine::resolve_config();
    const bool graph_enabled = config.decode_graph.enabled;
    program.configure(std::move(config));
    MfqSamplingParams params;
    params.max_tokens = generated_tokens;
    params.temperature = 0.;
    params.top_k = 1;
    params.top_p = 1.;
    params.enable_mtp = enable_mtp;
    params.seed = 20260907;
    const char* profiler_env = env_lookup("TYLOINFER_CUDA_PROFILER_RANGE");
    const bool profiler_range = profiler_env && std::atoi(profiler_env) != 0;
    std::cout << "mtp_bench config mode=" << (enable_mtp ? "mtp" : "ordinary")
        << " full_window_batching=1 runtime_graph=" << (graph_enabled ? 1 : 0)
        << " gen=" << generated_tokens << " reps=" << repetitions
        << " warmup=1 seed=20260907 synthetic_ids=1 eos_stop=0 session_cache=0\n";
    const auto device = executor.device();
    const auto mode = executor.forward_mode(device.index);
    const models::ForwardOptions options{mode.kv_capacity, mode.trace, mode.trace_layer, device.index};
    for (const auto& prompt : std::vector<std::vector<std::int64_t>>{
            {1, 2, 3}, {100, 200, 300, 400, 500, 600, 700}, std::vector<std::int64_t>(17, 10)}) {
        work.retire();
        state.reset(1);
        auto input = work.keep(ops::contiguous(work.keep(ops::reshape(
            work.keep(ops::tensor(prompt, device)), {1, -1}))));
        std::vector<std::int64_t> reference;
        reference.reserve(generated_tokens);
        for (int step = 0; step < generated_tokens; ++step) {
            models::Input arguments;
            arguments.ids = input;
            arguments.advance = true;
            auto hidden = work.keep(model.forward(std::move(arguments), state, options));
            auto last = work.keep(ops::slice(hidden, 1, hidden.size(1) - 1, hidden.size(1)));
            last = work.keep(ops::contiguous(work.keep(ops::cast(
                work.keep(ops::reshape(last, {hidden.size(0), hidden.size(2)})), kFloat16))));
            auto logits = work.keep(model.logits(last, false));
            auto scores = work.keep(ops::reshape(work.keep(ops::contiguous(logits)), {logits.size(0), -1}));
            auto next = work.keep(ops::sample_greedy(scores));
            reference.push_back(ops::item(next));
            work.tensors.clear(); // item() is the original per-token synchronization point.
            input = work.keep(ops::reshape(next, {1, 1}));
        }
        cudaStreamSynchronize(current_stream().stream());
        for (int repeat = -1; repeat < repetitions; ++repeat) {
            work.retire();
            std::vector<std::int64_t> tokens;
            tokens.reserve(generated_tokens);
            MfqPrefillTiming prefill;
            Clock::time_point first_token;
            cudaStreamSynchronize(current_stream().stream());
            const bool capture = profiler_range && repeat >= 0;
            if (capture) {
                std::cout << "mtp_bench profiler_begin prompt=" << prompt.size() << " repeat=" << repeat << '\n';
                if (cudaProfilerStart() != cudaSuccess) throw std::runtime_error("cudaProfilerStart failed");
            }
            const auto started = Clock::now();
            work.request.emplace();
            work.request->prompt = prompt;
            work.request->sampling = params;
            work.output.emplace(*work.request, nullptr, "bench");
            work.sequence = program.generate(*work.request, *work.output, true, "bench");
            while (auto step = work.sequence.next()) {
                if (!step.value) continue;
                if (auto* progress = std::get_if<engine::PrefillProgress>(&*step.value)) prefill = progress->timing;
                if (auto* delta = std::get_if<engine::OutputDelta>(&*step.value)) {
                    if (tokens.empty()) first_token = Clock::now();
                    tokens.insert(tokens.end(), delta->token_ids.begin(), delta->token_ids.end());
                }
            }
            const int produced = static_cast<int>(tokens.size());
            cudaStreamSynchronize(current_stream().stream());
            const auto finished = Clock::now();
            if (capture) {
                if (cudaProfilerStop() != cudaSuccess) throw std::runtime_error("cudaProfilerStop failed");
                std::cout << "mtp_bench profiler_end prompt=" << prompt.size() << " repeat=" << repeat << '\n';
            }
            const auto stats = program.stats();
            if (produced != generated_tokens || tokens != reference)
                throw std::runtime_error("MTP benchmark output differs from ordinary serial oracle");
            if (enable_mtp && stats.cycles == 0)
                throw std::runtime_error("MTP benchmark did not execute speculative cycles");
            const double total_ms = milliseconds(started, finished);
            const double first_ms = milliseconds(started, first_token);
            const double decode_ms = milliseconds(first_token, finished);
            if (!std::isfinite(total_ms) || total_ms <= 0. || !std::isfinite(decode_ms) || decode_ms <= 0.)
                throw std::runtime_error("invalid benchmark clock interval");
            std::cout << "mtp_bench sample mode=" << (enable_mtp ? "mtp" : "ordinary")
                << " prompt=" << prompt.size() << " gen=" << produced
                << " repeat=" << repeat << " warmup=" << (repeat < 0)
                << " total_ms=" << total_ms << " ttft_ms=" << first_ms
                << " decode_ms=" << decode_ms << " prefill_gpu_ms=" << prefill.llm_ms
                << " total_tps=" << produced * 1000. / total_ms
                << " decode_tps=" << (produced - 1) * 1000. / decode_ms
                << " exact=1 cycles=" << (enable_mtp ? stats.cycles : 0)
                << " accepted=" << (enable_mtp ? stats.accepted : 0)
                << " rejected=" << (enable_mtp ? stats.rejected : 0)
                << " graph_captures=" << stats.captures << " graph_reuses=" << stats.reuses << '\n';
        }
    }
    std::cout << "mtp_bench PASS samples=" << repetitions * 3 << '\n';
    return 0;
}
}  // namespace tyloi::cuda::diagnostics
