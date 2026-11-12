#include "diagnostics/ffn.h"
#include "diagnostics/gemma_moe.h"
#include "engine/program.h"
#include "engine/models/ffn.h"
#include "engine/models/common/load.h"
#include "engine/models/gemma4/config.h"
#include "engine/models/source.h"
#include "engine/models/source/model_config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/cache.h"
#include "engine/weights/loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
struct Work {
    tyloi::cuda::Program& program;
    tyloi::DenseTensor input, output, candidate, baseline;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "moe_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "moe_cleanup error=unknown\n"; }
    }
};
struct ProfileScope {
    CudaProfiler& profiler;
    bool enabled;
    std::unordered_map<std::string, ProfileStat> stats;
    std::vector<std::string> order;
    std::vector<CudaProfiler::PendingEvent> pending;
    explicit ProfileScope(CudaProfiler& value) : profiler(value), enabled(value.enabled),
        stats(std::move(value.stats)), order(std::move(value.order)), pending(std::move(value.pending)) {
        profiler.stats.clear();
        profiler.order.clear();
        profiler.pending.clear();
        profiler.enabled = true;
    }
    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;
    ~ProfileScope() {
        profiler.reset();
        profiler.stats = std::move(stats);
        profiler.order = std::move(order);
        profiler.pending = std::move(pending);
        profiler.enabled = enabled;
    }
};
}

int run_moe_check(tyloi::cuda::Program& program, const std::string& config_path, int layer,
                  const std::vector<std::int64_t>& token_sizes, int reps) {
    if (layer < 0) throw std::runtime_error("--check-moe-layer must be nonnegative");
    if (reps < 1) throw std::runtime_error("--check-moe-reps must be positive");
    if (token_sizes.empty() || std::any_of(token_sizes.begin(), token_sizes.end(),
            [](int64_t value) { return value < 1 || value > 4096; }))
        throw std::runtime_error("--check-moe-tokens values must be in [1, 4096]");
    const auto loader = program.loader();
    const auto& source = dynamic_cast<const tyloi::ModelSource&>(loader.source);
    tyloi::cuda::validate_model_source(source);
    if (tyloi::models::graph(source).backbone == "gemma4") {
        const auto config = tyloi::models::gemma4::Config::from_json(
            tyloi::models::config_json(source, config_path));
        return run_gemma_moe_check(program, config, layer, token_sizes, reps);
    }
    auto& executor = program.executor();
    auto scope = executor.enter();
    Work work{program};
    program.load_ffn(layer);
    auto& ffn = program.ffn();
    const auto info = ffn.info();
    std::cout << "moe_bench_config"
              << " layer=" << layer << " experts=" << info.experts
              << " top_k=" << info.routes << " hidden=" << info.hidden
              << " intermediate=" << info.intermediate << " gate_up_pools=" << info.gate_pools
              << " down_pools=" << info.down_pools << " routed_weight_bytes="
              << std::fixed << std::setprecision(0) << info.routed_bytes << "\n";

    tyloi::cuda::manual_seed(20260720 + layer);
    const auto primary = executor.device();
    for (const int64_t tokens : token_sizes) {
        work.input = tyloi::cuda::randn({tokens, info.input_width},
            tyloi::TensorOptions().device(primary).dtype(tyloi::kFloat16));
        for (int warmup = 0; warmup < 10; ++warmup) work.output = ffn.forward(work.input).dense();
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(tyloi::cuda::current_stream().stream()));
        tyloi::cuda::Event start(cudaEventDefault), stop(cudaEventDefault);
        const auto stream = tyloi::cuda::current_stream().stream();
        const auto wall_start = std::chrono::steady_clock::now();
        start.record(stream);
        for (int iteration = 0; iteration < reps; ++iteration) work.output = ffn.forward(work.input).dense();
        stop.record(stream);
        stop.synchronize();
        const auto wall_stop = std::chrono::steady_clock::now();
        float elapsed_ms = 0;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get()));
        const double cuda_ms = static_cast<double>(elapsed_ms) / reps;
        const double wall_ms = std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / reps;
        const double checksum = work.output.to(tyloi::kFloat32).sum().item<double>();
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite MoE benchmark output");
        std::cout << std::setprecision(6) << "moe_bench_result"
                  << " layer=" << layer << " tokens=" << tokens << " reps=" << reps
                  << " cuda_ms=" << cuda_ms << " wall_ms=" << wall_ms
                  << " tokens_per_second=" << (1000.0 * tokens / cuda_ms)
                  << " checksum=" << checksum << "\n";

        const char* prefill = tyloi::cuda::env_lookup("TYLOINFER_CHECK_MOE_PREFILL_MMA_AB");
        if (prefill && std::atoi(prefill) != 0 && tokens >= 9) {
            work.candidate = work.output.clone();
            auto mode = tyloi::cuda::Executor::current_ffn();
            mode.moe_prefill = false;
            {
                auto baseline_scope = executor.enter(mode);
                work.baseline = ffn.forward(work.input).dense();
                TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            auto diff = (work.candidate - work.baseline).to(tyloi::kFloat32);
            const double baseline_norm = work.baseline.to(tyloi::kFloat32).norm().item<double>();
            std::cout << "moe_prefill_mma_ab" << " tokens=" << tokens
                      << " equal=" << (work.candidate.equal(work.baseline) ? 1 : 0)
                      << " differing=" << work.candidate.ne(work.baseline).sum().item<int64_t>()
                      << " rel_l2=" << (diff.norm().item<double>() / baseline_norm)
                      << " mean_abs=" << diff.abs().mean().item<float>()
                      << " max_abs=" << diff.abs().max().item<float>() << "\n";
            work.candidate = {}; work.baseline = {};
        }

        const char* exact = tyloi::cuda::env_lookup("TYLOINFER_CHECK_MOE_POOL_EXACT");
        if (exact && std::atoi(exact) != 0 && tokens <= 8) {
            work.candidate = work.output;
            auto mode = tyloi::cuda::Executor::current_ffn();
            mode.moe_nvq_decode = false;
            mode.moe_nint_decode = false;
            mode.unfused = true;
            mode.materialized = true;
            {
                auto baseline_scope = executor.enter(mode);
                work.baseline = ffn.forward(work.input).dense();
                TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            auto diff = (work.candidate - work.baseline).abs().to(tyloi::kFloat32);
            std::cout << "moe_exact_result"
                      << " equal=" << (work.candidate.equal(work.baseline) ? 1 : 0)
                      << " differing=" << work.candidate.ne(work.baseline).sum().item<int64_t>()
                      << " max_abs=" << diff.max().item<float>() << "\n";
            work.candidate = {}; work.baseline = {};
        }

        {
            ProfileScope profiling(executor.profiler());
            const int profile_reps = std::min(reps, 10);
            for (int iteration = 0; iteration < profile_reps; ++iteration) work.output = ffn.forward(work.input).dense();
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
            executor.profiler().report("moe_layer" + std::to_string(layer) + "_m" + std::to_string(tokens));
        }
    }
    program.weights().report(std::cout);
    return 0;
}
