#include "diagnostics/experts.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/ffn/routes.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/loader.h"
#include "engine/weights/cache.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/encoding/format_features.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/moe/load.h"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {
struct Work {
    tyloi::cuda::Program& program;
    tyloi::Tensor full, sharded, weight;
    tyloi::DenseTensor x, host_ids, ids, output, reference, candidate, baseline, gate_up;
    MoeRoutePlan route;
    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "expert_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "expert_cleanup error=unknown\n"; }
    }
};
const tyloi::MfeTensor& mfe(const tyloi::Tensor& value) {
    const auto* result = value.representation<tyloi::MfeTensor>();
    if (!result) throw std::runtime_error("expert diagnostic requires an MFE tensor");
    return *result;
}
double mfe_weight_bytes(const tyloi::MfeTensor& weight) {
    double bytes = 0;
    for (const auto& shard : weight.shards) if (shard.weight) bytes += mfe_weight_bytes(*shard.weight);
    if (weight.compact) for (const auto& pool : *weight.compact) {
        const auto& w = pool.weight;
        for (const auto* value : {&w.q_packed, &w.row_q_bits, &w.row_q_bit_offsets,
                                  &w.sub_scale, &w.sub_min, &w.neuron_scale, &w.neuron_min})
            if (value->defined()) bytes += double(value->nbytes());
    }
    if (!weight.pools.empty() || weight.cached_source) bytes += double(tyloi::cuda::logical_bytes(weight));
    return bytes;
}
} // namespace

int run_expert_parallel_moe_check(
        tyloi::cuda::Program& program,
        const std::string & tensor_name,
        int tokens,
        int routes) {
    TYLOINFER_RUNTIME_CHECK(
        tyloi::cuda::moe_parallel_config(program.config()).enabled(),
        "--check-ep-moe requires --expert-parallel or --tensor-parallel");
    TYLOINFER_RUNTIME_CHECK(
        tokens >= 1 && tokens <= 4096,
        "--check-ep-moe-tokens must be in [1, 4096]");
    TYLOINFER_RUNTIME_CHECK(
        routes >= 1,
        "--check-ep-moe-routes must be positive");
    auto& executor = program.executor();
    const auto saved = tyloi::cuda::moe_parallel_config(program.config());
    auto scope = executor.enter(saved.primary_device());
    auto loader = program.loader();
    Work work{program};
    const std::string role =
        tensor_name.find("gate_up") != std::string::npos
        ? "gate_up" : "diagnostic";
    auto full_loader = loader;
    full_loader.placement.device = saved.primary_device();
    full_loader.placement.tensor = {};
    full_loader.placement.experts = {};
    full_loader.placement.experts.devices = {saved.primary_device()};
    work.full = full_loader.experts(tensor_name, false, 0, role);
    work.sharded = loader.experts(tensor_name, false, 0, role);
    const auto& full = mfe(work.full);
    const auto& sharded = mfe(work.sharded);
    auto projector = executor.projector(tyloi::cuda::Executor::current_ffn(), saved.primary_device());
    TYLOINFER_RUNTIME_CHECK(
        !sharded.shards.empty(),
        "expert-parallel MoE diagnostic did not create shards");
    TYLOINFER_RUNTIME_CHECK(
        full.n_experts == sharded.n_experts &&
        full.out_per_expert == sharded.out_per_expert &&
        full.neuron_len == sharded.neuron_len,
        "expert-parallel MoE metadata differs");
    TYLOINFER_RUNTIME_CHECK(
        routes <= full.n_experts,
        "--check-ep-moe-routes exceeds the expert count");

    MfqCudaGuard primary_guard(
        saved.primary_device());
    const int64_t count =
        static_cast<int64_t>(tokens) *
        full.neuron_len;
    auto sequence = tyloi::cuda::arange(
        count,
        tyloi::TensorOptions()
            .device(tyloi::Device(
                tyloi::kCUDA,
                saved.primary_device()))
            .dtype(tyloi::kFloat32));
    work.x = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * tyloi::cuda::sin(sequence * 0.015625))
        .to(tyloi::kFloat16)
        .reshape({tokens, full.neuron_len})
        .contiguous();
    std::vector<int32_t> host_ids(
        static_cast<size_t>(tokens) *
        static_cast<size_t>(routes));
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[
                static_cast<size_t>(token) * routes +
                static_cast<size_t>(route)] =
                (token * routes + route * 3) %
                full.n_experts;
        }
    }
    work.host_ids = tyloi::cuda::from_blob(
        host_ids.data(), {tokens, routes},
        tyloi::TensorOptions().dtype(tyloi::kInt32))
        .clone();
    work.ids = work.host_ids.to(tyloi::Device(
            tyloi::kCUDA,
            saved.primary_device()))
        .contiguous();
    const auto& x = work.x;
    const auto& ids = work.ids;
    work.route = build_moe_route_plan(ids, full.n_experts);
    auto& route = work.route;
    work.reference =
        projector.run(full, x, route)
            .to(tyloi::kFloat32);
    work.output =
        projector.run(sharded, x, route)
            .to(tyloi::kFloat32);
    auto& reference = work.reference;
    auto& test = work.output;
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    auto difference = test - reference;
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        difference.norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.abs().mean().item<double>();
    const double max_abs =
        difference.abs().max().item<double>();
    const int64_t differing =
        test.ne(reference).sum().item<int64_t>();
    std::cout
        << "expert_parallel_moe_check=1"
        << " tensor=" << tensor_name
        << " tokens=" << tokens
        << " routes=" << routes
        << " experts=" << full.n_experts
        << " logical_shape=["
        << full.out_per_expert << ','
        << full.neuron_len << ']'
        << " shards="
        << sharded.shards.size()
        << " differing=" << differing
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    if (!tyloi::cuda::isfinite(test).all().item<bool>() ||
        relative > 1.0e-6) {
        throw std::runtime_error(
            "expert-parallel MoE numerical check failed");
    }
    return 0;
}

int run_mfe_tensor_check(
        tyloi::cuda::Program& program,
        const std::string & tensor_name,
        int tokens,
        int routes,
        int reps,
        int split_width,
        bool routed_input,
        bool benchmark_only) {
    const int max_tokens = benchmark_only ? 131072 : 4096;
    if (tokens < 1 || tokens > max_tokens || routes < 1 || reps < 1 ||
            split_width < 0) {
        throw std::runtime_error("MFE tensor check dimensions are invalid");
    }
    if (benchmark_only && split_width != 0) {
        throw std::runtime_error(
            "MFE benchmark-only mode does not support split checks");
    }
    auto& executor = program.executor();
    const int primary = program.config().layer_placement.primary_device();
    auto scope = executor.enter(primary);
    MfqCudaGuard primary_guard(primary);
    auto loader = program.loader();
    const auto& mfq = loader.source;
    Work work{program};
    work.weight = loader.experts(tensor_name, true, 0, "diagnostic");
    if (program.weights().needs_finalize()) program.weights().finalize();
    const auto& weight = mfe(work.weight);
    auto projector = executor.projector(tyloi::cuda::Executor::current_ffn());
    if (routes > weight.n_experts) {
        throw std::runtime_error("MFE tensor check routes exceed expert count");
    }
    if (split_width > 0 &&
            (split_width >= weight.out_per_expert || (!weight.pools.empty() || weight.cached_source))) {
        throw std::runtime_error(
            "MFE merged/split check requires a resident NINT tensor "
            "and an interior split width");
    }
    const int64_t count =
        (int64_t)tokens * (routed_input ? routes : 1) * weight.neuron_len;
    const auto input_shape = routed_input
        ? std::vector<int64_t>{tokens, routes, weight.neuron_len}
        : std::vector<int64_t>{tokens, weight.neuron_len};
    auto& x = work.x;
    if (benchmark_only) {
        x = tyloi::cuda::zeros(
            input_shape,
            tyloi::TensorOptions()
                .device(tyloi::Device(tyloi::kCUDA, primary))
                .dtype(tyloi::kFloat16));
    } else {
        auto sequence = tyloi::cuda::arange(
            count,
            tyloi::TensorOptions().device(tyloi::Device(tyloi::kCUDA, primary)).dtype(tyloi::kFloat32));
        x = (
            (sequence.remainder(257) - 128.0) / 127.0 +
            0.03125 * tyloi::cuda::sin(sequence * 0.015625))
            .to(tyloi::kFloat16)
            .reshape(input_shape)
            .contiguous();
    }
    std::vector<int32_t> host_ids((size_t)tokens * routes);
    for (int token = 0; token < tokens; ++token) {
        for (int route = 0; route < routes; ++route) {
            host_ids[(size_t)token * routes + route] =
                (token * routes + route * 3) % weight.n_experts;
        }
    }
    work.host_ids = tyloi::cuda::from_blob(
        host_ids.data(), {tokens, routes},
        tyloi::TensorOptions().dtype(tyloi::kInt32))
        .clone();
    work.ids = work.host_ids.to(tyloi::Device(tyloi::kCUDA, primary)).contiguous();
    const auto& ids = work.ids;
    work.route = build_moe_route_plan(ids, weight.n_experts);
    auto& route = work.route;
    auto& output = work.output;
    for (int warmup = 0; warmup < 5; ++warmup) {
        output = projector.run(weight, x, route);
        if (warmup == 0) tyloi::cuda::prefetch_experts(weight, route);
    }
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    tyloi::cuda::Event start(cudaEventDefault), stop(cudaEventDefault);
    auto stream = tyloi::cuda::current_stream().stream();
    start.record(stream);
    for (int index = 0; index < reps; ++index) output = projector.run(weight, x, route);
    stop.record(stream);
    stop.synchronize();
    float elapsed = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), stop.get()));
    if (benchmark_only) {
        std::cout << std::fixed << std::setprecision(9)
                  << "mfe_tensor_benchmark"
                  << " tensor=" << tensor_name
                  << " tokens=" << tokens
                  << " routes=" << routes
                  << " experts=" << weight.n_experts
                  << " out=" << weight.out_per_expert
                  << " k=" << weight.neuron_len
                  << " routed_input=" << (routed_input ? 1 : 0)
                  << " mixed=" << ((!weight.pools.empty() || weight.cached_source) ? 1 : 0)
                  << " weight_bytes=" << mfe_weight_bytes(weight)
                  << " cuda_ms=" << elapsed / reps
                  << '\n';
        return 0;
    }
    double dense_reference_rel = -1.0;
    double dense_reference_mean_abs = -1.0;
    double dense_reference_max_abs = -1.0;
    if (split_width == 0) {
        work.reference = mfe_dense_reference(
            mfq, tensor_name, x, host_ids,
            tokens, routes, routed_input);
        const auto& reference = work.reference;
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto actual_f32 =
            output.reshape({tokens * routes, weight.out_per_expert})
                .to(tyloi::kFloat32);
        auto reference_f32 = reference.to(tyloi::kFloat32);
        auto difference = actual_f32 - reference_f32;
        dense_reference_rel =
            (difference.norm() / reference_f32.norm()).item<double>();
        dense_reference_mean_abs =
            difference.abs().mean().item<double>();
        dense_reference_max_abs =
            difference.abs().max().item<double>();
    }
    auto output_f32 = output.to(tyloi::kFloat32);
    if (!tyloi::cuda::isfinite(output_f32).all().item<bool>()) {
        throw std::runtime_error("MFE tensor check produced a non-finite value");
    }
    const double checksum = output_f32.sum().item<double>();
    const double squared_checksum = output_f32.square().sum().item<double>();
    auto flat = output_f32.cpu().reshape({-1});
    std::cout << std::fixed << std::setprecision(9)
              << "mfe_tensor_check"
              << " tensor=" << tensor_name
              << " tokens=" << tokens
              << " routes=" << routes
              << " experts=" << weight.n_experts
              << " out=" << weight.out_per_expert
              << " k=" << weight.neuron_len
              << " routed_input=" << (routed_input ? 1 : 0)
              << " mixed=" << ((!weight.pools.empty() || weight.cached_source) ? 1 : 0)
              << " weight_bytes=" << mfe_weight_bytes(weight)
              << " cuda_ms=" << elapsed / reps
              << " dense_reference_rel=" << dense_reference_rel
              << " dense_reference_mean_abs=" << dense_reference_mean_abs
              << " dense_reference_max_abs=" << dense_reference_max_abs
              << " checksum=" << checksum
              << " sqsum=" << squared_checksum
              << " values=";
    const int64_t shown = std::min<int64_t>(flat.numel(), 128);
    const float * values = flat.data_ptr<float>();
    for (int64_t index = 0; index < shown; ++index) {
        if (index) std::cout << ",";
        std::cout << values[index];
    }
    std::cout << "\n";

    if (split_width > 0) {
        auto run_segment = [&](int width, int row_offset) {
            return projector.run(weight, x, route)
                .narrow(2, row_offset, width).contiguous();
        };
        auto& merged = work.reference;
        auto& left = work.candidate;
        auto& right = work.baseline;
        for (int warmup = 0; warmup < 3; ++warmup) {
            merged = run_segment(weight.out_per_expert, 0);
            left = run_segment(split_width, 0);
            right = run_segment(
                weight.out_per_expert - split_width, split_width);
        }
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto split = tyloi::cuda::cat({left, right}, 2).contiguous();
        auto difference =
            merged.to(tyloi::kFloat32) - split.to(tyloi::kFloat32);
        const int64_t left_differing =
            merged.slice(2, 0, split_width).ne(left).sum().item<int64_t>();
        const int64_t right_differing =
            merged.slice(2, split_width, weight.out_per_expert)
                .ne(right).sum().item<int64_t>();
        const int64_t differing = left_differing + right_differing;
        const double merged_norm = merged.to(tyloi::kFloat32).norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "mfe_merged_split_check"
                  << " tensor=" << tensor_name
                  << " path=nint_matmul_kernel"
                  << " tokens=" << tokens
                  << " routes=" << routes
                  << " experts=" << weight.n_experts
                  << " bm=" << (tokens <= 128 ? 64 : (tokens <= 512 ? 32 : 64))
                  << " bn=64"
                  << " k=" << weight.neuron_len
                  << " full_width=" << weight.out_per_expert
                  << " split_width=" << split_width
                  << " right_width=" << weight.out_per_expert - split_width
                  << " weight_out_stride=" << weight.out_per_expert
                  << " left_row_offset=0"
                  << " right_row_offset=" << split_width
                  << " equal=" << (merged.equal(split) ? 1 : 0)
                  << " differing=" << differing
                  << " values=" << merged.numel()
                  << " left_differing=" << left_differing
                  << " right_differing=" << right_differing
                  << " rel_l2="
                  << (merged_norm == 0.0
                          ? 0.0
                          : difference.norm().item<double>() / merged_norm)
                  << " mean_abs=" << difference.abs().mean().item<double>()
                  << " max_abs=" << difference.abs().max().item<double>()
                  << "\n";
    }

    const char * warp_ab_env = tyloi::cuda::env_lookup("TYLOINFER_CHECK_NVQ_MOE_WARP_AB");
    if (warp_ab_env != nullptr && std::atoi(warp_ab_env) != 0) {
        const char * original_env = tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_WARPS");
        const bool had_original_env = original_env != nullptr;
        const std::string original_value = had_original_env ? original_env : "";
        const char * original_exact_env =
            tyloi::cuda::env_lookup("TYLOINFER_NVQ_MOE_EXACT_REDUCTION");
        const bool had_original_exact_env = original_exact_env != nullptr;
        const std::string original_exact_value =
            had_original_exact_env ? original_exact_env : "";
        auto set_env = [](const char * name, const char * value) {
            tyloi::cuda::env_set(name, value);
        };
        auto restore_env = [&](const char * name, bool existed, const std::string & value) {
            if (existed) {
                tyloi::cuda::env_set(name, value.c_str());
            } else {
                tyloi::cuda::env_clear(name);
            }
        };
        try {
            set_env("TYLOINFER_NVQ_MOE_WARPS", "0");
            set_env("TYLOINFER_NVQ_MOE_EXACT_REDUCTION", "1");
            work.candidate = projector.run(weight, x, route);
            set_env("TYLOINFER_NVQ_MOE_EXACT_REDUCTION", "0");
            work.baseline = projector.run(weight, x, route);
            cudaStreamSynchronize(::tyloi::cuda::current_stream());
        } catch (...) {
            try { program.drain(); } catch (...) {}
            restore_env("TYLOINFER_NVQ_MOE_WARPS", had_original_env, original_value);
            restore_env("TYLOINFER_NVQ_MOE_EXACT_REDUCTION", had_original_exact_env, original_exact_value);
            throw;
        }
        restore_env(
            "TYLOINFER_NVQ_MOE_WARPS", had_original_env, original_value);
        restore_env(
            "TYLOINFER_NVQ_MOE_EXACT_REDUCTION",
            had_original_exact_env, original_exact_value);
        auto& candidate = work.candidate;
        auto& baseline = work.baseline;
        auto candidate_f32 = candidate.to(tyloi::kFloat32);
        auto baseline_f32 = baseline.to(tyloi::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "nvq_moe_warp_ab"
                  << " candidate_physical_warps=2"
                  << " baseline_warps=" << (weight.neuron_len >= 4096 ? 8 : 4)
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }

    const char * clamped_ab_env =
        tyloi::cuda::env_lookup("TYLOINFER_CHECK_MFE_CLAMPED_SWIGLU_AB");
    if (clamped_ab_env != nullptr && tyloi::cuda::supports_clamped_swiglu(weight)) {
        const double limit = std::atof(clamped_ab_env);
        if (!std::isfinite(limit) || limit <= 0.0) {
            throw std::runtime_error(
                "TYLOINFER_CHECK_MFE_CLAMPED_SWIGLU_AB must be positive");
        }
        const int64_t gate_up_count =
            static_cast<int64_t>(tokens) * routes * 2 * weight.neuron_len;
        auto gate_up_sequence = tyloi::cuda::arange(
            gate_up_count,
            tyloi::TensorOptions().device(tyloi::Device(tyloi::kCUDA, primary)).dtype(tyloi::kFloat32));
        auto gate_up = (
            12.0 * tyloi::cuda::sin(gate_up_sequence * 0.013671875) +
            0.5 * tyloi::cuda::cos(gate_up_sequence * 0.00390625))
            .to(tyloi::kFloat16)
            .reshape({tokens, routes, 2 * weight.neuron_len})
            .contiguous();
        work.gate_up = gate_up;
        auto run_candidate = [&]() {
            return projector.clamped(weight, gate_up, route, limit);
        };
        auto run_baseline = [&]() {
            const int64_t width = weight.neuron_len;
            auto gate = tyloi::cuda::clamp_max(
                gate_up.slice(-1, 0, width).to(tyloi::kFloat32), limit);
            auto up = tyloi::cuda::clamp(
                gate_up.slice(-1, width, 2 * width).to(tyloi::kFloat32),
                -limit, limit);
            auto hidden = (tyloi::cuda::silu(gate) * up)
                .to(tyloi::kFloat16).contiguous();
            return projector.run(weight, hidden, route);
        };
        auto& candidate = work.candidate;
        auto& baseline = work.baseline;
        for (int warmup = 0; warmup < 5; ++warmup) {
            candidate = run_candidate();
            baseline = run_baseline();
        }
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto time_ms = [&](auto && fn) {
            tyloi::cuda::Event begin(cudaEventDefault), end(cudaEventDefault);
            auto cuda_stream = tyloi::cuda::current_stream().stream();
            begin.record(cuda_stream);
            tyloi::DenseTensor value;
            for (int iteration = 0; iteration < reps; ++iteration) {
                value = fn();
            }
            end.record(cuda_stream);
            end.synchronize();
            float elapsed_ms = 0.0f;
            TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms, begin.get(), end.get()));
            return std::pair<double, tyloi::DenseTensor>(
                elapsed_ms / reps, std::move(value));
        };
        auto candidate_timing = time_ms(run_candidate);
        auto baseline_timing = time_ms(run_baseline);
        candidate = std::move(candidate_timing.second);
        baseline = std::move(baseline_timing.second);
        auto candidate_f32 = candidate.to(tyloi::kFloat32);
        auto baseline_f32 = baseline.to(tyloi::kFloat32);
        auto diff = candidate_f32 - baseline_f32;
        const double baseline_norm = baseline_f32.norm().item<double>();
        std::cout << std::scientific << std::setprecision(9)
                  << "mfe_clamped_swiglu_ab"
                  << " limit=" << limit
                  << " candidate_ms=" << candidate_timing.first
                  << " baseline_ms=" << baseline_timing.first
                  << " speedup=" << baseline_timing.first / candidate_timing.first
                  << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                  << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                  << " rel_l2=" << diff.norm().item<double>() / baseline_norm
                  << " mean_abs=" << diff.abs().mean().item<double>()
                  << " max_abs=" << diff.abs().max().item<double>()
                  << "\n";
    }
    program.weights().report(std::cout);
    return 0;
}

#else
int run_expert_parallel_moe_check(tyloi::cuda::Program&, const std::string&, int, int) {
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
}
int run_mfe_tensor_check(tyloi::cuda::Program&, const std::string&, int, int, int, int, bool, bool) {
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
}
#endif
