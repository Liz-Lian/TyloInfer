#include "diagnostics/gemma_moe.h"
#include "engine/program.h"
#include "engine/models/gemma4/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/ffn/routes.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/native/env.h"
#include "engine/backends/cuda/native/stream.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/weights/cache.h"
#include "engine/weights/loader.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/moe/load.h"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {
using tyloi::DenseTensor;
using tyloi::MfeTensor;

const MfeTensor& expert(const tyloi::Tensor& tensor) {
    const auto* value = tensor.representation<MfeTensor>();
    if (!value) throw std::runtime_error("Gemma MoE benchmark requires MFE experts");
    return *value;
}

enum class Path { materialized, fused, gate_stage, down_stage, reduce_stage, topk };

struct Frame {
    MoeRoutePlan route;
    DenseTensor gate_pair, hidden, down_pair, output;
};

struct Work {
    tyloi::cuda::Program& program;
    tyloi::Tensor gate_bank, down_bank;
    DenseTensor dense_gate, dense_down, topk_logits, topk_ids, topk_weights;
    DenseTensor topk_reference_ids, topk_reference_weights;
    DenseTensor input, host_ids, ids, weights, stage_hidden, stage_down;
    DenseTensor fused, materialized, gate_glu, mma_first, mma_repeat, baseline;
    DenseTensor dense_gate_pair, dense_hidden, dense_down_pair, dense_output;
    MoeRoutePlan stage_route;
    int experts = 0, routes = 0;

    ~Work() noexcept {
        try { program.drain(); }
        catch (const std::exception& error) { std::cerr << "gemma_moe_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "gemma_moe_cleanup error=unknown\n"; }
    }

    DenseTensor run(Path path) {
        // Native temporary allocations are released on their allocation stream.
        Frame frame;
        if (path == Path::topk) {
            auto selected = moe_topk_cuda(topk_logits, routes, false, false, false,
                true, mfq_nullopt, 1e-20, 1.0);
            frame.gate_pair = selected.at(0);
            frame.output = selected.at(1);
            return frame.output;
        }
        auto project = program.executor().projector(tyloi::cuda::Executor::current_ffn());
        const auto& gate = expert(gate_bank);
        const auto& down = expert(down_bank);
        if (path == Path::gate_stage) {
            frame.output = project.glu(gate, input, stage_route, true);
        } else if (path == Path::down_stage) {
            frame.output = project.run(down, stage_hidden, stage_route);
        } else if (path == Path::reduce_stage) {
            frame.output = moe_weighted_reduce_cuda(stage_down, weights);
        } else {
            frame.route = build_moe_route_plan(ids, experts);
            const bool prefetched = tyloi::cuda::prefetch_cached_moe_projection_bundle(gate, down, frame.route);
            if (path == Path::fused) {
                frame.hidden = project.glu(gate, input, frame.route, true);
            } else {
                frame.gate_pair = project.run(gate, input, frame.route);
                frame.hidden = moe_geglu_split_cuda(frame.gate_pair);
            }
            if (!prefetched) tyloi::cuda::prefetch_experts(down, frame.route);
            frame.down_pair = project.run(down, frame.hidden, frame.route);
            frame.output = moe_weighted_reduce_cuda(frame.down_pair, weights);
        }
        return frame.output;
    }

    std::pair<double, DenseTensor> time(Path path, int repetitions) {
        DenseTensor output;
        for (int i = 0; i < 5; ++i) output = run(path);
        tyloi::cuda::Event start(cudaEventDefault), stop(cudaEventDefault);
        const auto stream = tyloi::cuda::current_stream().stream();
        start.record(stream);
        for (int i = 0; i < repetitions; ++i) output = run(path);
        stop.record(stream);
        stop.synchronize();
        float elapsed = 0;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), stop.get()));
        return {elapsed / repetitions, output};
    }

    /** Release one completed token-size case before the next case allocates. */
    void clear_case() {
        for (auto* value : {&input, &host_ids, &ids, &weights, &stage_hidden, &stage_down,
                &fused, &materialized, &gate_glu, &mma_first, &mma_repeat, &baseline,
                &dense_gate_pair, &dense_hidden, &dense_down_pair, &dense_output})
            *value = {};
        stage_route = {};
    }
};

DenseTensor dense_routes(Work& work, const DenseTensor& dense, const DenseTensor& input,
                        int64_t tokens) {
    auto result = tyloi::cuda::empty({tokens, work.routes, dense.size(1)},
        input.options().dtype(tyloi::kFloat16));
    // The original dense check uses the same expert for each route across tokens;
    // it therefore does not generally compare the benchmark's actual route IDs.
    for (int route = 0; route < work.routes; ++route) {
        const int expert_id = (route * 17) % work.experts;
        auto selected = input.dim() == 3 ? input.select(1, route) : input;
        auto product = tyloi::cuda::matmul(selected, dense.index({expert_id}).transpose(0, 1));
        result.select(1, route).copy_(product);
    }
    return result;
}
} // namespace

int run_gemma_moe_check(tyloi::cuda::Program& program,
    const tyloi::models::gemma4::Config& config, int layer,
    const std::vector<std::int64_t>& token_sizes, int reps) {
    if (layer < 0 || layer >= config.num_hidden_layers)
        throw std::runtime_error("MoE benchmark layer is out of range");
    if (reps < 1) throw std::runtime_error("--check-moe-reps must be positive");
    if (token_sizes.empty() || std::any_of(token_sizes.begin(), token_sizes.end(),
            [](int64_t value) { return value < 1 || value > 4096; }))
        throw std::runtime_error("--check-moe-tokens values must be in [1, 4096]");
    auto& executor = program.executor();
    const auto device = executor.device();
    auto scope = executor.enter();
    const MfqCudaGuard guard(device.index);
    auto loader = program.loader();
    Work work{program};
    const std::string prefix = "model.block." + std::to_string(layer) + ".mlp.experts.";
    work.gate_bank = loader.experts(prefix + "gate_up.weight", true, layer, "gate_up");
    work.down_bank = loader.experts(prefix + "down.weight", true, layer, "down");
    if (program.weights().needs_finalize()) program.weights().finalize();
    const auto& gate = expert(work.gate_bank);
    const auto& down = expert(work.down_bank);
    work.routes = static_cast<int>(config.num_experts_per_tok);
    work.experts = static_cast<int>(config.num_experts);
    TYLOINFER_RUNTIME_CHECK(work.routes > 0 && work.routes <= 8 && work.experts > work.routes,
        "Gemma MoE benchmark requires 1..8 routes and more experts than routes");
    const char* dense_env = tyloi::cuda::env_lookup("TYLOINFER_CHECK_GEMMA_MOE_DENSE_REFERENCE");
    const bool dense_enabled = dense_env && std::atoi(dense_env) != 0;
    if (dense_enabled) {
        work.dense_gate = materialize_mfe_dense(loader.source, prefix + "gate_up.weight", device.index);
        work.dense_down = materialize_mfe_dense(loader.source, prefix + "down.weight", device.index);
    }
    std::cout << "gemma_moe_bench_config" << " layer=" << layer << " experts=" << work.experts
              << " top_k=" << work.routes << " hidden=" << gate.neuron_len
              << " intermediate=" << down.neuron_len << " gate_up_pools=" << (gate.compact ? gate.compact->size() : 0)
              << " down_pools=" << (down.compact ? down.compact->size() : 0) << " routed_weight_bytes="
              << std::fixed << std::setprecision(0)
              << tyloi::cuda::logical_bytes(gate) + tyloi::cuda::logical_bytes(down) << "\n";

    work.topk_logits = tyloi::cuda::randn({1, work.experts},
        tyloi::TensorOptions().device(device).dtype(tyloi::kFloat32));
    auto selected = moe_topk_cuda(work.topk_logits, work.routes, false, false, false,
        true, mfq_nullopt, 1e-20, 1.0);
    work.topk_ids = selected.at(0); work.topk_weights = selected.at(1);
    auto reference = tyloi::cuda::topk(work.topk_logits, work.routes, 1, true, true);
    work.topk_reference_ids = std::get<1>(reference).to(tyloi::kInt32);
    work.topk_reference_weights = tyloi::cuda::softmax(std::get<0>(reference), 1);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(tyloi::cuda::current_stream().stream()));
    auto topk_timing = work.time(Path::topk, reps);
    std::cout << std::fixed << std::setprecision(6) << "gemma_topk_check"
              << " ids_equal=" << (work.topk_ids.equal(work.topk_reference_ids) ? 1 : 0)
              << " weights_max_abs=" << (work.topk_weights - work.topk_reference_weights).abs().max().item<float>()
              << " cuda_ms=" << topk_timing.first << "\n";

    tyloi::cuda::manual_seed(20260721 + layer);
    for (int64_t tokens : token_sizes) {
        work.input = tyloi::cuda::randn({tokens, gate.neuron_len},
            tyloi::TensorOptions().device(device).dtype(tyloi::kFloat16));
        std::vector<int32_t> ids(static_cast<size_t>(tokens) * work.routes);
        for (int64_t token = 0; token < tokens; ++token)
            for (int route = 0; route < work.routes; ++route)
                ids[static_cast<size_t>(token) * work.routes + route] =
                    static_cast<int32_t>((token * work.routes + route) % work.experts);
        work.host_ids = tyloi::cuda::from_blob(ids.data(), {tokens, work.routes},
            tyloi::TensorOptions().dtype(tyloi::kInt32)).clone();
        work.ids = work.host_ids.to(device).contiguous();
        work.weights = tyloi::cuda::full({tokens, work.routes}, 1.0 / work.routes,
            tyloi::TensorOptions().device(device).dtype(tyloi::kFloat32));
        const bool supported = tokens <= 4;
        const Path forward = supported ? Path::fused : Path::materialized;
        work.fused = work.run(forward);
        work.materialized = work.run(Path::materialized);
        work.gate_glu = supported ? work.run(Path::fused) : work.fused;
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(tyloi::cuda::current_stream().stream()));
        auto fused_diff = (work.fused - work.materialized).abs().to(tyloi::kFloat32);
        auto fused_time = work.time(forward, reps);
        auto materialized_time = work.time(Path::materialized, reps);
        auto glu_time = supported ? work.time(Path::fused, reps) : fused_time;
        auto glu_diff = (work.gate_glu - work.materialized).abs().to(tyloi::kFloat32);
        std::cout << std::fixed << std::setprecision(6) << "gemma_moe_geglu_quant_fusion"
                  << " tokens=" << tokens << " equal=" << (work.fused.equal(work.materialized) ? 1 : 0)
                  << " max_abs=" << fused_diff.max().item<float>() << " fused_ms=" << fused_time.first
                  << " materialized_ms=" << materialized_time.first << " speedup=" << materialized_time.first / fused_time.first
                  << " gate_glu_supported=" << (supported ? 1 : 0) << " gate_glu_equal=" << (work.gate_glu.equal(work.materialized) ? 1 : 0)
                  << " gate_glu_max_abs=" << glu_diff.max().item<float>() << " gate_glu_ms=" << glu_time.first << "\n";
        if (tokens == 1) {
            work.stage_route = build_moe_route_plan(work.ids, work.experts);
            work.stage_hidden = work.run(Path::gate_stage);
            work.stage_down = work.run(Path::down_stage);
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(tyloi::cuda::current_stream().stream()));
            auto gate_stage = work.time(Path::gate_stage, reps);
            auto down_stage = work.time(Path::down_stage, reps);
            auto reduce_stage = work.time(Path::reduce_stage, reps);
            std::cout << "gemma_moe_stage" << " gate_up_geglu_ms=" << gate_stage.first
                      << " down_ms=" << down_stage.first << " reduce_ms=" << reduce_stage.first << "\n";
        }
        if (tokens != 1) {
            auto mma = work.time(forward, reps);
            work.mma_first = mma.second.clone();
            work.mma_repeat = work.run(forward).clone();
            TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(tyloi::cuda::current_stream().stream()));
            auto mode = tyloi::cuda::Executor::current_ffn();
            mode.moe_prefill = false;
            std::pair<double, DenseTensor> baseline;
            {
                auto baseline_scope = executor.enter(mode);
                baseline = work.time(forward, reps);
                work.baseline = baseline.second;
                TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(tyloi::cuda::current_stream().stream()));
            }
            auto repeat_diff = (work.mma_repeat - work.mma_first).abs().to(tyloi::kFloat32);
            auto baseline_diff = (work.mma_first - work.baseline).abs().to(tyloi::kFloat32);
            std::cout << std::setprecision(6) << "gemma_moe_prefill_ab" << " tokens=" << tokens
                      << " mma_ms=" << mma.first << " baseline_ms=" << baseline.first
                      << " speedup=" << baseline.first / mma.first
                      << " repeat_equal=" << (work.mma_repeat.equal(work.mma_first) ? 1 : 0)
                      << " repeat_max_abs=" << repeat_diff.max().item<float>()
                      << " baseline_rel=" << ((work.mma_first - work.baseline).to(tyloi::kFloat32).norm() /
                          work.baseline.to(tyloi::kFloat32).norm()).item<float>()
                      << " baseline_max_abs=" << baseline_diff.max().item<float>() << "\n";
            if (dense_enabled) {
                work.dense_gate_pair = dense_routes(work, work.dense_gate, work.input, tokens);
                work.dense_hidden = moe_geglu_split_cuda(work.dense_gate_pair);
                work.dense_down_pair = dense_routes(work, work.dense_down, work.dense_hidden, tokens);
                work.dense_output = moe_weighted_reduce_cuda(work.dense_down_pair, work.weights);
                auto difference = work.mma_first.to(tyloi::kFloat32) - work.dense_output.to(tyloi::kFloat32);
                const double norm = work.dense_output.to(tyloi::kFloat32).norm().item<double>();
                std::cout << std::scientific << std::setprecision(9) << "gemma_moe_dense_reference"
                          << " tokens=" << tokens << " rel_l2=" << difference.norm().item<double>() / norm
                          << " mean_abs=" << difference.abs().mean().item<double>()
                          << " max_abs=" << difference.abs().max().item<double>() << "\n";
            }
        }
        program.drain();
        work.clear_case();
    }
    program.weights().report(std::cout);
    return 0;
}
#else
int run_gemma_moe_check(tyloi::cuda::Program&, const tyloi::models::gemma4::Config&,
    int, const std::vector<std::int64_t>&, int) {
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
}
#endif
