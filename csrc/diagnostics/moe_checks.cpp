#include "engine/models/common/load.h"
#include "engine/models/source/model_config.h"
#include "engine/models/source/model_graph.h"
#include "engine/weights/moe/experts.h"
#include "engine/backends/cuda/ops/ffn/ffn.h"
#include "env.h"
#include "engine/weights/loader.h"
#include "model_checks.h"

#include "engine/models/deepseek_v4/ops.h"
#include "engine/models/deepseek_v41/ops.h"
#include "engine/models/gemma4/config.h"
#include "engine/models/glm_dsa/ops.h"
#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/backends/cuda/executor/execution.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static double mfe_weight_bytes(const tyloi::cuda::Experts & weight) {
    double bytes = static_cast<double>(weight.mixed_weight_bytes);
    for (const auto & shard :
         weight.expert_parallel_shards) {
        if (shard.weight) {
            bytes += mfe_weight_bytes(
                *shard.weight);
        }
    }
    for (const auto & pool : weight.pools) {
        bytes += static_cast<double>(pool.weight.q_packed.numel()) * pool.weight.q_packed.element_size();
        bytes += static_cast<double>(pool.weight.row_q_bits.numel()) * pool.weight.row_q_bits.element_size();
        bytes += static_cast<double>(pool.weight.row_q_bit_offsets.numel()) * pool.weight.row_q_bit_offsets.element_size();
        bytes += static_cast<double>(pool.weight.sub_scale.numel()) * pool.weight.sub_scale.element_size();
        bytes += static_cast<double>(pool.weight.sub_min.numel()) * pool.weight.sub_min.element_size();
        bytes += static_cast<double>(pool.weight.neuron_scale.numel()) * pool.weight.neuron_scale.element_size();
        bytes += static_cast<double>(pool.weight.neuron_min.numel()) * pool.weight.neuron_min.element_size();
    }
    return bytes;
}

static int run_gemma_moe_check(
        CudaExecutionContext& execution,
        const tyloi::ModelSource & mfq,
        const tyloi::models::gemma4::Config & config,
        int layer,
        const std::vector<int64_t> & token_sizes,
    int reps) {
    const std::string prefix =
        "model.block." + std::to_string(layer) + ".mlp.experts.";
    auto gate_up = load_mfe_gpu(execution,
        mfq, prefix + "gate_up.weight",
        true, layer, "gate_up");
    auto down = load_mfe_gpu(execution,
        mfq, prefix + "down.weight",
        true, layer, "down");
    if (execution.weights.moe &&
            !moe_expert_cache_finalized(execution.weights.moe)) {
        finalize_moe_expert_cache(execution.weights.moe);
    }
    const int routes = static_cast<int>(config.num_experts_per_tok);
    const int experts = static_cast<int>(config.num_experts);
    TYLOINFER_RUNTIME_CHECK(routes > 0 && routes <= 8 && experts > routes,
        "Gemma MoE benchmark requires 1..8 routes and more experts than routes");
    const char * dense_reference_env =
        tyloi::cuda::env_lookup("TYLOINFER_CHECK_GEMMA_MOE_DENSE_REFERENCE");
    const bool dense_reference_enabled =
        dense_reference_env != nullptr && std::atoi(dense_reference_env) != 0;
    tyloi::DenseTensor dense_gate_up;
    tyloi::DenseTensor dense_down;
    if (dense_reference_enabled) {
        dense_gate_up = materialize_mfe_dense(
            mfq, prefix + "gate_up.weight", mfq_current_cuda_device());
        dense_down = materialize_mfe_dense(
            mfq, prefix + "down.weight", mfq_current_cuda_device());
    }
    std::cout << "gemma_moe_bench_config"
              << " layer=" << layer
              << " experts=" << experts
              << " top_k=" << routes
              << " hidden=" << gate_up.neuron_len
              << " intermediate=" << down.neuron_len
              << " gate_up_pools=" << gate_up.pools.size()
              << " down_pools=" << down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << mfe_weight_bytes(gate_up) + mfe_weight_bytes(down)
              << "\n";

    auto stream = mfq_get_current_cuda_stream().stream();
    auto time_ms = [&](auto && fn, int iterations) {
        tyloi::DenseTensor output;
        for (int warmup = 0; warmup < 5; ++warmup) output = fn();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < iterations; ++iteration) output = fn();
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return std::pair<double, tyloi::DenseTensor>(elapsed / iterations, output);
    };

    auto topk_logits = tyloi::cuda::randn(
        {1, experts}, tyloi::TensorOptions().device(tyloi::kCUDA).dtype(tyloi::kFloat32));
    auto selected = moe_topk_cuda(
        topk_logits, routes, false, false, false, true, mfq_nullopt, 1e-20, 1.0);
    auto reference = tyloi::cuda::topk(topk_logits, routes, 1, true, true);
    auto reference_weights = tyloi::cuda::softmax(std::get<0>(reference), 1);
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    auto topk_timing = time_ms([&]() {
        return moe_topk_cuda(
            topk_logits, routes, false, false, false, true,
            mfq_nullopt, 1e-20, 1.0).at(1);
    }, reps);
    std::cout << std::fixed << std::setprecision(6)
              << "gemma_topk_check"
              << " ids_equal="
              << (selected.at(0).equal(std::get<1>(reference).to(tyloi::kInt32)) ? 1 : 0)
              << " weights_max_abs="
              << (selected.at(1) - reference_weights).abs().max().item<float>()
              << " cuda_ms=" << topk_timing.first << "\n";

    tyloi::cuda::manual_seed(20260721 + layer);
    for (int64_t tokens : token_sizes) {
        auto x = tyloi::cuda::randn(
            {tokens, gate_up.neuron_len},
            tyloi::TensorOptions().device(tyloi::kCUDA).dtype(tyloi::kFloat16));
        std::vector<int32_t> host_ids(static_cast<size_t>(tokens) * routes);
        for (int64_t token = 0; token < tokens; ++token) {
            for (int route_index = 0; route_index < routes; ++route_index) {
                host_ids[static_cast<size_t>(token) * routes + route_index] =
                    static_cast<int32_t>(
                        (token * routes + route_index) % experts);
            }
        }
        auto ids = tyloi::cuda::from_blob(
            host_ids.data(), {tokens, routes}, tyloi::TensorOptions().dtype(tyloi::kInt32))
            .clone().to(tyloi::kCUDA).contiguous();
        auto weights = tyloi::cuda::full(
            {tokens, routes}, 1.0 / routes,
            tyloi::TensorOptions().device(tyloi::kCUDA).dtype(tyloi::kFloat32));
        auto forward_materialized = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            const bool projection_bundle_prefetched =
                prefetch_cached_moe_projection_bundle(
                    gate_up, down, route);
            auto gate_pair = execution.projector().run(gate_up, x, route);
            auto hidden = moe_geglu_split_cuda(gate_pair);
            if (!projection_bundle_prefetched) tyloi::cuda::prefetch_experts(down, route);
            auto down_pair = execution.projector().run(down, hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        auto forward_gate_glu = [&]() {
            auto route = build_moe_route_plan(ids, experts);
            const bool projection_bundle_prefetched =
                prefetch_cached_moe_projection_bundle(
                    gate_up, down, route);
            auto hidden = execution.projector().glu(gate_up, x, route, true);
            if (!projection_bundle_prefetched) tyloi::cuda::prefetch_experts(down, route);
            auto down_pair = execution.projector().run(down, hidden, route);
            return moe_weighted_reduce_cuda(down_pair, weights);
        };
        const bool gate_glu_supported = tokens <= 4;
        auto forward = [&]() {
            return gate_glu_supported
                ? forward_gate_glu()
                : forward_materialized();
        };
        auto fused_check = forward();
        auto materialized_check = forward_materialized();
        auto gate_glu_check = gate_glu_supported
            ? forward_gate_glu()
            : fused_check;
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        auto fused_diff = (fused_check - materialized_check).abs().to(tyloi::kFloat32);
        auto fused_time = time_ms(forward, reps);
        auto materialized_time = time_ms(forward_materialized, reps);
        auto gate_glu_time = gate_glu_supported
            ? time_ms(forward_gate_glu, reps)
            : fused_time;
        auto gate_glu_diff = (gate_glu_check - materialized_check).abs().to(tyloi::kFloat32);
        std::cout << std::fixed << std::setprecision(6)
                  << "gemma_moe_geglu_quant_fusion"
                  << " tokens=" << tokens
                  << " equal=" << (fused_check.equal(materialized_check) ? 1 : 0)
                  << " max_abs=" << fused_diff.max().item<float>()
                  << " fused_ms=" << fused_time.first
                  << " materialized_ms=" << materialized_time.first
                  << " speedup=" << materialized_time.first / fused_time.first
                  << " gate_glu_supported=" << (gate_glu_supported ? 1 : 0)
                  << " gate_glu_equal=" << (gate_glu_check.equal(materialized_check) ? 1 : 0)
                  << " gate_glu_max_abs=" << gate_glu_diff.max().item<float>()
                  << " gate_glu_ms=" << gate_glu_time.first << "\n";

        if (tokens == 1) {
            auto stage_route = build_moe_route_plan(ids, experts);
            auto stage_hidden = execution.projector().glu(gate_up, x, stage_route, true);
            auto stage_down = execution.projector().run(down, stage_hidden, stage_route);
            cudaStreamSynchronize(::tyloi::cuda::current_stream());
            auto gate_stage = time_ms(
                [&]() { return execution.projector().glu(gate_up, x, stage_route, true); }, reps);
            auto down_stage = time_ms(
                [&]() { return execution.projector().run(down, stage_hidden, stage_route); }, reps);
            auto reduce_stage = time_ms(
                [&]() { return moe_weighted_reduce_cuda(stage_down, weights); }, reps);
            std::cout << "gemma_moe_stage"
                      << " gate_up_geglu_ms=" << gate_stage.first
                      << " down_ms=" << down_stage.first
                      << " reduce_ms=" << reduce_stage.first << "\n";
        }

        if (tokens != 1) {
            auto& options = execution.config.tasks.projection.moe;
            const bool prefill = options.prefill;
            auto mma = time_ms(forward, reps);
            auto mma_first = mma.second.clone();
            auto mma_repeat = forward().clone();
            cudaStreamSynchronize(::tyloi::cuda::current_stream());
            options.prefill = false;
            decltype(mma) baseline;
            try {
                baseline = time_ms(forward, reps);
                cudaStreamSynchronize(::tyloi::cuda::current_stream());
            } catch (...) {
                options.prefill = prefill;
                throw;
            }
            options.prefill = prefill;
            auto repeat_diff = (mma_repeat - mma_first).abs().to(tyloi::kFloat32);
            auto baseline_diff = (mma_first - baseline.second).abs().to(tyloi::kFloat32);
            std::cout << std::setprecision(6)
                      << "gemma_moe_prefill_ab"
                      << " tokens=" << tokens
                      << " mma_ms=" << mma.first
                      << " baseline_ms=" << baseline.first
                      << " speedup=" << baseline.first / mma.first
                      << " repeat_equal=" << (mma_repeat.equal(mma_first) ? 1 : 0)
                      << " repeat_max_abs=" << repeat_diff.max().item<float>()
                      << " baseline_rel="
                      << ((mma_first - baseline.second).to(tyloi::kFloat32).norm() /
                          baseline.second.to(tyloi::kFloat32).norm()).item<float>()
                      << " baseline_max_abs=" << baseline_diff.max().item<float>()
                      << "\n";
            if (dense_reference_enabled) {
                auto dense_route_forward = [&](const tyloi::DenseTensor & dense,
                                               const tyloi::DenseTensor & input) {
                    auto result = tyloi::cuda::empty(
                        {tokens, routes, dense.size(1)},
                        input.options().dtype(tyloi::kFloat16));
                    for (int route_index = 0; route_index < routes; ++route_index) {
                        const int expert = (route_index * 17) % experts;
                        auto selected = input.dim() == 3
                            ? input.select(1, route_index)
                            : input;
                        result.select(1, route_index).copy_(tyloi::cuda::matmul(
                            selected,
                            dense.index({expert}).transpose(0, 1)));
                    }
                    return result;
                };
                auto dense_gate_pair =
                    dense_route_forward(dense_gate_up, x);
                auto dense_hidden =
                    moe_geglu_split_cuda(dense_gate_pair);
                auto dense_down_pair =
                    dense_route_forward(dense_down, dense_hidden);
                auto dense_output =
                    moe_weighted_reduce_cuda(dense_down_pair, weights);
                auto dense_difference =
                    mma_first.to(tyloi::kFloat32) -
                    dense_output.to(tyloi::kFloat32);
                const double dense_norm =
                    dense_output.to(tyloi::kFloat32).norm().item<double>();
                std::cout << std::scientific << std::setprecision(9)
                          << "gemma_moe_dense_reference"
                          << " tokens=" << tokens
                          << " rel_l2="
                          << dense_difference.norm().item<double>() / dense_norm
                          << " mean_abs="
                          << dense_difference.abs().mean().item<double>()
                          << " max_abs="
                          << dense_difference.abs().max().item<double>()
                          << "\n";
            }
        }
    }
    if (execution.weights.moe) {
        print_moe_expert_cache_stats(execution.weights.moe, std::cout);
    }
    return 0;
}

int run_moe_check(
        CudaExecutionContext& execution,
        const std::string & model_path,
        const std::string & config_path,
        int layer,
        const std::vector<int64_t> & token_sizes,
        int reps) {
    if (layer < 0) throw std::runtime_error("--check-moe-layer must be nonnegative");
    if (reps < 1) throw std::runtime_error("--check-moe-reps must be positive");
    if (token_sizes.empty() || std::any_of(token_sizes.begin(), token_sizes.end(),
            [](int64_t value) { return value < 1 || value > 4096; })) {
        throw std::runtime_error("--check-moe-tokens values must be in [1, 4096]");
    }
    auto model_source = tyloi::open_model_source(model_path);
    const auto& mfq = *model_source;
    tyloi::cuda::validate_model_source(mfq);
    const auto graph = tyloi::models::graph(mfq);
    const auto payload = tyloi::models::config_json(mfq, config_path);
    tyloi::cuda::Ffn ffn;
    int64_t hidden_size = 0;
    if (graph.backbone == "gemma4") {
        const auto config = tyloi::models::gemma4::Config::from_json(payload);
        if (layer >= config.num_hidden_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        return run_gemma_moe_check(
            execution, mfq, config, layer, token_sizes, reps);
    }
    if (graph.backbone == "glm_dsa") {
        const auto config = tyloi::models::glm_dsa::Config::from_json(payload);
        if (layer >= config.num_hidden_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        tyloi::cuda::glm_dsa::load_ffn(
            execution, mfq, config, layer, ffn);
        hidden_size = config.hidden_size;
    } else if (graph.backbone == "deepseek_v4") {
        const auto config =
            tyloi::models::deepseek_v4::Config::from_json(payload);
        if (layer >= config.num_hidden_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        auto block = tyloi::cuda::deepseek_v4::load_block(
            execution, mfq, config, layer, "deepseek_v4",
            std::make_shared<Dsv4SharedState>());
        ffn = std::move(static_cast<Dsv4Block&>(*block).ffn);
        hidden_size = config.hidden_size;
    } else if (graph.backbone == "deepseek_v41") {
        const auto config =
            tyloi::models::deepseek_v41::Config::from_json(payload);
        if (layer >= config.n_layers) {
            throw std::runtime_error(
                "MoE benchmark layer is out of range");
        }
        ffn = tyloi::cuda::deepseek_v41_runtime::load_moe_at(
            mfq, config,
            "model.block." + std::to_string(layer) + ".mlp.",
            layer, config.top_k);
        hidden_size = config.hidden;
    } else {
        throw std::runtime_error(
            "selected CUDA backbone has no FFN MoE benchmark adapter");
    }
    if (!ffn.is_moe) {
        throw std::runtime_error(
            "selected layer does not contain MFE MoE weights");
    }
    if (execution.weights.moe &&
            !moe_expert_cache_finalized(execution.weights.moe)) {
        finalize_moe_expert_cache(execution.weights.moe);
    }
    const double routed_weight_bytes =
        mfe_weight_bytes(ffn.moe_gate_up) + mfe_weight_bytes(ffn.moe_down);
    std::cout << "moe_bench_config"
              << " layer=" << layer
              << " experts=" << ffn.moe_gate_up.n_experts
              << " top_k=" << ffn.moe_top_k
              << " hidden=" << ffn.moe_gate_up.neuron_len
              << " intermediate=" << ffn.moe_down.neuron_len
              << " gate_up_pools=" << ffn.moe_gate_up.pools.size()
              << " down_pools=" << ffn.moe_down.pools.size()
              << " routed_weight_bytes=" << std::fixed << std::setprecision(0)
              << routed_weight_bytes << "\n";

    tyloi::cuda::manual_seed(20260720 + layer);
    tyloi::DenseTensor output;
    for (int64_t tokens : token_sizes) {
        auto x = tyloi::cuda::randn(
            {tokens, hidden_size},
            tyloi::TensorOptions().device(tyloi::kCUDA).dtype(tyloi::kFloat16));
        for (int warmup = 0; warmup < 10; ++warmup) output = execution.ffn().run(ffn, x);
        cudaStreamSynchronize(::tyloi::cuda::current_stream());

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        auto stream = mfq_get_current_cuda_stream().stream();
        auto wall_start = std::chrono::steady_clock::now();
        cudaEventRecord(start, stream);
        for (int iteration = 0; iteration < reps; ++iteration) output = execution.ffn().run(ffn, x);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        auto wall_stop = std::chrono::steady_clock::now();
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        const double cuda_ms = static_cast<double>(elapsed_ms) / reps;
        const double wall_ms =
            std::chrono::duration<double, std::milli>(wall_stop - wall_start).count() / reps;
        const double checksum = output.to(tyloi::kFloat32).sum().item<double>();
        if (!std::isfinite(checksum)) throw std::runtime_error("non-finite MoE benchmark output");
        std::cout << std::setprecision(6)
                  << "moe_bench_result"
                  << " layer=" << layer
                  << " tokens=" << tokens
                  << " reps=" << reps
                  << " cuda_ms=" << cuda_ms
                  << " wall_ms=" << wall_ms
                  << " tokens_per_second=" << (1000.0 * tokens / cuda_ms)
                  << " checksum=" << checksum << "\n";

        const char * prefill_ab_env = tyloi::cuda::env_lookup("TYLOINFER_CHECK_MOE_PREFILL_MMA_AB");
        if (prefill_ab_env != nullptr && std::atoi(prefill_ab_env) != 0 && tokens >= 9) {
            auto candidate = output.clone();
            auto& options = execution.config.tasks.projection.moe;
            const bool prefill = options.prefill;
            options.prefill = false;
            tyloi::DenseTensor baseline;
            try {
                baseline = execution.ffn().run(ffn, x);
                cudaStreamSynchronize(::tyloi::cuda::current_stream());
            } catch (...) {
                options.prefill = prefill;
                throw;
            }
            options.prefill = prefill;
            auto diff = (candidate - baseline).to(tyloi::kFloat32);
            const double baseline_norm = baseline.to(tyloi::kFloat32).norm().item<double>();
            std::cout << "moe_prefill_mma_ab"
                      << " tokens=" << tokens
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " rel_l2=" << (diff.norm().item<double>() / baseline_norm)
                      << " mean_abs=" << diff.abs().mean().item<float>()
                      << " max_abs=" << diff.abs().max().item<float>()
                      << "\n";
        }

        const char * exact_env = tyloi::cuda::env_lookup("TYLOINFER_CHECK_MOE_POOL_EXACT");
        if (exact_env != nullptr && std::atoi(exact_env) != 0 && tokens <= 8) {
            auto candidate = output;
            auto& options = execution.config.tasks.projection.moe;
            const auto saved_options = options;
            options.nvq_decode = false;
            options.nint_decode = false;
            execution.force_moe_unfused_reduce = true;
            execution.force_moe_materialized_swiglu = true;
            tyloi::DenseTensor baseline;
            try {
                baseline = execution.ffn().run(ffn, x);
                cudaStreamSynchronize(::tyloi::cuda::current_stream());
            } catch (...) {
                options = saved_options;
                execution.force_moe_unfused_reduce = false;
                execution.force_moe_materialized_swiglu = false;
                throw;
            }
            options = saved_options;
            execution.force_moe_unfused_reduce = false;
            execution.force_moe_materialized_swiglu = false;
            auto diff = (candidate - baseline).abs().to(tyloi::kFloat32);
            std::cout << "moe_exact_result"
                      << " equal=" << (candidate.equal(baseline) ? 1 : 0)
                      << " differing=" << candidate.ne(baseline).sum().item<int64_t>()
                      << " max_abs=" << diff.max().item<float>()
                      << "\n";
        }

        execution.profiler.reset();
        execution.profiler.enabled = true;
        const int profile_reps = std::min(reps, 10);
        for (int iteration = 0; iteration < profile_reps; ++iteration) output = execution.ffn().run(ffn, x);
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        execution.profiler.report(
            "moe_layer" + std::to_string(layer) + "_m" +
            std::to_string(tokens));
        execution.profiler.enabled = false;
        execution.profiler.reset();
    }
    if (execution.weights.moe) {
        print_moe_expert_cache_stats(execution.weights.moe, std::cout);
    }
    return 0;
}
