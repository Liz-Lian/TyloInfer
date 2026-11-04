#include "ffn.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/profile.h"
#include "diagnostics/route_stats.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/weights/moe/experts.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/executor/moe_ffn_pipeline.h"
#include "engine/backends/cuda/ops/projection/parallel.h"
#include "engine/backends/cuda/ops/ffn/mlp.h"
#include "engine/ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda {

using ActivationKind = tyloi::ops::MlpActivation;

bool uses_expert_cache(const Ffn& f) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!f.is_moe)
        return false;
    if (f.moe_split_gate_up) {
        return f.moe_gate.cached_source || f.moe_up.cached_source || f.moe_down.cached_source;
    }
    return f.moe_gate_up.cached_source || f.moe_down.cached_source;
#else
    return false;
#endif
}

bool expert_parallel(const Ffn& f) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!f.is_moe || f.moe_split_gate_up ||
        f.moe_gate_up.expert_parallel_shards.size() != f.moe_down.expert_parallel_shards.size() ||
        f.moe_gate_up.expert_parallel_shards.empty()) {
        return false;
    }
    for (size_t index = 0; index < f.moe_gate_up.expert_parallel_shards.size(); ++index) {
        const auto &gate = f.moe_gate_up.expert_parallel_shards[index];
        const auto &down = f.moe_down.expert_parallel_shards[index];
        if (!gate.weight || !down.weight || gate.device != down.device ||
            gate.expert_begin != down.expert_begin || gate.expert_end != down.expert_end) {
            return false;
        }
    }
    return true;
#else
    return false;
#endif
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
void Ffn::prefetch(const DenseTensor& x, const Mode& mode, bool enabled,
        MfqOptional<DenseTensor> input_ids) const {
#else
void Ffn::prefetch(const DenseTensor& x, Executor& executor, const Mode& mode, bool enabled,
        MfqOptional<DenseTensor> input_ids) const {
#endif
#if TYLOINFER_ENABLE_MFQ_COMPAT
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    auto& executor = Executor::current();
#endif
    const auto& f = *this;
    const auto project_ = executor.projector(mode);
    auto pipeline=f.moe_pipeline.lock();
    if(!enabled || !pipeline ||
            x.dim()<2 || x.numel()/x.size(-1)>32)return;
    ::tyloi::DenseTensor ids;
    // Cover each row's leading candidates without reading low-confidence
    // tails just because the leading experts are already cached.
    constexpr int64_t candidates_per_row=4;
    if(f.moe_hash_ids.defined()) {
        if(!input_ids.has_value() || !input_ids.value().numel())return;
        ids=f.moe_hash_ids.index_select(0,input_ids.value().reshape({-1}).to(::tyloi::kCUDA,::tyloi::kInt64));
        ids=ids.narrow(1,0,std::min(candidates_per_row,ids.size(1))).to(::tyloi::kInt32).contiguous();
    }else {
        auto row=x.reshape({-1,x.size(-1)}).to(::tyloi::kFloat32);
        auto logits=f.moe_router_projection?project_.run(*f.moe_router_projection, row).contiguous():
            ::tyloi::cuda::matmul(row,f.moe_router.transpose(0,1)).contiguous();
        ids=moe_topk_cuda(logits,std::min({int64_t(f.moe_top_k),logits.size(-1),candidates_per_row}),f.moe_use_sigmoid,
            f.moe_use_sqrt_softplus,f.moe_normalize,f.moe_delayed_softmax,
            f.moe_router_bias.defined()?MfqOptional<::tyloi::DenseTensor>(f.moe_router_bias):mfq_nullopt,
            1e-20,f.moe_router_scale).at(0).to(::tyloi::kInt32).contiguous();
    }
    // Visit each row's highest-ranked candidate before lower ranks, so one
    // token cannot consume the entire speculative read budget for a batch.
    static std::atomic<bool> reported{false};
    if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
        std::fprintf(stderr,"moe_ssd_router_candidates scope=first_forecast rows=%lld candidates_per_row=%lld\n",
            static_cast<long long>(ids.size(0)),static_cast<long long>(ids.size(1)));
    auto task=prefetch_moe_ffn_pipeline(pipeline,ids.transpose(0,1).contiguous().reshape({1,-1}));
    if(auto* window=tyloi::cuda::DecodeWindow::recording())window->enroll(std::move(task));
    else task.serve();
#else
    return;
#endif
}

Mlp dense(const Ffn& f) {
    return {f.gate_up, f.down, f.activation, f.geglu, f.swiglu_limit};
}

namespace {

DenseTensor half(DenseTensor x, CudaProfiler& profiler) {
    if (x.scalar_type() == ::tyloi::kFloat16) return x;
    return profiler.measure("ffn.input_cast", [&] { return x.to(::tyloi::kFloat16); });
}

DenseTensor sharded(const Mlp& mlp, DenseTensor x, Executor& executor, const Projector& project) {
    auto& profiler = executor.profiler();
    const auto& projection = executor.options().projection;
    return profiler.measure("ffn.tensor_parallel", [&] {
        return mlp.sharded(profiler, executor.mmq(), *executor.tq(), executor.collectives(),
                           project.primary_device(), projection.peer_first,
                           projection.reduce_to_primary, projection.fp16_reduce, x);
    });
}

DenseTensor local(const Mlp& mlp, DenseTensor x, Executor& executor, const Projector& project) {
    const auto& options = executor.options();
    const Mlp::Options mlp_options{options.projection.nvq_fusion,
                                   options.dense.mlp.gelu_fusion,
                                   options.dense.mlp.silu_fusion};
    return mlp.forward(project, executor.profiler(), mlp_options, x);
}

bool dense_linear_defined(const QuantLinear& linear) {
    return linear.out() > 0 && linear.neuron_len() > 0;
}

} // namespace

DenseTensor Ffn::forward_experts(DenseTensor x, Executor& executor, const Mode& mode_,
                            const Projector& project_, const MoeRoutePlan& route,
                            DenseTensor route_weights) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto& f = *this;
    auto& stats = executor.routes_;
    auto& collectives_ = executor.collectives_;
    const auto& projection_ = executor.options_.projection;
    const auto& options_ = executor.options_.ffn;
    const size_t shard_count = f.moe_gate_up.expert_parallel_shards.size();
    std::vector<::tyloi::DenseTensor> routed_partials(shard_count);
    std::vector<::tyloi::DenseTensor> down_partials;
    const bool collect_output_energy = stats.energy();
    if (collect_output_energy) {
        down_partials.resize(shard_count);
    }
    for (size_t launch_position = 0; launch_position < shard_count; ++launch_position) {
        const size_t index =
            parallel_index(
            projection_.peer_first, launch_position, shard_count);
        const auto &gate_shard = f.moe_gate_up.expert_parallel_shards[index];
        const auto &down_shard = f.moe_down.expert_parallel_shards[index];
        MfqCudaGuard guard(gate_shard.device);
        auto local_x =
            collectives_.copy(x, gate_shard.device);
        auto local_weights = collectives_.copy(route_weights, gate_shard.device);
        const auto &local_route =
            moe_route_to_device(collectives_, route, gate_shard.device);
        using DenseTensor = ::tyloi::DenseTensor;
        DenseTensor gate_up_pair;
        DenseTensor down_pair;
        const double limit = f.swiglu_limit;
        if (limit <= 0.0 && !mode_.materialized &&
            gate_shard.weight->supports_projection_glu_epilogue()) {
            auto hidden = project_.glu(*gate_shard.weight, local_x, local_route, false);
            down_pair = project_.run(*down_shard.weight, hidden, local_route);
        } else {
            gate_up_pair = project_.run(*gate_shard.weight, local_x, local_route);
            const auto width = gate_up_pair.size(-1) / 2;
            auto parts = std::array<DenseTensor, 2>{gate_up_pair.narrow(-1, 0, width),
                                               gate_up_pair.narrow(-1, width, width)};
            const bool allow_fusion =
                !mode_.materialized && options_.swiglu_fusion &&
                (gate_up_pair.size(0) == 1 ||
                 (gate_up_pair.size(0) <= 4 && options_.small_heterogeneous));
            if (allow_fusion && limit <= 0.0) {
                down_pair = project_.gated(*down_shard.weight, gate_up_pair, local_route, false);
            } else if (allow_fusion && down_shard.weight->supports_clamped_swiglu()) {
                down_pair = project_.clamped(*down_shard.weight, gate_up_pair, local_route, limit);
            } else {
                DenseTensor hidden;
                if (limit <= 0.0) {
                    hidden = moe_swiglu_split_cuda(gate_up_pair);
                } else {
                    auto gate = ::tyloi::cuda::clamp_max(parts[0].to(::tyloi::kFloat32), limit);
                    auto up = ::tyloi::cuda::clamp(parts[1].to(::tyloi::kFloat32), -limit, limit);
                    hidden = (::tyloi::cuda::silu(gate) * up).to(::tyloi::kFloat16).contiguous();
                }
                down_pair = project_.run(*down_shard.weight, hidden, local_route);
            }
        }
        if (collect_output_energy) {
            down_partials[index] = down_pair;
        }
        routed_partials[index] = moe_weighted_reduce_cuda(down_pair, local_weights);
    }
    if (stats.enabled()) {
        ::tyloi::DenseTensor complete_down;
        if (collect_output_energy) {
            complete_down = collectives_.sum(std::move(down_partials), project_.primary_device(),
            projection_.reduce_to_primary, projection_.fp16_reduce);
        }
        stats.record(f.moe_layer, route.ids, route_weights, complete_down,
                     f.moe_gate_up.n_experts);
    }
    return collectives_.sum(std::move(routed_partials), project_.primary_device(),
            projection_.reduce_to_primary, projection_.fp16_reduce);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Ffn::forward_impl(DenseTensor x, Executor& executor, const Mode& mode_,
                         const Projector& project_, MfqOptional<DenseTensor> input_ids,
                         bool allow_important_neurons,
                         MfqOptional<DenseTensor> router_bias) const {
    const auto& f = *this;
    auto &profiler = executor.profiler_;
    auto& stats = executor.routes_;
    auto& mmq_ = executor.mmq_;
    auto& branches_ = executor.branches_;
    const auto& projection_ = executor.options_.projection;
    const auto& dense_ = executor.options_.dense;
    const auto& options_ = executor.options_.ffn;
    auto xh = half(x, profiler);
    if (allow_important_neurons && f.important_neurons) {
        if (f.is_moe) {
            throw std::runtime_error("important-neuron branches are only supported for dense FFNs");
        }
        const int64_t rows = xh.numel() / xh.size(-1);
        const bool use_f32_down = rows >= 16 && mmq_.mode == KlMmqMode::Fp16 &&
                                  dense_.important_fp32;
        auto run_branch = [&](size_t index) {
                const auto &branch = index == 0 ? f : *f.important_neurons;
                if (use_f32_down) {
                    TYLOINFER_RUNTIME_CHECK(!branch.is_moe,
                        "FP32-output IN diagnostic requires a local dense SiLU FFN");
                    return dense(branch).kld(project_, profiler, mmq_, xh);
                }
                return branch.forward_impl(xh, executor, mode_, project_, input_ids, false, mfq_nullopt);
        };
        std::vector<DenseTensor> outputs;
        const bool ran_parallel = !use_f32_down &&
                       tyloi::cuda::decode_branches(
                       projection_.parallel, mode_.serial, rows) &&
                       dense_.important_parallel &&
                       branches_.run(2, run_branch, outputs);
        if (!ran_parallel) {
            outputs.push_back(run_branch(0));
            outputs.push_back(run_branch(1));
        }
        if (outputs.size() != 2)
            throw std::runtime_error("additive FFN requires two branch outputs");
        auto low = std::move(outputs[0]);
        auto high = std::move(outputs[1]);
        if (use_f32_down)
            return profiler.measure("ffn.in.combine.fp32", [&] {
                return (low + high).to(::tyloi::kFloat16).contiguous();
            });
        return profiler.measure("ffn.in.combine", [&] {
            return acc_cuda(low.contiguous(), high.contiguous());
        });
    }
    if (!f.is_moe && dense(f).parallel()) {
        return sharded(dense(f), xh, executor, project_);
    }
    if (f.is_moe) {
        const bool tq_routed=bool(f.moe_tq);
        const int64_t rows = xh.numel() / xh.size(-1);
        if (mode_.batch_serial && uses_expert_cache(f) && rows > 1) {
            // A bounded expert cache cannot safely admit the union of an
            // arbitrary request batch: a miss in that union otherwise
            // falls back to staging the complete projection. Keep the
            // outer request/KV/attention batch intact and execute only
            // each routed FFN row independently, so cache admission is
            // bounded by one token's top-k experts.
            auto flat = xh.reshape({rows, xh.size(-1)}).contiguous();
            std::vector<::tyloi::DenseTensor> outputs;
            outputs.reserve(static_cast<size_t>(rows));
            for (int64_t row = 0; row < rows; ++row) {
                auto row_input = flat.narrow(0, row, 1).contiguous();
                if (input_ids.has_value()) {
                    TYLOINFER_RUNTIME_CHECK(input_ids.value().numel() == rows,
                                      "continuous-batch MoE token ids do not match rows");
                    auto row_ids = input_ids.value().reshape({rows}).narrow(0, row, 1).contiguous();
                    auto bias=router_bias;
                    if(bias.has_value() && bias->dim()==2)bias=bias->narrow(0,row,1);
                    outputs.push_back(forward_impl(row_input, executor, mode_, project_, row_ids, false, bias));
                } else {
                    auto bias=router_bias;
                    if(bias.has_value() && bias->dim()==2)bias=bias->narrow(0,row,1);
                    outputs.push_back(forward_impl(row_input, executor, mode_, project_, mfq_nullopt, false, bias));
                }
            }
            return ::tyloi::cuda::cat(outputs, 0).reshape(xh.sizes());
        }
        if (!f.shared || f.moe_top_k <= 0 || (!f.moe_router.defined() && !f.moe_router_projection) ||
            (!f.moe_shared_ungated && !f.moe_shared_gate.defined())) {
            throw std::runtime_error("incomplete MoE FFN state");
        }
        auto xf = xh.reshape({-1, xh.size(-1)}).contiguous();
        auto xf32 = profiler.measure("moe.input_f32",
                                     [&]() { return xf.to(::tyloi::kFloat32); });
        using DenseTensor = ::tyloi::DenseTensor;
        MoeRoutePlan route;
        DenseTensor shared_gate_logits;
        bool projection_bundle_prefetched = false;
        const bool sharded_experts = expert_parallel(f);
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto pipeline=f.moe_pipeline.lock();
        const bool use_pipeline=bool(pipeline) &&
            mmq_.mode == KlMmqMode::Default && !mode_.materialized;
#else
        const bool use_pipeline=false;
#endif
        struct ExpertOutput {
            DenseTensor values;
            bool reduced;
        };
        // Start the tiny route readback before the shared expert.  The
        // cache consumes it after the shared branch has been enqueued,
        // then launches H2D on its separate stream.  The projection
        // forward below still performs the normal cache check and stream
        // wait, preserving the existing execution semantics.
        auto prefetch_projection_bundle = [&]() {
#if TYLOINFER_ENABLE_MFQ_COMPAT
            if (use_pipeline || !options_.bundle_prefetch || f.cpu_moe_down ||
                mode_.batch_serial) {
                return false;
            }
            if (f.moe_ungated_experts) {
                if (f.cpu_moe_up || f.cpu_moe_down)
                    return false;
                return prefetch_cached_moe_projection_bundle(f.moe_up, f.moe_down, route);
            }
            if (f.moe_split_gate_up) {
                if (f.cpu_moe_gate || f.cpu_moe_up)
                    return false;
                return prefetch_cached_moe_projection_bundle(f.moe_gate, f.moe_up, f.moe_down, route);
            }
            if (f.cpu_moe_gate_up)
                return false;
            return prefetch_cached_moe_projection_bundle(f.moe_gate_up, f.moe_down, route);
        #else
    return false;
#endif
};

        auto shared_gate = [&] {
            if (!shared_gate_logits.defined())
                shared_gate_logits = profiler.measure("moe.shared_gate", [&] {
                    return ::tyloi::cuda::matmul(xf32, f.moe_shared_gate.transpose(0, 1))
                        .contiguous();
                });
            return shared_gate_logits;
        };
        std::vector<::tyloi::DenseTensor> selected;
        {
            auto router_logits = profiler.measure("moe.router", [&]() {
                return f.moe_router_projection?project_.run(*f.moe_router_projection, xf32):
                    ::tyloi::cuda::matmul(xf32, f.moe_router.transpose(0, 1));
            });
            if (f.moe_hash_ids.defined()) {
                if (!input_ids.has_value()) {
                    throw std::runtime_error("hash-routed MoE requires the current token ids");
                }
                selected = profiler.measure("moe.hash_route", [&]() {
                    auto ids = f.moe_hash_ids
                                   .index_select(0, input_ids.value().reshape({-1}).to(
                                                        ::tyloi::kCUDA,
                                                        ::tyloi::kInt64))
                                   .to(::tyloi::kInt32)
                                   .contiguous();
                    auto weights = moe_sqrtsoftplus_weights_cuda(router_logits.contiguous(),
                                                                 ids, 1e-20, f.moe_router_scale);
                    return std::vector<::tyloi::DenseTensor>{ids, weights};
                });
            } else {
                selected = profiler.measure("moe.topk", [&]() {
                    return moe_topk_cuda(
                        router_logits.contiguous(), f.moe_top_k, f.moe_use_sigmoid,
                        f.moe_use_sqrt_softplus, f.moe_normalize, f.moe_delayed_softmax,
                        router_bias.has_value()?router_bias:f.moe_router_bias.defined()
                            ? MfqOptional<::tyloi::DenseTensor>(f.moe_router_bias)
                            : mfq_nullopt,
                        1e-20, f.moe_router_scale);
                });
            }
            if (!use_pipeline && !tq_routed) {
                // Dense (source-precision) banks are not MFE cohorts, so
                // their expert count comes from the bank itself.
                const auto routed_experts =
                    f.moe_dense_up.defined()
                        ? static_cast<int>(f.moe_dense_up.size(0))
                        : (f.moe_ungated_experts
                               ? f.moe_up.n_experts
                               : (f.moe_split_gate_up ? f.moe_gate.n_experts
                                                    : f.moe_gate_up.n_experts));
                route = profiler.measure("moe.route_map", [&]() {
                    return build_moe_route_plan(selected.at(0), routed_experts);
                });
            }
        }
        DenseTensor shared_half;
        {
            if (tyloi::mfq_compat_enabled && !sharded_experts && !use_pipeline && !tq_routed) {
                if (f.moe_split_gate_up) {
                    if (options_.delayed_readback) {
                        tyloi::cuda::prefetch_experts(f.moe_gate, route, true);
                    } else {
                        projection_bundle_prefetched = prefetch_projection_bundle();
                        if (!projection_bundle_prefetched) {
                            tyloi::cuda::prefetch_experts(f.moe_gate, route);
                        }
                    }
                } else {
                    if (options_.delayed_readback) {
                        tyloi::cuda::prefetch_experts(f.moe_gate_up, route, true);
                    } else {
                        projection_bundle_prefetched = prefetch_projection_bundle();
                        if (!projection_bundle_prefetched) {
                            tyloi::cuda::prefetch_experts(f.moe_gate_up, route);
                        }
                    }
                }
            }
            auto shared_output = profiler.measure(
                "moe.shared", [&]() {
                    return f.shared->forward_impl(xf, executor, mode_, project_,
                                                  mfq_nullopt, true, mfq_nullopt);
                });
            if (!f.moe_split_gate_up && !f.moe_shared_ungated) {
                shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                    return ::tyloi::cuda::matmul(xf32, f.moe_shared_gate.transpose(0, 1))
                        .contiguous();
                });
            }
            if (tyloi::mfq_compat_enabled && !sharded_experts && !use_pipeline && !tq_routed && options_.delayed_readback) {
                projection_bundle_prefetched = prefetch_projection_bundle();
                if (!projection_bundle_prefetched) {
                    if (f.moe_split_gate_up) {
                        tyloi::cuda::prefetch_experts(f.moe_gate, route);
                    } else {
                        tyloi::cuda::prefetch_experts(f.moe_gate_up, route);
                    }
                }
            }
            shared_half = shared_output.reshape({xf.size(0), xf.size(1)})
                .contiguous()
                .to(::tyloi::kFloat16);
        }
        ExpertOutput experts{};
        {
            const bool dense_routed =
                f.moe_dense_up.defined() || f.moe_dense_down.defined() ||
                f.moe_dense_gate.defined();
            // Only routed experts run in the latent space; the router and
            // the shared expert keep reading the original hidden state.
            const bool latent_routed =
                !dense_routed && dense_linear_defined(f.moe_latent_in);
            auto routed_input = xf;
            if (latent_routed)
                routed_input = profiler.measure("moe.latent_down", [&] {
                    return project_.run(f.moe_latent_in, xf);
                });
            const auto finish_latent = [&](DenseTensor reduced) {
                if (latent_routed)
                    return profiler.measure("moe.latent_up", [&] {
                        return project_.run(f.moe_latent_out, reduced);
                    });
                return reduced;
            };
            if(tq_routed) {
                TYLOINFER_RUNTIME_CHECK(f.moe_tq->up && f.moe_tq->down,
                    "TQ routed FFN requires matching Up and Down projections");
                if(!f.moe_ungated_experts && f.activation==ActivationKind::silu_gated) {
                    auto reduced=profiler.measure("moe.tq",[&] {
                        return project_.run(*f.moe_tq,routed_input,selected.at(0),selected.at(1),
                            static_cast<float>(f.swiglu_limit));
                    });
                    experts = {finish_latent(std::move(reduced)),true};
                } else {
                    TYLOINFER_RUNTIME_CHECK(f.moe_tq->slices.empty(),
                        "TQ expert parallelism currently requires SiLU-gated experts");
                    DenseTensor gate,up;
                    if(f.moe_ungated_experts) {
                        TYLOINFER_RUNTIME_CHECK(!f.moe_tq->gate,"ungated TQ experts cannot carry a Gate tensor");
                    }else {
                        TYLOINFER_RUNTIME_CHECK(f.moe_tq->gate,"gated TQ experts require a Gate tensor");
                        gate=f.moe_tq->gate->forward(routed_input,selected.at(0));
                    }
                    up=f.moe_tq->up->forward(routed_input,selected.at(0));
                    auto hidden=expert_activation(f.activation,std::move(gate),std::move(up),f.swiglu_limit);
                    auto pairs=f.moe_tq->down->forward(hidden,selected.at(0));
                    auto reduced=moe_weighted_reduce_cuda(pairs,selected.at(1));
                    experts = {finish_latent(std::move(reduced)),true};
                }
            } else if (dense_routed) {
                if (f.moe_ungated_experts &&
                    (f.moe_dense_gate.defined() || f.moe_gate_up.n_experts > 0 ||
                     f.moe_gate.n_experts > 0 || f.moe_up.n_experts > 0 ||
                     f.moe_gate_up.cached_source || f.moe_up.cached_source))
                    throw std::runtime_error(
                        "ungated routed MoE cannot mix a gate with a dense expert bank");
                TYLOINFER_RUNTIME_CHECK(f.moe_dense_up.defined() && f.moe_dense_down.defined(),
                                  "dense routed MoE requires Up and Down banks");
                auto expert_input = xf;
                if (dense_linear_defined(f.moe_latent_in))
                    expert_input = profiler.measure("moe.latent_down", [&] {
                        return project_.run(f.moe_latent_in, xf);
                    });
                auto up = profiler.measure("moe.dense_up", [&] {
                    return dense_routed_projection(f.moe_dense_up, expert_input,
                                                   selected.at(0));
                });
                DenseTensor gate;
                if (!f.moe_ungated_experts) {
                    TYLOINFER_RUNTIME_CHECK(f.moe_dense_gate.defined(),
                                      "gated dense routed MoE requires a Gate bank");
                    gate = profiler.measure("moe.dense_gate", [&] {
                        return dense_routed_projection(f.moe_dense_gate, expert_input,
                                                       selected.at(0));
                    });
                }
                auto hidden = profiler.measure("moe.dense_activation", [&] {
                    return expert_activation(f.activation, std::move(gate),
                                                   std::move(up), f.swiglu_limit);
                });
                auto down_pair = profiler.measure("moe.dense_down", [&] {
                    return dense_routed_projection(f.moe_dense_down, hidden, selected.at(0));
                });
                auto reduced = profiler.measure("moe.dense_reduce", [&] {
                    return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                });
                if (dense_linear_defined(f.moe_latent_out))
                    reduced = profiler.measure("moe.latent_up", [&] {
                        return project_.run(f.moe_latent_out, reduced);
                    });
                stats.record(f.moe_layer, selected.at(0), selected.at(1),
                             down_pair, static_cast<int>(f.moe_dense_up.size(0)));
                experts = {reduced, true};
            }
#if TYLOINFER_ENABLE_MFQ_COMPAT
            else if (f.moe_ungated_experts && !use_pipeline) {
                // relu-squared experts store Up and Down only: one Up matmul,
                // no gate projection, and the shared expert stays outside
                // the routed latent space.
                if (f.moe_gate.n_experts > 0 || f.moe_gate_up.n_experts > 0 ||
                    f.moe_gate.cached_source || f.moe_gate_up.cached_source)
                    throw std::runtime_error(
                        "ungated routed MoE cannot carry a gate projection");
                TYLOINFER_RUNTIME_CHECK(f.moe_up.n_experts > 0,
                                  "ungated routed MoE requires an Up bank");
                if (!projection_bundle_prefetched) {
                    tyloi::cuda::prefetch_experts(f.moe_up, route);
                    tyloi::cuda::prefetch_experts(f.moe_down, route);
                }
                std::optional<tyloi::cuda::Experts> staged_up;
                const tyloi::cuda::Experts *active_up = &f.moe_up;
                if (f.cpu_moe_up) {
                    staged_up.emplace(profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                        return stage_cpu_mixed_moe(f.cpu_moe_up, projection_.moe.heterogeneous);
                    }));
                    active_up = &staged_up.value();
                }
                auto up = profiler.measure("moe.up", [&]() {
                    return project_.run(*active_up, routed_input, route);
                });
                auto hidden = profiler.measure("moe.relu_squared", [&]() {
                    return relu_squared_cuda(up.contiguous());
                });
                staged_up.reset();
                std::optional<tyloi::cuda::Experts> staged_down;
                const tyloi::cuda::Experts *active_down = &f.moe_down;
                if (f.cpu_moe_down) {
                    staged_down.emplace(profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(f.cpu_moe_down, projection_.moe.heterogeneous);
                    }));
                    active_down = &staged_down.value();
                }
                auto down_pair = profiler.measure("moe.down", [&] {
                    return project_.run(*active_down, hidden, route);
                });
                stats.record(f.moe_layer, selected.at(0),
                             selected.at(1), down_pair, f.moe_up.n_experts);
                staged_down.reset();
                if (latent_routed)
                    experts = {finish_latent(profiler.measure("moe.reduce", [&] {
                                return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                            })),
                            true};
                else
                    experts = {down_pair, false};
            }
            else if(use_pipeline) {
                auto result=run_moe_ffn_pipeline(pipeline,project_,routed_input,selected.at(0),
                    selected.at(1),mode_.batch_serial,mode_.tile,mode_.tiles);
                if(result.dim()==3)result=moe_weighted_reduce_cuda(result,selected.at(1));
                experts = {finish_latent(result),true};
            }
            else if (sharded_experts) {
                auto result = profiler.measure("moe.sharded_experts", [&] {
                    return forward_experts(routed_input, executor, mode_, project_, route,
                                           selected.at(1));
                });
                if (latent_routed && result.dim() == 3)
                    result = moe_weighted_reduce_cuda(result, selected.at(1));
                experts = {finish_latent(result), true};
            }
            else {
                std::optional<tyloi::cuda::Experts> staged_gate_up;
                std::optional<tyloi::cuda::Experts> staged_gate;
                std::optional<tyloi::cuda::Experts> staged_up;
                const tyloi::cuda::Experts *active_gate_up = &f.moe_gate_up;
                const tyloi::cuda::Experts *active_gate = &f.moe_gate;
                const tyloi::cuda::Experts *active_up = &f.moe_up;
                ::tyloi::DenseTensor gate_up_pair;
                ::tyloi::DenseTensor projected_hidden;
                if (f.moe_split_gate_up) {
                    if (f.cpu_moe_gate) {
                        staged_gate.emplace(profiler.measure("moe.cpu_offload_gate_h2d", [&]() {
                            return stage_cpu_mixed_moe(f.cpu_moe_gate, projection_.moe.heterogeneous);
                        }));
                        active_gate = &staged_gate.value();
                    }
                    if (f.cpu_moe_up) {
                        staged_up.emplace(profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(f.cpu_moe_up, projection_.moe.heterogeneous);
                        }));
                        active_up = &staged_up.value();
                    }
                    gate_up_pair = profiler.measure("moe.gate_up_split", [&]() {
                        const bool reuse_gate_activation =
                            mode_.reuse_input && mmq_.mode == KlMmqMode::Default && xf.size(0) <= 8 &&
                            active_gate->can_reuse_activation_for(*active_up);
                        auto gate = project_.run(*active_gate, xf, route);
                        if (!projection_bundle_prefetched) {
                            tyloi::cuda::prefetch_experts(*active_up, route);
                        }
                        if (!f.moe_shared_ungated && !shared_gate_logits.defined()) {
                            shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                                return ::tyloi::cuda::matmul(xf32,
                                                                  f.moe_shared_gate.transpose(0, 1))
                                    .contiguous();
                            });
                        }
                        auto up = reuse_gate_activation
                                      ? project_.run(*active_up, xf, route, true)
                                      : project_.run(*active_up, xf, route);
                        return ::tyloi::cuda::cat({gate, up}, -1).contiguous();
                    });
                } else {
                    if (f.cpu_moe_gate_up) {
                        staged_gate_up.emplace(
                            profiler.measure("moe.cpu_offload_gate_up_h2d", [&]() {
                                return stage_cpu_mixed_moe(f.cpu_moe_gate_up, projection_.moe.heterogeneous);
                            }));
                        active_gate_up = &staged_gate_up.value();
                    }
                    const bool fuse_projection_glu =
                        f.swiglu_limit <= 0.0 && !mode_.materialized &&
                        active_gate_up->supports_projection_glu_epilogue();
                    if (fuse_projection_glu) {
                        projected_hidden = profiler.measure("moe.gate_up_swiglu", [&]() {
                            return project_.glu(*active_gate_up, xf, route, false);
                        });
                    } else {
                        gate_up_pair = profiler.measure("moe.gate_up", [&]() {
                            return project_.run(*active_gate_up, xf, route);
                        });
                    }
                }
                staged_gate_up.reset();
                staged_gate.reset();
                staged_up.reset();
                if (!projection_bundle_prefetched) {
                    tyloi::cuda::prefetch_experts(f.moe_down, route);
                }
                std::optional<tyloi::cuda::Experts> staged_down;
                const tyloi::cuda::Experts *active_down = &f.moe_down;
                if (f.cpu_moe_down) {
                    staged_down.emplace(profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(f.cpu_moe_down, projection_.moe.heterogeneous);
                    }));
                    active_down = &staged_down.value();
                }
                DenseTensor down_pair;
                if (projected_hidden.defined()) {
                    down_pair = profiler.measure("moe.down", [&] {
                        return project_.run(*active_down, projected_hidden, route);
                    });
                } else {
                    const double limit = f.swiglu_limit;
                    const bool allow_fusion = !mode_.materialized && options_.swiglu_fusion &&
                        (gate_up_pair.size(0) == 1 ||
                         (gate_up_pair.size(0) <= 4 && options_.small_heterogeneous));
                    if (allow_fusion && limit <= 0.0) {
                        down_pair = profiler.measure("moe.swiglu_down", [&] {
                            return project_.gated(*active_down, gate_up_pair, route, false);
                        });
                    } else if (allow_fusion && active_down->supports_clamped_swiglu()) {
                        down_pair = profiler.measure("moe.swiglu_down", [&] {
                            return project_.clamped(*active_down, gate_up_pair, route, limit);
                        });
                    } else {
                        auto hidden = profiler.measure("moe.swiglu", [&]() {
                            if (limit <= 0.0)
                                return moe_swiglu_split_cuda(gate_up_pair);
                            const int64_t width = gate_up_pair.size(-1) / 2;
                            auto gate = ::tyloi::cuda::clamp_max(
                                gate_up_pair.slice(-1, 0, width).to(::tyloi::kFloat32), limit);
                            auto up = ::tyloi::cuda::clamp(
                                gate_up_pair.slice(-1, width, 2 * width).to(::tyloi::kFloat32),
                                -limit, limit);
                            return (::tyloi::cuda::silu(gate) * up).to(::tyloi::kFloat16).contiguous();
                        });
                        down_pair = profiler.measure("moe.down", [&] {
                            return project_.run(*active_down, hidden, route);
                        });
                    }
                }
                stats.record(f.moe_layer, selected.at(0), selected.at(1), down_pair,
                    f.moe_split_gate_up ? f.moe_gate.n_experts : f.moe_gate_up.n_experts);
                staged_down.reset();
                experts = {down_pair, false};
            }
#else
            else {
                throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
            }
#endif
        }
        if (!experts.reduced && !mode_.unfused && options_.reduce_fusion &&
            !f.moe_shared_ungated && experts.values.size(0) <= 8) {
            auto logits = shared_gate();
            return profiler.measure("moe.reduce_combine", [&] {
                return moe_weighted_reduce_shared_gate_cuda(experts.values, selected.at(1),
                                                            shared_half, logits);
            });
        }
        auto routed = experts.reduced ? std::move(experts.values) : profiler.measure("moe.reduce", [&] {
            return moe_weighted_reduce_cuda(experts.values, selected.at(1));
        });
        if (f.moe_shared_ungated) {
            auto shared = shared_half;
            return profiler.measure("moe.combine", [&] {
                // Dense BF16 expert banks may return a different storage
                // type than the shared branch; align without changing it.
                if (routed.scalar_type() != shared.scalar_type())
                    shared = shared.to(routed.scalar_type()).contiguous();
                return acc_cuda(routed.contiguous(), shared);
            });
        }
        auto logits = shared_gate();
        return profiler.measure("moe.combine", [&] {
            return moe_add_shared_gate_cuda(routed.contiguous(), shared_half, logits);
        });
    }
    TYLOINFER_RUNTIME_CHECK(f.activation != ActivationKind::relu_squared || !f.important_neurons,
                      "relu-squared FFN does not support important neurons");
    return local(dense(f), xh, executor, project_);
}

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
DenseTensor Ffn::forward(DenseTensor x, const Mode& mode, MfqOptional<DenseTensor> input_ids,
                    MfqOptional<DenseTensor> router_bias) const {
    auto& executor = Executor::current();
#else
DenseTensor Ffn::forward(DenseTensor x, Executor& executor, const Mode& mode,
                    MfqOptional<DenseTensor> input_ids,
                    MfqOptional<DenseTensor> router_bias) const {
#endif
    const auto& f = *this;
    if (router_bias.has_value()) {
        const auto& bias = router_bias.value();
        TYLOINFER_RUNTIME_CHECK(f.is_moe && !f.moe_hash_ids.defined() && bias.defined() &&
            ((bias.dim()==1 && bias.size(0)==(f.moe_router_projection?f.moe_router_projection->out():f.moe_router.size(0))) ||
             (bias.dim()==2 && bias.size(0)==x.numel()/x.size(-1) && bias.size(1)==(f.moe_router_projection?f.moe_router_projection->out():f.moe_router.size(0)))),
                          "MoE router correction bias shape mismatch");
    }
    const auto project = executor.projector(mode);
    return forward_impl(std::move(x), executor, mode, project, std::move(input_ids),
                        true, std::move(router_bias));
}

} // namespace tyloi::cuda

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
namespace tyloi::cuda::ops {

tyloi::DenseTensor mlp(tyloi::DenseTensor input, const ::QuantLinearGroup& gate_up,
                const ::QuantLinear& down, tyloi::ops::MlpActivation activation,
                double swiglu_limit) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    const Mlp weights{gate_up, down, activation,
                      activation == tyloi::ops::MlpActivation::gelu_gated, swiglu_limit};
    weights.check(DenseTensor(input), project.primary_device());
    auto x = half(DenseTensor(input), executor.profiler());
    return weights.parallel() ? sharded(weights, x, executor, project)
                              : local(weights, x, executor, project);
}

tyloi::ops::SelectedExperts topk(tyloi::DenseTensor logits, std::int64_t count,
                               const tyloi::ops::MoeRouting& routing) {
    MfqCudaGuard guard(logits.device().index);
    auto result = moe_topk_cuda(DenseTensor(logits).contiguous(), count,
        routing.activation == tyloi::ops::RouterActivation::sigmoid,
        routing.activation == tyloi::ops::RouterActivation::sqrt_softplus,
        routing.normalize, routing.delayed_softmax, mfq_nullopt, 1e-20, routing.scale);
    return {result.at(0), result.at(1)};
}
tyloi::DenseTensor expert_scale(tyloi::DenseTensor weights, tyloi::DenseTensor ids, const tyloi::DenseTensor& scales) {
    MfqCudaGuard guard(weights.device().index);
    return moe_apply_expert_scale_cuda(DenseTensor(weights), DenseTensor(ids), DenseTensor(scales));
}
tyloi::DenseTensor weighted_reduce(tyloi::DenseTensor pairs, tyloi::DenseTensor weights) {
    MfqCudaGuard guard(pairs.device().index);
    return moe_weighted_reduce_cuda(DenseTensor(pairs), DenseTensor(weights));
}
tyloi::ops::RoutedGeglu routed_geglu(tyloi::DenseTensor input, tyloi::DenseTensor ids,
    const Experts& gate_bank, const Experts& down_bank, bool trace) {
    // Expert banks are not projection weights, but the same rule applies: every
    // buffer they keep alive must be resident where the tokens are.
    check_placement(gate_bank.placement(), input.device());
    check_placement(down_bank.placement(), input.device());
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto* gate = &gate_bank;
    const auto* down = &down_bank;
    MfqCudaGuard guard(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    auto& profiler = executor.profiler();
    auto route = profiler.measure("gemma.route_map", [&] {
        return build_moe_route_plan(DenseTensor(ids), gate->n_experts);
    });
    const bool prefetched = prefetch_cached_moe_projection_bundle(*gate, *down, route);
    auto prefetch_down = [&] { if (!prefetched) prefetch_experts(*down, route); };
    tyloi::ops::RoutedGeglu result;
    if (!trace && gate->supports_projection_glu_epilogue()) {
        auto hidden = profiler.measure("gemma.moe_gate_up_geglu", [&] {
            return project.glu(*gate, DenseTensor(input), route, true);
        });
        prefetch_down();
        result.pairs = profiler.measure("gemma.moe_down", [&] {
            return project.run(*down, hidden, route);
        });
        return result;
    }
    auto packed = profiler.measure("gemma.moe_gate_up", [&] {
        return project.run(*gate, DenseTensor(input), route);
    });
    prefetch_down();
    if (trace) result.gate_up = packed.to(::tyloi::kFloat32).contiguous().clone();
    if (!trace && packed.size(0) <= 4) {
        result.pairs = profiler.measure("gemma.moe_geglu_down", [&] {
            return project.gated(*down, packed, route, true);
        });
        return result;
    }
    auto hidden = profiler.measure("gemma.moe_geglu", [&] { return moe_geglu_split_cuda(packed); });
    if (trace) result.hidden = hidden.to(::tyloi::kFloat32).contiguous().clone();
    result.pairs = profiler.measure("gemma.moe_down", [&] { return project.run(*down, hidden, route); });
    return result;
#else
    throw std::runtime_error("Routed GEGLU expert weights require a compatibility build");
#endif
}

} // namespace tyloi::cuda::ops
#endif
