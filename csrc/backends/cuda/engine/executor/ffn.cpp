#include "ffn.h"
#include "engine/weights/moe/experts.h"
#include "execution.h"
#include "tq_moe.h"
#include "engine/weights/moe/load.h"
#include "engine/weights/moe/expert_cache.h"
#include "moe_ffn_pipeline.h"
#include "ops/ffn/mlp.h"
#include "ops/ffn/dense.h"
#include "ops/ffn/residual.h"
#include "engine/models/common/moe.h"
#include "mfq_cuda_activation_ops.h"
#include "mfq_cuda_moe_ops.h"
#include "mfq_cuda_norm_ops.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda {

using Tensor = mfq_tensor_backend::Tensor;
using ActivationKind = mfq::models::MlpActivation;

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

void prefetch_ffn(CudaExecutionContext& execution, const Ffn& f,const mfq_tensor_backend::Tensor& x,
        MfqOptional<mfq_tensor_backend::Tensor> input_ids) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    auto pipeline=f.moe_pipeline.lock();
    if(!pipeline || !moe_expert_cache_ssd_router_prefetch(execution.stores.moe_expert_cache) ||
            x.dim()<2 || x.numel()/x.size(-1)>32)return;
    namespace tb=mfq_tensor_backend;
    tb::Tensor ids;
    // Cover each row's leading candidates without reading low-confidence
    // tails just because the leading experts are already cached.
    constexpr int64_t candidates_per_row=4;
    if(f.moe_hash_ids.defined()) {
        if(!input_ids.has_value() || !input_ids.value().numel())return;
        ids=f.moe_hash_ids.index_select(0,input_ids.value().reshape({-1}).to(tb::kCUDA,tb::kInt64));
        ids=ids.narrow(1,0,std::min(candidates_per_row,ids.size(1))).to(tb::kInt32).contiguous();
    }else {
        auto row=x.reshape({-1,x.size(-1)}).to(tb::kFloat32);
        auto logits=f.moe_router_projection?execution.projector().run(*f.moe_router_projection, row).contiguous():
            tb::matmul(row,f.moe_router.transpose(0,1)).contiguous();
        ids=moe_topk_cuda(logits,std::min({int64_t(f.moe_top_k),logits.size(-1),candidates_per_row}),f.moe_use_sigmoid,
            f.moe_use_sqrt_softplus,f.moe_normalize,f.moe_delayed_softmax,
            f.moe_router_bias.defined()?MfqOptional<tb::Tensor>(f.moe_router_bias):mfq_nullopt,
            1e-20,f.moe_router_scale).at(0).to(tb::kInt32).contiguous();
    }
    // Visit each row's highest-ranked candidate before lower ranks, so one
    // token cannot consume the entire speculative read budget for a batch.
    static std::atomic<bool> reported{false};
    if(!reported.load(std::memory_order_relaxed) && !reported.exchange(true,std::memory_order_relaxed))
        std::fprintf(stderr,"moe_ssd_router_candidates scope=first_forecast rows=%lld candidates_per_row=%lld\n",
            static_cast<long long>(ids.size(0)),static_cast<long long>(ids.size(1)));
    auto task=prefetch_moe_ffn_pipeline(pipeline,ids.transpose(0,1).contiguous().reshape({1,-1}));
    if(auto* window=mfq::cuda::DecodeWindow::recording())window->enroll(std::move(task));
    else task.serve();
#else
    return;
#endif
}

namespace {

Mlp dense(const Ffn& f) {
    return {f.gate_up, f.down, f.activation, f.geglu, f.swiglu_limit};
}

bool dense_linear_defined(const QuantLinear& linear) {
    return linear.out() > 0 && linear.neuron_len() > 0;
}

mfq_tensor_backend::Tensor
run_experts(CudaExecutionContext &execution, const Ffn& f, mfq_tensor_backend::Tensor x,
                                 const MoeRoutePlan &route,
                                 mfq_tensor_backend::Tensor route_weights) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const size_t shard_count = f.moe_gate_up.expert_parallel_shards.size();
    std::vector<mfq_tensor_backend::Tensor> routed_partials(shard_count);
    std::vector<mfq_tensor_backend::Tensor> down_partials;
    const bool collect_output_energy =
        !execution.config.moe_route_stats_path.empty() && execution.config.moe_route_output_energy;
    if (collect_output_energy) {
        down_partials.resize(shard_count);
    }
    for (size_t launch_position = 0; launch_position < shard_count; ++launch_position) {
        const size_t index =
            parallel_index(
            execution.config.tasks.projection.peer_first, launch_position, shard_count);
        const auto &gate_shard = f.moe_gate_up.expert_parallel_shards[index];
        const auto &down_shard = f.moe_down.expert_parallel_shards[index];
        MfqCudaGuard guard(gate_shard.device);
        auto local_x =
            execution.collectives.copy(x, gate_shard.device);
        auto local_weights = execution.collectives.copy(route_weights, gate_shard.device);
        const auto &local_route =
            moe_route_to_device(execution.collectives, route, gate_shard.device);
        using Tensor = mfq_tensor_backend::Tensor;
        Tensor gate_up_pair;
        auto down_pair = mfq::models::gated_mlp(
            local_x, false, f.swiglu_limit, [](const auto &...) { return std::optional<Tensor>{}; },
            [&](const Tensor &input, auto, double limit) -> std::optional<Tensor> {
                if (limit <= 0.0 && !execution.force_moe_materialized_swiglu &&
                    gate_shard.weight->supports_projection_glu_epilogue())
                    return execution.projector().glu(*gate_shard.weight, input, local_route, false);
                return {};
            },
            [&](Tensor input) {
                gate_up_pair = execution.projector().run(*gate_shard.weight, input, local_route);
                const auto width = gate_up_pair.size(-1) / 2;
                return std::array<Tensor, 2>{gate_up_pair.narrow(-1, 0, width),
                                             gate_up_pair.narrow(-1, width, width)};
            },
            [&](Tensor gate, Tensor up, auto, double limit) {
                if (limit <= 0.0)
                    return moe_swiglu_split_cuda(gate_up_pair);
                gate = mfq_tensor_backend::clamp_max(gate.to(mfq_tensor_backend::kFloat32), limit);
                up = mfq_tensor_backend::clamp(up.to(mfq_tensor_backend::kFloat32), -limit, limit);
                return (mfq_tensor_backend::silu(gate) * up)
                    .to(mfq_tensor_backend::kFloat16)
                    .contiguous();
            },
            [&](Tensor hidden) {
                return execution.projector().run(*down_shard.weight, hidden, local_route);
            },
            [&](const Tensor &, const Tensor &, auto, double limit) -> std::optional<Tensor> {
                const bool allow_fusion =
                    !execution.force_moe_materialized_swiglu &&
                    execution.config.moe_swiglu_quant_fusion &&
                    (gate_up_pair.size(0) == 1 ||
                     (gate_up_pair.size(0) <= 4 && execution.config.moe_small_heterogeneous));
                if (!allow_fusion)
                    return {};
                if (limit <= 0.0)
                    return execution.projector().gated(*down_shard.weight, gate_up_pair, local_route, false);
                if (down_shard.weight->supports_clamped_swiglu())
                    return execution.projector().clamped(*down_shard.weight, gate_up_pair, local_route, limit);
                return {};
            });
        if (collect_output_energy) {
            down_partials[index] = down_pair;
        }
        routed_partials[index] = moe_weighted_reduce_cuda(down_pair, local_weights);
    }
    if (!execution.config.moe_route_stats_path.empty()) {
        mfq_tensor_backend::Tensor complete_down;
        if (collect_output_energy) {
            complete_down = execution.collectives.sum(std::move(down_partials), model_parallel_primary_device(execution),
            execution.config.tasks.projection.reduce_to_primary, execution.config.tasks.projection.fp16_reduce);
        }
        record_moe_route_stats(execution, f.moe_layer, route.ids, route_weights, complete_down,
                               f.moe_gate_up.n_experts);
    }
    return execution.collectives.sum(std::move(routed_partials), model_parallel_primary_device(execution),
            execution.config.tasks.projection.reduce_to_primary, execution.config.tasks.projection.fp16_reduce);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

mfq_tensor_backend::Tensor run_ffn_impl(CudaExecutionContext &execution, const Ffn& f,
                                             mfq_tensor_backend::Tensor x,
                                             MfqOptional<mfq_tensor_backend::Tensor> input_ids,
                                             bool allow_important_neurons,
                                             MfqOptional<mfq_tensor_backend::Tensor> router_bias) {
    auto &profiler = execution.profiler;
    mfq_tensor_backend::Tensor xh;
    if (x.scalar_type() == mfq_tensor_backend::kFloat16) {
        xh = x;
    } else {
        xh = profiler.measure("ffn.input_cast",
                              [&]() { return x.to(mfq_tensor_backend::kFloat16); });
    }
    if (allow_important_neurons && f.important_neurons) {
        if (f.is_moe) {
            throw std::runtime_error("important-neuron branches are only supported for dense FFNs");
        }
        const int64_t rows = xh.numel() / xh.size(-1);
        const bool use_f32_down = rows >= 16 && execution.kl_mmq.mode == KlMmqMode::Fp16 &&
                                  execution.config.tasks.dense.important_fp32;
        return mfq::models::additive_branches(
            [&](size_t index) {
                const auto &branch = index == 0 ? f : *f.important_neurons;
                if (use_f32_down) {
                    TYLOINFER_RUNTIME_CHECK(!branch.is_moe,
                        "FP32-output IN diagnostic requires a local dense SiLU FFN");
                    return dense(branch).kld(execution.projector(), profiler, execution.kl_mmq, xh);
                }
                return run_ffn_impl(execution, branch, xh, input_ids, false, mfq_nullopt);
            },
            [&](auto &run_branch, auto &outputs) {
                return !use_f32_down &&
                       mfq::cuda::decode_branches(
                    execution.config.tasks.projection.parallel, execution.decode_graph_serial_branches, rows) &&
                       execution.config.tasks.dense.important_parallel &&
                       execution.branches.run(2, run_branch, outputs);
            },
            [&](auto low, auto high) {
                if (use_f32_down)
                    return profiler.measure("ffn.in.combine.fp32", [&] {
                        return (low + high).to(mfq_tensor_backend::kFloat16).contiguous();
                    });
                return profiler.measure("ffn.in.combine", [&] {
                    return acc_cuda(low.contiguous(), high.contiguous());
                });
            });
    }
    if (!f.is_moe && dense(f).parallel()) {
        return profiler.measure("ffn.tensor_parallel", [&] {
            return dense(f).sharded(
                profiler, execution.kl_mmq, *execution.tq, execution.collectives,
                model_parallel_primary_device(execution), execution.config.tasks.projection.peer_first,
                execution.config.tasks.projection.reduce_to_primary, execution.config.tasks.projection.fp16_reduce, xh);
        });
    }
    if (f.is_moe) {
        const bool tq_routed=bool(f.moe_tq);
        const int64_t rows = xh.numel() / xh.size(-1);
        if (execution.continuous_batch_cache_serial && uses_expert_cache(f) && rows > 1) {
            // A bounded expert cache cannot safely admit the union of an
            // arbitrary request batch: a miss in that union otherwise
            // falls back to staging the complete projection. Keep the
            // outer request/KV/attention batch intact and execute only
            // each routed FFN row independently, so cache admission is
            // bounded by one token's top-k experts.
            auto flat = xh.reshape({rows, xh.size(-1)}).contiguous();
            std::vector<mfq_tensor_backend::Tensor> outputs;
            outputs.reserve(static_cast<size_t>(rows));
            for (int64_t row = 0; row < rows; ++row) {
                auto row_input = flat.narrow(0, row, 1).contiguous();
                if (input_ids.has_value()) {
                    TYLOINFER_RUNTIME_CHECK(input_ids.value().numel() == rows,
                                      "continuous-batch MoE token ids do not match rows");
                    auto row_ids = input_ids.value().reshape({rows}).narrow(0, row, 1).contiguous();
                    auto bias=router_bias;
                    if(bias.has_value() && bias->dim()==2)bias=bias->narrow(0,row,1);
                    outputs.push_back(run_ffn_impl(execution, f, row_input, row_ids, false,bias));
                } else {
                    auto bias=router_bias;
                    if(bias.has_value() && bias->dim()==2)bias=bias->narrow(0,row,1);
                    outputs.push_back(run_ffn_impl(execution, f, row_input, mfq_nullopt, false,bias));
                }
            }
            return mfq_tensor_backend::cat(outputs, 0).reshape(xh.sizes());
        }
        if (!f.shared || f.moe_top_k <= 0 || (!f.moe_router.defined() && !f.moe_router_projection) ||
            (!f.moe_shared_ungated && !f.moe_shared_gate.defined())) {
            throw std::runtime_error("incomplete MoE FFN state");
        }
        auto xf = xh.reshape({-1, xh.size(-1)}).contiguous();
        auto xf32 = profiler.measure("moe.input_f32",
                                     [&]() { return xf.to(mfq_tensor_backend::kFloat32); });
        using Tensor = mfq_tensor_backend::Tensor;
        MoeRoutePlan route;
        Tensor shared_gate_logits;
        bool projection_bundle_prefetched = false;
        const bool sharded_experts = expert_parallel(f);
#if TYLOINFER_ENABLE_MFQ_COMPAT
        auto pipeline=f.moe_pipeline.lock();
        const bool use_pipeline=bool(pipeline) &&
            execution.kl_mmq.mode==KlMmqMode::Default && !execution.force_moe_materialized_swiglu;
#else
        const bool use_pipeline=false;
#endif
        struct ExpertOutput {
            Tensor values;
            bool reduced;
        };
        // Start the tiny route readback before the shared expert.  The
        // cache consumes it after the shared branch has been enqueued,
        // then launches H2D on its separate stream.  The projection
        // forward below still performs the normal cache check and stream
        // wait, preserving the existing execution semantics.
        auto prefetch_projection_bundle = [&]() {
#if TYLOINFER_ENABLE_MFQ_COMPAT
            if (use_pipeline || !execution.config.moe_projection_bundle_prefetch || f.cpu_moe_down ||
                execution.continuous_batch_cache_serial) {
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
                    return mfq_tensor_backend::matmul(xf32, f.moe_shared_gate.transpose(0, 1))
                        .contiguous();
                });
            return shared_gate_logits;
        };
        return mfq::models::mixture_of_experts(
            [&] {
                return profiler.measure("moe.router", [&]() {
                    return f.moe_router_projection?execution.projector().run(*f.moe_router_projection, xf32):
                        mfq_tensor_backend::matmul(xf32, f.moe_router.transpose(0, 1));
                });
            },
            [&](Tensor router_logits) {
                std::vector<mfq_tensor_backend::Tensor> selected;
                if (f.moe_hash_ids.defined()) {
                    if (!input_ids.has_value()) {
                        throw std::runtime_error("hash-routed MoE requires the current token ids");
                    }
                    selected = profiler.measure("moe.hash_route", [&]() {
                        auto ids = f.moe_hash_ids
                                       .index_select(0, input_ids.value().reshape({-1}).to(
                                                            mfq_tensor_backend::kCUDA,
                                                            mfq_tensor_backend::kInt64))
                                       .to(mfq_tensor_backend::kInt32)
                                       .contiguous();
                        auto weights = moe_sqrtsoftplus_weights_cuda(router_logits.contiguous(),
                                                                     ids, 1e-20, f.moe_router_scale);
                        return std::vector<mfq_tensor_backend::Tensor>{ids, weights};
                    });
                } else {
                    selected = profiler.measure("moe.topk", [&]() {
                        return moe_topk_cuda(
                            router_logits.contiguous(), f.moe_top_k, f.moe_use_sigmoid,
                            f.moe_use_sqrt_softplus, f.moe_normalize, f.moe_delayed_softmax,
                            router_bias.has_value()?router_bias:f.moe_router_bias.defined()
                                ? MfqOptional<mfq_tensor_backend::Tensor>(f.moe_router_bias)
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
                return selected;
            },
            [&](const auto &) {
                if (mfq::mfq_compat_enabled && !sharded_experts && !use_pipeline && !tq_routed) {
                    if (f.moe_split_gate_up) {
                        if (execution.config.moe_delayed_route_readback) {
                            mfq::cuda::prefetch_experts(f.moe_gate, route, true);
                        } else {
                            projection_bundle_prefetched = prefetch_projection_bundle();
                            if (!projection_bundle_prefetched) {
                                mfq::cuda::prefetch_experts(f.moe_gate, route);
                            }
                        }
                    } else {
                        if (execution.config.moe_delayed_route_readback) {
                            mfq::cuda::prefetch_experts(f.moe_gate_up, route, true);
                        } else {
                            projection_bundle_prefetched = prefetch_projection_bundle();
                            if (!projection_bundle_prefetched) {
                                mfq::cuda::prefetch_experts(f.moe_gate_up, route);
                            }
                        }
                    }
                }
                auto shared_output = profiler.measure(
                    "moe.shared", [&]() { return run_ffn(execution, *f.shared, xf); });
                if (!f.moe_split_gate_up && !f.moe_shared_ungated) {
                    shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                        return mfq_tensor_backend::matmul(xf32, f.moe_shared_gate.transpose(0, 1))
                            .contiguous();
                    });
                }
                if (mfq::mfq_compat_enabled && !sharded_experts && !use_pipeline && !tq_routed && execution.config.moe_delayed_route_readback) {
                    projection_bundle_prefetched = prefetch_projection_bundle();
                    if (!projection_bundle_prefetched) {
                        if (f.moe_split_gate_up) {
                            mfq::cuda::prefetch_experts(f.moe_gate, route);
                        } else {
                            mfq::cuda::prefetch_experts(f.moe_gate_up, route);
                        }
                    }
                }
                return shared_output.reshape({xf.size(0), xf.size(1)})
                    .contiguous()
                    .to(mfq_tensor_backend::kFloat16);
            },
            [&](const auto &selected) -> ExpertOutput {
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
                        return execution.projector().run(f.moe_latent_in, xf);
                    });
                const auto finish_latent = [&](Tensor reduced) {
                    if (latent_routed)
                        return profiler.measure("moe.latent_up", [&] {
                            return execution.projector().run(f.moe_latent_out, reduced);
                        });
                    return reduced;
                };
                if(tq_routed) {
                    TYLOINFER_RUNTIME_CHECK(f.moe_tq->up && f.moe_tq->down,
                        "TQ routed FFN requires matching Up and Down projections");
                    if(!f.moe_ungated_experts && f.activation==ActivationKind::silu_gated) {
                        auto reduced=profiler.measure("moe.tq",[&] {
                            return execution.projector().run(*f.moe_tq,routed_input,selected.at(0),selected.at(1),
                                static_cast<float>(f.swiglu_limit));
                        });
                        return {finish_latent(std::move(reduced)),true};
                    }
                    TYLOINFER_RUNTIME_CHECK(f.moe_tq->slices.empty(),
                        "TQ expert parallelism currently requires SiLU-gated experts");
                    Tensor gate,up;
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
                    return {finish_latent(std::move(reduced)),true};
                }
                if (dense_routed) {
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
                            return execution.projector().run(f.moe_latent_in, xf);
                        });
                    auto up = profiler.measure("moe.dense_up", [&] {
                        return dense_routed_projection(f.moe_dense_up, expert_input,
                                                       selected.at(0));
                    });
                    Tensor gate;
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
                            return execution.projector().run(f.moe_latent_out, reduced);
                        });
                    record_moe_route_stats(execution, f.moe_layer, selected.at(0), selected.at(1),
                                           down_pair, static_cast<int>(f.moe_dense_up.size(0)));
                    return {reduced, true};
                }
#if TYLOINFER_ENABLE_MFQ_COMPAT
                if (f.moe_ungated_experts && !use_pipeline) {
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
                        mfq::cuda::prefetch_experts(f.moe_up, route);
                        mfq::cuda::prefetch_experts(f.moe_down, route);
                    }
                    std::optional<mfq::cuda::Experts> staged_up;
                    const mfq::cuda::Experts *active_up = &f.moe_up;
                    if (f.cpu_moe_up) {
                        staged_up.emplace(profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(f.cpu_moe_up, execution.config.tasks.projection.moe.heterogeneous);
                        }));
                        active_up = &staged_up.value();
                    }
                    auto up = profiler.measure("moe.up", [&]() {
                        return execution.projector().run(*active_up, routed_input, route);
                    });
                    auto hidden = profiler.measure("moe.relu_squared", [&]() {
                        return relu_squared_cuda(up.contiguous());
                    });
                    staged_up.reset();
                    std::optional<mfq::cuda::Experts> staged_down;
                    const mfq::cuda::Experts *active_down = &f.moe_down;
                    if (f.cpu_moe_down) {
                        staged_down.emplace(profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                            return stage_cpu_mixed_moe(f.cpu_moe_down, execution.config.tasks.projection.moe.heterogeneous);
                        }));
                        active_down = &staged_down.value();
                    }
                    auto down_pair = profiler.measure("moe.down", [&] {
                        return execution.projector().run(*active_down, hidden, route);
                    });
                    record_moe_route_stats(execution, f.moe_layer, selected.at(0),
                                           selected.at(1), down_pair, f.moe_up.n_experts);
                    staged_down.reset();
                    if (latent_routed)
                        return {finish_latent(profiler.measure("moe.reduce", [&] {
                                    return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                                })),
                                true};
                    return {down_pair, false};
                }
                if(use_pipeline) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
                    auto result=run_moe_ffn_pipeline(pipeline,execution,routed_input,selected.at(0),
                                                     selected.at(1));
                    if(result.dim()==3)result=moe_weighted_reduce_cuda(result,selected.at(1));
                    return {finish_latent(result),true};
#endif
                }
                if (sharded_experts) {
                    auto result = profiler.measure("moe.sharded_experts", [&] {
                        return run_experts(execution, f, routed_input, route,
                                                           selected.at(1));
                    });
                    if (latent_routed && result.dim() == 3)
                        result = moe_weighted_reduce_cuda(result, selected.at(1));
                    return {finish_latent(result), true};
                }
                std::optional<mfq::cuda::Experts> staged_gate_up;
                std::optional<mfq::cuda::Experts> staged_gate;
                std::optional<mfq::cuda::Experts> staged_up;
                const mfq::cuda::Experts *active_gate_up = &f.moe_gate_up;
                const mfq::cuda::Experts *active_gate = &f.moe_gate;
                const mfq::cuda::Experts *active_up = &f.moe_up;
                mfq_tensor_backend::Tensor gate_up_pair;
                mfq_tensor_backend::Tensor projected_hidden;
                if (f.moe_split_gate_up) {
                    if (f.cpu_moe_gate) {
                        staged_gate.emplace(profiler.measure("moe.cpu_offload_gate_h2d", [&]() {
                            return stage_cpu_mixed_moe(f.cpu_moe_gate, execution.config.tasks.projection.moe.heterogeneous);
                        }));
                        active_gate = &staged_gate.value();
                    }
                    if (f.cpu_moe_up) {
                        staged_up.emplace(profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(f.cpu_moe_up, execution.config.tasks.projection.moe.heterogeneous);
                        }));
                        active_up = &staged_up.value();
                    }
                    gate_up_pair = profiler.measure("moe.gate_up_split", [&]() {
                        const bool reuse_gate_activation =
                            execution.config.split_moe_activation_reuse &&
                            execution.kl_mmq.mode == KlMmqMode::Default && xf.size(0) <= 8 &&
                            active_gate->can_reuse_activation_for(*active_up);
                        auto gate = execution.projector().run(*active_gate, xf, route);
                        if (!projection_bundle_prefetched) {
                            mfq::cuda::prefetch_experts(*active_up, route);
                        }
                        if (!f.moe_shared_ungated && !shared_gate_logits.defined()) {
                            shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                                return mfq_tensor_backend::matmul(xf32,
                                                                  f.moe_shared_gate.transpose(0, 1))
                                    .contiguous();
                            });
                        }
                        auto up = reuse_gate_activation
                                      ? execution.projector().run(*active_up, xf, route, true)
                                      : execution.projector().run(*active_up, xf, route);
                        return mfq_tensor_backend::cat({gate, up}, -1).contiguous();
                    });
                } else {
                    if (f.cpu_moe_gate_up) {
                        staged_gate_up.emplace(
                            profiler.measure("moe.cpu_offload_gate_up_h2d", [&]() {
                                return stage_cpu_mixed_moe(f.cpu_moe_gate_up, execution.config.tasks.projection.moe.heterogeneous);
                            }));
                        active_gate_up = &staged_gate_up.value();
                    }
                    const bool fuse_projection_glu =
                        f.swiglu_limit <= 0.0 && !execution.force_moe_materialized_swiglu &&
                        active_gate_up->supports_projection_glu_epilogue();
                    if (fuse_projection_glu) {
                        projected_hidden = profiler.measure("moe.gate_up_swiglu", [&]() {
                            return execution.projector().glu(*active_gate_up, xf, route, false);
                        });
                    } else {
                        gate_up_pair = profiler.measure("moe.gate_up", [&]() {
                            return execution.projector().run(*active_gate_up, xf, route);
                        });
                    }
                }
                staged_gate_up.reset();
                staged_gate.reset();
                staged_up.reset();
                if (!projection_bundle_prefetched) {
                    mfq::cuda::prefetch_experts(f.moe_down, route);
                }
                std::optional<mfq::cuda::Experts> staged_down;
                const mfq::cuda::Experts *active_down = &f.moe_down;
                if (f.cpu_moe_down) {
                    staged_down.emplace(profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(f.cpu_moe_down, execution.config.tasks.projection.moe.heterogeneous);
                    }));
                    active_down = &staged_down.value();
                }
                auto down_pair = mfq::models::gated_mlp(
                    xf, false, f.swiglu_limit,
                    [](const auto &...) { return std::optional<Tensor>{}; },
                    [&](const Tensor &, auto, double) -> std::optional<Tensor> {
                        if (projected_hidden.defined())
                            return projected_hidden;
                        return {};
                    },
                    [&](const Tensor &) {
                        const auto width = gate_up_pair.size(-1) / 2;
                        return std::array<Tensor, 2>{gate_up_pair.narrow(-1, 0, width),
                                                     gate_up_pair.narrow(-1, width, width)};
                    },
                    [&](const Tensor &, const Tensor &, auto, double limit) {
                        return profiler.measure("moe.swiglu", [&]() {
                            if (limit <= 0.0) {
                                return moe_swiglu_split_cuda(gate_up_pair);
                            }
                            const int64_t width = gate_up_pair.size(-1) / 2;
                            auto gate = mfq_tensor_backend::clamp_max(
                                gate_up_pair.slice(-1, 0, width).to(mfq_tensor_backend::kFloat32),
                                limit);
                            auto up =
                                mfq_tensor_backend::clamp(gate_up_pair.slice(-1, width, 2 * width)
                                                              .to(mfq_tensor_backend::kFloat32),
                                                          -limit, limit);
                            return (mfq_tensor_backend::silu(gate) * up)
                                .to(mfq_tensor_backend::kFloat16)
                                .contiguous();
                        });
                    },
                    [&](Tensor hidden) {
                        return profiler.measure("moe.down", [&] {
                            return execution.projector().run(*active_down, hidden, route);
                        });
                    },
                    [&](const Tensor &, const Tensor &, auto,
                        double limit) -> std::optional<Tensor> {
                        const bool allow_fusion = !execution.force_moe_materialized_swiglu &&
                                                  execution.config.moe_swiglu_quant_fusion &&
                                                  (gate_up_pair.size(0) == 1 ||
                                                   (gate_up_pair.size(0) <= 4 &&
                                                    execution.config.moe_small_heterogeneous));
                        if (!allow_fusion)
                            return {};
                        if (limit <= 0.0)
                            return profiler.measure("moe.swiglu_down", [&] {
                                return execution.projector().gated(*active_down, gate_up_pair, route, false);
                            });
                        if (active_down->supports_clamped_swiglu())
                            return profiler.measure("moe.swiglu_down", [&] {
                                return execution.projector().clamped(*active_down, gate_up_pair, route, limit);
                            });
                        return {};
                    });
                record_moe_route_stats(
                    execution, f.moe_layer, selected.at(0), selected.at(1), down_pair,
                    f.moe_split_gate_up ? f.moe_gate.n_experts : f.moe_gate_up.n_experts);
                staged_down.reset();
                return {down_pair, false};
#else
                throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
            },
            [&](ExpertOutput experts, Tensor shared_half, const auto &selected) {
                return mfq::models::combine_experts(
                    experts.values, shared_half, experts.reduced, f.moe_shared_ungated,
                    [&](Tensor pairs, Tensor shared) -> std::optional<Tensor> {
                        if (execution.force_moe_unfused_reduce ||
                            !execution.config.moe_reduce_gate_fusion || f.moe_shared_ungated ||
                            pairs.size(0) > 8)
                            return {};
                        auto logits = shared_gate();
                        return profiler.measure("moe.reduce_combine", [&] {
                            return moe_weighted_reduce_shared_gate_cuda(pairs, selected.at(1),
                                                                        shared, logits);
                        });
                    },
                    [&](Tensor pairs) {
                        return profiler.measure("moe.reduce", [&] {
                            return moe_weighted_reduce_cuda(pairs, selected.at(1));
                        });
                    },
                    shared_gate,
                    [&](Tensor routed, Tensor shared) {
                        return profiler.measure("moe.combine", [&] {
                            // Dense BF16 expert banks may return a different storage
                            // type than the shared branch; align without changing it.
                            if (routed.scalar_type() != shared.scalar_type())
                                shared = shared.to(routed.scalar_type()).contiguous();
                            return acc_cuda(routed.contiguous(), shared);
                        });
                    },
                    [&](Tensor routed, Tensor shared, Tensor logits) {
                        return profiler.measure("moe.combine", [&] {
                            return moe_add_shared_gate_cuda(routed.contiguous(), shared, logits);
                        });
                    });
            });
    }
    TYLOINFER_RUNTIME_CHECK(f.activation != ActivationKind::relu_squared || !f.important_neurons,
                      "relu-squared FFN does not support important neurons");
    const Mlp::Options options{execution.config.tasks.projection.nvq_fusion,
                               execution.config.tasks.dense.mlp.gelu_fusion,
                               execution.config.tasks.dense.mlp.silu_fusion};
    return dense(f).forward(execution.projector(), profiler, options, xh);
}

} // namespace

Tensor run_ffn(CudaExecutionContext& execution, const Ffn& f, Tensor x,
               MfqOptional<Tensor> input_ids, MfqOptional<Tensor> router_bias) {
    if (router_bias.has_value()) {
        const auto& bias = router_bias.value();
        TYLOINFER_RUNTIME_CHECK(f.is_moe && !f.moe_hash_ids.defined() && bias.defined() &&
            ((bias.dim()==1 && bias.size(0)==(f.moe_router_projection?f.moe_router_projection->out():f.moe_router.size(0))) ||
             (bias.dim()==2 && bias.size(0)==x.numel()/x.size(-1) && bias.size(1)==(f.moe_router_projection?f.moe_router_projection->out():f.moe_router.size(0)))),
                          "MoE router correction bias shape mismatch");
    }
    return run_ffn_impl(execution, f, std::move(x), std::move(input_ids), true, std::move(router_bias));
}

Tensor full_ffn(
    CudaExecutionContext& execution, const Ffn& ffn, const Tensor& ffn_norm,
    const FullAttention& attention, Tensor residual, Tensor output) {
    if (!ffn.is_moe) {
        Dense op{dense(ffn), {}, ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset,
            attention.official_bf16, attention.attention_output_gate};
        if (ffn.important_neurons) op.important.emplace(dense(*ffn.important_neurons));
        auto parallel = execution.config.tasks.projection;
        parallel.primary = model_parallel_primary_device(execution);
        auto options = execution.config.tasks.dense;
        options.mlp.nvq_fusion = parallel.nvq_fusion;
        options.bf16_norm = execution.config.tasks.attention.fused_norm;
        return op.forward(execution.projector(), execution.profiler, execution.kl_mmq, *execution.tq,
            execution.branches, execution.collectives, parallel, options,
            execution.decode_graph_serial_branches, std::move(residual), std::move(output),
            attention.layer == execution.gemma_trace_layer ? execution.gemma_stage_trace : nullptr);
    }
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    auto& profiler = execution.profiler;
    const auto& config = execution.config;
    auto trace = [&](const char* name, const Tensor& value) {
        if (attention.attention_output_gate && execution.gemma_stage_trace &&
            attention.layer == execution.gemma_trace_layer)
            trace_gemma_stage(execution, attention.layer, name, value.reshape({B, T, -1}));
    };
    auto pair = full_norm(profiler, ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset,
                          attention.official_bf16, config.tasks.attention.fused_norm,
                          config.tasks.dense.fp32_residual, std::move(residual), std::move(output));
    residual = pair[0];
    auto input = pair[1].reshape({B * T, H});
    trace("qwen.ffn_norm", input);
    auto mlp = dense(ffn);
    auto ff = attention.official_bf16
        ? mlp.bf16(execution.projector(), profiler, config.tasks.dense.bf16_swiglu, input)
              .reshape({B, T, H}).to(mfq_tensor_backend::kBFloat16).contiguous()
        : run_ffn(execution, ffn, input).reshape({B, T, H});
    trace("qwen.ffn_output", ff);
    return full_add(profiler, config.tasks.dense.fp32_residual, config.tasks.dense.bf16_residual,
                    std::move(residual), std::move(ff));
}

Tensor linear_ffn(
    CudaExecutionContext& execution, const Ffn& ffn, const Tensor& ffn_norm,
    const LinearAttention& attention, Tensor residual, Tensor output) {
    if (!ffn.is_moe) {
        Dense op{dense(ffn), {}, ffn_norm, attention.rms_norm_eps,
            attention.norm_weight_offset, false, false, true};
        if (ffn.important_neurons) op.important.emplace(dense(*ffn.important_neurons));
        auto parallel = execution.config.tasks.projection;
        parallel.primary = model_parallel_primary_device(execution);
        auto options = execution.config.tasks.dense;
        options.mlp.nvq_fusion = parallel.nvq_fusion;
        return op.forward(execution.projector(), execution.profiler, execution.kl_mmq, *execution.tq,
            execution.branches, execution.collectives, parallel, options,
            execution.decode_graph_serial_branches, std::move(residual), std::move(output));
    }
    const int64_t B = residual.size(0), T = residual.size(1), H = residual.size(2);
    auto& profiler = execution.profiler;
    const auto& config = execution.config;
    auto pair = linear_norm(profiler, ffn_norm, attention.rms_norm_eps, attention.norm_weight_offset,
                            config.tasks.dense.fp32_residual, std::move(residual), std::move(output));
    residual = pair[0];
    auto input = pair[1].reshape({B * T, H});
    auto ff = run_ffn(execution, ffn, input).reshape({B, T, H});
    return linear_add(profiler, config.tasks.dense.fp32_residual, std::move(residual), std::move(ff));
}

} // namespace mfq::cuda
