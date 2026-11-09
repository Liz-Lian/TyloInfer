#include "ffn.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/native/profile.h"
#include "diagnostics/route_stats.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/weights/moe/experts.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/tq/routed.h"
#include "engine/backends/cuda/ops/ffn/tq.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/moe/cached_source_internal.h"
#endif
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/ops/projection/parallel.h"
#include "engine/backends/cuda/ops/ffn/mlp.h"
#ifndef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/ops/projection/projection.h"
#endif
#include "engine/ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <algorithm>
#include <array>
#include <climits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda {

using ActivationKind = tyloi::ops::MlpActivation;

std::shared_ptr<const ::tyloi::MfeTensor> cpu_expert(const std::shared_ptr<const ::tyloi::Tensor>& weight) {
    auto mfe = std::dynamic_pointer_cast<const ::tyloi::MfeTensor>(weight);
    return mfe && !mfe->cached_source && mfe->device().is_cpu() ? mfe : nullptr;
}

static std::shared_ptr<const ::tyloi::MfeTensor> mfe_runtime(const std::shared_ptr<const ::tyloi::Tensor>& weight) {
    return std::dynamic_pointer_cast<const ::tyloi::MfeTensor>(weight);
}

bool uses_expert_cache(const Ffn& f) {
    const auto moe_gate = mfe_runtime(f.moe_gate);
    const auto moe_up = mfe_runtime(f.moe_up);
    const auto moe_gate_up = mfe_runtime(f.moe_gate_up);
    const auto moe_down = mfe_runtime(f.moe_down);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!f.is_moe)
        return false;
    if (f.moe_split_gate_up) {
        return (moe_gate && moe_gate->cached_source) || (moe_up && moe_up->cached_source) ||
            (moe_down && moe_down->cached_source);
    }
    return (moe_gate_up && moe_gate_up->cached_source) || (moe_down && moe_down->cached_source);
#else
    return false;
#endif
}

bool expert_parallel(const Ffn& f) {
    const auto moe_gate_up = mfe_runtime(f.moe_gate_up);
    const auto moe_down = mfe_runtime(f.moe_down);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!f.is_moe || f.moe_split_gate_up || !moe_gate_up || !moe_down ||
        moe_gate_up->shards.size() != moe_down->shards.size() ||
        moe_gate_up->shards.empty()) {
        return false;
    }
    for (size_t index = 0; index < moe_gate_up->shards.size(); ++index) {
        const auto &gate = moe_gate_up->shards[index];
        const auto &down = moe_down->shards[index];
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

Mlp dense(const Ffn& f, const Projector& project) {
    TYLOINFER_RUNTIME_CHECK(f.down, "FFN down projection is missing");
#ifdef TYLOINFER_NATIVE_CUDA
    return {project.group(f.gate_up), *f.down, f.activation, f.geglu, f.swiglu_limit};
#else
    return {make_quant_group(f.gate_up, true, false), *f.down, f.activation, f.geglu, f.swiglu_limit};
#endif
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

} // namespace

DenseTensor Ffn::forward_experts(DenseTensor x, Executor& executor, const Mode& mode_,
                            const Projector& project_, const MoeRoutePlan& route,
                            DenseTensor route_weights) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto& f = *this;
    const auto moe_gate_up = mfe_runtime(f.moe_gate_up);
    const auto moe_down = mfe_runtime(f.moe_down);
    auto& stats = executor.routes_;
    auto& collectives_ = executor.collectives_;
    const auto& projection_ = executor.options_.projection;
    const auto& options_ = executor.options_.ffn;
    const size_t shard_count = moe_gate_up->shards.size();
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
        const auto &gate_shard = moe_gate_up->shards[index];
        const auto &down_shard = moe_down->shards[index];
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
            supports_projection_glu_epilogue(*gate_shard.weight)) {
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
            } else if (allow_fusion && supports_clamped_swiglu(*down_shard.weight)) {
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
                     moe_gate_up->n_experts);
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
    const auto moe_gate = mfe_runtime(f.moe_gate);
    const auto moe_up = mfe_runtime(f.moe_up);
    const auto moe_gate_up = mfe_runtime(f.moe_gate_up);
    const auto moe_down = mfe_runtime(f.moe_down);
    const auto cpu_moe_gate = cpu_expert(f.moe_gate);
    const auto cpu_moe_up = cpu_expert(f.moe_up);
    const auto cpu_moe_gate_up = cpu_expert(f.moe_gate_up);
    const auto cpu_moe_down = cpu_expert(f.moe_down);
    const auto dense_gate = std::dynamic_pointer_cast<const ::tyloi::DenseTensor>(f.moe_gate);
    const auto dense_up = std::dynamic_pointer_cast<const ::tyloi::DenseTensor>(f.moe_up);
    const auto dense_down = std::dynamic_pointer_cast<const ::tyloi::DenseTensor>(f.moe_down);
    const auto tq = tq_banks(f.moe_gate, f.moe_up, f.moe_down);
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
                    return dense(branch, project_).kld(project_, profiler, mmq_, xh);
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
    std::optional<Mlp> mlp;
    if (!f.is_moe) {
        mlp.emplace(dense(f, project_));
        if (mlp->parallel()) return sharded(*mlp, xh, executor, project_);
    }
    if (f.is_moe) {
        const bool tq_routed=bool(tq.gate || tq.up || tq.down);
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
        if (!f.shared || f.moe_top_k <= 0 || !f.moe_router ||
            (!f.moe_shared_ungated && !f.moe_shared_gate)) {
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
            if (!options_.bundle_prefetch || cpu_moe_down ||
                mode_.batch_serial) {
                return false;
            }
            if (f.moe_ungated_experts) {
                if (cpu_moe_up || cpu_moe_down || !moe_up || !moe_down)
                    return false;
                return prefetch_cached_moe_projection_bundle(*moe_up, *moe_down, route);
            }
            if (f.moe_split_gate_up) {
                if (cpu_moe_gate || cpu_moe_up || !moe_gate || !moe_up || !moe_down)
                    return false;
                return prefetch_cached_moe_projection_bundle(*moe_gate, *moe_up, *moe_down, route);
            }
            if (cpu_moe_gate_up || !moe_gate_up || !moe_down)
                return false;
            return prefetch_cached_moe_projection_bundle(*moe_gate_up, *moe_down, route);
        #else
    return false;
#endif
};

        auto shared_gate = [&] {
            if (!shared_gate_logits.defined())
                shared_gate_logits = profiler.measure("moe.shared_gate", [&] {
                    return ::tyloi::cuda::matmul(xf32, dynamic_cast<const DenseTensor&>(*f.moe_shared_gate).transpose(0, 1))
                        .contiguous();
                });
            return shared_gate_logits;
        };
        std::vector<::tyloi::DenseTensor> selected;
        {
            auto router_logits = profiler.measure("moe.router", [&]() {
                const auto* router = dynamic_cast<const DenseTensor*>(f.moe_router.get());
                return router ? ::tyloi::cuda::matmul(xf32, router->transpose(0, 1)) :
                    project_.run(*f.moe_router, xf32);
            });
            if (f.moe_hash_ids) {
                if (!input_ids.has_value()) {
                    throw std::runtime_error("hash-routed MoE requires the current token ids");
                }
                selected = profiler.measure("moe.hash_route", [&]() {
                    auto ids = dynamic_cast<const DenseTensor&>(*f.moe_hash_ids)
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
                        router_bias.has_value()?router_bias:f.moe_router_bias
                            ? MfqOptional<::tyloi::DenseTensor>(dynamic_cast<const DenseTensor&>(*f.moe_router_bias))
                            : mfq_nullopt,
                        1e-20, f.moe_router_scale);
                });
            }
            if (!tq_routed) {
                // Dense (source-precision) banks are not MFE cohorts, so
                // their expert count comes from the bank itself.
                const auto routed_experts =
                    bool(dense_up)
                        ? static_cast<int>(dense_up->size(0))
                        : (f.moe_ungated_experts
                               ? (moe_up ? moe_up->n_experts : 0)
                               : (f.moe_split_gate_up ? (moe_gate ? moe_gate->n_experts : 0)
                                                    : (moe_gate_up ? moe_gate_up->n_experts : 0)));
                route = profiler.measure("moe.route_map", [&]() {
                    return build_moe_route_plan(selected.at(0), routed_experts);
                });
            }
        }
        DenseTensor shared_half;
        {
            if (tyloi::mfq_compat_enabled && !sharded_experts && !tq_routed) {
                if (f.moe_split_gate_up) {
                    if (options_.delayed_readback) {
                        if (moe_gate) tyloi::cuda::prefetch_experts(*moe_gate, route, true);
                    } else {
                        projection_bundle_prefetched = prefetch_projection_bundle();
                        if (!projection_bundle_prefetched) {
                            if (moe_gate) tyloi::cuda::prefetch_experts(*moe_gate, route);
                        }
                    }
                } else {
                    if (options_.delayed_readback) {
                        if (moe_gate_up) tyloi::cuda::prefetch_experts(*moe_gate_up, route, true);
                    } else {
                        projection_bundle_prefetched = prefetch_projection_bundle();
                        if (!projection_bundle_prefetched) {
                            if (moe_gate_up) tyloi::cuda::prefetch_experts(*moe_gate_up, route);
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
                    return ::tyloi::cuda::matmul(xf32, dynamic_cast<const DenseTensor&>(*f.moe_shared_gate).transpose(0, 1))
                        .contiguous();
                });
            }
            if (tyloi::mfq_compat_enabled && !sharded_experts && !tq_routed && options_.delayed_readback) {
                projection_bundle_prefetched = prefetch_projection_bundle();
                if (!projection_bundle_prefetched) {
                    if (f.moe_split_gate_up) {
                        if (moe_gate) tyloi::cuda::prefetch_experts(*moe_gate, route);
                    } else {
                        if (moe_gate_up) tyloi::cuda::prefetch_experts(*moe_gate_up, route);
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
                bool(dense_up) || bool(dense_down) ||
                bool(dense_gate);
            // Only routed experts run in the latent space; the router and
            // the shared expert keep reading the original hidden state.
            const bool latent_routed =
                !dense_routed && f.moe_latent_in;
            TYLOINFER_RUNTIME_CHECK(!latent_routed || f.moe_latent_out,
                "MoE latent output projection is missing");
            auto routed_input = xf;
            if (latent_routed)
                routed_input = profiler.measure("moe.latent_down", [&] {
                    return project_.run(*f.moe_latent_in, xf);
                });
            const auto finish_latent = [&](DenseTensor reduced) {
                if (latent_routed)
                    return profiler.measure("moe.latent_up", [&] {
                        return project_.run(*f.moe_latent_out, reduced);
                    });
                return reduced;
            };
            if(tq_routed) {
                TYLOINFER_RUNTIME_CHECK(tq.up && tq.down,
                    "TQ routed FFN requires matching Up and Down projections");
                if(!f.moe_ungated_experts && f.activation==ActivationKind::silu_gated) {
                    auto reduced=profiler.measure("moe.tq",[&] {
                        return project_.run(tq,routed_input,selected.at(0),selected.at(1),
                            static_cast<float>(f.swiglu_limit));
                    });
                    experts = {finish_latent(std::move(reduced)),true};
                } else {
                    TYLOINFER_RUNTIME_CHECK(tq.slices.empty(),
                        "TQ expert parallelism currently requires SiLU-gated experts");
                    DenseTensor gate,up;
                    if(f.moe_ungated_experts) {
                        TYLOINFER_RUNTIME_CHECK(!tq.gate,"ungated TQ experts cannot carry a Gate tensor");
                    }else {
                        TYLOINFER_RUNTIME_CHECK(tq.gate,"gated TQ experts require a Gate tensor");
                        gate=tq_ops::routed_projection(*tq.gate,routed_input,selected.at(0));
                    }
                    up=tq_ops::routed_projection(*tq.up,routed_input,selected.at(0));
                    auto hidden=expert_activation(f.activation,std::move(gate),std::move(up),f.swiglu_limit);
                    auto pairs=tq_ops::routed_projection(*tq.down,hidden,selected.at(0));
                    auto reduced=moe_weighted_reduce_cuda(pairs,selected.at(1));
                    experts = {finish_latent(std::move(reduced)),true};
                }
            } else if (dense_routed) {
                if (f.moe_ungated_experts &&
                    (bool(dense_gate) || (moe_gate_up && (moe_gate_up->n_experts > 0 || moe_gate_up->cached_source)) ||
                     (moe_gate && moe_gate->n_experts > 0) ||
                     (moe_up && (moe_up->n_experts > 0 || moe_up->cached_source))))
                    throw std::runtime_error(
                        "ungated routed MoE cannot mix a gate with a dense expert bank");
                TYLOINFER_RUNTIME_CHECK(bool(dense_up) && bool(dense_down),
                                  "dense routed MoE requires Up and Down banks");
                auto expert_input = xf;
                if (f.moe_latent_in)
                    expert_input = profiler.measure("moe.latent_down", [&] {
                        return project_.run(*f.moe_latent_in, xf);
                    });
                auto up = profiler.measure("moe.dense_up", [&] {
                    return dense_routed_projection(*dense_up, expert_input,
                                                   selected.at(0));
                });
                DenseTensor gate;
                if (!f.moe_ungated_experts) {
                    TYLOINFER_RUNTIME_CHECK(bool(dense_gate),
                                      "gated dense routed MoE requires a Gate bank");
                    gate = profiler.measure("moe.dense_gate", [&] {
                        return dense_routed_projection(*dense_gate, expert_input,
                                                       selected.at(0));
                    });
                }
                auto hidden = profiler.measure("moe.dense_activation", [&] {
                    return expert_activation(f.activation, std::move(gate),
                                                   std::move(up), f.swiglu_limit);
                });
                auto down_pair = profiler.measure("moe.dense_down", [&] {
                    return dense_routed_projection(*dense_down, hidden, selected.at(0));
                });
                auto reduced = profiler.measure("moe.dense_reduce", [&] {
                    return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                });
                if (f.moe_latent_out)
                    reduced = profiler.measure("moe.latent_up", [&] {
                        return project_.run(*f.moe_latent_out, reduced);
                    });
                stats.record(f.moe_layer, selected.at(0), selected.at(1),
                             down_pair, static_cast<int>(dense_up->size(0)));
                experts = {reduced, true};
            }
#if TYLOINFER_ENABLE_MFQ_COMPAT
            else if (f.moe_ungated_experts) {
                // relu-squared experts store Up and Down only: one Up matmul,
                // no gate projection, and the shared expert stays outside
                // the routed latent space.
                if ((moe_gate && (moe_gate->n_experts > 0 || moe_gate->cached_source)) ||
                    (moe_gate_up && (moe_gate_up->n_experts > 0 || moe_gate_up->cached_source)))
                    throw std::runtime_error(
                        "ungated routed MoE cannot carry a gate projection");
                TYLOINFER_RUNTIME_CHECK(moe_up && moe_up->n_experts > 0,
                                  "ungated routed MoE requires an Up bank");
                if (!projection_bundle_prefetched) {
                    tyloi::cuda::prefetch_experts(*moe_up, route);
                    if (moe_down) tyloi::cuda::prefetch_experts(*moe_down, route);
                }
                auto active_up = moe_up;
                if (cpu_moe_up) {
                    active_up = profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                        return stage_cpu_mixed_moe(cpu_moe_up, projection_.moe.heterogeneous);
                    });
                }
                auto up = profiler.measure("moe.up", [&]() {
                    return project_.run(*active_up, routed_input, route);
                });
                auto hidden = profiler.measure("moe.relu_squared", [&]() {
                    return relu_squared_cuda(up.contiguous());
                });
                active_up.reset();
                auto active_down = moe_down;
                if (cpu_moe_down) {
                    active_down = profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(cpu_moe_down, projection_.moe.heterogeneous);
                    });
                }
                auto down_pair = profiler.measure("moe.down", [&] {
                    TYLOINFER_RUNTIME_CHECK(active_down, "MFE input must be contiguous CUDA f16 with exact K");
                    return project_.run(*active_down, hidden, route);
                });
                stats.record(f.moe_layer, selected.at(0),
                             selected.at(1), down_pair, moe_up->n_experts);
                active_down.reset();
                if (latent_routed)
                    experts = {finish_latent(profiler.measure("moe.reduce", [&] {
                                return moe_weighted_reduce_cuda(down_pair, selected.at(1));
                            })),
                            true};
                else
                    experts = {down_pair, false};
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
                auto active_gate_up = moe_gate_up;
                auto active_gate = moe_gate;
                auto active_up = moe_up;
                ::tyloi::DenseTensor gate_up_pair;
                ::tyloi::DenseTensor projected_hidden;
                if (f.moe_split_gate_up) {
                    if (cpu_moe_gate) {
                        active_gate = profiler.measure("moe.cpu_offload_gate_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate, projection_.moe.heterogeneous);
                        });
                    }
                    if (cpu_moe_up) {
                        active_up = profiler.measure("moe.cpu_offload_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_up, projection_.moe.heterogeneous);
                        });
                    }
                    gate_up_pair = profiler.measure("moe.gate_up_split", [&]() {
                        const bool reuse_gate_activation =
                            mode_.reuse_input && mmq_.mode == KlMmqMode::Default && xf.size(0) <= 8 &&
                            active_gate && active_up && can_reuse_activation_for(*active_gate, *active_up);
                        TYLOINFER_RUNTIME_CHECK(active_gate, "MFE input must be contiguous CUDA f16 with exact K");
                        auto gate = project_.run(*active_gate, xf, route);
                        if (!projection_bundle_prefetched) {
                            if (active_up) tyloi::cuda::prefetch_experts(*active_up, route);
                        }
                        if (!f.moe_shared_ungated && !shared_gate_logits.defined()) {
                            shared_gate_logits = profiler.measure("moe.shared_gate", [&]() {
                                return ::tyloi::cuda::matmul(xf32,
                                                                  dynamic_cast<const DenseTensor&>(*f.moe_shared_gate).transpose(0, 1))
                                    .contiguous();
                            });
                        }
                        TYLOINFER_RUNTIME_CHECK(active_up, "MFE input must be contiguous CUDA f16 with exact K");
                        auto up = reuse_gate_activation
                                      ? project_.run(*active_up, xf, route, true)
                                      : project_.run(*active_up, xf, route);
                        return ::tyloi::cuda::cat({gate, up}, -1).contiguous();
                    });
                } else {
                    if (cpu_moe_gate_up) {
                        active_gate_up = profiler.measure("moe.cpu_offload_gate_up_h2d", [&]() {
                            return stage_cpu_mixed_moe(cpu_moe_gate_up, projection_.moe.heterogeneous);
                        });
                    }
                    const bool fuse_projection_glu =
                        f.swiglu_limit <= 0.0 && !mode_.materialized && active_gate_up &&
                        supports_projection_glu_epilogue(*active_gate_up);
                    if (fuse_projection_glu) {
                        projected_hidden = profiler.measure("moe.gate_up_swiglu", [&]() {
                            return project_.glu(*active_gate_up, xf, route, false);
                        });
                    } else {
                        gate_up_pair = profiler.measure("moe.gate_up", [&]() {
                            TYLOINFER_RUNTIME_CHECK(active_gate_up, "MFE input must be contiguous CUDA f16 with exact K");
                            return project_.run(*active_gate_up, xf, route);
                        });
                    }
                }
                active_gate_up.reset();
                active_gate.reset();
                active_up.reset();
                if (!projection_bundle_prefetched) {
                    if (moe_down) tyloi::cuda::prefetch_experts(*moe_down, route);
                }
                auto active_down = moe_down;
                if (cpu_moe_down) {
                    active_down = profiler.measure("moe.cpu_offload_down_h2d", [&]() {
                        return stage_cpu_mixed_moe(cpu_moe_down, projection_.moe.heterogeneous);
                    });
                }
                DenseTensor down_pair;
                if (projected_hidden.defined()) {
                    down_pair = profiler.measure("moe.down", [&] {
                        TYLOINFER_RUNTIME_CHECK(active_down, "MFE input must be contiguous CUDA f16 with exact K");
                        return project_.run(*active_down, projected_hidden, route);
                    });
                } else {
                    const double limit = f.swiglu_limit;
                    const bool allow_fusion = !mode_.materialized && options_.swiglu_fusion &&
                        (gate_up_pair.size(0) == 1 ||
                         (gate_up_pair.size(0) <= 4 && options_.small_heterogeneous));
                    if (allow_fusion && limit <= 0.0) {
                        down_pair = profiler.measure("moe.swiglu_down", [&] {
                            TYLOINFER_RUNTIME_CHECK(active_down, "fused MFE GLU input has an unsupported layout");
                            return project_.gated(*active_down, gate_up_pair, route, false);
                        });
                    } else if (allow_fusion && active_down && supports_clamped_swiglu(*active_down)) {
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
                            TYLOINFER_RUNTIME_CHECK(active_down, "MFE input must be contiguous CUDA f16 with exact K");
                            return project_.run(*active_down, hidden, route);
                        });
                    }
                }
                stats.record(f.moe_layer, selected.at(0), selected.at(1), down_pair,
                    f.moe_split_gate_up ? moe_gate->n_experts : moe_gate_up->n_experts);
                active_down.reset();
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
    return local(*mlp, xh, executor, project_);
}

#ifdef TYLOINFER_NATIVE_CUDA
DenseTensor Ffn::forward(DenseTensor x, MfqOptional<DenseTensor> input_ids,
                    MfqOptional<DenseTensor> router_bias) const {
    auto& executor = Executor::current();
    const auto& mode = Executor::current_ffn();
#else
DenseTensor Ffn::forward(DenseTensor x, Executor& executor, const Mode& mode,
                    MfqOptional<DenseTensor> input_ids,
                    MfqOptional<DenseTensor> router_bias) const {
#endif
    const auto& f = *this;
    if (router_bias.has_value()) {
        const auto& bias = router_bias.value();
        TYLOINFER_RUNTIME_CHECK(f.is_moe && !f.moe_hash_ids && f.moe_router && bias.defined() &&
            ((bias.dim()==1 && bias.size(0)==f.moe_router->size(0)) ||
             (bias.dim()==2 && bias.size(0)==x.numel()/x.size(-1) && bias.size(1)==f.moe_router->size(0))),
                          "MoE router correction bias shape mismatch");
    }
    const auto project = executor.projector(mode);
    return forward_impl(std::move(x), executor, mode, project, std::move(input_ids),
                        true, std::move(router_bias));
}

} // namespace tyloi::cuda

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::cuda::ops {

tyloi::DenseTensor mlp(tyloi::DenseTensor input, std::span<const std::shared_ptr<const tyloi::Tensor>> gate_up,
                const ::tyloi::Tensor& down, tyloi::ops::MlpActivation activation,
                double swiglu_limit) {
    std::optional<MfqCudaGuard> guard;
    if (input.is_cuda()) guard.emplace(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    const Mlp weights{project.group(gate_up), down, activation,
                      activation == tyloi::ops::MlpActivation::gelu_gated, swiglu_limit};
    weights.check(DenseTensor(input), project.primary_device());
    auto x = half(DenseTensor(input), executor.profiler());
    return weights.parallel() ? sharded(weights, x, executor, project)
                              : local(weights, x, executor, project);
}

tyloi::ops::SelectedExperts topk(tyloi::DenseTensor logits, std::int64_t count,
                               const tyloi::ops::MoeRouting& routing, const tyloi::DenseTensor* bias) {
    MfqCudaGuard guard(logits.device().index);
    MfqOptional<DenseTensor> selection_bias = mfq_nullopt;
    if (bias) selection_bias = *bias;
    auto result = moe_topk_cuda(logits.dense().contiguous(), count,
        routing.activation == tyloi::ops::RouterActivation::sigmoid,
        routing.activation == tyloi::ops::RouterActivation::sqrt_softplus,
        routing.normalize, routing.delayed_softmax, selection_bias, 1e-20, routing.scale);
    return {std::make_shared<DenseTensor>(result.at(0)), std::make_shared<DenseTensor>(result.at(1))};
}
tyloi::DenseTensor expert_scale(tyloi::DenseTensor weights, tyloi::DenseTensor ids, const tyloi::DenseTensor& scales) {
    MfqCudaGuard guard(weights.device().index);
    return moe_apply_expert_scale_cuda(DenseTensor(weights), DenseTensor(ids), DenseTensor(scales));
}
tyloi::DenseTensor weighted_reduce(tyloi::DenseTensor pairs, tyloi::DenseTensor weights) {
    MfqCudaGuard guard(pairs.device().index);
    return moe_weighted_reduce_cuda(DenseTensor(pairs), DenseTensor(weights));
}
#if TYLOINFER_ENABLE_MFQ_COMPAT
static weights::Placement geglu_placement(const MfeTensor& weight) {
    weights::Placement result;
    if (!weight.shards.empty()) {
        std::int64_t end = 0;
        for (const auto& shard : weight.shards) {
            if (!shard.weight || shard.device < 0 || shard.expert_begin != end ||
                shard.expert_end <= end || shard.expert_end > weight.n_experts ||
                shard.weight->size(0) != weight.size(0) || shard.weight->size(1) != weight.size(1) ||
                shard.weight->size(2) != weight.size(2))
                throw std::invalid_argument("invalid expert-parallel MFE shard");
            check_placement(geglu_placement(*shard.weight), Device(DeviceType::cuda, shard.device));
            result.devices.emplace_back(DeviceType::cuda, shard.device);
            end = shard.expert_end;
        }
        if (end != weight.n_experts) throw std::invalid_argument("incomplete expert-parallel MFE shards");
    } else if (weight.cached_source) return weight.cached_source->placement();
    else {
        const auto add_payload = [&](const Tensor& payload) {
            for (const auto* buffer : payload.buffers()) {
                if (!buffer || !buffer->defined()) continue;
                if (!buffer->is_cuda()) throw std::invalid_argument("expert weight is not prepared for CUDA execution");
                if (std::find(result.devices.begin(), result.devices.end(), buffer->device()) == result.devices.end())
                    result.devices.push_back(buffer->device());
            }
        };
        if (weight.compact) for (const auto& pool : *weight.compact) add_payload(pool.weight);
        for (const auto& pool : weight.pools) if (const auto* payload = pool.payload()) add_payload(*payload);
        if (result.devices.empty()) throw std::invalid_argument("expert weight is not prepared");
        if (result.devices.size() != 1) throw std::invalid_argument("expert weight fields are on different devices");
    }
    return result;
}

static bool supports_glu(const MfeTensor& weight) {
    if (!weight.shards.empty()) return std::all_of(weight.shards.begin(), weight.shards.end(),
        [](const auto& shard) { return shard.weight && supports_glu(*shard.weight); });
    return (weight.compact && !weight.compact->empty()) ||
        (weight.cached_source ? weight.cached_source->source_tensor().nint_only() : weight.nint_only());
}
#endif

tyloi::ops::RoutedGeglu routed_geglu(tyloi::DenseTensor input, tyloi::DenseTensor ids,
    const ::tyloi::Tensor& gate_bank, const ::tyloi::Tensor& down_bank, bool trace) {
    const auto* dense_gate = dynamic_cast<const DenseTensor*>(&gate_bank);
    const auto* dense_down = dynamic_cast<const DenseTensor*>(&down_bank);
    const auto* tq_gate = dynamic_cast<const TqBank*>(&gate_bank);
    const auto* tq_down = dynamic_cast<const TqBank*>(&down_bank);
#if TYLOINFER_ENABLE_MFQ_COMPAT
    const auto* gate = dynamic_cast<const MfeTensor*>(&gate_bank);
    const auto* down = dynamic_cast<const MfeTensor*>(&down_bank);
    if (gate) check_placement(geglu_placement(*gate), input.device());
    if (down) check_placement(geglu_placement(*down), input.device());
#else
    constexpr const MfeTensor* gate = nullptr;
    constexpr const MfeTensor* down = nullptr;
#endif
    if ((!dense_gate && !tq_gate && !gate) || (!dense_down && !tq_down && !down))
        throw std::invalid_argument("unsupported routed GEGLU expert tensor layout");
    for (const auto* bank : {dense_gate, dense_down}) {
        if (!bank) continue;
        if (bank->is_encoded() || !bank->is_contiguous() || bank->device() != input.device() ||
            (bank->scalar_type() != kFloat16 && bank->scalar_type() != kBFloat16 &&
             bank->scalar_type() != kFloat32))
            throw std::invalid_argument("dense expert bank must be contiguous CUDA F16/BF16/F32 on the input device");
    }
    for (const auto* bank : {tq_gate, tq_down}) {
        if (bank && (!bank->defined() || !bank->slices.empty()))
            throw std::invalid_argument("routed GEGLU does not support TQ expert-parallel slices");
    }
    if (gate_bank.size(0) > INT_MAX || gate_bank.size(1) > INT_MAX ||
        gate_bank.size(2) > INT_MAX || down_bank.size(1) > INT_MAX ||
        ids.size(0) > INT_MAX / ids.size(1) ||
        ((tq_gate || tq_down) && ids.size(1) > 65535))
        throw std::invalid_argument("routed GEGLU geometry exceeds CUDA limits");
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if ((gate && gate->shards.empty() && gate->partial_experts) ||
        (down && down->shards.empty() && down->partial_experts))
        throw std::invalid_argument("routed GEGLU requires complete expert banks");
#endif
    MfqCudaGuard guard(input.device().index);
    auto& executor = Executor::current();
    const auto project = executor.projector(Executor::current_ffn());
    auto& profiler = executor.profiler();
    input = input.contiguous();
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if ((gate && !gate->shards.empty()) || (down && !down->shards.empty())) {
        if (project.primary_device() != input.device().index)
            throw std::invalid_argument("expert-parallel projection must reduce to the input device");
    }
    if (gate && down) {
        auto route = profiler.measure("gemma.route_map", [&] {
            return build_moe_route_plan(DenseTensor(ids), gate->n_experts);
        });
        const bool prefetched = gate->cached_source && down->cached_source &&
            MoeCachedSource::prefetch_bundle({gate->cached_source, down->cached_source}, route);
        auto prefetch_down = [&] { if (!prefetched && down->cached_source) down->cached_source->prefetch(route); };
        tyloi::ops::RoutedGeglu result;
        if (!trace && supports_glu(*gate)) {
            auto hidden = profiler.measure("gemma.moe_gate_up_geglu", [&] {
                return project.glu(*gate, DenseTensor(input), route, true);
            });
            prefetch_down();
            result.pairs = std::make_shared<DenseTensor>(profiler.measure("gemma.moe_down", [&] {
                return project.run(*down, hidden, route);
            }));
            return result;
        }
        auto packed = profiler.measure("gemma.moe_gate_up", [&] {
            return project.run(*gate, DenseTensor(input), route);
        });
        prefetch_down();
        if (trace) result.gate_up = std::make_shared<DenseTensor>(packed.to(::tyloi::kFloat32).contiguous().clone());
        if (!trace && packed.size(0) <= 4) {
            result.pairs = std::make_shared<DenseTensor>(profiler.measure("gemma.moe_geglu_down", [&] {
                return project.gated(*down, packed, route, true);
            }));
            return result;
        }
        auto hidden = profiler.measure("gemma.moe_geglu", [&] { return moe_geglu_split_cuda(packed); });
        if (trace) result.hidden = std::make_shared<DenseTensor>(hidden.to(::tyloi::kFloat32).contiguous().clone());
        result.pairs = std::make_shared<DenseTensor>(profiler.measure("gemma.moe_down", [&] { return project.run(*down, hidden, route); }));
        return result;
    }
#endif
    tyloi::ops::RoutedGeglu result;
    if (tq_gate || tq_down) {
        const auto host_ids = ids.contiguous().cpu().to(kInt64);
        const auto* selected = host_ids.data_ptr<std::int64_t>();
        for (std::int64_t i = 0; i < host_ids.numel(); ++i)
            if (selected[i] < 0 || selected[i] >= gate_bank.size(0))
                throw std::out_of_range("routed GEGLU expert ID is out of bounds");
    }
    DenseTensor packed;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    std::optional<MoeRoutePlan> route;
    if (gate || down) route.emplace(profiler.measure("gemma.route_map", [&] {
        return build_moe_route_plan(DenseTensor(ids), static_cast<int>(gate_bank.size(0)));
    }));
    if (gate) packed = project.run(*gate, input, *route);
    else
#endif
    if (dense_gate) packed = dense_routed_projection(*dense_gate, input, ids);
    else packed = tq_ops::routed_projection(*tq_gate, input, ids);
    if (trace) result.gate_up = std::make_shared<DenseTensor>(packed.to(kFloat32).contiguous().clone());
    auto hidden = moe_geglu_split_cuda(packed.to(kFloat16).contiguous());
    if (trace) result.hidden = std::make_shared<DenseTensor>(hidden.to(kFloat32).contiguous().clone());
    DenseTensor pairs;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (down) pairs = project.run(*down, hidden, *route);
    else
#endif
    if (dense_down) pairs = dense_routed_projection(*dense_down, hidden, ids);
    else pairs = tq_ops::routed_projection(*tq_down, hidden, ids);
    result.pairs = std::make_shared<DenseTensor>(std::move(pairs));
    return result;
}

} // namespace tyloi::cuda::ops
#endif
