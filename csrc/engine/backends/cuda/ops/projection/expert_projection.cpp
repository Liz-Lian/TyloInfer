#include "projection.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/backends/cuda/executor/tq_moe.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/backends/cuda/native/collectives.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/profile.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#include "engine/ops.h"
#endif

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/backends/cuda/ops/ffn/mfe.h"
#include "engine/weights/moe/cached_source_internal.h"
#endif

#include <stdexcept>
#include <utility>

namespace tyloi::cuda {

DenseTensor Projector::run(const ::tyloi::Tensor& weight, DenseTensor input,
                           const MoeRoutePlan& route) const {
    const auto* mfe = weight.representation<MfeTensor>();
    if (!mfe) throw std::invalid_argument("unsupported expert tensor layout");
    return run(*mfe, std::move(input), route);
}

#if TYLOINFER_ENABLE_MFQ_COMPAT
namespace {

std::shared_ptr<const void> mixed_source(const MfeTensor& weight) {
    if (weight.nint_dispatch) return weight.nint_dispatch;
    if (weight.nvq_dispatch) return weight.nvq_dispatch;
    if (!weight.pools.empty() && weight.pools.front().expert_local.storage())
        return weight.pools.front().expert_local.storage();
    throw std::runtime_error("mixed MFE has no backing identity");
}

} // namespace
#endif

DenseTensor Projector::run(const MfeTensor& weight, DenseTensor input,
                           const MoeRoutePlan& route, bool prequantized, int epilogue) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        if (prequantized)
            throw std::runtime_error("prequantized MoE activation reuse is unavailable with expert parallelism");
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = run(*shard.weight, std::move(local), local_route, false, epilogue);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (epilogue < 0 || epilogue > 2 || (epilogue && weight.out_per_expert % 2))
        throw std::runtime_error("invalid MFE NINT projection epilogue");
    if (!weight.cached_source && (weight.compact || weight.pools.empty()))
        return Mfe{weight.n_experts, weight.out_per_expert, weight.neuron_len,
            weight.partial_experts, weight.compact}.forward(mfe_scratch_, input, route, prequantized, epilogue);
    if (epilogue) throw std::runtime_error(prequantized
        ? "mixed prequantized MFE GLU epilogue is unavailable"
        : "mixed MFE GLU epilogue must use its native dispatch");
    const auto options = ::tyloi::cuda::options::moe(options_.moe);
    if (weight.cached_source) {
        auto scratch = moe_scratch_.get(weight.cached_source, input);
        if (prequantized)
            return weight.cached_source->forward_prequantized(
                *scratch, options, mmq_, options.heterogeneous, input, route);
        return weight.cached_source->forward(*scratch, options, mmq_, options.heterogeneous, input, route);
    }
    return mixed_project(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route, prequantized);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::glu(const MfeTensor& weight, DenseTensor input,
                           const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = glu(*shard.weight, std::move(local), local_route, gelu);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (!weight.cached_source && (weight.compact || weight.pools.empty())) return run(weight, input, route, false, gelu ? 2 : 1);
    const auto options = ::tyloi::cuda::options::moe(options_.moe);
    if (weight.cached_source)
        return weight.cached_source->forward_glu_output(*moe_scratch_.get(weight.cached_source, input),
            options, mmq_, options.heterogeneous, input, route, gelu);
    return mixed_glu(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route, gelu);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::gated(const MfeTensor& weight, DenseTensor input,
                             const MoeRoutePlan& route, bool gelu) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = gated(*shard.weight, std::move(local), local_route, gelu);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (weight.cached_source) {
        const auto options = ::tyloi::cuda::options::moe(options_.moe);
        return weight.cached_source->forward_glu(*moe_scratch_.get(weight.cached_source, input),
            options, mmq_, options.heterogeneous, input, route, gelu);
    }
    if (!input.is_cuda() || !input.is_contiguous() || input.scalar_type() != ::tyloi::kFloat16 ||
        input.dim() != 3 || input.size(2) != 2 * weight.neuron_len)
        throw std::runtime_error("fused MFE GLU input has an unsupported layout");
    auto hidden = gelu ? moe_geglu_split_cuda(input) : moe_swiglu_split_cuda(input);
    return run(weight, hidden, route);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::clamped(const MfeTensor& weight, DenseTensor input,
                               const MoeRoutePlan& route, double limit) const {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (!weight.shards.empty()) {
        std::vector<DenseTensor> outputs(weight.shards.size());
        for (size_t launch = 0; launch < weight.shards.size(); ++launch) {
            const auto index = parallel_index(options_.peer_first, launch, weight.shards.size());
            const auto& shard = weight.shards[index];
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            MfqCudaGuard guard(shard.device);
            auto local = input.defined() ? collectives_.copy(input, shard.device) : input;
            const auto& local_route = moe_route_to_device(collectives_, route, shard.device);
            outputs[index] = clamped(*shard.weight, std::move(local), local_route, limit);
        }
        return collectives_.sum(std::move(outputs), primary_, options_.reduce_to_primary, options_.fp16_reduce);
    }
    if (weight.cached_source && weight.cached_source->supports_clamped_swiglu()) {
        const auto options = ::tyloi::cuda::options::moe(options_.moe);
        return weight.cached_source->forward_clamped_swiglu(
            *moe_scratch_.get(weight.cached_source, input), options, mmq_, options.heterogeneous, input, route, limit);
    }
    if (!weight.compact && !weight.cached_source && supports_clamp(weight)) {
        const auto options = ::tyloi::cuda::options::moe(options_.moe);
        return mixed_clamp(weight, *moe_scratch_.get(mixed_source(weight), input), options, mmq_, input, route, limit);
    }
    return Mfe{weight.n_experts, weight.out_per_expert, weight.neuron_len,
        weight.partial_experts, weight.compact}.clamped(mfe_scratch_, input, route, limit);
#else
    throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
#endif
}

DenseTensor Projector::run(const TqMoeBanks& banks, DenseTensor input, DenseTensor ids,
                      DenseTensor weights, float limit) const {
    if (banks.slices.empty())
        return banks.forward(tq_, std::move(input), std::move(ids), std::move(weights),
                             limit, &profiler_);
    if (!banks.gate)
        throw std::invalid_argument("TQ expert parallelism requires a Gate bank");
    std::vector<DenseTensor> partials(banks.slices.size());
    for (size_t launch = 0; launch < banks.slices.size(); ++launch) {
        const auto index = parallel_index(options_.peer_first, launch, banks.slices.size());
        const auto& slice = banks.slices[index];
        const MfqCudaGuard guard(slice.device);
        auto local_input = collectives_.copy(input, slice.device);
        auto local_ids = collectives_.copy(ids, slice.device);
        auto local_weights = collectives_.copy(weights, slice.device);
        partials[index] = banks.forward(tq_, std::move(local_input), std::move(local_ids),
            std::move(local_weights), limit, &profiler_, slice.begin, slice.end);
    }
    return collectives_.sum(std::move(partials), primary_,
                            options_.reduce_to_primary, options_.fp16_reduce);
}

} // namespace tyloi::cuda

#ifdef TYLOINFER_NATIVE_CUDA
namespace tyloi::ops {
namespace {

void validate_expert_inputs(const Tensor& input, const Tensor& ids,
                            const Tensor& weights,
                            std::span<const Tensor> gate_up,
                            const Tensor& down, std::int64_t width) {
    TYLOINFER_RUNTIME_CHECK(gate_up.size() >= 1 && gate_up.size() <= 2 && down.defined() &&
        input.defined() && ids.defined() && weights.defined() && width > 0 &&
        input.dim() == 2 && ids.dim() == 2 && weights.dim() == 2 &&
        input.size(0) == ids.size(0) && ids.size(0) == weights.size(0) &&
        ids.size(1) == weights.size(1) && input.device() == ids.device() &&
        input.device() == weights.device(), "invalid routed expert inputs");
    for (const auto& weight : gate_up)
        TYLOINFER_RUNTIME_CHECK(weight.defined(), "missing routed expert projection");
}

} // namespace
} // namespace tyloi::ops

tyloi::Tensor tyloi::ops::experts(const Tensor& input, const Tensor& ids, const Tensor& weights,
                                  std::span<const Tensor> gate_up, const Tensor& down, std::int64_t width) {
    validate_expert_inputs(input, ids, weights, gate_up, down, width);
    auto& executor = cuda::Executor::current();
    const auto project = executor.projector(cuda::Executor::current_ffn());
    auto banks = tq_banks(gate_up.empty() ? nullptr : gate_up.front(),
                          gate_up.empty() ? nullptr : gate_up.back(), down);
    if (banks.gate || banks.up || banks.down) {
        TYLOINFER_RUNTIME_CHECK(gate_up.size() == 2 && down && banks.gate && banks.up && banks.down,
                                "TQ MoE requires separate Gate, Up and Down banks");
        return executor.profiler().measure("tq.moe", [&] {
            return project.run(banks, input.dense(), ids.dense(), weights.dense());
        });
    }
    DenseTensor hidden;
    {
        auto gu = project.run(gate_up[0], input.dense(), ids.dense());
        if (gate_up.size() == 2) {
            auto up = project.run(gate_up[1], input.dense(), ids.dense());
            gu = cuda::cat({gu, up}, -1);
        }
        auto gate = gu.narrow(-1, 0, width), up = gu.narrow(-1, width, width);
        hidden = (gate * cuda::sigmoid(gate)) * up;
    }
    auto pairs = project.run(down, hidden, ids.dense());
    hidden = {};
    auto reduced = cuda::zeros({input.size(0), input.size(1)},
                               input.options().dtype(kFloat32));
    for (int64_t r = 0; r < ids.size(1); ++r)
        reduced = reduced + pairs.select(1, r).to(kFloat32) *
                            weights.dense().select(1, r).unsqueeze(-1);
    return reduced.to(pairs.scalar_type());
}

tyloi::Tensor tyloi::ops::moe(const Tensor& input, const Tensor& ids, const Tensor& weights,
                              std::span<const Tensor> gate_up, const Tensor& down,
                              std::span<const Tensor> shared, const Tensor& shared_gate,
                              std::int64_t width, std::int64_t shared_rows) {
    validate_expert_inputs(input, ids, weights, gate_up, down, width);
    TYLOINFER_RUNTIME_CHECK(shared.size() == 3 && shared_gate.defined(),
                            "invalid shared expert inputs");
    for (const auto& weight : shared)
        TYLOINFER_RUNTIME_CHECK(weight.defined(), "missing shared expert projection");

    const auto shared_output = cuda::MoeSharedExpert(
        shared[0], shared[1], shared[2], shared_gate).forward(input.dense(), shared_rows);
    const auto routed = experts(input, ids, weights, gate_up, down, width).dense();
    return routed + shared_output;
}
#endif
