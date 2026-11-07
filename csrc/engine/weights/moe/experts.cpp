#include "experts.h"
#include "engine/weights/encoding/mfe.h"
#include "engine/weights/moe/dispatch.h"
#include "engine/weights/moe/prepare.h"
#include "engine/weights/encoding/format_features.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "cached_source_internal.h"
#include "engine/backends/cuda/ops/ffn/moe.h"
#endif

#include <algorithm>
#include <limits>
#include <stdexcept>

bool tyloi::MfeTensor::nint_only() const {
    return !pools.empty() && std::all_of(pools.begin(), pools.end(),
        [](const MixedMoePool& pool) { return pool.family == MixedMoeFamily::Nint; });
}

bool tyloi::MfeTensor::defined() const noexcept {
    if (n_experts <= 0 || out_per_expert <= 0 || neuron_len <= 0) return false;
    if (cached_source) return true;
    if (!shards.empty()) {
        for (const auto& shard : shards) if (!shard.weight || !shard.weight->defined()) return false;
        return true;
    }
    if (compact) {
        if (compact->empty()) return false;
        for (const auto& pool : *compact)
            if (!pool.weight.defined() || !pool.expert_local.defined()) return false;
        return true;
    }
    if (pools.empty()) return false;
    for (const auto& pool : pools) {
        const auto* payload = pool.payload();
        if (!(payload ? payload->defined() : pool.nepq.indices_packed.defined()) ||
            !pool.expert_local.defined()) return false;
    }
    return true;
}

std::int64_t tyloi::MfeTensor::numel() const {
    TensorView view;
    view.rank = 3;
    view.sizes[0] = n_experts;
    view.sizes[1] = out_per_expert;
    view.sizes[2] = neuron_len;
    return view.numel();
}

std::size_t tyloi::MfeTensor::nbytes() const {
    const auto elements = numel();
    if (static_cast<std::uint64_t>(elements) > std::numeric_limits<std::size_t>::max() / 2)
        throw std::overflow_error("tensor byte count overflow");
    return static_cast<std::size_t>(elements) * 2;
}

std::size_t tyloi::MfeTensor::backing_nbytes() const {
    return backing_bytes(buffers());
}

std::int64_t tyloi::MfeTensor::size(std::int64_t dimension) const {
    if (dimension < 0) dimension += 3;
    switch (dimension) {
        case 0: return n_experts;
        case 1: return out_per_expert;
        case 2: return neuron_len;
        default: throw std::out_of_range("tensor dimension is out of range");
    }
}

const tyloi::Device& tyloi::MfeTensor::device() const noexcept {
    if (cached_source) return source_device;
    if (!shards.empty() && shards.front().weight) return shards.front().weight->device();
    if (compact && !compact->empty()) return compact->front().expert_local.device();
    for (const auto& pool : pools) {
        const auto* payload = pool.payload();
        if (payload && payload->defined()) return payload->device();
        if (!payload && pool.nepq.indices_packed.defined()) return pool.nepq.indices_packed.device();
    }
    static const Device unprepared{};
    return unprepared;
}

std::vector<const tyloi::DenseTensor*> tyloi::MfeTensor::buffers() const {
    std::vector<const DenseTensor*> result;
    if (compact) for (const auto& pool : *compact) {
        auto values = pool.weight.buffers();
        result.insert(result.end(), values.begin(), values.end());
        result.push_back(&pool.expert_local);
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (cached_source) {
        auto values = cached_source->source_tensor().buffers();
        result.insert(result.end(), values.begin(), values.end());
    }
#endif
    for (const auto& shard : shards) if (shard.weight) {
        auto values = shard.weight->buffers();
        result.insert(result.end(), values.begin(), values.end());
    }
    for (const auto& pool : pools) {
        if (const auto* payload = pool.payload()) {
            auto values = payload->buffers();
            result.insert(result.end(), values.begin(), values.end());
        } else {
            auto values = nepq_buffers(pool.nepq);
            result.insert(result.end(), values.begin(), values.end());
        }
        result.push_back(&pool.expert_local);
    }
    if (nvq_dispatch) {
        const auto& d = *nvq_dispatch;
        result.insert(result.end(), {&d.metadata_device, &d.metadata_host, &d.map_ptrs,
            &d.weight_ptrs, &d.weight_sizes, &d.pool_params,
            &d.initial_expert_pool, &d.initial_expert_local});
    }
    if (nint_dispatch) {
        const auto& d = *nint_dispatch;
        result.insert(result.end(), {&d.metadata_device, &d.metadata_host, &d.map_ptrs,
            &d.pool_params, &d.initial_expert_pool, &d.initial_expert_local});
    }
    return result;
}

void tyloi::MfeTensor::record_stream(std::uintptr_t stream) const {
    int cuda_device = -1;
    for (const auto& shard : shards) {
        if (cuda_device >= 0 && cuda_device != shard.device)
            throw std::invalid_argument("MFE stream recording across CUDA devices is unsupported");
        cuda_device = shard.device;
    }
    for (const auto* buffer : buffers()) if (buffer && buffer->defined() && buffer->is_cuda()) {
        if (cuda_device >= 0 && cuda_device != buffer->get_device())
            throw std::invalid_argument("MFE stream recording across CUDA devices is unsupported");
        cuda_device = buffer->get_device();
    }
    for (const auto* buffer : buffers())
        if (buffer && buffer->defined()) buffer->record_stream(stream);
}

std::shared_ptr<const tyloi::Tensor> tyloi::MfeTensor::relocated(Device target) const {
    if (!defined())
        throw std::invalid_argument("MFE relocation before preparation is unsupported");
    auto moved = std::make_shared<MfeTensor>(*this);
    if (device() == target) return moved;
    if (nvq_dispatch || nint_dispatch || cached_source || !shards.empty())
        throw std::invalid_argument("MFE relocation with dispatch, cached source, or shards is unsupported");
    if (compact) moved->compact = std::make_shared<std::vector<MfePoolWeight>>(*compact);
    for (const auto* buffer : moved->buffers()) {
        auto& writable = const_cast<DenseTensor&>(*buffer);
        if (writable.defined() && writable.device() != target) writable = writable.to(target);
    }
    return moved;
}

namespace tyloi::cuda {

bool supports_projection_glu_epilogue(const MfeTensor& weight) {
    if (!weight.shards.empty()) {
        return std::all_of(weight.shards.begin(), weight.shards.end(), [](const auto& shard) {
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            return supports_projection_glu_epilogue(*shard.weight);
        });
    }
    if (weight.compact) return true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.cached_source) return weight.cached_source->source_tensor().nint_only();
    if (!weight.pools.empty()) return weight.nint_only();
#endif
    return false;
}

static bool supports_prequantized_input(const MfeTensor& weight) {
    if (!weight.shards.empty()) return false;
    if ((weight.compact && !weight.compact->empty()) || weight.cached_source) return true;
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return !weight.compact && !weight.pools.empty();
#else
    return false;
#endif
}

bool can_reuse_activation_for(const MfeTensor& producer, const MfeTensor& consumer) {
    if (!supports_prequantized_input(producer) || !supports_prequantized_input(consumer) ||
        producer.neuron_len != consumer.neuron_len || bool(producer.compact) != bool(consumer.compact))
        return false;
    const bool shared = (producer.cached_source && producer.cached_source == consumer.cached_source) ||
        (!producer.cached_source && !consumer.cached_source && !producer.compact &&
         !consumer.compact && &producer == &consumer) ||
        (!producer.cached_source && !consumer.cached_source && producer.compact &&
         producer.compact == consumer.compact);
    if (!shared) return false;
    const auto geometry = [](const MfeTensor& weight) {
        std::vector<MoeActivationGeometry> result;
        if (weight.compact) {
            for (const auto& pool : *weight.compact) {
                MoeActivationGeometry value{static_cast<int>(pool.weight.ng), static_cast<int>(pool.weight.gs), 0, 0};
                if (std::find(result.begin(), result.end(), value) == result.end()) result.push_back(value);
            }
        }
#if TYLOINFER_ENABLE_MFQ_COMPAT
        else if (weight.cached_source || !weight.pools.empty()) {
            result = moe_activation_geometry(weight.cached_source ? weight.cached_source->source_tensor().pools : weight.pools);
        }
#endif
        return result;
    };
    const auto available = geometry(producer);
    const auto required = geometry(consumer);
    return !available.empty() && !required.empty() &&
        std::all_of(required.begin(), required.end(), [&](const auto& value) {
            return std::find(available.begin(), available.end(), value) != available.end();
        });
}

bool supports_clamped_swiglu(const MfeTensor& weight) {
    if (!weight.shards.empty()) {
        return std::all_of(weight.shards.begin(), weight.shards.end(), [](const auto& shard) {
            if (!shard.weight) throw std::runtime_error("expert-parallel MoE shard is missing");
            return supports_clamped_swiglu(*shard.weight);
        });
    }
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return (weight.compact && !weight.compact->empty()) ||
        (!weight.compact && !weight.cached_source && !weight.pools.empty() && supports_clamp(weight)) ||
        (weight.cached_source && weight.cached_source->supports_clamped_swiglu());
#else
    return weight.compact && !weight.compact->empty();
#endif
}

void prefetch_experts(const MfeTensor& weight, const MoeRoutePlan& route, bool begin) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    if (weight.cached_source) {
        if (begin) weight.cached_source->begin_prefetch(route);
        else weight.cached_source->prefetch(route);
    }
#endif
}

} // namespace tyloi::cuda
