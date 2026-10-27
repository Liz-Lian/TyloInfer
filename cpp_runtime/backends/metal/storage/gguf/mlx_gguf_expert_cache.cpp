#include "mlx_gguf_expert_cache.h"
#include "mlx_gguf_prefill.h"
#include "mfq_container.h"
#include "mlx_expert_segment_cache.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe_ops.h"

#include <algorithm>
#include <array>
#include <climits>

namespace mfq::metal {
namespace {
using namespace mlx::core;
struct Projection {
    std::string dtype;
    Shape shape;
    int stride;
};
using Layer = std::array<Projection, 3>;

array ffn(const std::array<MlxGgufWeight, 3>& w, const array& input,
    const array& ids, const array& weights, bool float_accumulation = false) {
    if (gguf_expert_prefill_available(input)) {
        const MlxGgufPrefillPlan plan(input, ids, w[0].expert_count());
        return plan.down(w[2], plan.gate_up(w[0], w[1]), weights, float_accumulation);
    }
    std::optional<MlxGgufRoutePlan> plan;
    if (ids.shape(0) > 6) plan = gguf_route_plan(ids, w[0].expert_count());
    const auto* routes = plan ? &*plan : nullptr;
    auto hidden = w[0].routed_swiglu(w[1], input, ids, routes);
    auto pairs = w[2].routed_matmul(hidden, ids, false, routes);
    return moe_weighted_reduce(float_accumulation ? astype(pairs, float32) : pairs, weights);
}
}

struct MlxGgufExpertCache::Impl {
    std::vector<Layer> layers;
    std::vector<std::optional<std::array<MlxGgufWeight, 3>>> weights;
    std::unique_ptr<MlxExpertSegmentCache> cache;

    Impl(const MfqContainer& model, const std::vector<std::string>& prefixes, std::size_t budget, int workers) {
        std::vector<MlxExpertSegments> segments;
        for (const auto& prefix : prefixes) {
            Layer layer;
            MlxExpertSegments definition;
            for (int p = 0; p < 3; ++p) {
                const auto name = prefix + ".mlp.experts." + std::array{"gate", "up", "down"}[p] + ".weight";
                auto& projection = layer[p];
                projection.dtype = model.record(name).dtype;
                const auto shape = model.tensor_shape(name);
                if (!is_gguf_dtype(projection.dtype) || shape.size() != 3 ||
                    std::any_of(shape.begin(), shape.end(), [](auto d) { return d < 1 || d > INT_MAX; }))
                    throw std::invalid_argument("GGUF expert cache requires native rank-three tensors");
                for (const auto d : shape) projection.shape.push_back(int(d));
                const auto stride = gguf_packed_row_bytes(projection.dtype, projection.shape[2]);
                if (stride > INT_MAX || std::size_t(shape[1]) > SIZE_MAX / stride)
                    throw std::overflow_error("GGUF expert extent is too large");
                projection.stride = int(stride);
                const auto bytes = std::size_t(shape[1]) * stride;
                if (bytes > SIZE_MAX / shape[0] || bytes * shape[0] != model.record(name).nbytes)
                    throw std::invalid_argument("GGUF expert extent differs from tensor geometry");
                definition.experts = int(shape[0]); definition.fields.push_back({name, 0, bytes});
                model.record_prepared(name);
            }
            if (layer[0].shape != layer[1].shape || layer[2].shape !=
                Shape{layer[0].shape[0], layer[0].shape[2], layer[0].shape[1]})
                throw std::invalid_argument("GGUF expert Gate/Up/Down geometry differs");
            layers.push_back(std::move(layer)); segments.push_back(std::move(definition));
        }
        cache = std::make_unique<MlxExpertSegmentCache>(model, std::move(segments), budget, workers, true);
        weights.resize(layers.size());
    }
    const std::array<MlxGgufWeight, 3>& layer_weights(std::size_t index) {
        if (!weights[index]) {
            const auto& layer = layers[index]; const auto& tables = cache->address_tables(index);
            weights[index] = std::array{
                MlxGgufWeight::from_address_table(layer[0].dtype, layer[0].shape, tables[0], cache->residency()),
                MlxGgufWeight::from_address_table(layer[1].dtype, layer[1].shape, tables[1], cache->residency()),
                MlxGgufWeight::from_address_table(layer[2].dtype, layer[2].shape, tables[2], cache->residency())};
        }
        return *weights[index];
    }
};

MlxGgufExpertCache::MlxGgufExpertCache(const MfqContainer& model,
    const std::vector<std::string>& prefixes, std::size_t budget, int workers)
    : impl_(std::make_shared<Impl>(model, prefixes, budget, workers)) {}

void MlxGgufExpertCache::validate_layer(std::size_t layer, int experts, int input, int intermediate) const {
    if (impl_->layers.at(layer)[0].shape != Shape{experts, intermediate, input})
        throw std::invalid_argument("GGUF expert cache layer does not match model geometry");
}
void MlxGgufExpertCache::prepare_layer(std::size_t index, int top_k) const {
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation) return;
    const auto& layer = impl_->layers.at(index);
    if (top_k < 1 || top_k > layer[0].shape[0]) throw std::invalid_argument("invalid GGUF preparation route count");
    std::array<std::vector<array>, 3> values;
    for (int p = 0; p < 3; ++p) values[p].push_back(zeros({layer[p].shape[1], layer[p].stride}, uint8));
    eval(values[0][0], values[1][0], values[2][0]);
    const std::vector<std::int32_t> selected{0};
    std::array<MlxGgufWeight, 3> weights{
        MlxGgufWeight::from_experts(layer[0].dtype, layer[0].shape, selected, values[0]),
        MlxGgufWeight::from_experts(layer[1].dtype, layer[1].shape, selected, values[1]),
        MlxGgufWeight::from_experts(layer[2].dtype, layer[2].shape, selected, values[2])};
    preparation->collect([&] {
        for (const auto rows : preparation->row_buckets())
            preparation->add(ffn(weights, zeros({rows, layer[0].shape[2]}, float16),
                zeros({rows, top_k}, int32), zeros({rows, top_k}, float32)));
    });
}
array MlxGgufExpertCache::apply_ffn(std::size_t layer_index, const array& input,
    const array& ids, const array& route_weights, const array* parallel) {
    const auto& layer = impl_->layers.at(layer_index);
    if (ids.ndim() != 2 || route_weights.shape() != ids.shape() || input.ndim() == 0 ||
        input.shape(-1) != layer[0].shape[2] || input.size() / layer[0].shape[2] != std::size_t(ids.shape(0)))
        throw std::invalid_argument("GGUF expert FFN inputs and routes disagree");
    std::optional<MlxGgufRoutePlan> plan;
    std::optional<MlxGgufPrefillPlan> prefill;
    auto output = impl_->cache->apply(layer_index, ids, 2, [&] {
        const auto& weights = impl_->layer_weights(layer_index);
        if (gguf_expert_prefill_available(input)) {
            if (!prefill) prefill.emplace(input, ids, weights[0].expert_count());
            return prefill->gate_up(weights[0], weights[1]);
        }
        if (ids.shape(0) > 6) plan = gguf_route_plan(ids, weights[0].expert_count());
        return weights[0].routed_swiglu(weights[1], input, ids, plan ? &*plan : nullptr);
    }, [&](const array& hidden, bool partial) {
        if (prefill) return prefill->down(impl_->layer_weights(layer_index)[2], hidden, route_weights, partial);
        auto pairs = impl_->layer_weights(layer_index)[2].routed_matmul(hidden, ids, false, plan ? &*plan : nullptr);
        return moe_weighted_reduce(partial ? astype(pairs, float32) : pairs, route_weights);
    }, parallel);
    return output.dtype() == input.dtype() ? output : astype(output, input.dtype());
}
std::size_t MlxGgufExpertCache::limit_bytes() const noexcept { return impl_->cache->limit_bytes(); }
std::size_t MlxGgufExpertCache::backing_bytes() const noexcept { return impl_->cache->backing_bytes(); }
std::size_t MlxGgufExpertCache::resident_bytes() const noexcept { return impl_->cache->resident_bytes(); }
std::size_t MlxGgufExpertCache::set_limit(std::size_t bytes) { return impl_->cache->set_limit(bytes); }
MlxSsdExpertCacheStats MlxGgufExpertCache::stats() const { return impl_->cache->stats(); }
void MlxGgufExpertCache::clear() { impl_->cache->clear(); }
} // namespace mfq::metal
