#include "mlx_gguf_expert_cache.h"
#include "mlx_gguf_prefill.h"
#include "mfq_container.h"
#include "mlx_expert_segment_cache.h"
#include "mlx_kernel_prepare.h"
#include "mlx_moe_ops.h"
#include "mlx_weight_residency.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>

namespace mfq::metal {
namespace {
using namespace mlx::core;
struct Projection {
    std::string dtype;
    std::string name;
    Shape shape;
    int stride = 0;
    std::vector<std::string> expert_dtypes;
};
using Layer = std::array<Projection, 3>;

array ffn(const std::array<MlxGgufWeight, 3>& w, const array& input,
    const array& ids, const array& weights, bool float_accumulation = false, float limit = 0.f) {
    if (limit == 0.f && gguf_expert_prefill_available(input) && !gguf_short_sparse_routes(ids, w[0].expert_count()) &&
        !w[0].has_mixed_formats() && !w[1].has_mixed_formats() && !w[2].has_mixed_formats() &&
        w[0].same_rotation(w[1])) {
        const MlxGgufPrefillPlan plan(input, ids, w[0].expert_count());
        return plan.down(w[2], plan.gate_up(w[0], w[1]), weights, float_accumulation);
    }
    std::optional<MlxGgufRoutePlan> plan;
    if (ids.shape(0) > 6) plan = gguf_route_plan(ids, w[0].expert_count(),
        w[0].has_mixed_formats() && w[1].has_mixed_formats() && w[2].has_mixed_formats());
    const auto* routes = plan ? &*plan : nullptr;
    auto hidden = w[0].routed_swiglu(w[1], input, ids, routes, limit);
    if (!float_accumulation) return w[2].routed_weighted_matmul(hidden, ids, weights, routes);
    auto pairs = w[2].routed_matmul(hidden, ids, false, routes);
    return moe_weighted_reduce(float_accumulation ? astype(pairs, float32) : pairs, weights);
}
}

struct MlxGgufExpertCache::Impl {
    MfqContainer model;
    std::vector<Layer> layers;
    std::vector<std::optional<std::array<MlxGgufWeight, 3>>> weights;
    std::unique_ptr<MlxExpertSegmentCache> cache;

    Impl(const MfqContainer& source, const std::vector<std::string>& prefixes, std::size_t budget, int workers) : model(source) {
        std::vector<MlxExpertSegments> segments;
        const auto mixed = model.header().extra_json.find("gguf_mixed_experts");
        const auto mixed_layouts = mixed == model.header().extra_json.end() ? nlohmann::json::object()
            : nlohmann::json::parse(mixed->second);
        for (const auto& prefix : prefixes) {
            Layer layer;
            MlxExpertSegments definition;
            for (int p = 0; p < 3; ++p) {
                const auto name = prefix + ".mlp.experts." + std::array{"gate", "up", "down"}[p] + ".weight";
                auto& projection = layer[p];
                projection.name = name;
                projection.dtype = model.record(name).dtype;
                const auto shape = model.tensor_shape(name);
                if (!is_gguf_dtype(projection.dtype) || shape.size() != 3 ||
                    std::any_of(shape.begin(), shape.end(), [](auto d) { return d < 1 || d > INT_MAX; }))
                    throw std::invalid_argument("GGUF expert cache requires native rank-three tensors");
                for (const auto d : shape) projection.shape.push_back(int(d));
                if (projection.dtype == "GGUF_MIXED") {
                    const auto& entries = mixed_layouts.at(name);
                    if (!entries.is_array() || entries.size() != std::size_t(shape[0]))
                        throw std::invalid_argument("GGUF mixed cache expert count differs");
                    MlxExpertSegment field{name};
                    std::size_t next = 0;
                    for (const auto& entry : entries) {
                        const auto dtype = entry.at("dtype").get<std::string>();
                        const auto offset = entry.at("offset").get<std::uint64_t>();
                        const auto bytes = entry.at("nbytes").get<std::size_t>();
                        const auto stride = gguf_packed_row_bytes(dtype, projection.shape[2]);
                        if (offset != next || offset > model.record(name).nbytes || bytes > model.record(name).nbytes - offset ||
                            stride > INT_MAX || bytes != std::size_t(shape[1]) * stride)
                            throw std::invalid_argument("GGUF mixed cache expert byte range differs");
                        field.expert_offsets.push_back(offset); field.expert_sizes.push_back(bytes);
                        projection.expert_dtypes.push_back(dtype); next += bytes;
                    }
                    if (next != model.record(name).nbytes) throw std::invalid_argument("GGUF mixed cache has trailing bytes");
                    definition.experts = int(shape[0]); definition.fields.push_back(std::move(field));
                    model.record_prepared(name);
                    continue;
                }
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
            const auto load = [&](int p) {
                auto weight = layer[p].expert_dtypes.empty()
                    ? MlxGgufWeight::from_address_table(layer[p].dtype, layer[p].shape, tables[p], cache->residency())
                    : MlxGgufWeight::from_mixed_address_table(layer[p].shape, layer[p].expert_dtypes, tables[p], cache->residency());
                weight.configure_hadamard(model, layer[p].name);
                return weight;
            };
            weights[index] = std::array{
                load(0), load(1), load(2)};
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
    prepare_layer(index, top_k, 0.f);
}
void MlxGgufExpertCache::prepare_layer(std::size_t index, int top_k, float limit) const {
    if (!std::isfinite(limit) || limit < 0.f) throw std::invalid_argument("invalid GGUF SwiGLU limit");
    auto* preparation = MlxKernelPreparation::current();
    if (!preparation) return;
    const auto& layer = impl_->layers.at(index);
    if (top_k < 1 || top_k > layer[0].shape[0]) throw std::invalid_argument("invalid GGUF preparation route count");
    if (std::any_of(layer.begin(), layer.end(), [](const Projection& p) { return !p.expert_dtypes.empty(); })) {
        const auto metadata_weight = [&](int p) {
            auto table = zeros({layer[p].shape[0]}, uint64);
            table.eval();
            const auto bytes = static_cast<MTL::Buffer*>(table.buffer().ptr())->allocatedSize();
            std::shared_ptr<MlxWeightResidency> residency = MlxWeightResidency::retain_arrays({table}, bytes, true);
            auto weight = layer[p].expert_dtypes.empty()
                ? MlxGgufWeight::from_address_table(layer[p].dtype, layer[p].shape, table, std::move(residency))
                : MlxGgufWeight::from_mixed_address_table(layer[p].shape, layer[p].expert_dtypes, table, std::move(residency));
            weight.configure_hadamard(impl_->model, layer[p].name);
            return weight;
        };
        const std::array weights{metadata_weight(0), metadata_weight(1), metadata_weight(2)};
        preparation->collect([&] {
            for (const auto rows : preparation->row_buckets())
                for (const bool partial : {false, true})
                    preparation->add(ffn(weights, zeros({rows, layer[0].shape[2]}, float16),
                        zeros({rows, top_k}, int32), zeros({rows, top_k}, float32), partial, limit));
        });
        return;
    }
    std::array<std::vector<array>, 3> values;
    for (int p = 0; p < 3; ++p) values[p].push_back(zeros({layer[p].shape[1], layer[p].stride}, uint8));
    eval(values[0][0], values[1][0], values[2][0]);
    const std::vector<std::int32_t> selected{0};
    std::array<MlxGgufWeight, 3> weights{
        MlxGgufWeight::from_experts(layer[0].dtype, layer[0].shape, selected, values[0]),
        MlxGgufWeight::from_experts(layer[1].dtype, layer[1].shape, selected, values[1]),
        MlxGgufWeight::from_experts(layer[2].dtype, layer[2].shape, selected, values[2])};
    for (int p = 0; p < 3; ++p) weights[p].configure_hadamard(impl_->model, layer[p].name);
    preparation->collect([&] {
        for (const auto rows : preparation->row_buckets())
            for (const bool partial : {false, true})
                preparation->add(ffn(weights, zeros({rows, layer[0].shape[2]}, float16),
                    zeros({rows, top_k}, int32), zeros({rows, top_k}, float32), partial, limit));
    });
}
array MlxGgufExpertCache::apply_ffn(std::size_t layer_index, const array& input,
    const array& ids, const array& route_weights, const array* parallel) {
    return apply_ffn(layer_index, input, ids, route_weights, parallel, 0.f);
}
array MlxGgufExpertCache::apply_ffn(std::size_t layer_index, const array& input,
    const array& ids, const array& route_weights, const array* parallel, float limit) {
    if (!std::isfinite(limit) || limit < 0.f) throw std::invalid_argument("invalid GGUF SwiGLU limit");
    const auto& layer = impl_->layers.at(layer_index);
    if (ids.ndim() != 2 || route_weights.shape() != ids.shape() || input.ndim() == 0 ||
        input.shape(-1) != layer[0].shape[2] || input.size() / layer[0].shape[2] != std::size_t(ids.shape(0)))
        throw std::invalid_argument("GGUF expert FFN inputs and routes disagree");
    std::optional<MlxGgufRoutePlan> plan;
    std::optional<MlxGgufPrefillPlan> prefill;
    auto output = impl_->cache->apply(layer_index, ids, 2, [&] {
        const auto& weights = impl_->layer_weights(layer_index);
        if (limit == 0.f && gguf_expert_prefill_available(input) && !gguf_short_sparse_routes(ids, weights[0].expert_count()) && !weights[0].has_mixed_formats() &&
            !weights[1].has_mixed_formats() && !weights[2].has_mixed_formats() &&
            weights[0].same_rotation(weights[1])) {
            if (!prefill) prefill.emplace(input, ids, weights[0].expert_count());
            return prefill->gate_up(weights[0], weights[1]);
        }
        if (!plan && ids.shape(0) > 6) plan = gguf_route_plan(ids, weights[0].expert_count(),
            weights[0].has_mixed_formats() && weights[1].has_mixed_formats() && weights[2].has_mixed_formats());
        return weights[0].routed_swiglu(weights[1], input, ids, plan ? &*plan : nullptr, limit);
    }, [&](const array& hidden, bool partial) {
        if (prefill) return prefill->down(impl_->layer_weights(layer_index)[2], hidden, route_weights, partial);
        if (!partial) return impl_->layer_weights(layer_index)[2].routed_weighted_matmul(hidden, ids, route_weights,
            plan ? &*plan : nullptr);
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
