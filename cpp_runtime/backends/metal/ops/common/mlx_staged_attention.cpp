#include "mlx_staged_attention.h"
#include <mlx/primitives.h>
#include "mlx_cached_kernel.h"
#include "mlx_platform.h"
#include "ops/gguf/native/gguf_sources.h"

#include <mlx/backend/metal/device.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace mfq::metal {
namespace {
using namespace mlx::core;

class MlxAttentionKvRows final : public Primitive {
public:
    explicit MlxAttentionKvRows(const array& value)
        : Primitive(default_stream(Device::gpu)), copy_(contiguous(value).primitive_ptr()) {}
    const char* name() const override { return "MlxAttentionKvRows"; }
    void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        if (compatible(inputs.front())) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_cpu(inputs, outputs);
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        if (compatible(inputs.front())) outputs.front().copy_shared_buffer(inputs.front());
        else copy_->eval_gpu(inputs, outputs);
    }
private:
    bool compatible(const array& value) const {
        return value.strides()[3] == 1 && value.strides()[2] == 256 &&
            value.strides()[1] >= 0 && value.strides()[1] <= std::numeric_limits<int>::max();
    }
    std::shared_ptr<Primitive> copy_;
};

array attention_kv_rows(const array& value) {
    return array(value.shape(), value.dtype(), std::make_shared<MlxAttentionKvRows>(value), {value});
}

const fast::CustomKernelFunction& attention_pack_kv() {
    static const auto kernel = cached_metal_kernel("tyloinfer_gqa_pack_tail_kv_v1",
        {"keys", "values", "params"}, {"packed_keys", "packed_values"}, R"METAL(
const uint index = thread_position_in_grid.x, head = index / (32 * 256);
const uint flat = index % (32 * 256), dim = flat % 256;
const uint token = (params[0] - 1) * 32 + flat / 256;
Input k = Input(0), v = Input(0);
if (token < params[1]) {
    k = keys[ulong(head) * keys_strides[1] + token * keys_strides[2] + dim * keys_strides[3]];
    v = values[ulong(head) * values_strides[1] + token * values_strides[2] + dim * values_strides[3]];
}
packed_keys[index] = k; packed_values[index] = v;
)METAL", "", false, false, CompileOptions(static_cast<int>(MathMode::Fast)));
    return kernel;
}

const fast::CustomKernelFunction& attention_partial() {
    static const auto kernel = cached_metal_kernel("tyloinfer_gqa_staged_partial_v2",
        {"queries", "keys", "values", "tail_keys", "tail_values", "params"}, {"partials", "statistics"}, R"METAL(
constexpr uint M = 8 * GROUP, N = 32;
threadgroup float scores[M * N], row_max[M], row_sum[M], previous_scale[M];
threadgroup bfloat probabilities[M * N];
threadgroup atomic_uint rescale;
const uint head = threadgroup_position_in_grid.x, split = threadgroup_position_in_grid.y;
splash_paged_attention_tile<KH, GROUP, 8, 256, 0, Input>(
    const_cast<device Input*>(queries) + head * M * 256,
    const_cast<device Input*>(keys), const_cast<device Input*>(values),
    const_cast<device Input*>(tail_keys), const_cast<device Input*>(tail_values),
    keys_strides[1], values_strides[1],
    head, params[0], params[1], params[2], split, partials, statistics, head * params[2] + split,
    scores, probabilities, row_max, row_sum, previous_scale, &rescale, thread_index_in_threadgroup);
)METAL", native_gguf::staged_attention_source, false, false, CompileOptions(static_cast<int>(MathMode::Fast)));
    return kernel;
}

const fast::CustomKernelFunction& attention_reduce() {
    static const auto kernel = cached_metal_kernel("tyloinfer_gqa_staged_reduce_v1",
        {"partials", "statistics", "params"}, {"output"}, R"METAL(
threadgroup float weights[128], group_values[8];
constexpr uint M = 8 * GROUP;
const uint task = threadgroup_position_in_grid.x, head = task / M, row = task % M;
const uint pages = (params[0] + params[1] + 31) / 32;
const uint per_split = (pages + params[2] - 1) / params[2];
const uint written = (pages + per_split - 1) / per_split;
splash_attention_combine_splits<GROUP, 8, 256>(partials, statistics, output + head * M * 256,
    params[1], written, head * params[2], row, thread_index_in_threadgroup, weights, group_values);
)METAL", native_gguf::staged_attention_source, true, false, CompileOptions(static_cast<int>(MathMode::Fast)));
    return kernel;
}
}

std::optional<mlx::core::array> mlx_staged_attention(
    const mlx::core::array& query, const mlx::core::array& key,
    const mlx::core::array& value, bool causal, float scale,
    const std::optional<mlx::core::array>& mask) {
    using namespace mlx::core;
    if (!causal || mask || query.ndim() != 4 || key.ndim() != 4 || value.shape() != key.shape() ||
        query.shape(0) != 1 || key.shape(0) != 1 || query.shape(3) != 256 || key.shape(3) != 256 || key.shape(1) < 1 ||
        query.shape(1) % key.shape(1) || query.shape(2) < 1 || query.shape(2) > 8 || scale != 1.f / 16.f ||
        (query.dtype() != float16 && query.dtype() != bfloat16) || key.dtype() != query.dtype() || value.dtype() != query.dtype())
        return std::nullopt;
    const int rows = query.shape(2), heads = key.shape(1), group = query.shape(1) / heads, live = key.shape(2);
    if ((group != 6 && group != 8) || rows * group <= 32 || live < rows || mlx_metal_nax_disabled() ||
        !mlx::core::metal::device(Device::gpu).mtl_device()->supportsFamily(MTL::GPUFamilyApple10)) return std::nullopt;
    if (live > std::numeric_limits<int>::max() - 31) return std::nullopt;
    const int capacity = (live + 31) / 32 * 32, pages = capacity / 32;
    if (capacity > std::numeric_limits<int>::max() / 256 ||
        std::uint64_t(heads) * pages * 8 * 256 > std::uint64_t(std::numeric_limits<int>::max())) return std::nullopt;
    auto row_keys = attention_kv_rows(key), row_values = attention_kv_rows(value);
    auto staged_query = transpose(reshape(query, {heads, group, rows, 256}), {0, 2, 1, 3});
    if (rows < 8) staged_query = pad(staged_query, {{0, 0}, {0, 8 - rows}, {0, 0}, {0, 0}});
    else staged_query = contiguous(staged_query);
    auto packed = attention_pack_kv()({row_keys, row_values, array({std::uint32_t(pages), std::uint32_t(live)})},
        {{heads, 32, 256}, {heads, 32, 256}}, {query.dtype(), query.dtype()},
        {heads * 32 * 256, 1, 1}, {256, 1, 1}, {{"Input", query.dtype()}}, std::nullopt, false, {});
    const int splits = std::min(128, std::max(32, (pages + 15) / 16));
    auto params = array({std::uint32_t(live - rows), std::uint32_t(rows), std::uint32_t(splits)});
    auto partial = attention_partial()({staged_query, row_keys, row_values, packed[0], packed[1], params},
        {{heads * splits, 8 * group, 256}, {heads * splits, 8 * group, 2}}, {float32, float32},
        {heads * 256, splits, 1}, {256, 1, 1}, {{"Input", query.dtype()}, {"KH", heads}, {"GROUP", group}}, std::nullopt, false, {});
    auto result = attention_reduce()({partial[0], partial[1], params}, {{heads, 8, group, 256}}, {query.dtype()},
        {heads * 8 * group * 256, 1, 1}, {256, 1, 1}, {{"GROUP", group}}, std::nullopt, false, {})[0];
    return reshape(transpose(slice(result, {0, 0, 0, 0}, {heads, rows, group, 256}), {0, 2, 1, 3}), query.shape());
}
}
