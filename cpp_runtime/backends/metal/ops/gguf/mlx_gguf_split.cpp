#include "mlx_gguf_staged.h"
#include "mlx_cached_kernel.h"
#include "mlx_kernel_prepare.h"
#include "../../platform/metal_device_info.h"

#include <mlx/allocator.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <climits>

namespace mfq::metal {
namespace {
using namespace mlx::core;
constexpr int kDeviceScratchElements = 64;

using WorkspaceSizes = std::array<std::pair<std::size_t, int>, 4>;

class SplitWorkspaceKernel final : public Primitive, public MlxPreparableKernel {
public:
    SplitWorkspaceKernel(std::shared_ptr<Primitive> kernel, WorkspaceSizes sizes)
        : Primitive(kernel->stream()), kernel_(std::move(kernel)), sizes_(sizes) {}
    const char* name() const override { return "GgufSplitWorkspace"; }
    std::string preparation_key() const override { return mlx_kernel_preparation_key(*kernel_); }
    void prepare_gpu() override { prepare_mlx_kernel_gpu(*kernel_); }
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("GGUF split projection requires Metal");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto checked = inputs;
        std::vector<array> workspaces;
        workspaces.reserve(8);
        for (int i = 0; i < 4; ++i) {
            const auto [elements, counters] = sizes_[i];
            if (elements) {
                checked[i * 5 + 3] = array(allocator::malloc(elements * sizeof(float)), Shape{int(elements)}, float32);
                checked[i * 5 + 4] = array(allocator::malloc(std::size_t(counters) * sizeof(uint32_t)), Shape{counters}, uint32);
                std::fill_n(checked[i * 5 + 4].data<uint32_t>(), counters, 0u);
                workspaces.push_back(checked[i * 5 + 3]);
                workspaces.push_back(checked[i * 5 + 4]);
            } else {
                checked[i * 5 + 3] = checked[(i - 1) * 5 + 3];
                checked[i * 5 + 4] = checked[(i - 1) * 5 + 4];
            }
        }
        kernel_->eval_gpu(checked, outputs);
        mlx::core::metal::get_command_encoder(stream()).add_temporaries(std::move(workspaces));
    }
private:
    std::shared_ptr<Primitive> kernel_;
    WorkspaceSizes sizes_;
};

int partition_count(int outputs, int width, int cores) {
    const auto tiles = std::size_t(outputs / 64);
    int splits = 1;
    while (splits < 8 && (width / 32) % (2 * splits) == 0 &&
        tiles * splits < std::size_t(cores) * 6 && width / (2 * splits) >= 512)
        splits *= 2;
    return splits;
}

const fast::CustomKernelFunction& split_kernel(const std::string& header) {
    static const auto kernel = cached_metal_kernel("tinfer_gguf_planes_split_projection",
        {"p00", "p01", "m0", "s0", "c0", "p10", "p11", "m1", "s1", "c1",
            "p20", "p21", "m2", "s2", "c2", "p30", "p31", "m3", "s3", "c3", "x", "params"},
        {"out0", "out1", "out2", "out3"}, R"METAL(
uint index = threadgroup_position_in_grid.x;
const uint t0 = params[1] / 64u * params[5], t1 = params[2] / 64u * params[6], t2 = params[3] / 64u * params[7];
threadgroup uint table[256], arrival;
const uint tid = thread_position_in_threadgroup.x;
if (index < t0) {
    load_staged_lookup<F0>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    plane_split<F0, ROWS>(p00, p01, m0, s0, c0, x, out0, table, &arrival, params[0], params[1], params[5], params[9], index, tid);
} else if ((index -= t0) < t1) {
    load_staged_lookup<F1>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    plane_split<F1, ROWS>(p10, p11, m1, s1, c1, x, out1, table, &arrival, params[0], params[2], params[6], params[9], index, tid);
} else if ((index -= t1) < t2) {
    load_staged_lookup<F2>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    plane_split<F2, ROWS>(p20, p21, m2, s2, c2, x, out2, table, &arrival, params[0], params[3], params[7], params[9], index, tid);
} else {
    index -= t2;
    load_staged_lookup<F3>(table, tid, 64u); threadgroup_barrier(mem_flags::mem_threadgroup);
    plane_split<F3, ROWS>(p30, p31, m3, s3, c3, x, out3, table, &arrival, params[0], params[4], params[8], params[9], index, tid);
}
)METAL", header + R"METAL(
template<int F, typename T, class Auxiliary, class Scales>
void plane_split_one(device const uchar* p0, Auxiliary p1, Scales meta,
    device const float* scratch, device const uint* counters, device const T* x, device T* out,
    threadgroup uint* table, threadgroup uint* arrival, uint k, uint n, uint splits, uint index, uint tid) {
    const uint tile = index % (n / 64u), split = index / (n / 64u), column = tile * 64u + tid;
    const uint groups = k / 32u, per = groups / splits, begin = split * per, end = begin + per;
    GroupPayload payload;
    fetch_group<F>(payload, p0, p1, meta, column, groups, begin, true);
    float sum = 0.f;
    for (uint g = begin; g < end; ++g) {
        decode_group_chunks<F, float>(payload, staged_lookup<F>(table), g,
            [&](uint c, float4 low, float4 high) __attribute__((always_inline)) {
                auto input = x + g * 32u + c * 4u;
                const float4 a(input[0], input[1], input[2], input[3]);
                const float4 b(input[16], input[17], input[18], input[19]);
                sum += dot(low, a) + dot(high, b);
            });
        if (g + 1u < end) fetch_group<F>(payload, p0, p1, meta, column, groups, g + 1u,
            (g + 1u) / PlaneLayout<F>::meta_groups != g / PlaneLayout<F>::meta_groups);
    }
    if (splits == 1u) { out[column] = T(sum); return; }
    auto published = reinterpret_cast<device coherent(device) float*>(const_cast<device float*>(scratch));
    auto counter = reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(counters)) + tile;
    published[ulong(split) * n + column] = sum;
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (*arrival != splits - 1u) return;
    float total = 0.f;
    for (uint s = 0u; s < splits; ++s) total += s == split ? sum : published[ulong(s) * n + column];
    out[column] = T(total);
    if (tid == 0u) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

template<int F, int Rows, typename T, class Auxiliary, class Scales>
void plane_split(device const uchar* p0, Auxiliary p1, Scales meta,
    device const float* scratch, device const uint* counters, device const T* x, device T* out,
    threadgroup uint* table, threadgroup uint* arrival, uint k, uint n, uint splits, uint rows, uint index, uint tid) {
    if constexpr (Rows == 1) {
        plane_split_one<F>(p0, p1, meta, scratch, counters, x, out, table, arrival, k, n, splits, index, tid);
    } else {
    const uint tile = index % (n / 64u), split = index / (n / 64u), column = tile * 64u + tid;
    const uint groups = k / 32u, per = groups / splits, begin = split * per, end = begin + per;
    GroupPayload payload;
    fetch_group<F>(payload, p0, p1, meta, column, groups, begin, true);
    float sums[Rows] = {0.f};
    for (uint g = begin; g < end; ++g) {
        decode_group_chunks<F, float>(payload, staged_lookup<F>(table), g,
            [&](uint c, float4 low, float4 high) __attribute__((always_inline)) {
                #pragma clang loop unroll(full)
                for (uint row = 0u; row < uint(Rows); ++row) {
                    if constexpr (Rows > 1) if (row >= rows) break;
                    auto input = x + ulong(row) * k + g * 32u + c * 4u;
                    const float4 a(input[0], input[1], input[2], input[3]);
                    const float4 b(input[16], input[17], input[18], input[19]);
                    sums[row] += dot(low, a) + dot(high, b);
                }
            });
        if (g + 1u < end) fetch_group<F>(payload, p0, p1, meta, column, groups, g + 1u,
            (g + 1u) / PlaneLayout<F>::meta_groups != g / PlaneLayout<F>::meta_groups);
    }
    if (splits == 1u) {
        #pragma clang loop unroll(full)
        for (uint row = 0u; row < uint(Rows); ++row) {
            if constexpr (Rows > 1) if (row >= rows) break;
            out[ulong(row) * n + column] = T(sums[row]);
        }
        return;
    }
    auto published = reinterpret_cast<device coherent(device) float*>(const_cast<device float*>(scratch));
    auto counter = reinterpret_cast<device atomic_uint*>(const_cast<device uint*>(counters)) + tile;
    #pragma clang loop unroll(full)
    for (uint row = 0u; row < uint(Rows); ++row) {
        if constexpr (Rows > 1) if (row >= rows) break;
        published[(ulong(row) * splits + split) * n + column] = sums[row];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        *arrival = atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    if (*arrival != splits - 1u) return;
    #pragma clang loop unroll(full)
    for (uint row = 0u; row < uint(Rows); ++row) {
        if constexpr (Rows > 1) if (row >= rows) break;
        float total = 0.f;
        for (uint s = 0u; s < splits; ++s)
            total += s == split ? sums[row] : published[(ulong(row) * splits + s) * n + column];
        out[ulong(row) * n + column] = T(total);
    }
    if (tid == 0u) atomic_store_explicit(counter, 0u, memory_order_relaxed);
    }
}
)METAL");
    return kernel;
}
}

std::optional<std::vector<array>> MlxGgufStagedPrefill::project_split_many(
    std::span<const MlxGgufWeight> weights, const array& input) {
    if (weights.empty() || weights.size() > 4 || input.ndim() < 1 ||
        input.shape(-1) != weights.front().input_size()) return std::nullopt;
    const auto rows = input.size() / input.shape(-1);
    if (rows < 1 || rows > 8) return std::nullopt;
    const int cores = metal_apple_gpu_core_count();
    if (cores == 0) return std::nullopt;
    bool partitioned = false;
    for (const auto& weight : weights) {
        if (!weight.planes_ || weight.shape_.size() != 2 || weight.input_size() != input.shape(-1)) return std::nullopt;
        partitioned |= partition_count(weight.output_size(), weight.input_size(), cores) > 1;
    }
    if (!partitioned) return std::nullopt;
    std::vector<array> inputs;
    std::vector<Shape> shapes;
    std::vector<std::pair<std::string, fast::TemplateArg>> arguments;
    std::vector<uint32_t> params{uint32_t(input.shape(-1))}, splits;
    WorkspaceSizes workspace_sizes{};
    static const array scratch = [] {
        const std::array<float, kDeviceScratchElements> values{};
        return array(values.data(), Shape{kDeviceScratchElements}, float32);
    }();
    static const array counters = [] {
        const std::array<uint32_t, kDeviceScratchElements> values{};
        return array(values.data(), Shape{kDeviceScratchElements}, uint32);
    }();
    inputs.reserve(22); shapes.reserve(4); arguments.reserve(5); params.reserve(10); splits.reserve(4);
    std::size_t tiles = 0;
    for (int i = 0; i < 4; ++i) {
        const auto& weight = weights[std::min(std::size_t(i), weights.size() - 1)];
        const int n = i < weights.size() ? weight.output_size() : 0;
        const int s = n ? partition_count(n, weight.input_size(), cores) : 1;
        const auto count = s > 1 ? std::max(std::size_t(kDeviceScratchElements), std::size_t(n) * s * rows)
            : std::size_t(kDeviceScratchElements);
        if (count > INT_MAX) throw std::overflow_error("GGUF split projection scratch exceeds the array limit");
        const int counter_count = s > 1 ? std::max(kDeviceScratchElements, n / 64) : kDeviceScratchElements;
        if (i < weights.size()) workspace_sizes[i] = {count, counter_count};
        inputs.insert(inputs.end(), {weight.bytes_, (*weight.planes_)[0], (*weight.planes_)[1],
            scratch, counters});
        auto shape = input.shape(); shape.back() = n;
        shapes.push_back(n ? std::move(shape) : Shape{1});
        params.push_back(uint32_t(n)); splits.push_back(uint32_t(s));
        tiles += std::size_t(n / 64) * s;
        arguments.emplace_back("F" + std::to_string(i), weight.format_);
    }
    if (tiles > INT_MAX / 64) throw std::overflow_error("GGUF split projection grid is too large");
    params.insert(params.end(), splits.begin(), splits.end());
    params.push_back(uint32_t(rows));
    arguments.emplace_back("ROWS", rows == 1 ? 1 : rows <= 6 ? 6 : 8);
    inputs.push_back(contiguous(input)); inputs.push_back(array(params.begin(), Shape{10}));
    auto outputs = split_kernel(decoder_header())(inputs, shapes, std::vector<Dtype>(4, input.dtype()),
        {int(tiles * 64), 1, 1}, {64, 1, 1}, arguments, std::nullopt, false, {});
    auto primitive = std::make_shared<SplitWorkspaceKernel>(outputs.front().primitive_ptr(), workspace_sizes);
    for (auto& output : outputs) output.primitive_ptr() = primitive;
    outputs.erase(outputs.begin() + weights.size(), outputs.end());
    return outputs;
}
}
