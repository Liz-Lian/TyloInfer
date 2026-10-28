#include "mlx_native_gguf.h"
#include "gguf_sources.h"
#include "mlx_kernel_prepare.h"
#include "../../../platform/metal_device_info.h"
#include "mlx_platform.h"
#include "mlx_cached_kernel.h"
#include "ops/Linear.hpp"
#include "ops/MoE.hpp"
#include "ops/HyperConnection.hpp"
#include "metal/abi/GgufRepack.h"
#include "metal/abi/Gguf.h"
#include "mfq_container.h"
#include "mlx_weight_residency.h"
#include "ggml.h"

#include <mlx/allocator.h>
#include <mlx/backend/metal/utils.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <climits>
#include <cctype>
#include <cstring>
#include <memory>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace mfq::metal {
namespace {
using namespace mlx::core;
using splash::metal::MetalBuffer;
using splash::metal::CommandGraph;
using namespace splash::ops;

array row_major_view(const array& input) {
    return mlx_row_contiguous_view(input, default_stream(Device::gpu));
}

const char* source_for(const std::string& name) {
    if (name.starts_with("hyper_connection")) return native_gguf::hyper_source;
    if (name == "gguf_repack" || name == "moe_expert_load") return native_gguf::repack_source;
    if (name.starts_with("gguf_decode_sg_") || name == "decode_linear_gguf_prepare" || name.starts_with("moe_expert_gguf_sg"))
        return native_gguf::register_source;
    if (name.starts_with("gguf_float")) return native_gguf::float_source;
    if (name.starts_with("moe_expert_gguf") || name.starts_with("moe_affine_vector_") ||
        name.starts_with("moe_native_vector_") || name.starts_with("moe_native_single_") ||
        name == "moe_single_reduce") return native_gguf::moe_projection_source;
    if (name.starts_with("moe_")) return native_gguf::moe_source;
    if (name.starts_with("gguf_")) return native_gguf::linear_source;
    throw std::invalid_argument("unavailable native GGUF pipeline: " + name);
}

struct Pipeline {
    std::string name;
    MTL::ComputePipelineState* value = nullptr;
    std::mutex mutex;
    explicit Pipeline(std::string name) : name(std::move(name)) {}
    MTL::ComputePipelineState* prepare(Stream stream) {
        std::lock_guard lock(mutex);
        if (!value) {
            const char* source = source_for(name);
            auto& device = kernel_device(stream.device);
            const auto hash = std::hash<std::string_view>{}(source);
            static std::mutex libraries_mutex;
            static std::unordered_map<const char*, std::shared_ptr<std::mutex>> library_mutexes;
            std::shared_ptr<std::mutex> library_mutex;
            {
                std::lock_guard libraries_lock(libraries_mutex);
                auto& mutex = library_mutexes[source];
                if (!mutex) mutex = std::make_shared<std::mutex>();
                library_mutex = mutex;
            }
            MTL::Library* library;
            {
                std::lock_guard library_lock(*library_mutex);
                library = device.get_library("tinfer_native_gguf_" + std::to_string(hash),
                    CompileOptions(static_cast<int>(MathMode::Fast)), [&] { return std::string(source); });
            }
            value = device.get_kernel(name, library);
        }
        return value;
    }
};

std::shared_ptr<Pipeline> pipeline(const std::string& name) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::shared_ptr<Pipeline>> values;
    std::lock_guard lock(mutex);
    auto& value = values[name];
    if (!value) value = std::make_shared<Pipeline>(name);
    return value;
}

struct Graph {
    CommandGraph commands;
    std::vector<std::size_t> scratch;
    std::vector<bool> clear;
    std::vector<std::shared_ptr<Pipeline>> pipelines;
    std::size_t inputs = 0;
    std::uint64_t sparse_weight_inputs = 0;
    bool pooled = false;
    std::uint32_t direct_tile_bound = 0;
    int rows = 0;
    MetalBuffer temporary(std::size_t bytes, bool zero = false) {
        if (!bytes) return {};
        if (bytes > INT_MAX) throw std::overflow_error("native GGUF scratch exceeds array size");
        const std::size_t slot = inputs + 1 + scratch.size();
        scratch.push_back(bytes);
        clear.push_back(zero);
        return {slot, bytes};
    }
    void finish() {
        if (!pipelines.empty()) return;
        if (!commands.command().events.empty()) throw std::invalid_argument("native GGUF graph contains unsupported host/event steps");
        for (const auto& dispatch : commands.dispatches()) pipelines.push_back(pipeline(dispatch.pipelineName));
    }
};

struct Workspace {
    std::mutex mutex;
    std::vector<std::optional<array>> buffers;
    std::size_t retained = 0;
};

std::shared_ptr<Workspace> workspace(Stream stream) {
    struct Entry { std::shared_ptr<Workspace> buffers; std::uint64_t used; };
    static std::mutex mutex;
    static std::map<Stream, Entry> streams;
    static std::uint64_t clock = 0;
    std::lock_guard lock(mutex);
    if (auto found = streams.find(stream); found != streams.end()) {
        found->second.used = ++clock;
        return found->second.buffers;
    }
    if (streams.size() >= 4) {
        const auto oldest = std::min_element(streams.begin(), streams.end(),
            [](const auto& a, const auto& b) { return a.second.used < b.second.used; });
        streams.erase(oldest);
    }
    auto result = std::make_shared<Workspace>();
    streams.emplace(stream, Entry{result, ++clock});
    return result;
}

using GraphKey = std::vector<std::uint64_t>;

template <typename Build>
std::shared_ptr<Graph> cached_graph(GraphKey key, bool enabled, Build build) {
    if (!enabled) return build();
    struct Entry { std::shared_ptr<Graph> graph; std::uint64_t used; };
    static std::mutex mutex;
    static std::map<GraphKey, Entry> plans;
    static std::uint64_t clock = 0;
    std::lock_guard lock(mutex);
    if (auto found = plans.find(key); found != plans.end()) {
        found->second.used = ++clock;
        return found->second.graph;
    }
    auto graph = build(); graph->finish();
    if (plans.size() >= 128) {
        const auto oldest = std::min_element(plans.begin(), plans.end(),
            [](const auto& a, const auto& b) { return a.second.used < b.second.used; });
        plans.erase(oldest);
    }
    plans.emplace(std::move(key), Entry{graph, ++clock});
    return graph;
}

GraphKey graph_key(std::uint64_t kind, std::uint64_t rows,
    std::span<const MlxNativeGgufWeight> weights) {
    auto& gpu = mlx::core::metal::device(Device::gpu);
    GraphKey key{kind, rows, std::uint64_t(!mlx_metal_nax_disabled() &&
        gpu.mtl_device()->supportsFamily(MTL::GPUFamilyApple10)), std::uint64_t(metal_apple_gpu_core_count())};
    for (const auto& weight : weights) {
        key.insert(key.end(), {std::uint64_t(weight.format), std::uint64_t(weight.width),
            std::uint64_t(weight.outputs), std::uint64_t(weight.storage_outputs()), std::uint64_t(weight.experts)});
        for (const auto& plane : weight.planes) key.push_back(plane.nbytes());
    }
    return key;
}

class NativeGraph final : public Primitive, public MlxPreparableKernel {
public:
    NativeGraph(Stream stream, std::shared_ptr<Graph> graph) : Primitive(stream), graph_(std::move(graph)) {}
    const char* name() const override { return "MlxNativeGgufGraph"; }
    std::string preparation_key() const override {
        std::string key = "native_gguf";
        for (const auto& pipeline : graph_->pipelines) key += ':' + pipeline->name;
        return key;
    }
    void prepare_gpu() override { for (const auto& pipeline : graph_->pipelines) pipeline->prepare(stream()); }
    void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
        throw std::runtime_error("native GGUF kernels require Metal");
    }
    void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override {
        auto& output = outputs.front(); output.set_data(allocator::malloc(output.nbytes()));
        std::vector<array> storage = inputs;
        storage.push_back(output);
        std::vector<array> temporary;
        const auto pool = graph_->pooled ? workspace(stream()) : nullptr;
        std::unique_lock<std::mutex> pool_lock;
        if (pool) {
            pool_lock = std::unique_lock(pool->mutex);
            if (pool->buffers.size() < graph_->scratch.size()) pool->buffers.resize(graph_->scratch.size());
        }
        for (std::size_t i = 0; i < graph_->scratch.size(); ++i) {
            const auto bytes = graph_->scratch[i];
            auto value = [&] {
                if (pool && !graph_->clear[i]) {
                    auto& slot = pool->buffers[i];
                    const auto previous = slot ? slot->nbytes() : 0;
                    if (previous >= bytes) return *slot;
                    if (pool->retained - previous + bytes <= (32u << 20)) {
                        slot.emplace(allocator::malloc(bytes), Shape{int(bytes)}, uint8);
                        pool->retained += bytes - previous;
                        return *slot;
                    }
                }
                return array(allocator::malloc(bytes), Shape{int(bytes)}, uint8);
            }();
            if (graph_->clear[i]) std::memset(value.data<std::uint8_t>(), 0, bytes);
            temporary.push_back(value); storage.push_back(std::move(value));
        }
        if (pool_lock.owns_lock()) pool_lock.unlock();
        auto& encoder = mlx::core::metal::get_command_encoder(stream());
        const bool bind_weights = graph_->pooled && graph_->sparse_weight_inputs;
        const auto bind_input = [&](const array& value, std::size_t slot, int index, std::int64_t offset = 0) {
            if (bind_weights && slot < 64 && (graph_->sparse_weight_inputs & (std::uint64_t(1) << slot)))
                encoder.set_buffer(static_cast<const MTL::Buffer*>(value.buffer().ptr()), index, value.offset() + offset);
            else encoder.set_input_array(value, index, offset);
        };
        if (bind_weights) encoder.barrier();
        if (!graph_->direct_tile_bound && std::any_of(graph_->commands.dispatches().begin(), graph_->commands.dispatches().end(),
                [](const auto& dispatch) { return bool(dispatch.indirectThreadgroups.buffer); })) {
            for (std::size_t slot = 0; slot < storage.size(); ++slot)
                if (storage[slot].size()) bind_input(storage[slot], slot, 0);
            auto& device = mlx::core::metal::device(stream().device);
            auto* update = device.get_kernel("fence_update");
            auto* wait = device.get_kernel("fence_wait");
            array fence(allocator::malloc(sizeof(uint32_t)), {1}, uint32);
            fence.data<uint32_t>()[0] = 0;
            encoder.barrier();
            encoder.set_compute_pipeline_state(update);
            encoder.set_output_array(fence, 0); encoder.set_bytes(uint32_t(1), 1);
            encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            encoder.end_encoding();
            auto* buffer = encoder.get_command_buffer();
            auto raw = NS::RetainPtr(buffer->computeCommandEncoder(MTL::DispatchTypeSerial));
            const auto* timestamp = static_cast<const MTL::Buffer*>(fence.buffer().ptr());
            const uint32_t ready = 1, done = 2;
            raw->setComputePipelineState(wait); raw->setBuffer(timestamp, fence.offset(), 0);
            raw->setBytes(&ready, sizeof(ready), 1);
            raw->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            raw->memoryBarrier(MTL::BarrierScopeBuffers);
            std::size_t index = 0;
            for (const auto& dispatch : graph_->commands.dispatches()) {
                raw->setComputePipelineState(graph_->pipelines[index++]->prepare(stream()));
                for (const auto& binding : dispatch.buffers) {
                    const auto slot = binding.buffer ? binding.buffer.slot() : 0;
                    const auto& value = storage.at(slot);
                    raw->setBuffer(static_cast<const MTL::Buffer*>(value.buffer().ptr()),
                        value.offset() + binding.buffer.offset(), binding.index);
                }
                for (const auto& bytes : dispatch.bytes) raw->setBytes(bytes.data, bytes.sizeBytes, bytes.index);
                const auto& threads = dispatch.threadsPerThreadgroup;
                const MTL::Size group(threads.x, threads.y, threads.z);
                if (dispatch.indirectThreadgroups.buffer) {
                    const auto& grid = dispatch.indirectThreadgroups;
                    const auto& value = storage.at(grid.buffer.slot());
                    raw->dispatchThreadgroups(static_cast<const MTL::Buffer*>(value.buffer().ptr()),
                        value.offset() + grid.buffer.offset() + grid.offsetBytes, group);
                } else {
                    const auto& groups = dispatch.threadgroups;
                    raw->dispatchThreadgroups(MTL::Size(groups.x, groups.y, groups.z), group);
                }
                raw->memoryBarrier(MTL::BarrierScopeBuffers);
            }
            raw->setComputePipelineState(update); raw->setBuffer(timestamp, fence.offset(), 0);
            raw->setBytes(&done, sizeof(done), 1);
            raw->dispatchThreads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            raw->endEncoding();
            encoder.set_compute_pipeline_state(wait); encoder.set_input_array(fence, 0); encoder.set_bytes(done, 1);
            encoder.register_output_array(output);
            if (graph_->pooled)
                for (const auto& value : temporary) encoder.register_output_array(value);
            encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
            buffer->addCompletedHandler([storage = std::move(storage), fence](MTL::CommandBuffer*) {});
            return;
        }
        std::size_t index = 0;
        for (const auto& dispatch : graph_->commands.dispatches()) {
            auto* state = graph_->pipelines[index++]->prepare(stream());
            encoder.set_compute_pipeline_state(state);
            for (const auto& binding : dispatch.buffers) {
                const auto slot = binding.buffer ? binding.buffer.slot() : 0;
                auto& value = storage.at(slot);
                if (slot < graph_->inputs) bind_input(value, slot, binding.index, binding.buffer.offset());
                else encoder.set_output_array(value, binding.index, binding.buffer.offset());
            }
            for (const auto& bytes : dispatch.bytes)
                encoder.set_bytes(static_cast<const std::uint8_t*>(bytes.data), int(bytes.sizeBytes), bytes.index);
            const auto& threads = dispatch.threadsPerThreadgroup;
            if (threads.x * threads.y * threads.z > state->maxTotalThreadsPerThreadgroup())
                throw std::invalid_argument("native GGUF threadgroup exceeds the pipeline limit");
            auto groups = dispatch.threadgroups;
            if (dispatch.indirectThreadgroups.buffer) groups.y = graph_->direct_tile_bound;
            encoder.dispatch_threadgroups(MTL::Size(groups.x, groups.y, groups.z), MTL::Size(threads.x, threads.y, threads.z));
        }
        if (graph_->pooled) {
            encoder.get_command_buffer()->addCompletedHandler(
                [temporary = std::move(temporary)](MTL::CommandBuffer*) {});
        } else encoder.add_temporaries(std::move(temporary));
    }
private:
    std::shared_ptr<Graph> graph_;
};

array execute(std::shared_ptr<Graph> graph, std::vector<array> inputs, Shape shape, Dtype dtype) {
    graph->finish();
    return array(std::move(shape), dtype, std::make_shared<NativeGraph>(default_stream(Device::gpu), std::move(graph)),
        std::move(inputs));
}

splash::DeviceCapabilities capabilities() {
    splash::DeviceCapabilities device;
    auto& gpu = mlx::core::metal::device(Device::gpu);
    device.appleGpuFamily = !mlx_metal_nax_disabled() && gpu.mtl_device()->supportsFamily(MTL::GPUFamilyApple10) ? 10 : 9;
    device.gpuCoreCount = metal_apple_gpu_core_count();
    return device;
}

LinearScratch scratch(Graph& graph, const LinearScratchSize& size) {
    return {graph.temporary(size.input), graph.temporary(size.sums), graph.temporary(size.partials),
        graph.temporary(size.counters, true), graph.temporary(size.rotated)};
}

bool single_row_moe(const MlxNativeGgufMoe& weights, int actual) {
    const int width = weights.routed[0].width, intermediate = weights.routed[0].outputs, experts = weights.routed[0].experts;
    if (actual != 1 || width <= 0 || experts <= 0 || experts > 512 ||
        weights.top_k <= 0 || weights.top_k > experts ||
        weights.slots.dtype() != uint32 || weights.slots.size() < std::size_t(experts) ||
        weights.router.size() != std::size_t(experts) * width ||
        (weights.router.dtype() != float32 && weights.router.dtype() != bfloat16) ||
        weights.scalar_gate.dtype() != float32 || weights.scalar_gate.size() != std::size_t(width)) return false;
    const auto valid = [](const MlxNativeGgufWeight& w, int count, int k, int n) {
        if (w.format < 0 || w.format >= GGUF_FMT_COUNT || w.experts != count || w.width != k || w.outputs != n ||
            w.storage_outputs() != n || k <= 0 || n <= 0 || k % 64 || n % 64) return false;
        const auto& format = kQuantFormats[w.format];
        const std::uint64_t groups = std::uint64_t(count) * n * k / 32;
        return w.planes[0].nbytes() >= groups * format.plane0_bytes &&
            w.planes[1].nbytes() >= groups * format.plane1_bytes &&
            w.planes[2].nbytes() >= groups / format.meta_groups * format.meta_bytes;
    };
    for (int i = 0; i < 3; ++i) {
        const int k = i == 2 ? intermediate : width, n = i == 2 ? width : intermediate;
        if (!valid(weights.routed[i], experts, k, n) || !valid(weights.shared[i], 1, k, n)) return false;
    }
    return true;
}

}

MlxNativeGgufWeight mlx_native_gguf_float(const array& weight) {
    if (weight.dtype() == float16) {
        auto exact = astype(weight, float32); exact.eval(); MlxWeightResidency::track(exact);
        return mlx_native_gguf_float(exact);
    }
    if (weight.ndim() != 2 || (weight.dtype() != bfloat16 && weight.dtype() != float32))
        throw std::invalid_argument("native float projection requires BF16 or F32 matrices");
    return {{weight, weight, weight}, int(weight.dtype() == bfloat16 ? QuantizedSegment::kBFloat16 : QuantizedSegment::kFloat32),
        weight.shape(1), weight.shape(0)};
}

std::array<array, 2> mlx_native_gguf_hyper(const array& streams, const array& norm,
    const MlxNativeGgufWeight& down, const MlxNativeGgufWeight& up, const array* inject) {
    const int width = down.width, hidden = width / 4, actual = int(streams.size() / width);
    if (streams.ndim() < 2 || streams.shape(-1) != width || hidden != HYPER_CONNECTION_NORM_HIDDEN ||
        norm.dtype() != float32 || norm.size() != width || actual < 1 ||
        down.outputs != up.width || up.outputs != width ||
        (inject && (inject->dtype() != float32 || inject->size() != 4 * width)))
        throw std::invalid_argument("native hyper-connection geometry differs");
    const int rows = actual <= 32 ? (actual + 7) / 8 * 8 : actual;
    const bool bfloat_input = streams.dtype() == bfloat16 && actual >= 8;
    const auto input_dtype = bfloat_input ? bfloat16 : float32;
    auto x = reshape(row_major_view(astype(streams, input_dtype)), {actual, width});
    std::vector<array> inputs{x, norm, inject ? *inject : norm};
    for (const auto* weight : {&down, &up}) inputs.insert(inputs.end(), weight->planes.begin(), weight->planes.end());
    const std::array projections{down, up};
    auto key = graph_key(3, rows, projections); key.push_back(inject != nullptr); key.push_back(bfloat_input);
    auto graph = cached_graph(std::move(key), actual <= 32, [&] {
        std::size_t slot = 3;
        const auto projection = [&](const MlxNativeGgufWeight& weight) {
            const auto at = slot; slot += 3;
            auto segment = weight.format < 0 ? QuantizedSegment::floats(weight.outputs, weight.width,
                {at, weight.planes[0].nbytes()}, uint32_t(weight.format)) :
                QuantizedSegment::planes(weight.format, weight.outputs, weight.width, {at, weight.planes[0].nbytes()},
                    {at + 1, weight.planes[1].nbytes()}, {at + 2, weight.planes[2].nbytes()});
            return Projection(weight.outputs, weight.width, BlockWeights{{segment}});
        };
        const auto lower = projection(down), upper = projection(up);
        Linear linear(capabilities());
        struct Part { int begin, rows; LinearPlan low, up; };
        std::vector<Part> parts;
        LinearScratchSize size;
        int stored = 0, largest = 0;
        for (int begin = 0; begin < rows; begin += SPLASH_PREFILL_TOKEN_BUDGET) {
            const int count = std::min(rows - begin, int(SPLASH_PREFILL_TOKEN_BUDGET));
            const int padded = count <= 32 ? (count + 7) / 8 * 8 : count;
            auto low_plan = count <= 32 ? linear.decodePlan(lower, padded / 8) : linear.prefillPlan(lower, count, LinearEpilogue::None);
            auto up_plan = count <= 32 ? linear.decodePlan(upper, padded / 8) : linear.prefillPlan(upper, count, LinearEpilogue::None);
            const int capacity = std::max(low_plan.storageRows(), up_plan.storageRows());
            stored = begin + capacity; largest = std::max(largest, capacity);
            size.include(low_plan.scratchSize()); size.include(up_plan.scratchSize());
            parts.push_back({begin, padded, std::move(low_plan), std::move(up_plan)});
        }
        auto result = std::make_shared<Graph>(); result->inputs = inputs.size(); result->pooled = actual <= 32; result->rows = stored;
        const auto work = scratch(*result, size);
        const auto normalized = result->temporary(std::size_t(largest) * width * 2);
        const auto low = result->temporary(std::size_t(largest) * down.outputs * 2);
        const auto gate = result->temporary(std::size_t(largest) * width * 2);
        const std::size_t branch_bytes = std::size_t(stored) * hidden * 2, weights_bytes = std::size_t(stored) * 4 * 4;
        const HyperShape shape{uint32_t(hidden), 4, uint32_t(down.outputs)};
        for (const auto& part : parts) {
            const MetalBuffer branch{result->inputs, branch_bytes - std::size_t(part.begin) * hidden * 2,
                std::size_t(part.begin) * hidden * 2};
            const MetalBuffer weights{result->inputs, weights_bytes - std::size_t(part.begin) * 4 * 4,
                branch_bytes + std::size_t(part.begin) * 4 * 4};
            const MetalBuffer source{0, std::uint64_t(stored - part.begin) * width * input_dtype.size(),
                std::uint64_t(part.begin) * width * input_dtype.size()};
            if (bfloat_input) {
                const bool quarters = inject && part.rows <= HYPER_CONNECTION_NORM_QUARTER_ROWS;
                result->commands.add(quarters ? "hyper_connection_norm_bfloat_quarters" : "hyper_connection_norm_bfloat_loaded",
                    {source, {1, norm.nbytes()}, {2, inject ? inject->nbytes() : norm.nbytes()}, normalized,
                        inject ? weights : MetalBuffer{1, norm.nbytes()}},
                    HyperConnectionParams{uint32_t(part.rows), uint32_t(hidden), 4, uint32_t(down.outputs), uint32_t(inject != nullptr)},
                    {uint32_t(part.rows), 1, 1},
                    {quarters ? HYPER_CONNECTION_NORM_QUARTER_THREADS : HYPER_CONNECTION_ROW_THREADS, 1, 1});
            } else
                HyperConnection::addNorm(result->commands, source,
                    {{1, norm.nbytes()}, true}, inject ? MetalBuffer{2, inject->nbytes()} : MetalBuffer{},
                    normalized, weights, shape, part.rows);
            linear.add(result->commands, {.input = normalized, .output = low, .scratch = work}, lower, part.low);
            const auto layout = part.rows <= 32 && up.format >= 0 ? part.up.input() : LinearInput::Plain;
            const auto prepared = HyperConnection::addLow(result->commands, low, shape, part.rows, work, layout);
            linear.add(result->commands, {.input = low, .output = gate, .scratch = work, .prepared = prepared}, upper, part.up);
            HyperConnection::addMix(result->commands, normalized, gate, branch, shape, part.rows);
        }
        return result;
    });
    const int stored = graph->rows;
    if (stored > actual) inputs[0] = concatenate({inputs[0], zeros({stored - actual, width}, input_dtype)}, 0);
    const std::size_t branch_bytes = std::size_t(stored) * hidden * 2, weights_bytes = std::size_t(stored) * 4 * 4;
    const auto result = execute(graph, std::move(inputs), {int(branch_bytes + weights_bytes)}, uint8);
    auto branch_shape = streams.shape(); branch_shape.back() = hidden;
    auto b = reshape(view(slice(result, {0}, {int(branch_bytes)}), bfloat16), {stored, hidden});
    auto w = inject ? slice(reshape(view(slice(result, {int(branch_bytes)}, {int(branch_bytes + weights_bytes)}), float32),
        {stored, 4}), {0, 0}, {actual, 4}) : zeros({actual, 4}, float32);
    return {reshape(slice(b, {0, 0}, {actual, hidden}), branch_shape), w};
}

bool mlx_native_gguf_supported(int type, int rows, int width) noexcept {
    if (rows <= 0 || width <= 0 || width % 64) return false;
    for (const auto& format : kQuantFormats)
        if (format.ggml_type == std::uint32_t(type)) return width % format.block_elements == 0;
    return false;
}

MlxNativeGgufWeight mlx_native_gguf_load(const MfqContainer& model, const std::string& name) {
    const auto& record = model.record(name);
    const auto shape = model.tensor_shape(name);
    if (shape.size() != 2 && shape.size() != 3) throw std::invalid_argument("native GGUF weight rank differs");
    const int experts = shape.size() == 3 ? int(shape[0]) : 1;
    const int rows = int(shape[shape.size() - 2]), width = int(shape.back());
    int type = -1;
    for (int i = 0; i < GGML_TYPE_COUNT; ++i) {
        const auto* base = ggml_type_name(ggml_type(i));
        if (!base) continue;
        std::string dtype = "GGUF_";
        for (const auto c : std::string_view(base)) dtype += char(std::toupper(static_cast<unsigned char>(c)));
        if (dtype == record.dtype) type = i;
    }
    if (type < 0 || record.nbytes > INT_MAX) throw std::invalid_argument("unsupported native GGUF source tensor");
    array packed(allocator::malloc(record.nbytes), {int(record.nbytes)}, uint8);
    model.read_range_into(name, 0, {reinterpret_cast<std::byte*>(packed.data<std::uint8_t>()), std::size_t(record.nbytes)});
    auto result = mlx_native_gguf_pack(packed, type, experts * rows, width, int(record.nbytes / (experts * rows)));
    eval(std::vector<array>(result.planes.begin(), result.planes.end()));
    for (const auto& plane : result.planes) MlxWeightResidency::track(plane);
    result.outputs = rows; result.experts = experts;
    model.record_prepared(name);
    return result;
}

MlxNativeGgufWeight mlx_native_gguf_pack(const array& packed, int type, int rows, int width, int row_bytes) {
    std::uint32_t format = GGUF_FMT_COUNT;
    for (std::uint32_t i = 0; i < GGUF_FMT_COUNT; ++i) if (kQuantFormats[i].ggml_type == std::uint32_t(type)) format = i;
    if (!mlx_native_gguf_supported(type, rows, width) ||
        packed.nbytes() != std::size_t(rows) * row_bytes)
        throw std::invalid_argument("unsupported native GGUF repack geometry");
    if (rows % QUANT_TILE_ROWS) {
        const int padded = (rows + QUANT_TILE_ROWS - 1) / QUANT_TILE_ROWS * QUANT_TILE_ROWS;
        auto values = concatenate({reshape(packed, {-1}), zeros({(padded - rows) * row_bytes}, uint8)}, 0);
        auto result = mlx_native_gguf_pack(values, type, padded, width, row_bytes);
        result.outputs = rows; result.padded_outputs = padded;
        return result;
    }
    const auto& f = kQuantFormats[format];
    const std::size_t groups = std::size_t(rows) * width / 32;
    const std::size_t p0 = groups * f.plane0_bytes, p1 = groups * f.plane1_bytes, meta = groups / f.meta_groups * f.meta_bytes;
    if (p0 + p1 + meta > INT_MAX) throw std::overflow_error("native GGUF repack exceeds array size");
    auto graph = std::make_shared<Graph>(); graph->inputs = 1;
    GgufRepackParams params{std::uint32_t(rows), std::uint32_t(width), format, std::uint32_t(row_bytes),
        std::uint32_t(p0), std::uint32_t(p0 + p1)};
    graph->commands.add("gguf_repack", {{0, packed.nbytes()}, {1, p0 + p1 + meta}}, params,
        {(groups + 255) / 256, 1, 1}, {256, 1, 1});
    auto all = execute(graph, {packed}, {int(p0 + p1 + meta)}, uint8);
    return {{slice(all, {0}, {int(p0)}), slice(all, {int(p0)}, {int(p0 + p1)}),
        slice(all, {int(p0 + p1)}, {int(p0 + p1 + meta)})}, int(format), width, rows};
}

bool mlx_native_affine_supported(int bits, int group, int outputs, int width) noexcept {
    return bits > 0 && bits <= 255 && group > 0 && group <= 255 &&
        mlx_native_gguf_supported(QUANT_AFFINE_TYPE(bits, group), outputs, width);
}

MlxNativeGgufWeight mlx_native_affine_pack(const array& values, const array& scales, const array& biases,
    int bits, int group) {
    if (bits <= 0 || group <= 0 || (values.ndim() != 2 && values.ndim() != 3) ||
        values.dtype() != uint32 || scales.dtype() != bfloat16 || biases.dtype() != bfloat16 ||
        (values.ndim() == 3 && values.shape(-2) % 64))
        throw std::invalid_argument("native affine planes require BF16 group coefficients");
    static const auto kernel = cached_metal_kernel("tinfer_native_affine_blocks", {"values", "scales", "biases", "params"},
        {"blocks"}, R"METAL(
    const uint i = thread_position_in_grid.x, k = params[0], n = params[1], bits = params[2], gs = params[3];
    const uint groups = k / gs;
    if (i >= n * groups) return;
    const uint words = gs / 32u * bits, bytes = 4u + words * 4u;
    auto block = blocks + ulong(i) * bytes;
    reinterpret_cast<device bfloat*>(block)[0] = scales[i];
    reinterpret_cast<device bfloat*>(block)[1] = biases[i];
    for (uint j = 0u; j < words; ++j) reinterpret_cast<device uint*>(block + 4u)[j] = values[ulong(i) * words + j];
    )METAL");
    const int width = values.shape(-1) / bits * 32, outputs = values.shape(-2);
    const int experts = values.ndim() == 3 ? values.shape(0) : 1, rows = experts * outputs;
    const int stride = width / group * (4 + group / 8 * bits);
    if (!mlx_native_affine_supported(bits, group, outputs, width))
        throw std::invalid_argument("unsupported native affine format");
    auto expected = values.shape(); expected.back() = width / group;
    if (scales.shape() != expected || biases.shape() != expected)
        throw std::invalid_argument("native affine group coefficient shapes differ");
    auto packed = kernel({values, scales, biases, array({uint32_t(width), uint32_t(rows), uint32_t(bits), uint32_t(group)})},
        {{rows * stride}}, {uint8}, {((rows * (width / group) + 127) / 128) * 128, 1, 1}, {128, 1, 1},
        {}, std::nullopt, false, {})[0];
    auto result = mlx_native_gguf_pack(packed, QUANT_AFFINE_TYPE(bits, group), rows, width, stride);
    result.outputs = outputs; result.experts = experts;
    return result;
}

std::vector<array> mlx_native_gguf_project(std::span<const MlxNativeGgufWeight> weights, const array& input) {
    if (weights.empty() || input.ndim() < 1) throw std::invalid_argument("native GGUF projection requires matrices");
    if (weights.size() > 3) {
        auto first = mlx_native_gguf_project(weights.first(3), input);
        auto rest = mlx_native_gguf_project(weights.subspan(3), input);
        first.insert(first.end(), rest.begin(), rest.end());
        return first;
    }
    const int width = weights.front().width, count = int(input.size() / width);
    if (input.shape(-1) != width || count < 1)
        throw std::invalid_argument("native GGUF projection input shape differs");
    if (count > SPLASH_PREFILL_TOKEN_BUDGET && input.dtype() == float32) {
        const auto flat = reshape(input, {count, width});
        std::vector<std::vector<array>> chunks(weights.size());
        for (int begin = 0; begin < count; begin += SPLASH_PREFILL_TOKEN_BUDGET) {
            auto part = mlx_native_gguf_project(weights, slice(flat, {begin, 0}, {std::min(count, begin + int(SPLASH_PREFILL_TOKEN_BUDGET)), width}));
            for (std::size_t i = 0; i < part.size(); ++i) chunks[i].push_back(std::move(part[i]));
        }
        std::vector<array> result;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            auto shape = input.shape(); shape.back() = weights[i].outputs;
            result.push_back(reshape(concatenate(chunks[i], 0), shape));
        }
        return result;
    }
    std::vector<array> inputs{reshape(row_major_view(astype(input, bfloat16)), {count, width})};
    inputs.reserve(1 + 3 * weights.size());
    int total = 0;
    const int requested = count <= 32 ? (count + 7) / 8 * 8 : count;
    for (const auto& weight : weights) {
        if (weight.width != width) throw std::invalid_argument("native GGUF projections have different inputs");
        inputs.insert(inputs.end(), weight.planes.begin(), weight.planes.end()); total += weight.storage_outputs();
    }
    const auto output_type = input.dtype() == float32 && count <= 32 && weights.size() == 1 ? float32 : bfloat16;
    const int gpu_cores = count == 1 ? std::max(1, metal_apple_gpu_core_count()) : 1;
    const bool vector_decode = count == 1 && total / 64 <= gpu_cores * 8 &&
        std::none_of(weights.begin(), weights.end(), [gpu_cores](const auto& weight) {
            return weight.format == GGUF_FMT_IQ4XS && weight.width >= 4096 && weight.width > weight.outputs &&
                weight.storage_outputs() / 64 * 8 < gpu_cores * 6;
        });
    auto key = graph_key(1, requested, weights); key.push_back(output_type == float32);
    key.push_back(vector_decode);
    auto graph = cached_graph(std::move(key), count <= 32, [&] {
        Linear linear(capabilities());
        auto result = std::make_shared<Graph>(); result->inputs = inputs.size(); result->pooled = count <= 32;
        if (vector_decode) {
            result->rows = 1;
            GgufDecodeFusedParams params{};
            params.input_size = width; params.out_stride = total; params.splits = 1;
            const auto target = gpu_cores * 6;
            while (params.splits < 8 && total / 64 * params.splits < target &&
                width / 32 % (params.splits * 2) == 0 && width / 32 / (params.splits * 2) >= 8)
                params.splits *= 2;
            std::vector<MetalBuffer> buffers{{0, std::uint64_t(width) * 2}};
            int offset = 0;
            for (std::size_t i = 0; i < 3; ++i) {
                const auto segment = std::min(i, weights.size() - 1);
                const auto& weight = weights[segment];
                for (int plane = 0; plane < 3; ++plane)
                    buffers.push_back({1 + 3 * segment + plane, weight.planes[plane].nbytes()});
                if (i < weights.size()) {
                    params.cols[i] = weight.storage_outputs(); params.fmt[i] = weight.format;
                    params.offset[i] = offset; offset += weight.storage_outputs();
                }
            }
            buffers.push_back({result->inputs, std::uint64_t(total) * size_of(output_type)});
            const auto partials = result->temporary(std::uint64_t(total) * params.splits * sizeof(float));
            buffers.push_back(partials);
            const auto suffix = output_type == float32 ? "f32" : "bf16";
            result->commands.add(std::string("moe_native_vector_") + suffix, std::move(buffers), params,
                {std::uint64_t(total / 64), params.splits, 1}, {64, 1, 1});
            if (params.splits > 1)
                result->commands.add(std::string("moe_native_vector_reduce_") + suffix,
                    {partials, {result->inputs, std::uint64_t(total) * size_of(output_type)}},
                    std::array<std::uint32_t, 2>{std::uint32_t(total), params.splits},
                    {std::uint64_t((total + 255) / 256), 1, 1}, {256, 1, 1});
            return result;
        }
        std::vector<QuantizedSegment> segments;
        std::size_t slot = 1; int offset = 0;
        for (const auto& weight : weights) {
            auto segment = QuantizedSegment::planes(weight.format, weight.storage_outputs(), width,
                {slot, weight.planes[0].nbytes()}, {slot + 1, weight.planes[1].nbytes()}, {slot + 2, weight.planes[2].nbytes()});
            segment.columnOffset = offset; slot += 3; offset += weight.storage_outputs(); segments.push_back(std::move(segment));
        }
        Projection projection(total, width, BlockWeights{std::move(segments)});
        if (output_type == float32) projection.destination = FloatOutput::Float32;
        std::vector<std::pair<int, LinearPlan>> parts;
        LinearScratchSize size;
        for (int begin = 0; begin < requested; begin += SPLASH_PREFILL_TOKEN_BUDGET) {
            const int rows = std::min(requested - begin, int(SPLASH_PREFILL_TOKEN_BUDGET));
            auto plan = rows <= 32 ? linear.decodePlan(projection, (rows + 7) / 8) :
                linear.prefillPlan(projection, rows, LinearEpilogue::None);
            result->rows = begin + plan.storageRows();
            size.include(plan.scratchSize()); parts.emplace_back(begin, std::move(plan));
        }
        const auto work = scratch(*result, size);
        for (const auto& [begin, plan] : parts)
            linear.add(result->commands,
                {.input = {0, std::uint64_t(result->rows - begin) * width * 2, std::uint64_t(begin) * width * 2},
                 .output = {result->inputs, std::uint64_t(result->rows - begin) * total * size_of(output_type),
                     std::uint64_t(begin) * total * size_of(output_type)}, .scratch = work}, projection, plan);
        return result;
    });
    const int rows = graph->rows;
    if (rows > count) inputs[0] = concatenate({inputs[0], zeros({rows - count, width}, bfloat16)}, 0);
    std::vector<array> result;
    auto all = execute(graph, std::move(inputs), {rows, total}, output_type);
    int begin = 0;
    for (const auto& weight : weights) {
        auto block = slice(all, {0, begin}, {count, begin + weight.outputs});
        auto shape = input.shape(); shape.back() = weight.outputs;
        result.push_back(reshape(astype(block, input.dtype()), shape)); begin += weight.storage_outputs();
    }
    return result;
}

array mlx_native_gguf_swiglu(const MlxNativeGgufWeight& gate, const MlxNativeGgufWeight& up, const array& input) {
    if (gate.width != up.width || gate.outputs != up.outputs || input.ndim() < 2 || input.shape(-1) != gate.width)
        throw std::invalid_argument("native GGUF Gate/Up shapes differ");
    const int width = gate.width, count = int(input.size() / width), output = gate.storage_outputs();
    if (output != up.storage_outputs()) throw std::invalid_argument("native GGUF Gate/Up storage differs");
    if (count > SPLASH_PREFILL_TOKEN_BUDGET) {
        auto flat = reshape(input, {count, width});
        std::vector<array> chunks;
        for (int begin = 0; begin < count; begin += SPLASH_PREFILL_TOKEN_BUDGET)
            chunks.push_back(mlx_native_gguf_swiglu(gate, up,
                slice(flat, {begin, 0}, {std::min(count, begin + int(SPLASH_PREFILL_TOKEN_BUDGET)), width})));
        auto shape = input.shape(); shape.back() = gate.outputs;
        return reshape(concatenate(chunks, 0), shape);
    }
    std::vector<array> inputs{reshape(row_major_view(astype(input, bfloat16)), {count, width})};
    for (const auto* weight : {&gate, &up}) {
        inputs.insert(inputs.end(), weight->planes.begin(), weight->planes.end());
    }
    const bool decode = count <= 32;
    const int requested = decode ? (count + 7) / 8 * 8 : count;
    const std::array weights{gate, up};
    auto graph = cached_graph(graph_key(2, requested, weights), decode, [&] {
        Linear linear(capabilities());
        auto result = std::make_shared<Graph>(); result->inputs = 7; result->pooled = decode;
        std::vector<Projection> projections;
        std::size_t slot = 1;
        for (const auto* weight : {&gate, &up}) {
            projections.emplace_back(output, width, BlockWeights{{QuantizedSegment::planes(weight->format, output, width,
                {slot, weight->planes[0].nbytes()}, {slot + 1, weight->planes[1].nbytes()}, {slot + 2, weight->planes[2].nbytes()})}});
            slot += 3;
        }
        auto plan = decode ? linear.decodePlan(projections[1], requested / 8, LinearEpilogue::GateUp, &projections[0]) :
            linear.prefillPlan(projections[1], requested, LinearEpilogue::UpWithGate);
        auto bound = plan.scratchSize();
        const int rows = result->rows = plan.storageRows();
        if (!decode) bound.include(linear.prefillPlan(projections[0], rows, LinearEpilogue::None).scratchSize());
        const auto work = scratch(*result, bound);
        const auto intermediate = result->temporary(decode ? plan.gateScratchBytes() : std::size_t(rows) * output * 2);
        const MetalBuffer x{0, std::uint64_t(rows) * width * 2}, y{7, std::uint64_t(rows) * output * 2};
        if (!decode) linear.add(result->commands, {.input = x, .output = intermediate, .scratch = work},
            projections[0], linear.prefillPlan(projections[0], rows, LinearEpilogue::None));
        linear.add(result->commands, {.input = x, .output = y, .gateScratch = intermediate, .scratch = work},
            projections[1], plan, decode ? &projections[0] : nullptr);
        return result;
    });
    const int rows = graph->rows;
    if (rows > count) inputs[0] = concatenate({inputs[0], zeros({rows - count, width}, bfloat16)}, 0);
    auto all = execute(graph, std::move(inputs), {rows, output}, bfloat16);
    auto shape = input.shape(); shape.back() = gate.outputs;
    return reshape(astype(slice(all, {0, 0}, {count, gate.outputs}), input.dtype()), shape);
}

array mlx_native_gguf_moe(const MlxNativeGgufMoe& weights, const array& input) {
    const int width = weights.routed[0].width, actual = int(input.size() / width);
    if (input.ndim() < 2 || input.shape(-1) != width || actual < 1)
        throw std::invalid_argument("native GGUF MoE input shape differs");
    if (actual > SPLASH_PREFILL_TOKEN_BUDGET) {
        auto flat = reshape(input, {actual, width});
        std::vector<array> chunks;
        for (int begin = 0; begin < actual; begin += SPLASH_PREFILL_TOKEN_BUDGET)
            chunks.push_back(mlx_native_gguf_moe(weights,
                slice(flat, {begin, 0}, {std::min(actual, begin + int(SPLASH_PREFILL_TOKEN_BUDGET)), width})));
        return reshape(concatenate(chunks, 0), input.shape());
    }
    const bool single = single_row_moe(weights, actual);
    const bool vector = single && std::all_of(weights.routed.begin(), weights.routed.end(),
        [](const auto& weight) { return quant_affine_format(weight.format); }) &&
        std::all_of(weights.shared.begin(), weights.shared.end(),
            [](const auto& weight) { return quant_affine_format(weight.format); });
    const int intermediate = weights.routed[0].outputs, experts = weights.routed[0].experts;
    const int rows = single ? 1 : actual <= 32 ? (actual + 7) / 8 * 8 : actual;
    auto x = reshape(row_major_view(astype(input, bfloat16)), {actual, width});
    if (rows > actual) x = concatenate({x, zeros({rows - actual, width}, bfloat16)}, 0);
    std::vector<array> inputs{x, weights.router, weights.scalar_gate, weights.slots};
    inputs.reserve(23);
    for (int i = 0; i < 3; ++i) {
        for (const auto* weight : {&weights.routed[i], &weights.shared[i]})
            inputs.insert(inputs.end(), weight->planes.begin(), weight->planes.end());
    }
    MetalBuffer verify;
    if (actual <= 32 && !single) {
        std::vector<uint32_t> valid(rows / 8);
        for (int i = 0; i < int(valid.size()); ++i) valid[i] = std::min(8, actual - i * 8);
        verify = {inputs.size(), valid.size() * sizeof(uint32_t)};
        inputs.emplace_back(valid.begin(), Shape{int(valid.size())}, uint32);
    }
    auto key = graph_key(4, rows, weights.routed);
    const auto shared_key = graph_key(4, rows, weights.shared);
    key.insert(key.end(), shared_key.begin(), shared_key.end());
    key.insert(key.end(), {std::uint64_t(weights.top_k), std::uint64_t(weights.router.dtype() == bfloat16),
        weights.router.nbytes(), weights.scalar_gate.nbytes(), weights.slots.nbytes(), std::uint64_t(actual == 1)});
    auto graph = cached_graph(std::move(key), actual <= 32, [&] {
        const auto device = capabilities();
        Linear linear(device);
        MoeShape shape{uint32_t(width), uint32_t(weights.routed[0].experts), uint32_t(weights.top_k),
            uint32_t(weights.routed[0].outputs), uint32_t(weights.routed[0].format), true};
        for (const auto* projection : {&weights.routed, &weights.shared})
            for (const auto& weight : *projection) shape.registerFormats &= quant_register_tensor(weight.format, weight.width);
        const auto tile = moeGgufTile(gpuFamilyClass(device.appleGpuFamily), shape);
        const MoeConfig configuration{actual <= 32 ? MoeExpertTile::M8 : moeGgufPrefillTile(shape, rows, tile),
            tile, linear.ggufFloatTile(rows, shape.experts)};
        BlockMoeWeights block;
        block.router = QuantizedSegment::floats(shape.experts, width, {1, weights.router.nbytes()},
            weights.router.dtype() == bfloat16 ? QuantizedSegment::kBFloat16 : QuantizedSegment::kFloat32);
        block.sharedScalarGate = QuantizedSegment::floats(1, width, {2, weights.scalar_gate.nbytes()});
        block.slotTable = {3, weights.slots.nbytes()}; block.slots = shape.experts;
        std::size_t slot = 4;
        std::uint64_t sparse_weight_inputs = 0;
        const auto bind = [&](const MlxNativeGgufWeight& weight, bool routed) {
            const auto at = slot; slot += 3;
            if (routed) sparse_weight_inputs |= std::uint64_t(7) << at;
            return QuantizedSegment::planes(weight.format, weight.outputs * (routed ? weight.experts : 1), weight.width,
                {at, weight.planes[0].nbytes()}, {at + 1, weight.planes[1].nbytes()}, {at + 2, weight.planes[2].nbytes()});
        };
        for (int i = 0; i < 3; ++i) {
            auto& projection = i == 0 ? block.gate : i == 1 ? block.up : block.down;
            projection.routed = bind(weights.routed[i], true); projection.shared = bind(weights.shared[i], false);
        }
        auto result = std::make_shared<Graph>(); result->inputs = inputs.size(); result->pooled = actual <= 32;
        if (actual == 1) result->direct_tile_bound = shape.routesPerToken();
        result->sparse_weight_inputs = sparse_weight_inputs;
        if (single) {
            const int routes = weights.top_k + 1;
            const auto scores = result->temporary(std::size_t(shape.routerWidth()) * 4);
            const auto ids = result->temporary(std::size_t(routes) * 4), probabilities = result->temporary(std::size_t(routes) * 4);
            const auto middle = result->temporary(std::size_t(routes) * intermediate * 2);
            const auto down = result->temporary(std::size_t(routes) * width * 2);
            MoE::addRoutes(result->commands, {0, x.nbytes()}, block.router, block.sharedScalarGate, scores,
                ids, probabilities, 1, shape, configuration.ggufRouterTile);
            const auto fields = [](const BlockExpertProjection& p) {
                return std::array{p.routed.plane0, p.routed.plane1Slot(), p.routed.meta,
                    p.shared.plane0, p.shared.plane1Slot(), p.shared.meta};
            };
            std::vector<MetalBuffer> upper{{0, x.nbytes()}, ids, block.slotTable};
            for (const auto* projection : {&block.gate, &block.up}) {
                const auto values = fields(*projection); upper.insert(upper.end(), values.begin(), values.end());
            }
            upper.push_back(middle);
            result->commands.add(vector ? "moe_affine_vector_gate_up" : "moe_native_single_gate_up", std::move(upper),
                MoeGgufGateUpParams{uint32_t(width), uint32_t(intermediate), uint32_t(experts),
                    block.gate.routed.formatId, block.gate.shared.formatId, block.up.routed.formatId, block.up.shared.formatId},
                {uint32_t(intermediate / (vector && width <= intermediate ? 64 : 32)), uint32_t(routes), 1},
                {uint32_t(vector ? 256 : 64), 1, 1});
            const auto values = fields(block.down);
            std::vector<MetalBuffer> lower{middle, ids, block.slotTable}; lower.insert(lower.end(), values.begin(), values.end());
            lower.push_back(down);
            result->commands.add(vector ? "moe_affine_vector_down" : "moe_native_single_down", std::move(lower),
                MoeGgufExpertParams{uint32_t(intermediate), uint32_t(width), uint32_t(experts),
                    block.down.routed.formatId, block.down.shared.formatId},
                {uint32_t(width / (vector && intermediate > width ? 32 : 64)), uint32_t(routes), 1},
                {uint32_t(vector ? 256 : 64), 1, 1});
            result->commands.add("moe_single_reduce", {down, probabilities, {result->inputs, std::uint64_t(width) * 2}},
                MoeCombineParams{1, uint32_t(width), uint32_t(routes)}, {uint32_t((width + 255) / 256), 1, 1}, {256, 1, 1});
            return result;
        }
        const auto plan = actual <= 32 ? MoE::decodePlan(shape, rows / 8, configuration) : MoE::prefillPlan(shape, rows, configuration);
        MoeScratch work;
        for (const auto& field : kMoeScratchFields)
            work.*field.buffer = result->temporary(plan.workspace().*field.bytes, field.buffer == &MoeScratch::routeRows);
        const auto residual = result->temporary(std::size_t(rows) * width * 2, true);
        MoE::add(result->commands, {.input = {0, x.nbytes()}, .residual = residual,
            .output = {result->inputs, std::uint64_t(rows) * width * 2}, .scratch = work, .verifyRows = verify}, block, plan);
        return result;
    });
    auto result = execute(graph, std::move(inputs), {rows, width}, bfloat16);
    return reshape(astype(slice(result, {0, 0}, {actual, width}), input.dtype()), input.shape());
}

}
