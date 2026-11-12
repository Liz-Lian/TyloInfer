#include "tq.h"
#include "engine/backends/cuda/kernels/tq_ops.h"
#include "engine/weights/encoding/tq_rows.h"
#include "engine/weights/tq/image.h"

#include <cublas_v2.h>
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::diagnostics::numerics {
namespace {
void status(int code, const char *name) {
    if (code)
        throw std::runtime_error(std::string(name) + " failed: " + std::to_string(code));
}
void blas_status(cublasStatus_t code, const char *name) {
    if (code != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(name) + " failed: " + std::to_string(int(code)));
}
struct Stream {
    cudaStream_t value{};
    Stream() {
        cuda_check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "cudaStreamCreate");
    }
    ~Stream() {
        if (value) {
            cudaStreamSynchronize(value);
            cudaStreamDestroy(value);
        }
    }
    void wait() { cuda_check(cudaStreamSynchronize(value), "cudaStreamSynchronize"); }
};
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    cudaStream_t capturing{};
    ~Graph() {
        if (capturing) {
            cudaGraph_t abandoned{};
            if (cudaStreamEndCapture(capturing, &abandoned) == cudaSuccess && abandoned)
                cudaGraphDestroy(abandoned);
        }
        if (exec)
            cudaGraphExecDestroy(exec);
        if (graph)
            cudaGraphDestroy(graph);
    }
    void begin(cudaStream_t stream) {
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
                   "cudaStreamBeginCapture");
        capturing = stream;
    }
    void end(cudaStream_t stream) {
        const auto status = cudaStreamEndCapture(stream, &graph);
        capturing = nullptr;
        cuda_check(status, "cudaStreamEndCapture");
        cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
    }
    void replay(cudaStream_t stream) {
        cuda_check(cudaGraphLaunch(exec, stream), "cudaGraphLaunch");
    }
};
struct Blas {
    cublasHandle_t value{};
    explicit Blas(cudaStream_t stream) {
        blas_status(cublasCreate(&value), "cublasCreate");
        const auto status = cublasSetStream(value, stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(value);
            value = nullptr;
            blas_status(status, "cublasSetStream");
        }
    }
    ~Blas() {
        if (value)
            cublasDestroy(value);
    }
};
void require_input(const Case &value, const char *name, const Json &shape,
                   const std::string &dtype) {
    const auto &spec = value.data().at("inputs").at(name);
    if (spec.at("shape") != shape || spec.at("dtype") != dtype)
        throw std::invalid_argument(std::string("input shape or dtype differs: ") + name);
}
int type(const std::string &name) {
    if (name == "float16")
        return 0;
    if (name == "bfloat16")
        return 1;
    if (name == "float32")
        return 2;
    throw std::invalid_argument("unsupported TQ dtype");
}
std::size_t size_of(int dtype) { return dtype == 2 ? 4 : 2; }
std::vector<std::uint8_t> device_values(std::vector<std::uint8_t> bytes, const std::string &dtype) {
    if (dtype != "bfloat16")
        return bytes;
    if (bytes.size() % 4)
        throw std::invalid_argument("invalid BF16 input size");
    std::vector<std::uint8_t> result(bytes.size() / 2);
    for (std::size_t i = 0; i < result.size() / 2; ++i) {
        std::uint32_t bits;
        std::memcpy(&bits, bytes.data() + i * 4, 4);
        if (bits & 0xffff)
            throw std::invalid_argument("invalid expanded BF16 input");
        const auto upper = std::uint16_t(bits >> 16);
        std::memcpy(result.data() + i * 2, &upper, 2);
    }
    return result;
}
void upload(Buffer &target, const void *source, std::size_t bytes, cudaStream_t stream,
            std::size_t offset = 0) {
    if (offset > target.size() || bytes > target.size() - offset)
        throw std::invalid_argument("upload exceeds buffer");
    cuda_check(cudaMemcpyAsync(static_cast<char *>(target.data()) + offset, source, bytes,
                               cudaMemcpyHostToDevice, stream),
               "upload");
}
struct Input {
    Buffer storage;
    std::size_t offset;
    Input(const Case &value, const std::string &name, cudaStream_t stream)
        : Input(value.data().at("inputs").at(name), value.input(name), stream) {}
    Input(const Json &spec, std::vector<std::uint8_t> bytes, cudaStream_t stream)
        : Input(spec, device_values(std::move(bytes), spec.at("dtype")), stream, 0) {}
    Input(const Json &spec, std::vector<std::uint8_t> bytes, cudaStream_t stream, int)
        : storage(bytes.size() + offset_bytes(spec)), offset(offset_bytes(spec)) {
        const auto layout = spec.at("layout");
        if (layout.is_object()) {
            auto stride = std::int64_t{1};
            const auto shape = spec.at("shape");
            const auto strides = layout.at("strides");
            if (shape.size() != strides.size())
                throw std::invalid_argument("invalid input strides");
            for (std::size_t i = shape.size(); i-- > 0;) {
                if (strides[i].get<std::int64_t>() != stride)
                    throw std::invalid_argument("unsupported input strides");
                const auto size = shape[i].get<std::int64_t>();
                if (size < 0 || (size && stride > std::numeric_limits<std::int64_t>::max() / size))
                    throw std::invalid_argument("input stride overflow");
                stride *= size;
            }
        } else if (layout != "contiguous")
            throw std::invalid_argument("invalid input layout");
        upload(storage, bytes.data(), bytes.size(), stream, offset);
        cuda_check(cudaStreamSynchronize(stream), "retain host upload");
    }
    static std::size_t offset_bytes(const Json &spec) {
        const auto layout = spec.at("layout");
        if (!layout.is_object())
            return 0;
        const auto offset = layout.at("offset").get<std::int64_t>();
        if (offset < 0)
            throw std::invalid_argument("negative input offset");
        const auto dtype = spec.at("dtype").get<std::string>();
        const auto bytes = dtype == "int64" ? 8 : size_of(type(dtype));
        if (std::uint64_t(offset) > std::numeric_limits<std::size_t>::max() / bytes)
            throw std::invalid_argument("input offset overflow");
        return std::size_t(offset) * bytes;
    }
    void *data() const { return static_cast<char *>(storage.data()) + offset; }
};
struct Weight {
    Buffer storage;
    TrellisProjectionView view{};
    Weight(const Case &value, const std::string &name, int width, cudaStream_t stream)
        : Weight(value.asset(name), width, stream) {}
    Weight(std::vector<std::uint8_t> blob, int width, cudaStream_t stream)
        : Weight(blob, tyloi::TqLayout(blob.data(), blob.size()), width, stream) {}
    Weight(const std::vector<std::uint8_t> &blob, const tyloi::TqLayout &layout, int width,
           cudaStream_t stream)
        : storage(((layout.packed + 3) & ~std::size_t(3)) + layout.packed_bytes + 32) {
        if (width <= 0 || std::uint32_t(width) > layout.width)
            throw std::invalid_argument("invalid TQ width");
        auto image = tq_image(blob, layout);
        upload(storage, image.data(), image.size(), stream);
        cuda_check(cudaStreamSynchronize(stream), "retain TQ image upload");
        auto *base = static_cast<std::uint8_t *>(storage.data());
        const auto words = (layout.packed + 3) & ~std::size_t(3);
        int gslog = 0;
        for (auto gs = layout.gs; gs > 1; gs >>= 1)
            ++gslog;
        view = {reinterpret_cast<const std::uint32_t *>(base + words),
                reinterpret_cast<const std::uint64_t *>(base + 64),
                base + layout.scales,
                base + words + layout.packed_bytes,
                nullptr,
                int(layout.rows),
                int(layout.width),
                width,
                gslog,
                int(layout.columns)};
    }
};
void result(Case &value, const std::string &name, const void *source,
            std::initializer_list<std::int64_t> shape, int dtype, cudaStream_t stream) {
    std::size_t count = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0 || std::uint64_t(dimension) >
                                  std::numeric_limits<std::size_t>::max() / count)
            throw std::invalid_argument("invalid output shape");
        count *= std::size_t(dimension);
    }
    if (dtype < 0 || dtype > 2 || count > std::numeric_limits<std::size_t>::max() / 4)
        throw std::invalid_argument("invalid output dtype or size");
    const std::string actual_dtype = dtype == 0 ? "float16" : dtype == 1 ? "bfloat16" : "float32";
    Json actual_shape = Json::array();
    for (const auto dimension : shape)
        actual_shape.push_back(dimension);
    std::vector<std::uint8_t> bytes(count * size_of(dtype));
    cuda_check(cudaMemcpyAsync(bytes.data(), source, bytes.size(), cudaMemcpyDeviceToHost, stream),
               "download output");
    cuda_check(cudaStreamSynchronize(stream), "download output synchronization");
    if (dtype == 1) {
        std::vector<std::uint8_t> expanded(count * 4);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint16_t upper;
            std::memcpy(&upper, bytes.data() + i * 2, 2);
            const auto bits = std::uint32_t(upper) << 16;
            std::memcpy(expanded.data() + i * 4, &bits, 4);
        }
        value.output(name, actual_shape, actual_dtype, expanded.data(), expanded.size());
    } else
        value.output(name, actual_shape, actual_dtype, bytes.data(), bytes.size());
}
std::vector<std::uint8_t> guard_data(std::size_t count, int dtype) {
    std::vector<std::uint8_t> data(count * size_of(dtype));
    if (dtype == 2) {
        const float v = 42;
        for (std::size_t i = 0; i < count; ++i)
            std::memcpy(data.data() + i * 4, &v, 4);
    } else {
        const std::uint16_t v = dtype == 0 ? 0x5140 : 0x4228;
        for (std::size_t i = 0; i < count; ++i)
            std::memcpy(data.data() + i * 2, &v, 2);
    }
    return data;
}
bool guards(Buffer &buffer, std::size_t count, int dtype, cudaStream_t stream) {
    std::vector<std::uint8_t> data(buffer.size());
    cuda_check(
        cudaMemcpyAsync(data.data(), buffer.data(), data.size(), cudaMemcpyDeviceToHost, stream),
        "guard download");
    cuda_check(cudaStreamSynchronize(stream), "guard synchronization");
    const auto expected = guard_data(1, dtype);
    const auto step = size_of(dtype);
    return std::equal(expected.begin(), expected.end(), data.begin()) &&
           std::equal(expected.begin(), expected.end(), data.begin() + (count + 1) * step);
}
void check(Json &checks, int index, bool pass, Json evidence) {
    checks["native_" + std::to_string(index)] = {{"status", pass ? "pass" : "fail"},
                                                 {"evidence", std::move(evidence)}};
}

struct TqCase {
    Case &value;
    const Json &data;
    const Json &config;
    const int dtype;
    const int width;
    const std::string actual_dtype;
    Stream &stream;
    Json &checks;

    void gemv() {
        require_input(value, "x", Json::array({width}), actual_dtype);
        Weight weight(value, "weight", width, stream.value);
        Input x(value, "x", stream.value);
        std::unique_ptr<Input> bias;
        if (data.at("inputs").contains("bias")) {
            require_input(value, "bias", Json::array({weight.view.rows}), "float32");
            bias = std::make_unique<Input>(value, "bias", stream.value);
            weight.view.bias = static_cast<const float *>(bias->data());
        }
        const auto count = std::size_t(weight.view.rows);
        auto initial = guard_data(count + 2, dtype);
        Buffer output(initial.size());
        upload(output, initial.data(), initial.size(), stream.value);
        stream.wait();
        auto *out = static_cast<char *>(output.data()) + size_of(dtype);
        int index = 0;
        for (const auto &launch : config.at("warps_rows_per_block")) {
            const int warps = launch[0], rows = launch[1];
            status(mfq_tq_gemv(weight.view, x.data(), out, dtype, warps, rows, stream.value),
                   "tq_gemv");
            const auto name =
                bias ? "output_w" + std::to_string(warps) + "_r" + std::to_string(rows) : "output";
            result(value, name, out, {weight.view.rows}, dtype, stream.value);
            if (bias)
                check(checks, index++, guards(output, count, dtype, stream.value),
                      {{"guard_values", {42, 42}}, {"warps", warps}, {"rows_per_block", rows}});
        }
    }

    void projection(bool q8) {
        Weight weight(value, q8 ? "weight0" : "weight", width, stream.value);
        const int tokens = config.at("tokens"), groups = config.at("groups");
        if (tokens <= 0 || groups <= 0 || weight.view.rows % groups)
            throw std::invalid_argument("invalid grouped projection geometry");
        require_input(value, "x", Json::array({tokens, groups, width}), actual_dtype);
        require_input(value, "bias", Json::array({weight.view.rows}), "float32");
        if (q8) {
            require_input(value, "x_changed", Json::array({tokens, groups, width}), actual_dtype);
            require_input(value, "x_zero", Json::array({tokens, groups, width}), actual_dtype);
        }
        Input x(value, "x", stream.value), bias(value, "bias", stream.value);
        weight.view.bias = static_cast<const float *>(bias.data());
        const auto count = std::size_t(tokens) * weight.view.rows;
        auto initial = guard_data(count + 2, dtype);
        Buffer output(initial.size());
        upload(output, initial.data(), initial.size(), stream.value);
        stream.wait();
        auto *out = static_cast<char *>(output.data()) + size_of(dtype);
        const auto bytes = config.at("workspace_bytes").get<std::size_t>();
        Buffer workspace(bytes);
        Blas blas(stream.value);
        const auto run = [&](std::size_t workspace_bytes) {
            return q8 ? mfq_tq_q8_tokens(weight.view, x.data(), out, tokens, groups, dtype,
                                         workspace.data(), workspace_bytes, stream.value)
                      : mfq_tq_float_gemm(weight.view, x.data(), out, tokens, groups, dtype,
                                          workspace.data(), workspace_bytes, blas.value,
                                          stream.value);
        };
        status(run(bytes), "TQ projection");
        result(value, "output", out, {tokens, weight.view.rows}, dtype, stream.value);
        bool intact = guards(output, count, dtype, stream.value);
        Graph graph;
        graph.begin(stream.value);
        status(run(bytes), "TQ graph capture");
        graph.end(stream.value);
        if (q8) {
            Input changed(value, "x_changed", stream.value);
            cuda_check(cudaMemcpyAsync(x.data(), changed.data(),
                                       std::size_t(tokens) * groups * width * size_of(dtype),
                                       cudaMemcpyDeviceToDevice, stream.value),
                       "changed input");
            graph.replay(stream.value);
            result(value, "changed", out, {tokens, weight.view.rows}, dtype, stream.value);
            intact &= guards(output, count, dtype, stream.value);
            Input zero(value, "x_zero", stream.value);
            cuda_check(cudaMemcpyAsync(x.data(), zero.data(),
                                       std::size_t(tokens) * groups * width * size_of(dtype),
                                       cudaMemcpyDeviceToDevice, stream.value),
                       "zero input");
            graph.replay(stream.value);
            result(value, "zero", out, {tokens, weight.view.rows}, dtype, stream.value);
            intact &= guards(output, count, dtype, stream.value);
            const auto short_status = run(bytes - 1);
            stream.wait();
            check(checks, 0, intact, {{"guard_values", {42, 42}}});
            check(checks, 1, true, {{"capture", 1}, {"changed_input_replay", 1}});
            check(checks, 2, short_status == -1,
                  {{"status", short_status}, {"workspace_bytes", bytes - 1}});
            check(checks, 3, true, {{"zero_input_replay", 1}});
        } else {
            graph.replay(stream.value);
            result(value, "replay", out, {tokens, weight.view.rows}, dtype, stream.value);
            intact &= guards(output, count, dtype, stream.value);
            check(checks, 0, intact, {{"guard_values", {42, 42}}});
            check(checks, 1, true, {{"capture", 1}, {"replay", 1}, {"workspace_bytes", bytes}});
        }
    }

    void parallel() {
        const int tokens = config.at("tokens"), joined = config.at("joined"),
                  joined_rows = config.at("joined_columns");
        const auto &rows = config.at("projection_rows");
        if (tokens <= 0 || rows.empty() || joined < 0 || std::size_t(joined) > rows.size())
            throw std::invalid_argument("invalid parallel projection geometry");
        require_input(value, "x", Json::array({tokens, width}), actual_dtype);
        require_input(value, "x_changed", Json::array({tokens, width}), actual_dtype);
        std::vector<std::unique_ptr<Weight>> weights;
        std::vector<TrellisProjectionJob> jobs;
        int blocks = 0;
        std::int64_t total_rows = 0;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            weights.push_back(
                std::make_unique<Weight>(value, "weight" + std::to_string(i), width, stream.value));
            if (weights.back()->view.rows != rows[i].get<int>())
                throw std::invalid_argument("projection rows differ");
            blocks += (weights.back()->view.rows + TRELLIS_GROUP_ROWS - 1) / TRELLIS_GROUP_ROWS;
            jobs.push_back({weights.back()->view, total_rows, blocks});
            total_rows += weights.back()->view.rows;
        }
        Buffer plan(jobs.size() * sizeof(TrellisProjectionJob));
        upload(plan, jobs.data(), plan.size(), stream.value);
        stream.wait();
        Input x(value, "x", stream.value);
        if (total_rows <= 0 || total_rows > std::numeric_limits<std::int64_t>::max() / tokens)
            throw std::invalid_argument("parallel output shape overflow");
        const auto count = std::size_t(tokens) * total_rows;
        auto initial = guard_data(count + 2, dtype);
        Buffer output(initial.size());
        upload(output, initial.data(), initial.size(), stream.value);
        stream.wait();
        auto *out = static_cast<char *>(output.data()) + size_of(dtype);
        const auto bytes = config.at("workspace_bytes").get<std::size_t>();
        Buffer workspace(bytes);
        const auto run = [&](std::size_t workspace_bytes) {
            return mfq_tq_q8_parallel_tokens(
                jobs.data(), static_cast<const TrellisProjectionJob *>(plan.data()),
                int(jobs.size()), blocks, x.data(), out, tokens, width, dtype, workspace.data(),
                workspace_bytes, joined, joined_rows, stream.value);
        };
        status(run(bytes), "TQ Q8 parallel");
        result(value, "output", out, {std::int64_t(tokens) * total_rows}, dtype, stream.value);
        bool intact = guards(output, count, dtype, stream.value);
        Graph graph;
        graph.begin(stream.value);
        status(run(bytes), "TQ Q8 parallel capture");
        graph.end(stream.value);
        Input changed(value, "x_changed", stream.value);
        cuda_check(cudaMemcpyAsync(x.data(), changed.data(),
                                   std::size_t(tokens) * width * size_of(dtype),
                                   cudaMemcpyDeviceToDevice, stream.value),
                   "changed input");
        graph.replay(stream.value);
        result(value, "changed", out, {std::int64_t(tokens) * total_rows}, dtype, stream.value);
        intact &= guards(output, count, dtype, stream.value);
        const auto short_status = run(bytes - 1);
        stream.wait();
        check(checks, 0, intact, {{"guard_values", {42, 42}}});
        check(checks, 1, true, {{"capture", 1}, {"changed_input_replay", 1}});
        check(checks, 2, short_status == -1,
              {{"status", short_status}, {"workspace_bytes", bytes - 1}});
    }

    void ffn() {
        const int hidden = config.at("hidden");
        Weight gate(value, "gate", width, stream.value), up(value, "up", width, stream.value),
            down(value, "down", hidden, stream.value);
        if (hidden <= 0 || gate.view.rows != hidden || up.view.rows != hidden)
            throw std::invalid_argument("invalid FFN geometry");
        require_input(value, "x", Json::array({width}), actual_dtype);
        Input x(value, "x", stream.value);
        Buffer workspace(std::size_t(hidden) * size_of(dtype));
        Buffer output(std::size_t(down.view.rows) * size_of(dtype));
        status(mfq_tq_swiglu(gate.view, up.view, x.data(), workspace.data(), dtype, 0, 0,
                             stream.value),
               "TQ swiglu");
        result(value, "workspace", workspace.data(), {hidden}, dtype, stream.value);
        status(mfq_tq_ffn(gate.view, up.view, down.view, x.data(), workspace.data(), output.data(),
                          dtype, stream.value),
               "TQ FFN");
        result(value, "output", output.data(), {down.view.rows}, dtype, stream.value);
        stream.wait();
        Stream extra;
        for (int i = 0; i < 3; ++i)
            status(mfq_tq_ffn(gate.view, up.view, down.view, x.data(), workspace.data(),
                              output.data(), dtype, extra.value),
                   "TQ FFN alternate stream");
        extra.wait();
        Graph graph;
        graph.begin(stream.value);
        status(mfq_tq_ffn(gate.view, up.view, down.view, x.data(), workspace.data(), output.data(),
                          dtype, stream.value),
               "TQ FFN capture");
        graph.end(stream.value);
        graph.replay(stream.value);
        result(value, "replay", output.data(), {down.view.rows}, dtype, stream.value);
        check(checks, 0, true, {{"capture", 1}, {"replay", 1}});
        check(checks, 1, true, {{"alternate_stream_runs", 3}, {"synchronized", true}});
    }

    void embedding() {
        require_input(value, "ids", Json::array({3}), "int64");
        require_input(value, "ids_changed", Json::array({3}), "int64");
        std::vector<std::unique_ptr<Weight>> weights;
        for (int i = 0; i < 3; ++i)
            weights.push_back(
                std::make_unique<Weight>(value, "weight" + std::to_string(i), width, stream.value));
        auto selection = [&](const char *name) {
            auto bytes = value.input(name);
            if (bytes.size() != 3 * sizeof(std::int64_t))
                throw std::invalid_argument("invalid embedding IDs");
            std::vector<TrellisProjectionView> views;
            for (int i = 0; i < 3; ++i) {
                std::int64_t id;
                std::memcpy(&id, bytes.data() + i * 8, 8);
                if (id < 0 || id >= 3)
                    throw std::out_of_range("embedding ID");
                views.push_back(weights[std::size_t(id)]->view);
            }
            return views;
        };
        auto views = selection("ids");
        Buffer plan(views.size() * sizeof(TrellisProjectionView));
        upload(plan, views.data(), plan.size(), stream.value);
        stream.wait();
        const auto count = std::size_t(3) * width;
        auto initial = guard_data(count + 2, dtype);
        Buffer output(initial.size());
        upload(output, initial.data(), initial.size(), stream.value);
        stream.wait();
        auto *out = static_cast<char *>(output.data()) + size_of(dtype);
        const auto run = [&] {
            return mfq_tq_embedding_rows(static_cast<const TrellisProjectionView *>(plan.data()), 3,
                                         width, out, dtype, stream.value);
        };
        status(run(), "TQ embedding");
        result(value, "output", out, {std::int64_t(3) * width}, dtype, stream.value);
        bool intact = guards(output, count, dtype, stream.value);
        Graph graph;
        graph.begin(stream.value);
        status(run(), "TQ embedding capture");
        graph.end(stream.value);
        views = selection("ids_changed");
        upload(plan, views.data(), plan.size(), stream.value);
        stream.wait();
        graph.replay(stream.value);
        result(value, "changed", out, {std::int64_t(3) * width}, dtype, stream.value);
        intact &= guards(output, count, dtype, stream.value);
        check(checks, 0, intact, {{"guard_values", {42, 42}}});
        check(checks, 1, true, {{"capture", 1}, {"changed_table_replay", 1}});
    }

    void routed() {
        const int hidden = config.at("hidden"), rows = config.at("output"),
                  routes = config.at("routes");
        if (routes != 3 || hidden <= 0 || rows <= 0)
            throw std::invalid_argument("unsupported routed geometry");
        require_input(value, "x", Json::array({width}), actual_dtype);
        require_input(value, "ids", Json::array({routes}), "int64");
        require_input(value, "ids_changed", Json::array({routes}), "int64");
        require_input(value, "weights", Json::array({routes}), "float32");
        std::vector<std::unique_ptr<Weight>> weights;
        std::vector<std::unique_ptr<Buffer>> tables;
        for (const std::string role : {"gate", "up", "down"}) {
            std::vector<TrellisProjectionView> views;
            for (int i = 0; i < 3; ++i) {
                weights.push_back(std::make_unique<Weight>(value, role + std::to_string(i),
                                                           role == "down" ? hidden : width,
                                                           stream.value));
                if (weights.back()->view.rows != (role == "down" ? rows : hidden))
                    throw std::invalid_argument("invalid routed weight geometry");
                views.push_back(weights.back()->view);
            }
            tables.push_back(
                std::make_unique<Buffer>(views.size() * sizeof(TrellisProjectionView)));
            upload(*tables.back(), views.data(), tables.back()->size(), stream.value);
            stream.wait();
        }
        for (const auto *name : {"ids", "ids_changed"}) {
            const auto raw = value.input(name);
            for (int i = 0; i < routes; ++i) {
                std::int64_t id;
                std::memcpy(&id, raw.data() + i * sizeof(id), sizeof(id));
                if (id < 0 || id >= 3)
                    throw std::out_of_range("routed expert ID");
            }
        }
        Input x(value, "x", stream.value), ids(value, "ids", stream.value),
            route_weights(value, "weights", stream.value);
        Buffer workspace(std::size_t(routes) * hidden * size_of(dtype)),
            pairs(std::size_t(routes) * rows * size_of(dtype));
        auto initial = guard_data(std::size_t(rows) + 2, dtype);
        Buffer output(initial.size());
        upload(output, initial.data(), initial.size(), stream.value);
        stream.wait();
        auto *out = static_cast<char *>(output.data()) + size_of(dtype);
        const auto run = [&] {
            auto *token_ids = static_cast<const std::int64_t *>(ids.data());
            status(
                mfq_tq_routed_swiglu(static_cast<const TrellisProjectionView *>(tables[0]->data()),
                                     static_cast<const TrellisProjectionView *>(tables[1]->data()),
                                     token_ids, x.data(), workspace.data(), routes, hidden, dtype,
                                     config.at("limit").get<float>(), stream.value),
                "TQ routed swiglu");
            status(mfq_tq_routed_down(static_cast<const TrellisProjectionView *>(tables[2]->data()),
                                      token_ids, workspace.data(), pairs.data(), routes, rows,
                                      hidden, dtype, stream.value),
                   "TQ routed down");
            status(mfq_tq_weighted_reduce(pairs.data(),
                                          static_cast<const float *>(route_weights.data()), out, 1,
                                          routes, rows, dtype, stream.value),
                   "TQ weighted reduce");
        };
        run();
        result(value, "output", out, {rows}, dtype, stream.value);
        bool intact = guards(output, rows, dtype, stream.value);
        Graph graph;
        graph.begin(stream.value);
        run();
        graph.end(stream.value);
        Input changed(value, "ids_changed", stream.value);
        cuda_check(cudaMemcpyAsync(ids.data(), changed.data(),
                                   std::size_t(routes) * sizeof(std::int64_t),
                                   cudaMemcpyDeviceToDevice, stream.value),
                   "changed expert IDs");
        graph.replay(stream.value);
        result(value, "changed", out, {rows}, dtype, stream.value);
        intact &= guards(output, rows, dtype, stream.value);
        check(checks, 0, intact, {{"guard_values", {42, 42}}});
        check(checks, 1, true, {{"capture", 1}, {"changed_ids_replay", 1}});
    }
};
} // namespace

void run_tq(Case &value) {
    const auto &data = value.data();
    const auto &config = data.at("config");
    const auto op = data.at("op").get<std::string>();
    const auto actual_dtype = config.at("actual_dtype").get<std::string>();
    const auto dtype = type(actual_dtype);
    const int width = config.at("width");
    if (width <= 0)
        throw std::invalid_argument("invalid TQ width");
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    int runtime = 0;
    int driver = 0;
    cuda_check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
    cuda_check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    Stream stream;
    Json checks = Json::object();
    TqCase task{value, data, config, dtype, width, actual_dtype, stream, checks};
    if (op == "tq_gemv")
        task.gemv();
    else if (op == "tq_float_prefill")
        task.projection(false);
    else if (op == "tq_q8")
        task.projection(true);
    else if (op == "tq_q8_parallel")
        task.parallel();
    else if (op == "tq_ffn")
        task.ffn();
    else if (op == "tq_embedding")
        task.embedding();
    else if (op == "tq_routed_ffn")
        task.routed();
    else
        throw std::invalid_argument("unsupported TQ operation: " + op);
    stream.wait();
    Json execution = {
        {"target", "mfq-numerics"},
        {"device", device},
        {"device_name", properties.name},
        {"capability", {properties.major, properties.minor}},
        {"cuda_runtime", runtime},
        {"cuda_driver", driver},
        {"entrypoint", op},
        {"build", {{"native_cuda", true}, {"cuda_runtime_version", CUDART_VERSION}}},
    };
    value.finish(std::move(execution), std::move(checks));
}
} // namespace tyloi::diagnostics::numerics
