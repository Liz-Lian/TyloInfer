#include "backend_checks.h"

#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/program.h"
#include "engine/models/deepseek_v41/engram.h"
#include "engine/ops/csa.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/graph.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef TYLOINFER_CUDA_CHECK
#define TYLOINFER_CUDA_CHECK(expr) do { \
    cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__)); \
    } \
} while (0)
#endif

namespace {
/** Retire work before the tensors and optional captured graph leave scope. */
class DrainGuard {
public:
    DrainGuard(tyloi::cuda::Program& program, tyloi::cuda::StreamHandle stream,
               tyloi::cuda::Graph* graph = nullptr)
        : program_(program), stream_(std::move(stream)), graph_(graph) {}
    ~DrainGuard() noexcept {
        if (!active_) return;
        if (graph_) graph_->reset();
        const auto error = cudaStreamSynchronize(stream_.stream());
        if (error != cudaSuccess) std::cerr << "backend_cleanup error=" << cudaGetErrorString(error) << '\n';
        try { program_.drain(); }
        catch (const std::exception& error) { std::cerr << "backend_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "backend_cleanup error=unknown\n"; }
    }
    /** Propagate stream and resource failures on a successful check. */
    void finish() {
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream_.stream()));
        program_.drain();
        active_ = false;
    }
private:
    tyloi::cuda::Program& program_;
    tyloi::cuda::StreamHandle stream_;
    tyloi::cuda::Graph* graph_;
    bool active_ = true;
};
}

int run_backend_dsv41_check(tyloi::cuda::Program& program) {
    auto scope = program.executor().enter();
    tyloi::DenseTensor blocks, positions;
    tyloi::Tensor candidates;
    DrainGuard drain(program, tyloi::cuda::current_stream());
    tyloi::cuda::deepseek_v41_runtime::EngramHashState::self_check();
    const auto options = tyloi::TensorOptions()
                             .device(program.executor().device())
                             .dtype(tyloi::kInt32);
    blocks = tyloi::cuda::tensor(std::vector<std::int32_t>{0, 2, -1}, options)
                 .reshape({1, 1, 3});
    candidates = tyloi::ops::csa_candidates(tyloi::Tensor{blocks}, 2, 5);
    positions = candidates.dense().to(tyloi::kCPU, tyloi::kInt64).contiguous();
    const std::vector<std::int64_t> expected{0, 1, 4, -1, -1, -1};
    const auto* values = positions.data_ptr<std::int64_t>();
    TYLOINFER_RUNTIME_CHECK(positions.numel() == static_cast<std::int64_t>(expected.size()) &&
                               std::equal(expected.begin(), expected.end(), values),
                           "DeepSeek-V4.1 candidate hierarchy contract failed");
    drain.finish();
    std::cout << "deepseek_v41_runtime_check=ok\n";
    return 0;
}

int run_backend_bf16_add_check(
        tyloi::cuda::Program& program,
        int64_t elements,
        int repetitions) {
    TYLOINFER_RUNTIME_CHECK(
        elements > 0 && elements <= (1LL << 30),
        "backend BF16 add element count is out of range");
    TYLOINFER_RUNTIME_CHECK(
        repetitions > 0,
        "backend BF16 add repetitions must be positive");
    const tyloi::Device device(tyloi::kCUDA, tyloi::cuda::current_stream().device_index());
    tyloi::DenseTensor sequence, left, right, eager;
    DrainGuard eager_drain(program, tyloi::cuda::current_stream());
    sequence = tyloi::cuda::arange(
        elements,
        tyloi::TensorOptions()
            .device(device)
            .dtype(tyloi::kFloat32));
    left = ((sequence.remainder(257) - 128.0) / 64.0)
        .to(tyloi::kBFloat16).contiguous();
    right = ((sequence.remainder(131) - 65.0) / 32.0)
        .to(tyloi::kBFloat16).contiguous();

    for (int warmup = 0; warmup < 30; ++warmup) {
        eager = (left + right).contiguous();
    }
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    tyloi::cuda::Event eager_start(cudaEventDefault);
    tyloi::cuda::Event eager_stop(cudaEventDefault);
    const auto eager_stream = tyloi::cuda::current_stream();
    eager_start.record(eager_stream.stream());
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        eager = (left + right).contiguous();
    }
    eager_stop.record(eager_stream.stream());
    eager_stop.synchronize();
    float eager_ms = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&eager_ms, eager_start.get(), eager_stop.get()));
    const auto graph_stream = tyloi::cuda::stream_from_pool(false);
    tyloi::cuda::StreamGuard graph_guard(graph_stream);
    tyloi::cuda::Graph graph;
    graph.prepare_memory();
    tyloi::DenseTensor graph_result = (left + right).contiguous();
    DrainGuard graph_drain(program, graph_stream, &graph);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    graph_result = tyloi::DenseTensor();
    graph.capture_begin();
    graph_result = (left + right).contiguous();
    graph.capture_end();
    graph.replay();
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    tyloi::cuda::Event graph_start(cudaEventDefault);
    tyloi::cuda::Event graph_stop(cudaEventDefault);
    graph_start.record(graph_stream.stream());
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        graph.replay();
    }
    graph_stop.record(graph_stream.stream());
    graph_stop.synchronize();
    float graph_ms = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&graph_ms, graph_start.get(), graph_stop.get()));
    const auto difference = (
        graph_result.to(tyloi::kFloat32) -
        eager.to(tyloi::kFloat32)).abs();
    const double maximum = difference.max().item<double>();
    TYLOINFER_RUNTIME_CHECK(
        maximum == 0.0 &&
            tyloi::cuda::isfinite(graph_result).all().item<bool>(),
        "backend BF16 add graph result differs from eager output");
    graph_drain.finish();
    graph_result = tyloi::DenseTensor();
    graph.reset();

    constexpr int chain_nodes = 512;
    const int chain_repetitions =
        std::max(100, repetitions / 50);
    tyloi::cuda::Graph chain_graph;
    chain_graph.prepare_memory();
    tyloi::DenseTensor chain_result;
    DrainGuard chain_drain(program, graph_stream, &chain_graph);
    for (int node = 0; node < chain_nodes; ++node) {
        chain_result = (left + right).contiguous();
    }
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    chain_result = tyloi::DenseTensor();
    chain_graph.capture_begin();
    for (int node = 0; node < chain_nodes; ++node) {
        chain_result = (left + right).contiguous();
    }
    chain_graph.capture_end();
    chain_graph.replay();
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    tyloi::cuda::Event chain_start(cudaEventDefault);
    tyloi::cuda::Event chain_stop(cudaEventDefault);
    chain_start.record(graph_stream.stream());
    for (int repetition = 0;
            repetition < chain_repetitions;
            ++repetition) {
        chain_graph.replay();
    }
    chain_stop.record(graph_stream.stream());
    chain_stop.synchronize();
    float chain_ms = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&chain_ms, chain_start.get(), chain_stop.get()));
    const auto chain_difference = (
        chain_result.to(tyloi::kFloat32) -
        eager.to(tyloi::kFloat32)).abs();
    const double chain_maximum =
        chain_difference.max().item<double>();
    TYLOINFER_RUNTIME_CHECK(
        chain_maximum == 0.0 &&
            tyloi::cuda::isfinite(chain_result).all().item<bool>(),
        "backend BF16 add chain Graph differs from eager output");
    const double chain_graph_us =
        1000.0 * static_cast<double>(chain_ms) /
        chain_repetitions;
    chain_drain.finish();
    chain_result = tyloi::DenseTensor();
    chain_graph.reset();

    tyloi::cuda::Graph shared_graph;
    tyloi::DenseTensor shared_reference, shared_result;
    DrainGuard shared_drain(program, graph_stream, &shared_graph);
    shared_reference = silu_mul_cuda(left, right);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    shared_graph.prepare_memory();
    for (int node = 0; node < chain_nodes; ++node) {
        shared_result = silu_mul_cuda(left, right);
    }
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    shared_result = tyloi::DenseTensor();
    shared_graph.capture_begin();
    for (int node = 0; node < chain_nodes; ++node) {
        shared_result = silu_mul_cuda(left, right);
    }
    shared_graph.capture_end();
    shared_graph.replay();
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    tyloi::cuda::Event shared_start(cudaEventDefault);
    tyloi::cuda::Event shared_stop(cudaEventDefault);
    shared_start.record(graph_stream.stream());
    for (int repetition = 0;
            repetition < chain_repetitions;
            ++repetition) {
        shared_graph.replay();
    }
    shared_stop.record(graph_stream.stream());
    shared_stop.synchronize();
    float shared_ms = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&shared_ms, shared_start.get(), shared_stop.get()));
    const auto shared_difference = (
        shared_result.to(tyloi::kFloat32) -
        shared_reference.to(tyloi::kFloat32)).abs();
    const double shared_maximum =
        shared_difference.max().item<double>();
    TYLOINFER_RUNTIME_CHECK(
        shared_maximum == 0.0 &&
            tyloi::cuda::isfinite(shared_result).all().item<bool>(),
        "shared BF16 SwiGLU chain Graph differs from eager output");
    const double shared_graph_us =
        1000.0 * static_cast<double>(shared_ms) /
        chain_repetitions;
    shared_drain.finish();
    eager_drain.finish();
    std::cout
        << "backend_bf16_add elements=" << elements
        << " repetitions=" << repetitions
        << " eager_us="
        << 1000.0 * static_cast<double>(eager_ms) / repetitions
        << " graph_us="
        << 1000.0 * static_cast<double>(graph_ms) / repetitions
        << " max_abs=" << maximum
        << " chain_nodes=" << chain_nodes
        << " chain_repetitions=" << chain_repetitions
        << " chain_graph_us=" << chain_graph_us
        << " chain_node_us=" << chain_graph_us / chain_nodes
        << " chain_max_abs=" << chain_maximum
        << " shared_chain_graph_us=" << shared_graph_us
        << " shared_chain_node_us=" << shared_graph_us / chain_nodes
        << " shared_chain_max_abs=" << shared_maximum << '\n';
    return 0;
}

static int run_backend_argmax_dtype_check(tyloi::cuda::Program& program,
        int64_t elements,
        int repetitions,
        tyloi::ScalarType dtype) {
    TYLOINFER_RUNTIME_CHECK(
        elements > 0 && elements <= (1LL << 30),
        "backend argmax element count is out of range");
    TYLOINFER_RUNTIME_CHECK(
        repetitions > 0,
        "backend argmax repetitions must be positive");
    const tyloi::Device device(tyloi::kCUDA, tyloi::cuda::current_stream().device_index());
    const int64_t expected = elements * 3 / 5;
    tyloi::DenseTensor sequence, logits, generic_input, generic_result, eager;
    DrainGuard generic_drain(program, tyloi::cuda::current_stream());
    sequence = tyloi::cuda::arange(
        elements,
        tyloi::TensorOptions()
            .device(device)
            .dtype(tyloi::kFloat32));
    logits = ((sequence - static_cast<double>(expected)).abs() * -1.0)
        .to(dtype)
        .contiguous()
        .view({1, elements});

    // A non-last axis with one output row forces the generic serial kernel
    // over exactly the same logits. Keep its timing short for large vocabularies.
    generic_input = logits.transpose(0, 1);
    tyloi::cuda::Event generic_start(cudaEventDefault), generic_stop(cudaEventDefault);
    constexpr int generic_repetitions = 8;
    const auto generic_stream = tyloi::cuda::current_stream();
    generic_start.record(generic_stream.stream());
    for (int i = 0; i < generic_repetitions; ++i)
        generic_result = generic_input.argmax(0, false);
    generic_stop.record(generic_stream.stream());
    generic_stop.synchronize();
    float generic_ms = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&generic_ms, generic_start.get(), generic_stop.get()));
    TYLOINFER_RUNTIME_CHECK(generic_result.item<int64_t>() == expected,
        "generic argmax reference returned an unexpected index");

    DrainGuard eager_drain(program, tyloi::cuda::current_stream());
    for (int warmup = 0; warmup < 30; ++warmup) {
        eager = tyloi::cuda::argmax(logits, -1);
    }
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    tyloi::cuda::Event eager_start(cudaEventDefault);
    tyloi::cuda::Event eager_stop(cudaEventDefault);
    const auto eager_stream = tyloi::cuda::current_stream();
    eager_start.record(eager_stream.stream());
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        eager = tyloi::cuda::argmax(logits, -1);
    }
    eager_stop.record(eager_stream.stream());
    eager_stop.synchronize();
    float eager_ms = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&eager_ms, eager_start.get(), eager_stop.get()));
    const auto graph_stream = tyloi::cuda::stream_from_pool(false);
    tyloi::cuda::StreamGuard graph_guard(graph_stream);
    tyloi::cuda::Graph graph;
    graph.prepare_memory();
    tyloi::DenseTensor graph_result =
        tyloi::cuda::argmax(logits, -1);
    DrainGuard graph_drain(program, graph_stream, &graph);
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
    graph_result = tyloi::DenseTensor();
    graph.capture_begin();
    graph_result = tyloi::cuda::argmax(logits, -1);
    graph.capture_end();
    graph.replay();
    TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));

    tyloi::cuda::Event graph_start(cudaEventDefault);
    tyloi::cuda::Event graph_stop(cudaEventDefault);
    graph_start.record(graph_stream.stream());
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        graph.replay();
    }
    graph_stop.record(graph_stream.stream());
    graph_stop.synchronize();
    float graph_ms = 0.0f;
    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&graph_ms, graph_start.get(), graph_stop.get()));
    const int64_t eager_index = eager.item<int64_t>();
    const int64_t graph_index = graph_result.item<int64_t>();
    TYLOINFER_RUNTIME_CHECK(
        eager_index == expected && graph_index == expected,
        "backend argmax returned an unexpected index");
    graph_drain.finish();
    eager_drain.finish();
    generic_drain.finish();
    std::cout << "backend_argmax_check"
              << " dtype=" << static_cast<int>(dtype)
              << " elements=" << elements
              << " repetitions=" << repetitions
              << " expected=" << expected
              << " eager_index=" << eager_index
              << " graph_index=" << graph_index
              << " generic_us=" << generic_ms * 1000.0f / generic_repetitions
              << " eager_us=" << eager_ms * 1000.0f / repetitions
              << " graph_us=" << graph_ms * 1000.0f / repetitions
              << "\n";
    return 0;
}

int run_backend_argmax_check(tyloi::cuda::Program& program, int64_t elements, int repetitions) {
    const tyloi::Device device(tyloi::kCUDA, tyloi::cuda::current_stream().device_index());
    int cases = 0;
    for (auto dtype : {tyloi::kBFloat16, tyloi::kFloat16, tyloi::kFloat32}) {
        for (int64_t width : {1LL, 3LL, 31LL, 255LL, 256LL, 257LL, 8193LL, 248320LL}) {
            std::vector<float> values(static_cast<size_t>(3 * width), -7.0f);
            std::vector<int64_t> expected(3);
            for (int row = 0; row < 3; ++row) {
                const auto first = std::min<int64_t>(row * 17, width - 1);
                values[row * width + first] = 5.0f;
                values[row * width + width - 1] = 5.0f;
                expected[row] = first;
            }
            tyloi::DenseTensor input, host_input;
            DrainGuard drain(program, tyloi::cuda::current_stream());
            host_input = tyloi::cuda::tensor(values).reshape({3, width}).to(dtype);
            input = host_input.to(device);
            const auto check = [&](const tyloi::DenseTensor& indices) {
                const auto host = indices.cpu().contiguous();
                TYLOINFER_RUNTIME_CHECK(host.scalar_type() == tyloi::kInt64 && host.numel() == 3,
                    "argmax result dtype or size changed");
                for (int row = 0; row < 3; ++row)
                    TYLOINFER_RUNTIME_CHECK(host.data_ptr<int64_t>()[row] == expected[row],
                        "argmax failed first-maximum or batched-row check");
            };
            check(input.argmax(-1, false));
            check(input.argmax(1, true));
            check(input.transpose(0, 1).contiguous().argmax(0, false));
            drain.finish();
            ++cases;
        }
        if (dtype != tyloi::kBFloat16) {
            // Match the existing generic floating reduction on NaNs, infinities,
            // signed zero and rows without a value greater than lowest().
            constexpr int64_t width = 513;
            std::vector<float> values(4 * width, std::numeric_limits<float>::quiet_NaN());
            std::fill(values.begin() + width, values.begin() + 2 * width,
                -std::numeric_limits<float>::infinity());
            values[width + 257] = values[width + 512] = std::numeric_limits<float>::infinity();
            std::fill(values.begin() + 2 * width, values.begin() + 3 * width, -0.0f);
            values[2 * width + 32] = 0.0f;
            std::fill(values.begin() + 3 * width, values.end(), std::numeric_limits<float>::lowest());
            tyloi::DenseTensor input, host_input;
            DrainGuard drain(program, tyloi::cuda::current_stream());
            host_input = tyloi::cuda::tensor(values).reshape({4, width}).to(dtype);
            input = host_input.to(device);
            auto reference = input.transpose(0, 1).contiguous().argmax(0, false).cpu().contiguous();
            auto actual = input.argmax(-1, false).cpu().contiguous();
            TYLOINFER_RUNTIME_CHECK(std::memcmp(reference.data_ptr(), actual.data_ptr(), 4 * sizeof(int64_t)) == 0,
                "parallel argmax changed generic nonfinite or signed-zero behavior");
            drain.finish();
            ++cases;
        }
        run_backend_argmax_dtype_check(program, elements, repetitions, dtype);
        for (int64_t width : {1LL, 3LL, 257LL, 16383LL, 16384LL, 16385LL, 248320LL, 248321LL}) {
            constexpr int rows = 6;
            std::vector<float> values(rows * width, -7.0f);
            for (int row = 0; row < rows; ++row) {
                auto begin = values.begin() + row * width;
                if (row == 0) {
                    values[std::min<int64_t>(4095, width - 1)] = 5.0f;
                    values[width - 1] = 5.0f;
                } else if (row == 1) std::fill(begin, begin + width, std::numeric_limits<float>::quiet_NaN());
                else if (row == 2) std::fill(begin, begin + width, -std::numeric_limits<float>::infinity());
                else if (row == 3) {
                    std::fill(begin, begin + width, std::numeric_limits<float>::quiet_NaN());
                    begin[std::min<int64_t>(4096, width - 1)] = std::numeric_limits<float>::infinity();
                    begin[width - 1] = std::numeric_limits<float>::infinity();
                } else if (row == 4) {
                    std::fill(begin, begin + width, -0.0f);
                    begin[width - 1] = 0.0f;
                } else std::fill(begin, begin + width, std::numeric_limits<float>::lowest());
            }
            tyloi::DenseTensor input, host_input;
            DrainGuard drain(program, tyloi::cuda::current_stream());
            host_input = tyloi::cuda::tensor(values).reshape({rows, width}).to(dtype);
            input = host_input.to(device);
            auto quantized = input.to(tyloi::kCPU, tyloi::kFloat32).contiguous();
            auto actual = sample_greedy_cuda(input).cpu().contiguous();
            for (int row = 0; row < rows; ++row) {
                float best = std::numeric_limits<float>::lowest();
                int64_t expected = 0;
                for (int64_t column = 0; column < width; ++column) {
                    const float value = quantized.data_ptr<float>()[row * width + column];
                    if (value > best) { best = value; expected = column; }
                }
                TYLOINFER_RUNTIME_CHECK(actual.data_ptr<int64_t>()[row] == expected,
                    "greedy sampler changed tail, tie, NaN, infinity or fallback behavior");
            }
            drain.finish();
            ++cases;
        }
        for (int rows : {1, 6}) {
            tyloi::DenseTensor input;
            DrainGuard drain(program, tyloi::cuda::current_stream());
            input = tyloi::cuda::zeros({rows, elements}, tyloi::TensorOptions().device(device).dtype(dtype));
            for (int i = 0; i < 20; ++i) (void)sample_greedy_cuda(input);
            cudaStreamSynchronize(::tyloi::cuda::current_stream());
            tyloi::cuda::Event begin(cudaEventDefault), end(cudaEventDefault);
            const auto stream = tyloi::cuda::current_stream().stream();
            begin.record(stream);
            for (int i = 0; i < repetitions; ++i) (void)sample_greedy_cuda(input);
            end.record(stream); end.synchronize();
            float ms = 0; TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms, begin.get(), end.get()));
            drain.finish();
            std::cout << "backend_greedy rows=" << rows << " dtype=" << static_cast<int>(dtype)
                      << " eager_us=" << ms * 1000 / repetitions << "\n";
        }
    }
    std::cout << "backend_argmax_cases=" << cases
              << " BF16/FP16/FP32, batched rows, tails, ties, keepdim, strided fallback, nonfinite PASS\n";
    return 0;
}
