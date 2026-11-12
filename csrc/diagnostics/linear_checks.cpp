#include "engine/models/common/load.h"
#include "engine/models/source.h"
#include "env.h"
#include "engine/weights/loader.h"
#include "model_checks.h"

#include "engine/backends/cuda/ops/projection/quant_linear.h"
#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/native/branches.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/graph.h"
#include "engine/backends/cuda/kernels/mfq_cuda_activation_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_embedding_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_linear_attention_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_quant_ops.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using tyloi::indexing::Slice;

namespace {
/** Keep diagnostic inputs and outputs alive until their owning executor drains. */
class Work {
public:
    explicit Work(tyloi::cuda::Program& program) : program_(program) { retained_.reserve(128); }
    ~Work() noexcept {
        if (graph_) graph_->reset();
        if (graph_stream_) {
            const auto error = cudaStreamSynchronize(graph_stream_.stream());
            if (error != cudaSuccess) std::cerr << "linear_cleanup error=" << cudaGetErrorString(error) << '\n';
        }
        if (!finished_) {
            try { program_.drain(); }
            catch (const std::exception& error) { std::cerr << "linear_cleanup error=" << error.what() << '\n'; }
            catch (...) { std::cerr << "linear_cleanup error=unknown\n"; }
        }
    }
    /** Retain a tensor representation without changing its storage or dtype. */
    void keep(const tyloi::Tensor& value) { if (value.defined()) retained_.push_back(value); }
    /** Retain a dense tensor across asynchronous work. */
    void keep(const tyloi::DenseTensor& value) { if (value.defined()) retained_.emplace_back(value); }
    /** Retain the current timed result through the next asynchronous launch. */
    void latest(const tyloi::DenseTensor& value) { latest_ = value; }
    /** Retain the current grouped timed result through the next launch. */
    void latest_all(const std::vector<tyloi::DenseTensor>& values) { latest_group_ = values; }
    /** Retain every tensor in one group or observation. */
    template <typename T> void keep_all(const std::vector<T>& values) {
        for (const auto& value : values) keep(value);
    }
    /** Retain graph warmup outputs until their stream has synchronized. */
    void hold_warmup(const std::vector<tyloi::DenseTensor>& values) { warmup_ = values; }
    /** Reserve warmup handle storage before any graph work is launched. */
    void reserve_warmup(std::size_t count) { warmup_.reserve(count); }
    /** Release synchronized warmup outputs before graph capture. */
    void release_warmup() { warmup_.clear(); }
    /** Retain a prepared group, including packed projection buffers. */
    void keep_group(const QuantLinearGroup& value) { groups_.push_back(value); }
    /** Begin ownership of a graph and its dedicated stream before capture. */
    tyloi::cuda::Graph& graph(tyloi::cuda::StreamHandle stream) {
        graph_stream_ = std::move(stream);
        graph_ = std::make_unique<tyloi::cuda::Graph>();
        return *graph_;
    }
    /** Drain and release graph resources after their last use. */
    void finish_graph() {
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream_.stream()));
        program_.drain();
        graph_->reset();
        graph_.reset();
        graph_stream_ = {};
    }
    /** Propagate drain failures before releasing the retained handles. */
    void finish() { program_.drain(); finished_ = true; }
private:
    tyloi::cuda::Program& program_;
    std::vector<tyloi::Tensor> retained_;
    std::vector<QuantLinearGroup> groups_;
    tyloi::DenseTensor latest_;
    std::vector<tyloi::DenseTensor> latest_group_;
    std::vector<tyloi::DenseTensor> warmup_;
    tyloi::cuda::StreamHandle graph_stream_;
    std::unique_ptr<tyloi::cuda::Graph> graph_;
    bool finished_ = false;
};
}

int run_linear_check(
    tyloi::cuda::Program& program,
    const std::string & name,
    int M,
    InputGateMode gate_mode,
    int reps) {
    Work work(program);
    auto loader = program.loader();
    auto projector = program.executor().projector(program.executor().mode().ffn);
    const auto device = program.executor().device();
    auto linear = loader.weight(name);
    work.keep(linear);
    TYLOINFER_RUNTIME_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    TYLOINFER_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    int64_t neuron_len = linear.size(1);
    auto x = tyloi::cuda::arange((int64_t)M * neuron_len,
                           tyloi::TensorOptions().device(device).dtype(tyloi::kFloat32))
                 .reshape({M, neuron_len});
    x = (x.remainder(97) - 48) / 512.0;
    work.keep(x);
    auto xh = x.to(tyloi::kFloat16).contiguous();
    work.keep(xh);
    tyloi::DenseTensor gateh;
    if (gate_mode != InputGateMode::None) {
        TYLOINFER_RUNTIME_CHECK(gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu, "linear check gate mode must be 0, 1, or 2");
        gateh = ((tyloi::cuda::arange((int64_t)M * neuron_len, x.options()).reshape({M, neuron_len})
                    .remainder(53) - 26) / 16.0)
                    .to(tyloi::kFloat16).contiguous();
        work.keep(gateh);
    }
    auto run = [&]() {
        return gate_mode == InputGateMode::None
            ? projector.run(linear, xh)
            : projector.gated(linear, xh, gateh, gate_mode);
    };
    const char * check_bf16_output_env =
        tyloi::cuda::env_lookup("TYLOINFER_CHECK_LINEAR_BF16_OUTPUT");
    if (check_bf16_output_env != nullptr &&
            check_bf16_output_env[0] == '1') {
        TYLOINFER_RUNTIME_CHECK(
            gate_mode == InputGateMode::None,
            "direct BF16 linear check does not support input gating");
        auto bf16_input = x.to(tyloi::kBFloat16).contiguous();
        work.keep(bf16_input);
        auto reference = projector.run(linear, bf16_input)
            .to(tyloi::kBFloat16).contiguous();
        work.keep(reference);
        auto candidate = projector.run(linear, bf16_input).to(tyloi::kBFloat16).contiguous();
        work.keep(candidate);
        auto difference =
            (candidate.to(tyloi::kFloat32) -
             reference.to(tyloi::kFloat32)).abs();
        std::cout << "linear_bf16_output_check"
                  << " max_abs="
                  << difference.max().item<double>()
                  << " equal="
                  << (candidate.equal(reference) ? 1 : 0)
                  << "\n";
        TYLOINFER_RUNTIME_CHECK(
            candidate.equal(reference),
            "direct BF16 NINT output differs from FP16-then-BF16 reference");
    }
    tyloi::DenseTensor y_test;
    const int warmups = std::min(30, std::max(1, reps));
    for (int i = 0; i < warmups; ++i) { y_test = run(); work.latest(y_test); }
    work.keep(y_test);
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    tyloi::cuda::Event start(cudaEventDefault), stop(cudaEventDefault);
    auto stream = tyloi::cuda::current_stream().stream();
    start.record(stream);
    for (int i = 0; i < reps; ++i) { y_test = run(); work.latest(y_test); }
    work.keep(y_test);
    stop.record(stream);
    stop.synchronize();
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start.get(), stop.get());
    y_test = y_test.to(tyloi::kFloat32);

    const auto* nint = linear.representation<tyloi::NintTensor>();
    const auto* nvq = linear.representation<tyloi::NvqTensor>();
    const auto* mxfp4 = linear.representation<tyloi::Mxfp4Tensor>();
    const auto* mxfp4_sq = linear.representation<tyloi::Mxfp4SqTensor>();
    const auto* fp8_sq = linear.representation<tyloi::Fp8SqTensor>();
    const auto* mxfp8 = linear.representation<tyloi::Mxfp8Tensor>();
    auto ww = quant_linear_reference_weight(linear);
    work.keep(ww);
    tyloi::DenseTensor ref_input = xh;
    if (gate_mode == InputGateMode::Sigmoid) ref_input = xh * tyloi::cuda::sigmoid(gateh);
    else if (gate_mode == InputGateMode::Silu) ref_input = xh * tyloi::cuda::silu(gateh);
    work.keep(ref_input);
    auto y_ref = tyloi::cuda::matmul(ref_input, ww.transpose(0, 1)).to(tyloi::kFloat32);
    work.keep(y_ref);
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    auto diff = (y_test - y_ref).abs();
    double per_ms = (double)elapsed_ms / (double)reps;
    double weight_bytes = nint != nullptr
        ? (double)nint->q_packed.numel() +
          (nint->q8_zero
              ? (double)nint->q8_zero_scale.numel() * sizeof(mfq_half)
              : (double)(nint->sub_scale.numel() +
                         nint->sub_min.numel()) +
                (double)(nint->neuron_scale.numel() +
                         nint->neuron_min.numel()) * sizeof(float))
        : nvq != nullptr
        ? (double)nvq->indices_packed.numel() + (double)nvq->aux_packed.numel() +
          (double)nvq->sub_scale_packed.numel() +
          (double)nvq->neuron_scale.numel() * sizeof(float) + (double)nvq->codebook.numel()
        : mxfp4 != nullptr
        ? (double)mxfp4->values.numel() + (double)mxfp4->scales.numel()
        : mxfp4_sq != nullptr
        ? (double)mxfp4_sq->blob.numel()
        : fp8_sq != nullptr
        ? (double)fp8_sq->blob.numel()
        : mxfp8 != nullptr
        ? (double)mxfp8->values.numel() + (double)mxfp8->scales.numel()
        : (double)linear.backing_nbytes();
    std::cout << "shape=" << y_ref.sizes() << "\n";
    if (nint != nullptr) {
        if (nint->q8_zero) {
            std::cout << "dtype=NINT8-0";
        } else {
            std::cout << "dtype=NINT format_version="
                      << nint->format_version
                      << " aggregate_bpw=" << nint->aggregate_bpw
                      << " distribution_entropy="
                      << nint->distribution_entropy
                      << " nominal_q=" << nint->bits;
        }
        std::cout << " gs=" << nint->gs << " m=" << M << "\n";
    } else if (nvq != nullptr) {
        std::cout << "dtype=NVQ profile=" << nvq->format << " gs=" << nvq->gs
                  << " sub_bits=" << nvq->sub_bits << " m=" << M << "\n";
    } else if (mxfp4 != nullptr) {
        std::cout << "dtype=MXFP4 block=1x32 m=" << M << "\n";
    } else if (mxfp4_sq != nullptr) {
        std::cout << "dtype=MXFP4-SQ format_version="
                  << mxfp4_sq->format_version
                  << " aggregate_bpw=" << mxfp4_sq->aggregate_bpw
                  << " distribution_entropy="
                  << mxfp4_sq->distribution_entropy;
        if (mxfp4_sq->bits != 0) {
            std::cout << " uniform_q=" << mxfp4_sq->bits;
        }
        std::cout << " block=1x32 m=" << M << "\n";
    } else if (fp8_sq != nullptr) {
        std::cout << "dtype=" << fp8_sq->dtype
                  << " format_version=" << fp8_sq->format_version
                  << " aggregate_bpw=" << fp8_sq->aggregate_bpw
                  << " distribution_entropy="
                  << fp8_sq->distribution_entropy
                  << " block=" << fp8_sq->block_rows
                  << "x" << fp8_sq->block_columns
                  << " m=" << M << "\n";
    } else if (mxfp8 != nullptr) {
        std::cout << "dtype=MXFP8 block=128x128 m=" << M << "\n";
    } else {
        std::cout << "dtype=" << (linear.is_encoded() ? "encoded" : "dense") << " m=" << M << "\n";
    }
    if (gate_mode != InputGateMode::None) std::cout << "gate=" << (gate_mode == InputGateMode::Sigmoid ? "sigmoid" : "silu") << "\n";
    std::cout << "production_ms=" << per_ms << "\n";
    std::cout << "production_weight_gbps=" << weight_bytes / (per_ms * 1.0e6) << "\n";
    std::cout << "production_rel=" << ((y_test - y_ref).norm() / y_ref.norm()).item<float>() << "\n";
    std::cout << "production_mean_abs=" << diff.mean().item<float>() << "\n";
    std::cout << "production_max_abs=" << diff.max().item<float>() << "\n";
    work.finish();
    return 0;
}

int run_cpu_linear_check(
        tyloi::cuda::Program& program,
        const std::string & name,
        int rows,
        InputGateMode gate_mode,
        int reps) {
    Work work(program);
    TYLOINFER_RUNTIME_CHECK(rows >= 1 && rows <= 4096, "--check-linear-m must be in [1, 4096]");
    TYLOINFER_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    auto loader = program.loader();
    auto projector = program.executor().projector(program.executor().mode().ffn);
    const auto device = program.executor().device();
    auto cpu_loader = loader.for_layer(device.index, true);
    auto cpu_linear = cpu_loader.weight(name);
    work.keep(cpu_linear);
    auto cuda_linear = loader.weight(name);
    work.keep(cuda_linear);
    const int64_t width = cpu_linear.size(1);
    auto x = tyloi::cuda::arange(
        static_cast<int64_t>(rows) * width,
        tyloi::TensorOptions().device(tyloi::kCPU).dtype(tyloi::kFloat32))
        .reshape({rows, width});
    x = ((x.remainder(97) - 48) / 512.0)
        .to(tyloi::kFloat16).contiguous();
    work.keep(x);
    tyloi::DenseTensor gate;
    if (gate_mode != InputGateMode::None) {
        TYLOINFER_RUNTIME_CHECK(gate_mode == InputGateMode::Sigmoid || gate_mode == InputGateMode::Silu, "linear check gate mode must be 0, 1, or 2");
        gate = ((tyloi::cuda::arange(
            static_cast<int64_t>(rows) * width,
            tyloi::TensorOptions().device(tyloi::kCPU).dtype(tyloi::kFloat32))
            .reshape({rows, width}).remainder(53) - 26) / 16.0)
            .to(tyloi::kFloat16).contiguous();
        work.keep(gate);
    }
    auto run_cpu = [&]() {
        return gate_mode == InputGateMode::None
            ? projector.run(cpu_linear, x)
            : projector.gated(cpu_linear, x, gate, gate_mode);
    };
    tyloi::DenseTensor actual = run_cpu();
    work.keep(actual);
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < reps; ++iteration) {
        actual = run_cpu();
        work.latest(actual);
    }
    work.keep(actual);
    const auto end = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(
        end - start).count() / static_cast<double>(reps);

    auto cuda_x = x.to(device).contiguous();
    work.keep(cuda_x);
    tyloi::DenseTensor cuda_gate;
    if (gate_mode != InputGateMode::None) cuda_gate = gate.to(device).contiguous();
    work.keep(cuda_gate);
    auto reference = gate_mode == InputGateMode::None
        ? projector.run(cuda_linear, cuda_x)
        : projector.gated(cuda_linear, cuda_x, cuda_gate, gate_mode);
    work.keep(reference);
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    reference = reference.to(tyloi::kCPU, tyloi::kFloat32).contiguous();
    actual = actual.to(tyloi::kFloat32).contiguous();
    auto difference = (actual - reference).abs();
    std::cout << "cpu_linear=" << name << "\n"
              << "shape=" << actual.sizes() << "\n"
              << "cpu_ms=" << cpu_ms << "\n"
              << "relative_error="
              << ((actual - reference).norm() / reference.norm()).item<float>() << "\n"
              << "mean_abs=" << difference.mean().item<float>() << "\n"
              << "max_abs=" << difference.max().item<float>() << "\n";
    work.finish();
    return 0;
}

int run_tensor_parallel_linear_check(
        tyloi::cuda::Program& program,
        const std::string & name,
        TensorParallelAxis axis,
        int M) {
    Work work(program);
    TYLOINFER_RUNTIME_CHECK(
        program.config().tensor_parallel.enabled(),
        "--check-tp-linear requires --tensor-parallel");
    TYLOINFER_RUNTIME_CHECK(
        axis == TensorParallelAxis::Output ||
        axis == TensorParallelAxis::Input,
        "--check-tp-axis must be output or input");
    TYLOINFER_RUNTIME_CHECK(
        M >= 1 && M <= 4096,
        "--check-tp-m must be in [1, 4096]");
    auto loader = program.loader();
    auto projector = program.executor().projector(program.executor().mode().ffn);
    const ParallelConfig saved = program.config().tensor_parallel;
    auto full_loader = loader;
    full_loader.placement.tensor = {};
    full_loader.placement.tensor.devices = {saved.primary_device()};
    auto full = full_loader.weight(name, TensorParallelAxis::Mirrored);
    work.keep(full);
    auto sharded = loader.weight(name, axis);
    work.keep(sharded);
    const auto* shards = sharded.representation<tyloi::ShardedTensor>();
    TYLOINFER_RUNTIME_CHECK(
        shards != nullptr,
        "tensor-parallel diagnostic did not create shards");

    const int64_t width = full.size(1);
    auto x = tyloi::cuda::arange(
        static_cast<int64_t>(M) * width,
        tyloi::TensorOptions()
            .device(tyloi::Device(
                tyloi::kCUDA,
                saved.primary_device()))
            .dtype(tyloi::kFloat32))
        .reshape({M, width});
    x = ((x.remainder(127) - 63) / 384.0)
        .to(tyloi::kFloat16).contiguous();
    work.keep(x);
    auto reference = projector.run(full, x).to(tyloi::kFloat32);
    work.keep(reference);
    auto test = projector.run(sharded, x).to(tyloi::kFloat32);
    work.keep(test);
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    const auto difference = (test - reference).abs();
    const double denominator =
        std::max(
            reference.norm().item<double>(),
            1.0e-30);
    const double relative =
        (test - reference).norm().item<double>() /
        denominator;
    const double mean_abs =
        difference.mean().item<double>();
    const double max_abs =
        difference.max().item<double>();
    std::cout
        << "tensor_parallel_check=1"
        << " tensor=" << name
        << " axis="
        << (axis == TensorParallelAxis::Output
            ? "output" : "input")
        << " logical_shape=[" << full.size(0)
        << ',' << full.size(1) << ']'
        << " shards="
        << shards->shards.size()
        << " m=" << M
        << " relative=" << relative
        << " mean_abs=" << mean_abs
        << " max_abs=" << max_abs
        << '\n';
    const double tolerance = full.representation<tyloi::Mxfp8Tensor>() != nullptr
        ? 5.0e-4
        : axis == TensorParallelAxis::Output
        ? 1.0e-6 : 5.0e-3;
    if (!tyloi::cuda::isfinite(test).all().item<bool>() ||
        relative > tolerance) {
        throw std::runtime_error(
            "tensor-parallel linear numerical check failed");
    }
    work.finish();
    return 0;
}

std::vector<std::string> parse_tensor_names(const std::string & value) {
    std::vector<std::string> names;
    size_t begin = 0;
    while (begin <= value.size()) {
        const size_t end = value.find(',', begin);
        const size_t count =
            end == std::string::npos ? value.size() - begin : end - begin;
        const std::string name = value.substr(begin, count);
        if (name.empty()) {
            throw std::runtime_error("empty tensor name in comma-separated list");
        }
        names.push_back(name);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return names;
}

int run_linear_group_check(
        tyloi::cuda::Program& program,
        const std::string & names_arg,
        int M,
        int reps) {
    Work work(program);
    TYLOINFER_RUNTIME_CHECK(M >= 1 && M <= 4096, "--check-linear-m must be in [1, 4096]");
    TYLOINFER_RUNTIME_CHECK(reps >= 1, "--check-linear-reps must be positive");
    const auto names = parse_tensor_names(names_arg);
    TYLOINFER_RUNTIME_CHECK(names.size() >= 2, "--check-linear-group requires at least two tensors");
    auto loader = program.loader();
    const auto device = program.executor().device();
    const char * preserve_env =
        tyloi::cuda::env_lookup("TYLOINFER_CHECK_LINEAR_GROUP_PRESERVE");
    const bool preserve_projection_boundaries =
        preserve_env != nullptr && preserve_env[0] == '1';
    const char * bf16_env =
        tyloi::cuda::env_lookup("TYLOINFER_CHECK_LINEAR_GROUP_BF16");
    const bool check_bf16 =
        bf16_env != nullptr && bf16_env[0] == '1';
    auto mode = program.executor().mode().ffn;
    mode.independent = preserve_projection_boundaries;
    auto projector = program.executor().projector(mode);
    auto group_weights = loader.group(names, names.size(), nullptr, preserve_projection_boundaries);
    work.keep_all(group_weights);
    auto group = projector.group(group_weights);
    work.keep_group(group);
    const int64_t width = group.nint_grouped
        ? (group.nint.split_w.empty()
            ? group.nint.w.neuron_len
            : group.nint.split_w.front().neuron_len)
        : group.layers.front().size(1);
    auto sequence = tyloi::cuda::arange(
        (int64_t)M * width,
        tyloi::TensorOptions().device(device).dtype(tyloi::kFloat32));
    auto x = (((sequence.remainder(257) - 128.0) / 127.0) +
              0.03125 * tyloi::cuda::sin(sequence * 0.015625))
                 .to(check_bf16
                     ? tyloi::kBFloat16
                     : tyloi::kFloat16)
                 .reshape({M, width})
                 .contiguous();
    work.keep(x);
    auto actual = projector.run(group, x);
    work.keep_all(actual);
    const char * check_bf16_swiglu_env =
        tyloi::cuda::env_lookup("TYLOINFER_CHECK_BF16_SWIGLU");
    if (check_bf16_swiglu_env != nullptr &&
            check_bf16_swiglu_env[0] == '1') {
        TYLOINFER_RUNTIME_CHECK(
            actual.size() == 2 &&
            actual[0].scalar_type() == tyloi::kBFloat16 &&
            actual[1].scalar_type() == tyloi::kBFloat16,
            "BF16 SwiGLU check requires two BF16 projection outputs");
        auto reference =
            (tyloi::cuda::silu(actual[0]) * actual[1]).contiguous();
        work.keep(reference);
        auto candidate = silu_mul_cuda(
            actual[0].contiguous(), actual[1].contiguous());
        work.keep(candidate);
        auto difference =
            (candidate.to(tyloi::kFloat32) -
             reference.to(tyloi::kFloat32)).abs();
        std::cout << "bf16_swiglu_check"
                  << " max_abs="
                  << difference.max().item<double>()
                  << " equal="
                  << (candidate.equal(reference) ? 1 : 0)
                  << "\n";
        TYLOINFER_RUNTIME_CHECK(
            candidate.equal(reference),
            "fused BF16 SwiGLU differs from the official BF16 expression");
    }
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    const auto started = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; ++rep) {
        actual = projector.run(group, x);
        work.latest_all(actual);
    }
    cudaStreamSynchronize(::tyloi::cuda::current_stream());
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::cout << "linear_group_timing"
              << " preserve=" << (preserve_projection_boundaries ? 1 : 0)
              << " m=" << M
              << " reps=" << reps
              << " mean_ms=" << elapsed_ms / reps << '\n';
    TYLOINFER_RUNTIME_CHECK(actual.size() == names.size(), "linear group output count mismatch");
    std::vector<tyloi::DenseTensor> graph_actual;
    if (M == 1 && tyloi::cuda::decode_branches(
                    program.executor().options().projection.parallel, program.executor().mode().ffn.serial, M)) {
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        const auto graph_stream =
            tyloi::cuda::stream_from_pool(false);
        tyloi::cuda::StreamGuard graph_guard(
            graph_stream);
        auto& graph = work.graph(graph_stream);
        graph.prepare_memory();
        work.reserve_warmup(names.size());
        graph_actual = projector.run(group, x);
        work.hold_warmup(graph_actual);
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(graph_stream.stream()));
        graph_actual.clear();
        work.release_warmup();
        graph.capture_begin();
        graph_actual = projector.run(group, x);
        work.keep_all(graph_actual);
        graph.capture_end();
        graph.replay();
        TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(
            graph_stream.stream()));
        TYLOINFER_RUNTIME_CHECK(
            graph_actual.size() == actual.size(),
            "linear group CUDA Graph output count mismatch");
        for (size_t index = 0;
             index < graph_actual.size(); ++index) {
            const auto difference =
                (graph_actual[index].to(tyloi::kFloat32) -
                 actual[index].to(tyloi::kFloat32)).abs();
            const double maximum =
                difference.max().item<double>();
            std::cout
                << "linear_group_graph_check"
                << " tensor=" << names[index]
                << " max_abs=" << maximum << "\n";
            TYLOINFER_RUNTIME_CHECK(
                maximum == 0.0,
                "linear group CUDA Graph replay differs from eager output");
        }
        work.finish_graph();
    }
    auto context = tyloi::cuda::default_context(device.index);
    context->blas().set_stream(tyloi::cuda::current_stream(device.index).stream());
    TYLOINFER_CUDA_CHECK(cublasSetMathMode(context->blas().get(), CUBLAS_DEFAULT_MATH));
    std::vector<tyloi::DenseTensor> dense_weights;
    std::vector<tyloi::DenseTensor> separate_dense_references;
    std::vector<tyloi::DenseTensor> fp32_references;
    std::vector<tyloi::DenseTensor> separate_production;
    dense_weights.reserve(names.size());
    separate_dense_references.reserve(names.size());
    fp32_references.reserve(names.size());
    separate_production.reserve(names.size());
    for (size_t index = 0; index < names.size(); ++index) {
        auto linear = loader.weight(names[index]);
        work.keep(linear);
        const auto* nint = linear.representation<tyloi::NintTensor>();
        TYLOINFER_RUNTIME_CHECK(
            nint != nullptr,
            "--check-linear-group currently requires NINT tensors");
        const auto & weight = *nint;
        tyloi::DenseTensor dense;
#if TYLOINFER_ENABLE_MFQ_COMPAT
        dense = weight.q8_zero
            ? nint8_zero_dequant_cuda(
                  weight.q_packed, weight.q8_zero_scale, weight.neuron_len)
            : nint_decode_cuda(
                  weight.q_packed, weight.row_q_bits,
                  weight.row_q_bit_offsets, weight.sub_scale,
                  weight.sub_min, weight.neuron_scale,
                  weight.neuron_min, weight.neuron_len, weight.gs);
#else
        throw std::runtime_error("NINT linear group diagnostic requires a compatibility build");
#endif
        work.keep(dense);
        auto separate = projector.run(linear, x);
        work.keep(separate);
        if (actual[index].scalar_type() == tyloi::kBFloat16) {
            separate = separate.to(tyloi::kBFloat16);
        }
        separate_production.push_back(separate.to(tyloi::kFloat32));
        auto dense_for_x = dense.to(x.scalar_type());
        separate_dense_references.push_back(
            tyloi::cuda::matmul(x, dense_for_x.transpose(0, 1))
                .to(tyloi::kFloat32));
        work.keep(separate_dense_references.back());
        fp32_references.push_back(tyloi::cuda::matmul(
            x.to(tyloi::kFloat32),
            dense.to(tyloi::kFloat32).transpose(0, 1)));
        work.keep(fp32_references.back());
        dense_weights.push_back(std::move(dense));
    }
    auto combined_dense = tyloi::cuda::cat(dense_weights, 0)
        .to(x.scalar_type()).contiguous();
    work.keep(combined_dense);
    auto combined_output =
        tyloi::cuda::matmul(x, combined_dense.transpose(0, 1)).to(tyloi::kFloat32);
    work.keep(combined_output);
    auto combined_references =
        combined_output.split_with_sizes(group.outs, -1);
    work.keep_all(combined_references);
    for (size_t index = 0; index < names.size(); ++index) {
        auto combined_reference = combined_references[index];
        auto separate_dense_reference = separate_dense_references[index];
        auto fp32_reference = fp32_references[index];
        auto separate_candidate = separate_production[index];
        auto candidate = actual[index].to(tyloi::kFloat32);
        auto combined_dense_difference = candidate - combined_reference;
        auto grouped_vs_separate = candidate - separate_candidate;
        auto separate_dense_difference =
            separate_candidate - separate_dense_reference;
        auto grouped_fp32_difference = candidate - fp32_reference;
        auto separate_fp32_difference = separate_candidate - fp32_reference;
        const double grouped_vs_separate_rel =
            (grouped_vs_separate.norm() /
             separate_candidate.norm().clamp_min(1.0e-30)).item<double>();
        const double grouped_vs_separate_snr =
            grouped_vs_separate_rel == 0.0
                ? std::numeric_limits<double>::infinity()
                : -20.0 * std::log10(grouped_vs_separate_rel);
        const double grouped_fp32_rel =
            (grouped_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        const double separate_fp32_rel =
            (separate_fp32_difference.norm() /
             fp32_reference.norm().clamp_min(1.0e-30)).item<double>();
        std::cout << std::fixed << std::setprecision(9)
                  << "linear_group_check"
                  << " tensor=" << names[index]
                  << " m=" << M
                  << " n=" << combined_reference.size(1)
                  << " k=" << width
                  << " grouped_vs_combined_dense_rel="
                  << (combined_dense_difference.norm() /
                      combined_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_separate_rel=" << grouped_vs_separate_rel
                  << " grouped_vs_separate_snr_db=" << grouped_vs_separate_snr
                  << " grouped_vs_separate_mean_abs="
                  << grouped_vs_separate.abs().mean().item<double>()
                  << " grouped_vs_separate_max_abs="
                  << grouped_vs_separate.abs().max().item<double>()
                  << " separate_vs_separate_dense_rel="
                  << (separate_dense_difference.norm() /
                      separate_dense_reference.norm().clamp_min(1.0e-30)).item<double>()
                  << " grouped_vs_fp32_rel=" << grouped_fp32_rel
                  << " grouped_vs_fp32_snr_db="
                  << (grouped_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(grouped_fp32_rel))
                  << " separate_vs_fp32_rel=" << separate_fp32_rel
                  << " separate_vs_fp32_snr_db="
                  << (separate_fp32_rel == 0.0
                          ? std::numeric_limits<double>::infinity()
                          : -20.0 * std::log10(separate_fp32_rel))
                  << "\n";
    }
    work.finish();
    return 0;
}

int run_q8_embedding_check(
        tyloi::cuda::Program& program,
        const std::string & name) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    Work work(program);
    auto loader = program.loader();
    const auto device = program.executor().device();
    auto linear = loader.weight(name);
    work.keep(linear);
    const auto* nint = linear.representation<tyloi::NintTensor>();
    TYLOINFER_RUNTIME_CHECK(
        nint && nint->q8_zero,
        "--check-q8-embedding requires an NINT8-0 tensor");
    const int64_t vocab = nint->out;
    std::vector<int64_t> host_ids = {
        0,
        std::min<int64_t>(1, vocab - 1),
        std::min<int64_t>(106, vocab - 1),
        std::min<int64_t>(12345, vocab - 1),
        std::min<int64_t>(255999, vocab - 1),
        vocab - 1,
    };
    auto host_id_tensor = tyloi::cuda::from_blob(
        host_ids.data(), {(int64_t)host_ids.size()},
        tyloi::TensorOptions().dtype(tyloi::kInt64))
                   .clone();
    work.keep(host_id_tensor);
    auto ids = host_id_tensor.to(device).contiguous();
    work.keep(ids);
    auto candidate = nint8_zero_embedding_lookup_cuda(
        nint->q_packed, nint->q8_zero_scale,
        ids, nint->neuron_len);
    work.keep(candidate);
    auto dense = nint8_zero_dequant_cuda(
        nint->q_packed, nint->q8_zero_scale,
        nint->neuron_len);
    work.keep(dense);
    auto reference = dense.index_select(0, ids);
    work.keep(reference);
    auto difference =
        candidate.to(tyloi::kFloat32) - reference.to(tyloi::kFloat32);
    std::cout << std::fixed << std::setprecision(9)
              << "q8_embedding_check"
              << " tensor=" << name
              << " ids=" << host_ids.size()
              << " vocab=" << vocab
              << " width=" << nint->neuron_len
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel="
              << (difference.norm() /
                  reference.to(tyloi::kFloat32).norm()).item<double>()
              << " mean_abs=" << difference.abs().mean().item<double>()
              << " max_abs=" << difference.abs().max().item<double>()
              << "\n";
    work.finish();
    return 0;
#else
    (void)program;
    (void)name;
    throw std::runtime_error("NINT8-zero embedding diagnostic requires a compatibility build");
#endif
}

int run_dsv4_output_a_check(
        tyloi::cuda::Program& program,
        const std::string & name,
        int batch,
        int reps) {
    Work work(program);
    constexpr int64_t kGroups = 8;
    TYLOINFER_RUNTIME_CHECK(batch > 0 && reps > 0, "DSV4 output_a check requires positive batch and reps");
    auto loader = program.loader();
    auto projector = program.executor().projector(program.executor().mode().ffn);
    const auto device = program.executor().device();
    auto linear = loader.weight(name, TensorParallelAxis::Input);
    work.keep(linear);
    const auto* nint = linear.representation<tyloi::NintTensor>();
    const bool mxfp8 = linear.representation<tyloi::Mxfp8Tensor>() != nullptr;
    const bool supported_nint =
        nint && nint->bits == 8 && nint->gs == 48;
    TYLOINFER_RUNTIME_CHECK(
        supported_nint || mxfp8,
        "DSV4 output_a check requires NINT8 gs48 or MXFP8");
    TYLOINFER_RUNTIME_CHECK(
        linear.size(0) % kGroups == 0,
        "DSV4 output_a rows must divide eight groups");

    const int64_t width = linear.size(1);
    const int64_t rows_per_group = linear.size(0) / kGroups;
    auto sequence = tyloi::cuda::arange(
        (int64_t)batch * kGroups * width,
        tyloi::TensorOptions().device(device).dtype(tyloi::kFloat32));
    auto grouped = (
        (sequence.remainder(257) - 128.0) / 127.0 +
        0.03125 * tyloi::cuda::sin(sequence * 0.015625))
        .to(tyloi::kFloat16)
        .reshape({batch, kGroups, width})
        .contiguous();
    work.keep(grouped);

    auto legacy = [&]() {
        auto expanded = projector.run(linear, grouped.reshape({batch * kGroups, width}))
            .reshape({batch, kGroups, kGroups, rows_per_group});
        std::vector<tyloi::DenseTensor> diagonal;
        diagonal.reserve(kGroups);
        for (int64_t group = 0; group < kGroups; ++group) {
            diagonal.push_back(
                expanded.index({Slice(), group, group, Slice()}));
        }
        return tyloi::cuda::stack(diagonal, 1).reshape({batch, linear.size(0)});
    };
    auto groupwise = [&]() {
        if (mxfp8) return projector.groupwise(linear, grouped, kGroups);
#if TYLOINFER_ENABLE_MFQ_COMPAT
        return nint_matmul_groupwise_u8(program.executor().profiler(), *nint, grouped, kGroups);
#else
        throw std::runtime_error("NINT groupwise diagnostic requires a compatibility build");
#endif
    };
    auto time_ms = [&](auto && fn) {
        tyloi::DenseTensor output;
        for (int warmup = 0; warmup < 10; ++warmup) { output = fn(); work.latest(output); }
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        tyloi::cuda::Event start(cudaEventDefault), stop(cudaEventDefault);
        auto stream = tyloi::cuda::current_stream().stream();
        start.record(stream);
        for (int iteration = 0; iteration < reps; ++iteration) { output = fn(); work.latest(output); }
        stop.record(stream);
        stop.synchronize();
        float elapsed = 0.0f;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), stop.get()));
        return std::pair<float, tyloi::DenseTensor>(elapsed / reps, output);
    };

    auto legacy_result = time_ms(legacy);
    work.keep(legacy_result.second);
    auto groupwise_result = time_ms(groupwise);
    work.keep(groupwise_result.second);
    auto reference = legacy_result.second.to(tyloi::kFloat32);
    auto candidate = groupwise_result.second.to(tyloi::kFloat32);
    auto diff = (candidate - reference).abs();
    const float relative =
        ((candidate - reference).norm() / reference.norm()).item<float>();
    std::cout << std::fixed << std::setprecision(9)
              << "dsv4_output_a_check"
              << " tensor=" << name
              << " format=" << (mxfp8 ? "MXFP8" : "NINT8")
              << " batch=" << batch
              << " groups=" << kGroups
              << " rows_per_group=" << rows_per_group
              << " k=" << width
              << " equal=" << (candidate.equal(reference) ? 1 : 0)
              << " rel=" << relative
              << " mean_abs=" << diff.mean().item<float>()
              << " max_abs=" << diff.max().item<float>()
              << " legacy_ms=" << legacy_result.first
              << " groupwise_ms=" << groupwise_result.first
              << " speedup=" << legacy_result.first / groupwise_result.first
              << " checksum=" << candidate.sum().item<double>()
              << "\n";
    if (mxfp8) {
        TYLOINFER_RUNTIME_CHECK(
            tyloi::cuda::isfinite(candidate).all().item<bool>() &&
                relative <= 5.0e-4f,
            "DSV4 MXFP8 groupwise output_a exceeded the FP16 GEMM "
            "reduction-order tolerance");
    } else {
        TYLOINFER_RUNTIME_CHECK(
            candidate.equal(reference),
            "DSV4 NINT groupwise output_a must be bit-exact with the "
            "legacy path");
    }
    work.finish();
    return 0;
}

int run_gemma_geglu_check(
    tyloi::cuda::Program& program,
    int layer,
    int reps) {
    Work work(program);
    if (layer < 0 || reps < 1) {
        throw std::runtime_error("Gemma GeGLU check requires a nonnegative layer and positive reps");
    }
    auto loader = program.loader();
    auto projector = program.executor().projector(program.executor().mode().ffn);
    const auto device = program.executor().device();
    tyloi::cuda::validate_model_source(dynamic_cast<const tyloi::ModelSource&>(loader.source));
    const std::string prefix =
        "model.block." + std::to_string(layer) + ".mlp.";
    auto group_weights = loader.group({prefix + "gate.weight", prefix + "up.weight"}, 2);
    work.keep_all(group_weights);
    auto gate_up = projector.group(group_weights);
    work.keep_group(gate_up);
    auto down = loader.weight(prefix + "down.weight");
    work.keep(down);
    const auto* down_nint = down.representation<tyloi::NintTensor>();
    if (!gate_up.nint_grouped || !gate_up.nint.split_w.empty() || !down_nint ||
        gate_up.outs.size() != 2 || gate_up.outs[0] != gate_up.outs[1]) {
        throw std::runtime_error("Gemma GeGLU check requires packed NINT gate/up and NINT down tensors");
    }

    const int64_t hidden = gate_up.nint.w.neuron_len;
    auto xf = tyloi::cuda::arange(
        hidden, tyloi::TensorOptions().device(device).dtype(tyloi::kFloat32));
    auto x = (((xf.remainder(257) - 128.0) / 64.0) +
              0.125 * tyloi::cuda::sin(xf * 0.03125)).to(tyloi::kFloat16).reshape({1, hidden}).contiguous();
    work.keep(x);

    auto materialized_activation = [&]() {
        auto parts = projector.run(gate_up, x);
        work.latest_all(parts);
        return gelu_mul_cuda(parts[0].contiguous(), parts[1].contiguous());
    };
    auto materialized = [&]() {
        auto activation = materialized_activation();
        work.latest(activation);
        return projector.run(down, activation);
    };
    auto fused_activation = [&]() {
        return projector.geglu(gate_up, x);
    };
    auto fused = [&]() {
        auto activation = fused_activation();
        work.latest(activation);
        return projector.run(down, activation);
    };

    auto reference_activation = materialized_activation();
    work.keep(reference_activation);
    auto reference_output = projector.run(down, reference_activation);
    work.keep(reference_output);
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_up_bits=" << gate_up.nint.w.bits
              << " gate_up_gs=" << gate_up.nint.w.gs
              << " gate_out=" << gate_up.outs[0]
              << " up_out=" << gate_up.outs[1]
              << " down_bits=" << down_nint->bits
              << " down_gs=" << down_nint->gs << "\n";
    auto report = [&](const char * name, tyloi::DenseTensor value, tyloi::DenseTensor reference) {
        work.keep(value);
        work.keep(reference);
        auto got = value.to(tyloi::kFloat64);
        auto ref = reference.to(tyloi::kFloat64);
        const double ref_norm = std::max(ref.norm().item<double>(), 1.0e-30);
        const double got_norm = std::max(got.norm().item<double>(), 1.0e-30);
        std::cout << "gemma_geglu_check path=" << name
                  << " relative_l2=" << (got - ref).norm().item<double>() / ref_norm
                  << " cosine=" << tyloi::cuda::dot(got.reshape({-1}), ref.reshape({-1})).item<double>() /
                         (got_norm * ref_norm)
                  << " max_abs=" << (got - ref).abs().max().item<double>() << "\n";
    };

    report("activation_combined", fused_activation(), reference_activation);
    report("output_combined", fused(), reference_output);
    report("activation_pair", fused_activation(), reference_activation);
    report("output_pair", fused(), reference_output);

    auto time_ms = [&](auto && fn) {
        for (int i = 0; i < 10; ++i) work.latest(fn());
        cudaStreamSynchronize(::tyloi::cuda::current_stream());
        tyloi::cuda::Event start(cudaEventDefault), stop(cudaEventDefault);
        auto stream = tyloi::cuda::current_stream().stream();
        start.record(stream);
        for (int i = 0; i < reps; ++i) work.latest(fn());
        stop.record(stream);
        stop.synchronize();
        float elapsed = 0.0f;
        TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.get(), stop.get()));
        return elapsed / reps;
    };
    const float materialized_ms = time_ms(materialized);
    const float combined_ms = time_ms(fused);
    const float pair_ms = time_ms(fused);
    std::cout << "gemma_geglu_check layer=" << layer
              << " gate_bits=" << gate_up.nint.w.bits
              << " gate_gs=" << gate_up.nint.w.gs
              << " down_bits=" << down_nint->bits
              << " down_gs=" << down_nint->gs
              << " materialized_ms=" << materialized_ms
              << " combined_ms=" << combined_ms
              << " pair_ms=" << pair_ms << "\n";
    work.finish();
    return 0;
}
