#include "trace.h"

#include "engine/program.h"
#include "engine/backends/cuda/executor/config.h"
#include "engine/backends/cuda/executor/executor.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/causal_model.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
/** Drain both executors before either owner's observations can be released. */
void drain_pair(tyloi::cuda::Program& first, tyloi::cuda::Program& second) {
    std::exception_ptr failure;
    try { first.drain(); } catch (...) { failure = std::current_exception(); }
    try { second.drain(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
}

/** Retain the input, observed stages and terminal outputs through executor drain. */
struct Work {
    tyloi::cuda::Program& program;
    tyloi::cuda::Program* peer = nullptr;
    tyloi::Tensor ids, hidden, logits;
    tyloi::models::Input::Trace trace;
    ~Work() noexcept {
        try {
            if (peer) drain_pair(*peer, program);
            else program.drain();
        }
        catch (const std::exception& error) { std::cerr << "trace_cleanup error=" << error.what() << '\n'; }
        catch (...) { std::cerr << "trace_cleanup error=unknown\n"; }
    }
};

/** Run a fresh request and retain each model-provided observation. */
void forward(Work& work, const tyloi::Tensor& ids) {
    auto& executor = work.program.executor();
    const auto primary = work.program.config().layer_placement.primary_device();
    auto scope = executor.enter(primary);
    work.ids = ids.dense().to(tyloi::Device(tyloi::kCUDA, primary)).contiguous();
    work.program.state().reset(work.ids.size(0));
    tyloi::models::Input input;
    input.ids = work.ids;
    input.trace = &work.trace;
    const auto mode = executor.forward_mode(primary);
    work.hidden = work.program.model().forward(std::move(input), work.program.state(),
        {mode.kv_capacity, mode.trace, mode.trace_layer, primary});
    work.program.drain();
}

/** Project retained normalized hidden states using the model's requested cap. */
void project(Work& work, const tyloi::Tensor& hidden, bool capped) {
    auto& executor = work.program.executor();
    auto scope = executor.enter(work.program.config().layer_placement.primary_device());
    work.logits = work.program.model().logits(hidden, capped);
    work.program.drain();
}

/** Slice a token-major observation exactly as the original dump did. */
tyloi::DenseTensor token_slice(const tyloi::Tensor& value, std::int64_t total,
        std::int64_t start, std::int64_t count) {
    auto dense = value.dense();
    return dense.dim() >= 3 && dense.size(1) == total ? dense.narrow(1, start, count) : dense;
}

/** Write one contiguous F32 observation and its JSONL record. */
void write_stage(const std::filesystem::path& root, std::ofstream& metadata,
        const std::string& stage, const tyloi::DenseTensor& tensor) {
    auto value = tensor.to(tyloi::kCPU, tyloi::kFloat32).contiguous();
    const auto file = root / (stage + ".f32");
    std::ofstream output(file, std::ios::binary);
    if (!output) throw std::runtime_error("cannot create block trace tensor: " + file.string());
    output.write(reinterpret_cast<const char*>(value.data_ptr<float>()),
        static_cast<std::streamsize>(value.nbytes()));
    if (!output) throw std::runtime_error("failed to write block trace tensor: " + file.string());
    metadata << "{\"stage\":\"" << stage << "\",\"file\":\"" << file.filename().string() << "\",\"shape\":[";
    for (std::int64_t dim = 0; dim < value.dim(); ++dim) {
        if (dim) metadata << ',';
        metadata << value.size(dim);
    }
    metadata << "],\"dtype\":\"float32\"}\n";
    std::cout << "block_trace_dump stage=" << stage << " values=" << value.numel() << "\n";
}

/** Name the embedding stage and each subsequent transformer block. */
std::string stage_name(std::size_t index) {
    return index == 0 ? "embedding" : "block_" + std::to_string(index - 1);
}
} // namespace

int run_block_trace_compare(tyloi::cuda::Program& test,
        tyloi::cuda::CudaLoadOptions options, tyloi::Tensor ids) {
    if (!ids || ids.dim() != 2 || ids.scalar_type() != tyloi::kInt64)
        throw std::invalid_argument("block trace IDs must be int64 [batch,tokens]");
    options.n_gpu_layers = -1;
    options.n_gpu_layers_set = false;
    auto reference = tyloi::cuda::Program::open(std::move(options));
    reference->load(false);
    Work actual{test, reference.get()}, expected{*reference, &test};
    forward(actual, ids);
    forward(expected, ids);
    if (actual.trace.size() != expected.trace.size())
        throw std::runtime_error("block trace stage count mismatch");
    auto scope = test.executor().enter(test.config().layer_placement.primary_device());
    const auto device = test.executor().device();
    for (std::size_t i = 0; i < actual.trace.size(); ++i) {
        auto ref = expected.trace[i].dense().to(device).reshape({-1}).to(tyloi::kFloat64);
        auto got = actual.trace[i].dense().to(device).reshape({-1}).to(tyloi::kFloat64);
        if (ref.numel() != got.numel()) throw std::runtime_error("block trace tensor size mismatch");
        const double ref_norm = ref.norm().item<double>();
        const double got_norm = got.norm().item<double>();
        const double denominator = std::max(ref_norm, 1.0e-30);
        const double relative = (got - ref).norm().item<double>() / denominator;
        const double cosine = tyloi::cuda::dot(ref, got).item<double>() / std::max(ref_norm * got_norm, 1.0e-30);
        const double reference_rms = ref.square().mean().sqrt().item<double>();
        const double test_rms = got.square().mean().sqrt().item<double>();
        std::cout << "block_trace stage=" << stage_name(i) << " relative_l2=" << relative
                  << " cosine=" << cosine << " norm_ratio=" << got_norm / denominator
                  << " reference_rms=" << reference_rms << " test_rms=" << test_rms << "\n";
    }
    project(expected, expected.hidden, false);
    project(actual, actual.hidden, false);
    auto ref_logits = expected.logits.dense().to(device).to(tyloi::kFloat32);
    auto got_logits = actual.logits.dense().to(device).to(tyloi::kFloat32);
    const std::int64_t tokens = ref_logits.numel() / ref_logits.size(-1);
    auto ref2 = ref_logits.reshape({tokens, -1});
    auto got2 = got_logits.reshape({tokens, -1});
    double kl_sum = 0.0;
    std::int64_t same_top = 0, rows = 0;
    for (std::int64_t start = 0; start < tokens; start += 8) {
        const auto end = std::min(start + 8, tokens);
        auto ref = ref2.narrow(0, start, end - start);
        auto got = got2.narrow(0, start, end - start);
        auto ref_logp = tyloi::cuda::log_softmax(ref, -1);
        auto got_logp = tyloi::cuda::log_softmax(got, -1);
        kl_sum += (ref_logp.exp() * (ref_logp - got_logp)).sum(-1)
            .to(tyloi::kFloat64).sum().item<double>();
        same_top += ref.argmax(-1).eq(got.argmax(-1)).sum().item<std::int64_t>();
        rows += end - start;
    }
    auto ref64 = ref_logits.to(tyloi::kFloat64);
    auto got64 = got_logits.to(tyloi::kFloat64);
    const double relative = (got64 - ref64).norm().item<double>() / std::max(ref64.norm().item<double>(), 1.0e-30);
    drain_pair(test, *reference);
    std::cout << "block_trace_logits kld=" << kl_sum / std::max<std::int64_t>(rows, 1)
              << " same_top=" << double(same_top) / std::max<std::int64_t>(rows, 1)
              << " relative_l2=" << relative << "\n";
    return 0;
}

int run_block_trace_dump(tyloi::cuda::Program& program, const std::string& output_dir,
        tyloi::Tensor ids, std::int64_t token_start, std::int64_t token_count) {
    if (!ids || ids.dim() != 2 || ids.scalar_type() != tyloi::kInt64)
        throw std::invalid_argument("block trace IDs must be int64 [batch,tokens]");
    const auto total = ids.size(1);
    if (token_start < 0 || token_start >= total)
        throw std::runtime_error("--dump-block-trace-start is outside the token range");
    if (token_count <= 0) token_count = total - token_start;
    if (token_count > total - token_start)
        throw std::runtime_error("--dump-block-trace-count exceeds the token range");
    const std::filesystem::path root(output_dir);
    std::error_code error;
    if (!std::filesystem::create_directories(root, error) || error)
        throw std::runtime_error("block trace output directory must be new: " + output_dir);
    Work work{program};
    forward(work, ids);
    auto scope = program.executor().enter(program.config().layer_placement.primary_device());
    auto ids_cpu = work.ids.dense().to(tyloi::kCPU, tyloi::kInt32).contiguous();
    {
        std::ofstream output(root / "tokens.i32", std::ios::binary);
        if (!output) throw std::runtime_error("cannot create block trace token file");
        output.write(reinterpret_cast<const char*>(ids_cpu.data_ptr<std::int32_t>()),
            static_cast<std::streamsize>(ids_cpu.nbytes()));
        if (!output) throw std::runtime_error("failed to write block trace tokens");
    }
    std::ofstream metadata(root / "trace_meta.jsonl");
    if (!metadata) throw std::runtime_error("cannot create block trace metadata");
    for (std::size_t i = 0; i < work.trace.size(); ++i)
        write_stage(root, metadata, stage_name(i), token_slice(work.trace[i], total, token_start, token_count));
    auto hidden = token_slice(work.hidden, total, token_start, token_count);
    write_stage(root, metadata, "final_norm", hidden);
    project(work, hidden, true);
    write_stage(root, metadata, "logits", work.logits.dense());
    metadata.flush();
    if (!metadata) throw std::runtime_error("failed to write block trace metadata");
    program.drain();
    return 0;
}
