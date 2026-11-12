#include "flash_graph.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/glm5_next.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/graph.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tyloi::diagnostics::numerics {
namespace {
using tyloi::DenseTensor;

struct Run {
    explicit Run(tyloi::cuda::StreamHandle selected) : stream(std::move(selected)) {}
    ~Run() noexcept {
        graph.reset();
        (void)cudaStreamSynchronize(stream.stream());
    }
    tyloi::cuda::StreamHandle stream;
    tyloi::cuda::Graph graph;
    DenseTensor warmup, result, changed_eager;
};

DenseTensor compute(const std::string& op, const std::vector<DenseTensor>& input,
                    const Json& config) {
    if (op == "qsa_block_scores")
        return mfq_qwen4_exp::block_scores(input.at(0), input.at(1));
    if (op == "qwen4_sparse_gqa_attention")
        return mfq_qwen4_exp::sparse_gqa_attention(input.at(0), input.at(1), input.at(2), input.at(3));
    if (op == "glm5_kda_forget_gate")
        return mfq_glm5_next::kda_forget_gate(input.at(0), input.at(1), input.at(2),
                                              input.at(3), input.at(4),
                                              config.at("num_heads").get<std::int64_t>(),
                                              config.at("head_dim").get<std::int64_t>(),
                                              config.at("lower_bound").get<double>());
    throw std::invalid_argument("unsupported Flash-Next graph operation");
}
} // namespace

bool run_flash_graph(Case& value) {
    if (value.data().at("op") != "flash_graph")
        return false;
    if (value.data().at("source").at("generator") != "tests/numerics/flash_ref.py")
        throw std::invalid_argument("unsupported Flash-Next graph source");
    const auto& config = value.data().at("config");
    const auto op = config.at("operation").get<std::string>();
    std::size_t count;
    if (op == "qsa_block_scores")
        count = 2;
    else if (op == "qwen4_sparse_gqa_attention")
        count = 4;
    else if (op == "glm5_kda_forget_gate")
        count = 5;
    else
        throw std::invalid_argument("unsupported Flash-Next graph operation");
    if (op == "glm5_kda_forget_gate" &&
        (config.at("num_heads").get<std::int64_t>() != 3 ||
         config.at("head_dim").get<std::int64_t>() != 4 ||
         config.at("lower_bound").get<double>() != -5.0))
        throw std::invalid_argument("unsupported Flash-Next graph KDA parameters");

    TensorWork work(value);
    const auto stream = tyloi::cuda::stream_from_pool(false);
    tyloi::cuda::StreamGuard guard(stream);
    Run run(stream);
    std::vector<DenseTensor> input, changed;
    for (std::size_t index = 0; index < count; ++index) {
        input.push_back(work.input("input_" + std::to_string(index)));
        changed.push_back(work.input("changed_" + std::to_string(index)));
    }
    run.changed_eager = compute(op, changed, config);
    run.warmup = compute(op, input, config);
    cuda_check(cudaStreamSynchronize(stream.stream()), "graph first warmup synchronization");
    run.warmup = {};
    run.graph.prepare_memory();
    run.warmup = compute(op, input, config);
    cuda_check(cudaStreamSynchronize(stream.stream()), "graph pool warmup synchronization");
    run.warmup = {};
    run.graph.capture_begin();
    run.result = compute(op, input, config);
    run.graph.capture_end();
    run.graph.replay();
    run.graph.replay();
    for (std::size_t index = 0; index < count; ++index)
        input[index].copy_(changed[index]);
    run.graph.replay();
    cuda_check(cudaStreamSynchronize(stream.stream()), "graph changed replay synchronization");
    work.output("eager_changed", run.changed_eager);
    work.output("replay_changed", run.result);
    const std::uint8_t completed = 1;
    value.output("graph_completed", Json::array(), "bool", &completed, 1);
    work.finish("flash_graph", {{"operation", op}, {"stream", "nondefault pool"},
                                 {"warmups", 2}, {"replays", 3}});
    return true;
}
} // namespace tyloi::diagnostics::numerics
