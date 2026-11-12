#include "sample.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_embedding_ops.h"
#include "engine/backends/cuda/kernels/mfq_cuda_sampling_ops.h"

#include <string>
#include <utility>

namespace tyloi::diagnostics::numerics {
bool run_sample(Case& value) {
    const auto& data = value.data();
    const auto op = data.at("op").get<std::string>();
    if (op != "sample_greedy" && op != "sample_softmax" && op != "sample_topk_topp" &&
        op != "sample_tie" && op != "sample_penalties" && op != "embedding")
        return false;
    TensorWork work(value);
    const auto& config = data.at("config");
    Json checks = Json::object();
    tyloi::DenseTensor output;
    if (op == "embedding") {
        auto weight = work.input("weight");
        auto ids = work.input("ids");
        output = work.keep(embedding_lookup_cuda(weight, ids));
    } else {
        auto logits = work.input("logits");
        if (op == "sample_greedy") {
            output = work.keep(sample_greedy_cuda(logits));
        } else if (op == "sample_penalties") {
            auto counts = work.input("counts");
            auto tokens = work.input("tokens");
            sample_token_counts_add_cuda(counts, tokens);
            output = work.keep(sample_apply_penalties_cuda(
                logits, counts, config.at("presence").get<double>(),
                config.at("frequency").get<double>(), config.at("repetition").get<double>()));
            const bool same_pointer = output.data_ptr() == logits.data_ptr();
            checks["native_0"] = {{"status", same_pointer ? "pass" : "fail"},
                                  {"evidence",
                                   {{"entrypoint", "sample_apply_penalties_cuda"},
                                    {"input_output_same_pointer", same_pointer}}}};
            work.output("counts", counts);
        } else {
            auto random = work.input("random");
            const auto temperature = config.at("temperature").get<double>();
            if (op == "sample_softmax")
                output = work.keep(sample_softmax_cuda(logits, random, temperature));
            else
                output = work.keep(sample_top_k_top_p_cuda(logits, random, temperature,
                                                           config.at("top_k").get<std::int64_t>(),
                                                           config.at("top_p").get<double>()));
        }
    }
    work.output("output", output);
    work.finish(op, Json::object(), std::move(checks));
    return true;
}
} // namespace tyloi::diagnostics::numerics
