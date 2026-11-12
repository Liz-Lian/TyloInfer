#include "flash_reject.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/glm5_next.h"
#include "engine/backends/cuda/kernels/qwen4_exp.h"
#include "engine/backends/cuda/native/device.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace tyloi::diagnostics::numerics {

bool run_flash_reject(Case& value) {
    if (value.data().at("op") != "flash_rejection")
        return false;
    if (value.data().at("source").at("generator") != "tests/numerics/flash_ref.py")
        throw std::invalid_argument("unsupported Flash-Next rejection source");
    const auto& config = value.data().at("config");
    const auto op = config.at("operation").get<std::string>();
    std::size_t count;
    std::int64_t parameter = 0;
    if (op == "qsa_block_scores")
        count = 2;
    else if (op == "glm5_kpool_states") {
        count = 3;
        parameter = config.at("pool_size").get<std::int64_t>();
    } else if (op == "qwen4_ple_dilated_conv_silu") {
        count = 3;
        parameter = config.at("dilation").get<std::int64_t>();
    } else if (op == "qwen4_sparse_gqa_attention")
        count = 4;
    else if (op == "qwen4_dense_gqa_attention") {
        count = 3;
        parameter = config.at("query_offset").get<std::int64_t>();
    } else
        throw std::invalid_argument("unsupported Flash-Next rejection operation");

    TensorWork work(value);
    std::vector<tyloi::DenseTensor> input;
    for (std::size_t index = 0; index < count; ++index)
        input.push_back(work.input("input_" + std::to_string(index)));
    tyloi::DenseTensor result;
    std::vector<tyloi::DenseTensor> results;
    std::uint8_t rejected = 0;
    Json evidence = {{"threw", false}};
    try {
        if (op == "qsa_block_scores")
            result = mfq_qwen4_exp::block_scores(input[0], input[1]);
        else if (op == "glm5_kpool_states")
            result = mfq_glm5_next::kpool_states(input[0], input[1], input[2], parameter);
        else if (op == "qwen4_ple_dilated_conv_silu")
            results = mfq_qwen4_exp::ple_dilated_conv_silu(input[0], input[1], input[2], parameter);
        else if (op == "qwen4_sparse_gqa_attention")
            result = mfq_qwen4_exp::sparse_gqa_attention(input[0], input[1], input[2], input[3]);
        else
            result = mfq_qwen4_exp::dense_gqa_attention(input[0], input[1], input[2], parameter);
    } catch (const tyloi::cuda::Error&) {
        throw;
    } catch (const std::invalid_argument& error) {
        rejected = 1;
        evidence = {{"threw", true}, {"type", typeid(error).name()}, {"message", error.what()}};
    } catch (const std::runtime_error& error) {
        rejected = 1;
        evidence = {{"threw", true}, {"type", typeid(error).name()}, {"message", error.what()}};
    }
    work.keep(result);
    for (auto& item : results)
        work.keep(item);
    value.output("rejected", Json::array(), "bool", &rejected, 1);
    work.finish("flash_rejection", {{"operation", op}, {"rejection", evidence}});
    return true;
}
} // namespace tyloi::diagnostics::numerics
