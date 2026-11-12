#include "gemma.h"
#include "tensor.h"
#include "engine/backends/cuda/kernels/mfq_cuda_norm_ops.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyloi::diagnostics::numerics {
namespace {
using tyloi::DenseTensor;

DenseTensor input(TensorWork& work, const std::string& name, tyloi::ScalarType type,
                  const std::vector<std::int64_t>& shape) {
    auto value = work.input(name);
    if (value.scalar_type() != type || value.sizes().vec() != shape ||
        value.device() != work.device() || !value.is_contiguous())
        throw std::invalid_argument("Gemma fixed input geometry or dtype mismatch: " + name);
    return value;
}
} // namespace

bool run_gemma(Case& value) {
    const auto op = value.data().at("op").get<std::string>();
    if (op != "gemma_pre_norms" && op != "gemma_ffn_merge")
        return false;
    if (value.data().at("source").at("generator") != "tests/numerics/gemma_ref.py")
        throw std::invalid_argument("unsupported Gemma case source");
    const auto& config = value.data().at("config");
    const auto rows = config.at("rows").get<std::int64_t>();
    const auto width = config.at("width").get<std::int64_t>();
    const auto eps = config.at("eps").get<double>();
    if ((rows != 1 && rows != 16) || width != 2816 || eps != 1e-6 ||
        config.at("category") != "native_execution_consistency")
        throw std::invalid_argument("unsupported Gemma case geometry or policy");

    std::vector<DenseTensor> results;
    DenseTensor output;
    TensorWork work(value);
    const std::vector<std::int64_t> shape{rows, width};
    const std::vector<std::int64_t> weight_shape{width};
    auto residual = input(work, "residual", tyloi::kFloat16, shape);
    auto weight0 = input(work, "weight0", tyloi::kFloat32, weight_shape);
    auto weight1 = input(work, "weight1", tyloi::kFloat32, weight_shape);
    auto weight2 = input(work, "weight2", tyloi::kFloat32, weight_shape);
    if (op == "gemma_pre_norms") {
        auto attention = input(work, "attn", tyloi::kFloat16, shape);
        auto weight3 = input(work, "weight3", tyloi::kFloat32, weight_shape);
        results = gemma4_attn_residual_pre_norms_f16_cuda(
            residual, attention, weight0, weight1, weight2, weight3, eps);
        for (auto& item : results)
            work.keep(item);
        if (results.size() != 4)
            throw std::runtime_error("Gemma fused pre norms returned an unexpected result count");
        work.output("residual", results[0]);
        work.output("dense", results[1]);
        work.output("router", results[2]);
        work.output("moe", results[3]);
    } else {
        auto dense = input(work, "dense", tyloi::kFloat16, shape);
        auto moe = input(work, "moe", tyloi::kFloat16, shape);
        auto scale = input(work, "layer_scale", tyloi::kFloat16, {1});
        output = gemma4_ffn_merge_f16_cuda(
            dense, moe, residual, weight0, weight1, weight2, scale, eps);
        work.keep(output);
        work.output("output", output);
    }
    work.finish(op, {{"category", "native_execution_consistency"},
                     {"provider", "Gemma fused CUDA"}});
    return true;
}
} // namespace tyloi::diagnostics::numerics
