#include "models/common/axial_patch_vision_ops.h"
#include "backends/cuda/engine/executor/execution.h"
#include "backends/cuda/engine/models/vision.h"
#include "backends/cuda/engine/weights/vision.h"
#include "engine/executor/sequence.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mfq::cuda::axial_patch_vision {
namespace {
using Tensor = mfq_tensor_backend::Tensor;
using vision::Affine;
using vision::RmsNorm;

struct HostLoading {
    bool& flag;
    bool previous;
    explicit HostLoading(CudaExecutionContext& execution)
        : flag(execution.loading_cpu_layer), previous(flag) { flag = true; }
    ~HostLoading() { flag = previous; }
};

Affine affine(CudaExecutionContext& execution, const mfq::ModelSource& source,
              const std::string& prefix, std::int64_t input, std::int64_t output) {
    auto result = vision::load_affine(execution, source, prefix + ".weight", prefix + ".bias");
    TYLOINFER_RUNTIME_CHECK(result.linear.neuron_len() == input && result.linear.out() == output,
                      "axial patch vision affine geometry mismatch: " + prefix);
    return result;
}

RmsNorm norm(CudaExecutionContext& execution, const mfq::ModelSource& source,
             const std::string& name, std::int64_t width, double epsilon) {
    return vision::load_rms_norm(execution, source, name, width, epsilon);
}

struct BlockWeights {
    RmsNorm norm1, norm2;
    Affine qkv, output, gate_up, down;
    template<class F> void visit_tensors(F visit) {
        norm1.visit_tensors(visit); norm2.visit_tensors(visit);
        vision::visit_tensors(qkv, visit); vision::visit_tensors(output, visit);
        vision::visit_tensors(gate_up, visit); vision::visit_tensors(down, visit);
    }
};

struct AlignerWeights {
    RmsNorm norm;
    Affine input, output;
    template<class F> void visit_tensors(F visit) {
        norm.visit_tensors(visit); vision::visit_tensors(input, visit);
        vision::visit_tensors(output, visit);
    }
};

template<class Weights, class Forward>
Tensor use_weights(const std::shared_ptr<MoeExpertCache>& cache, Weights weights,
                   std::size_t activations, Forward forward) {
    return vision::workspace(cache, vision::weight_bytes(weights) + activations, [&] {
        auto device = vision::to_cuda(std::move(weights));
        return forward(device);
    });
}
} // namespace

struct CudaAxialPatchVisionTower::Impl {
    // The model owns the source for the lifetime of its vision component.
    const mfq::ModelSource* source;
    Config config;

    mfq::StepSequence<Tensor> encode_steps(CudaExecutionContext& execution, Tensor patches,
                                          int height, int width) const {
        if (height <= 0 || width <= 0 || height > std::numeric_limits<int>::max() / width)
            throw std::invalid_argument("invalid axial patch vision grid");
        const std::int64_t count = std::int64_t(height) * width;
        const auto hidden = config.hidden, heads = config.heads, dim = config.head_dim();
        if (patches.dim() == 4) patches = patches.reshape({count, config.patch_width()});
        if (patches.dim() != 2 || patches.size(0) != count || patches.size(1) != config.patch_width())
            throw std::invalid_argument("axial patch vision patch geometry mismatch");
        patches = patches.to(mfq_tensor_backend::kFloat16);
        const auto activations = std::size_t(count) * std::size_t(std::max(6 * hidden, 2 * config.intermediate)) * sizeof(float)
            + (std::size_t(32) << 20);
        Tensor value;
        {
            Affine patch;
            { HostLoading loading(execution);
              patch = affine(execution, *source, "vision.patch_embedding", config.patch_width(), hidden); }
            value = use_weights(execution.stores.weights.moe, std::move(patch), activations,
                                [&](const auto& weight) { return weight(execution.projector(), patches); });
        }
        co_yield mfq::StepState::advanced;
        std::vector<std::int32_t> positions;
        positions.reserve(std::size_t(count) * 2);
        for (int row = 0; row < height; ++row)
            for (int column = 0; column < width; ++column) {
                positions.push_back(row); positions.push_back(column);
            }
        for (std::int64_t layer = 0; layer < config.layers; ++layer) {
            BlockWeights block;
            {
                HostLoading loading(execution);
                const auto prefix = "vision.block." + std::to_string(layer);
                block.norm1 = norm(execution, *source, prefix + ".norm1.weight", hidden, config.rms_eps);
                block.norm2 = norm(execution, *source, prefix + ".norm2.weight", hidden, config.rms_eps);
                block.qkv = affine(execution, *source, prefix + ".attention.qkv", hidden, 3 * hidden);
                block.output = affine(execution, *source, prefix + ".attention.output", hidden, hidden);
                block.gate_up = affine(execution, *source, prefix + ".mlp.gate_up", hidden, 2 * config.intermediate);
                block.down = affine(execution, *source, prefix + ".mlp.down", config.intermediate, hidden);
            }
            value = use_weights(execution.stores.weights.moe, std::move(block), activations, [&](const auto& weights) {
                const auto projector = execution.projector();
                auto qkv = weights.qkv(projector, weights.norm1(value));
                const auto rope = [&](int index) {
                    return vision::apply_axial_rope(
                        qkv.narrow(-1, index * hidden, hidden).reshape({count, heads, dim}), positions, config.rope_theta);
                };
                auto query = rope(0), key = rope(1);
                auto v = qkv.narrow(-1, 2 * hidden, hidden).reshape({count, heads, dim});
                auto attended = vision::segmented_attention(query, key, v, {std::int32_t(count)}).reshape({count, hidden});
                auto result = value + weights.output(projector, std::move(attended));
                auto gate_up = weights.gate_up(projector, weights.norm2(result));
                return result + weights.down(projector,
                    mfq_tensor_backend::silu(gate_up.narrow(-1, 0, config.intermediate)) *
                    gate_up.narrow(-1, config.intermediate, config.intermediate));
            });
            co_yield mfq::StepState::advanced;
        }
        {
            const auto ratio = int(config.downsample_ratio);
            const auto grid = mfq::models::axial_patch_vision::aligned_grid(height, width, ratio);
            const auto indices = mfq::models::axial_patch_vision::gather_indices(height, width, ratio);
            AlignerWeights weights;
            {
                HostLoading loading(execution);
                weights.norm = norm(execution, *source, "vision.output_norm.weight", hidden, config.rms_eps);
                weights.input = affine(execution, *source, "vision.aligner.input", hidden * ratio * ratio, config.output_hidden);
                weights.output = affine(execution, *source, "vision.aligner.output", config.output_hidden, config.output_hidden);
            }
            value = use_weights(execution.stores.weights.moe, std::move(weights), activations, [&](const auto& device) {
                const auto projector = execution.projector();
                auto padded = mfq_tensor_backend::cat({device.norm(value), mfq_tensor_backend::zeros({1, hidden}, value.options())}, 0);
                auto index = mfq_tensor_backend::tensor(indices,
                    mfq_tensor_backend::TensorOptions().dtype(mfq_tensor_backend::kInt32).device(mfq_tensor_backend::kCPU))
                    .to(mfq_tensor_backend::kInt64).to(mfq_tensor_backend::kCUDA);
                auto gathered = padded.index_select(0, index).reshape({grid.count(), ratio * ratio, hidden})
                    .transpose(1, 2).contiguous().reshape({grid.count(), hidden * ratio * ratio});
                auto projected = device.input(projector, std::move(gathered));
                return device.output(projector, mfq_tensor_backend::gelu(projected, "none"));
            });
        }
        co_yield std::move(value);
    }
};

CudaAxialPatchVisionTower CudaAxialPatchVisionTower::load(
    CudaExecutionContext& execution, const mfq::ModelSource& model, const Config& config) {
    (void)execution;
    config.validate();
    for (const auto* name : {"vision.patch_embedding.weight", "vision.output_norm.weight",
                            "vision.aligner.input.weight", "vision.aligner.output.weight"})
        TYLOINFER_RUNTIME_CHECK(has_tensor(model, name), std::string("axial patch vision tensor missing: ") + name);
    return CudaAxialPatchVisionTower(std::make_unique<Impl>(Impl{&model, config}));
}

CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(CudaAxialPatchVisionTower&&) noexcept = default;
CudaAxialPatchVisionTower& CudaAxialPatchVisionTower::operator=(CudaAxialPatchVisionTower&&) noexcept = default;
CudaAxialPatchVisionTower::~CudaAxialPatchVisionTower() = default;

mfq::StepSequence<Tensor> CudaAxialPatchVisionTower::encode_steps(
    CudaExecutionContext& execution, Tensor patches, int height, int width) const {
    return impl_->encode_steps(execution, std::move(patches), height, width);
}

Tensor CudaAxialPatchVisionTower::encode(CudaExecutionContext& execution, Tensor patches, int height, int width) const {
    auto steps = encode_steps(execution, std::move(patches), height, width);
    Tensor output;
    while (auto step = steps.next()) if (step.value) output = std::move(*step.value);
    return output;
}

const Config& CudaAxialPatchVisionTower::config() const noexcept { return impl_->config; }
} // namespace mfq::cuda::axial_patch_vision
