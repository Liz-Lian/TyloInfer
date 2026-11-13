#include "engine/models/deepseek_v41/axial_patch_encoder.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/projection/projection.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/ops.h"
#include "engine/weights/fields.h"
#include "engine/weights/loader.h"
#include "engine/generation/sequence.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::axial_patch_vision {
namespace {
using vision::RmsNorm;

weights::BiasWeight affine(const ::tyloi::weights::Loader& loader,
              const std::string& prefix, std::int64_t input, std::int64_t output) {
    auto result = loader.affine(prefix + ".weight", prefix + ".bias");
    TYLOINFER_RUNTIME_CHECK(result.weight.size(1) == input && result.weight.size(0) == output,
                      "axial patch vision affine geometry mismatch: " + prefix);
    return result;
}

RmsNorm norm(const ::tyloi::weights::Loader& loader,
             const std::string& name, std::int64_t width, double epsilon) {
    return RmsNorm{loader.norm(name, width), epsilon};
}

struct BlockWeights {
    RmsNorm norm1, norm2;
    weights::BiasWeight qkv, output, gate_up, down;
    template<class F> void visit_tensors(F visit) const {
        norm1.visit_tensors(visit); norm2.visit_tensors(visit);
        ::tyloi::weights::fields::tensors(qkv, visit); ::tyloi::weights::fields::tensors(output, visit);
        ::tyloi::weights::fields::tensors(gate_up, visit); ::tyloi::weights::fields::tensors(down, visit);
    }
    void to_device(::tyloi::Device device) {
        norm1.to_device(device); norm2.to_device(device);
        ::tyloi::weights::fields::to_device(qkv, device); ::tyloi::weights::fields::to_device(output, device);
        ::tyloi::weights::fields::to_device(gate_up, device); ::tyloi::weights::fields::to_device(down, device);
    }
};

struct AlignerWeights {
    RmsNorm norm;
    weights::BiasWeight input, output;
    template<class F> void visit_tensors(F visit) const {
        norm.visit_tensors(visit); ::tyloi::weights::fields::tensors(input, visit);
        ::tyloi::weights::fields::tensors(output, visit);
    }
    void to_device(::tyloi::Device device) {
        norm.to_device(device); ::tyloi::weights::fields::to_device(input, device);
        ::tyloi::weights::fields::to_device(output, device);
    }
};

} // namespace

struct CudaAxialPatchVisionTower::Impl {
    // The model owns the source for the lifetime of its vision component.
    ::tyloi::weights::Loader loader;
    Config config;

    tyloi::StepSequence<::tyloi::Tensor> encode_steps(
        ::tyloi::Tensor patches, int height, int width) const {
        if (height <= 0 || width <= 0 || height > std::numeric_limits<int>::max() / width)
            throw std::invalid_argument("invalid axial patch vision grid");
        const std::int64_t count = std::int64_t(height) * width;
        const auto hidden = config.hidden, heads = config.heads, dim = config.head_dim();
        if (!patches || !patches.defined())
            throw std::invalid_argument("axial patch vision patch geometry mismatch");
        if (patches.dim() == 4) patches = tyloi::ops::reshape(patches, {count, config.patch_width()});
        if (patches.dim() != 2 || patches.size(0) != count || patches.size(1) != config.patch_width())
            throw std::invalid_argument("axial patch vision patch geometry mismatch");
        const auto host = loader.for_layer(loader.placement.device, true);
        patches = tyloi::ops::cast(patches, ::tyloi::kFloat16);
        const auto activations = std::size_t(count) * std::size_t(std::max(6 * hidden, 2 * config.intermediate)) * sizeof(float)
            + (std::size_t(32) << 20);
        ::tyloi::Tensor value;
        {
            auto patch = affine(host, "vision.patch_embedding", config.patch_width(), hidden);
            co_yield tyloi::Workspace{::tyloi::weights::fields::bytes(patch) + activations};
            ::tyloi::Tensor output;
            {
                auto staged = std::move(patch);
                ::tyloi::weights::fields::to_device(staged, patches.device());
                output = tyloi::ops::affine(patches, staged.weight, staged.bias);
            }
            co_yield tyloi::Workspace{0, &output};
            value = std::move(output);
        }
        co_yield tyloi::StepState::advanced;
        std::vector<std::int32_t> positions;
        positions.reserve(std::size_t(count) * 2);
        for (int row = 0; row < height; ++row)
            for (int column = 0; column < width; ++column) {
                positions.push_back(row); positions.push_back(column);
            }
        for (std::int64_t layer = 0; layer < config.layers; ++layer) {
            const auto prefix = "vision.block." + std::to_string(layer);
            BlockWeights block{
                norm(host, prefix + ".norm1.weight", hidden, config.rms_eps),
                norm(host, prefix + ".norm2.weight", hidden, config.rms_eps),
                affine(host, prefix + ".attention.qkv", hidden, 3 * hidden),
                affine(host, prefix + ".attention.output", hidden, hidden),
                affine(host, prefix + ".mlp.gate_up", hidden, 2 * config.intermediate),
                affine(host, prefix + ".mlp.down", config.intermediate, hidden)};
            co_yield tyloi::Workspace{::tyloi::weights::fields::bytes(block) + activations};
            ::tyloi::Tensor output;
            {
                auto weights = std::move(block);
                ::tyloi::weights::fields::to_device(weights, value.device());
                auto normalized = weights.norm1(value);
                auto qkv = tyloi::ops::affine(normalized, weights.qkv.weight, weights.qkv.bias);
                const auto rope = [&](int index) {
                    auto sliced = tyloi::ops::slice(qkv, -1, index * hidden, (index + 1) * hidden);
                    auto shaped = tyloi::ops::reshape(sliced, {count, heads, dim});
                    return tyloi::ops::axial_rope(shaped, positions, config.rope_theta);
                };
                auto query = rope(0), key = rope(1);
                auto v_slice = tyloi::ops::slice(qkv, -1, 2 * hidden, 3 * hidden);
                auto v = tyloi::ops::reshape(v_slice, {count, heads, dim});
                auto attention = tyloi::ops::segmented_attention(query, key, v, {std::int32_t(count)});
                auto attended = tyloi::ops::reshape(attention, {count, hidden});
                auto projected = tyloi::ops::affine(attended, weights.output.weight, weights.output.bias);
                auto result = tyloi::ops::add(value, projected);
                auto normalized2 = weights.norm2(result);
                auto gate_up = tyloi::ops::affine(normalized2, weights.gate_up.weight, weights.gate_up.bias);
                auto gate = tyloi::ops::slice(gate_up, -1, 0, config.intermediate);
                auto up = tyloi::ops::slice(gate_up, -1, config.intermediate, 2 * config.intermediate);
                auto activated = tyloi::ops::silu(gate);
                auto product = tyloi::ops::multiply(activated, up);
                auto down = tyloi::ops::affine(product, weights.down.weight, weights.down.bias);
                output = tyloi::ops::add(result, down);
            }
            co_yield tyloi::Workspace{0, &output};
            value = std::move(output);
            co_yield tyloi::StepState::advanced;
        }
        {
            const auto ratio = int(config.downsample_ratio);
            const auto grid = tyloi::models::axial_patch_vision::aligned_grid(height, width, ratio);
            const auto indices = tyloi::models::axial_patch_vision::gather_indices(height, width, ratio);
            AlignerWeights weights{
                norm(host, "vision.output_norm.weight", hidden, config.rms_eps),
                affine(host, "vision.aligner.input", hidden * ratio * ratio, config.output_hidden),
                affine(host, "vision.aligner.output", config.output_hidden, config.output_hidden)};
            co_yield tyloi::Workspace{::tyloi::weights::fields::bytes(weights) + activations};
            ::tyloi::Tensor output;
            {
                auto staged = std::move(weights);
                ::tyloi::weights::fields::to_device(staged, value.device());
                auto normalized = tyloi::Tensor(staged.norm(value));
                auto zero = tyloi::ops::zeros({1, hidden}, value.options());
                auto padded = tyloi::ops::cat(std::array<tyloi::Tensor, 2>{normalized, zero}, 0);
                auto index = tyloi::ops::tensor(indices, tyloi::Device{tyloi::DeviceType::cpu, 0});
                index = tyloi::ops::cast(index, ::tyloi::kInt64);
                index = tyloi::ops::cast(index, ::tyloi::kInt64, value.device());
                auto selected = tyloi::ops::index_select(padded, 0, index);
                auto shaped = tyloi::ops::reshape(selected, {grid.count(), ratio * ratio, hidden});
                auto transposed = tyloi::ops::transpose(shaped, 1, 2);
                auto contiguous = tyloi::ops::contiguous(transposed);
                auto gathered = tyloi::ops::reshape(contiguous, {grid.count(), hidden * ratio * ratio});
                auto projected = tyloi::ops::affine(gathered, staged.input.weight, staged.input.bias);
                auto activated = tyloi::ops::gelu(projected);
                output = tyloi::ops::affine(activated, staged.output.weight, staged.output.bias);
            }
            co_yield tyloi::Workspace{0, &output};
            value = std::move(output);
        }
        co_yield std::move(value);
    }
};

CudaAxialPatchVisionTower CudaAxialPatchVisionTower::load(
    const ::tyloi::weights::Loader& loader,
    const Config& config) {
    config.validate();
    for (const auto* name : {"vision.patch_embedding.weight", "vision.output_norm.weight",
                            "vision.aligner.input.weight", "vision.aligner.output.weight"})
        TYLOINFER_RUNTIME_CHECK(loader.has(name), std::string("axial patch vision tensor missing: ") + name);
    return CudaAxialPatchVisionTower(std::make_unique<Impl>(Impl{loader,
        config}));
}

CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(CudaAxialPatchVisionTower&&) noexcept = default;
CudaAxialPatchVisionTower& CudaAxialPatchVisionTower::operator=(CudaAxialPatchVisionTower&&) noexcept = default;
CudaAxialPatchVisionTower::~CudaAxialPatchVisionTower() = default;

tyloi::StepSequence<::tyloi::Tensor> CudaAxialPatchVisionTower::encode_steps(
    ::tyloi::Tensor patches, int height, int width) const {
    return impl_->encode_steps(
        std::move(patches), height, width);
}


const Config& CudaAxialPatchVisionTower::config() const noexcept { return impl_->config; }
} // namespace tyloi::cuda::axial_patch_vision
