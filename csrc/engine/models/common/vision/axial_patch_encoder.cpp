#include "engine/models/common/vision/axial_patch_encoder.h"
#include "engine/backends/cuda/ops/projection/bias.h"
#include "engine/backends/cuda/ops/norm/layer.h"
#include "engine/backends/cuda/ops/vision/attention.h"
#include "engine/ops.h"
#include "engine/weights/fields.h"
#include "engine/weights/loader.h"
#include "engine/weights/workspace.h"
#include "engine/generation/sequence.h"

#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tyloi::cuda::axial_patch_vision {
namespace {
using vision::RmsNorm;

BiasWeight affine(const weight_loader::Loader& loader,
              const std::string& prefix, std::int64_t input, std::int64_t output) {
    auto result = loader.affine(prefix + ".weight", prefix + ".bias");
    TYLOINFER_RUNTIME_CHECK(result.weight.neuron_len() == input && result.weight.out() == output,
                      "axial patch vision affine geometry mismatch: " + prefix);
    return result;
}

RmsNorm norm(const weight_loader::Loader& loader,
             const std::string& name, std::int64_t width, double epsilon) {
    return RmsNorm{loader.norm(name, width), epsilon};
}

struct BlockWeights {
    RmsNorm norm1, norm2;
    BiasWeight qkv, output, gate_up, down;
    template<class F> void visit_tensors(F visit) {
        norm1.visit_tensors(visit); norm2.visit_tensors(visit);
        fields::tensors(qkv, visit); fields::tensors(output, visit);
        fields::tensors(gate_up, visit); fields::tensors(down, visit);
    }
};

struct AlignerWeights {
    RmsNorm norm;
    BiasWeight input, output;
    template<class F> void visit_tensors(F visit) {
        norm.visit_tensors(visit); fields::tensors(input, visit);
        fields::tensors(output, visit);
    }
};

} // namespace

struct CudaAxialPatchVisionTower::Impl {
    // The model owns the source for the lifetime of its vision component.
    weight_loader::Loader loader;
    Config config;

    tyloi::StepSequence<DenseTensor> encode_steps(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
                                          Projector projector,
#endif
                                          DenseTensor patches,
                                          int height, int width) const {
        if (height <= 0 || width <= 0 || height > std::numeric_limits<int>::max() / width)
            throw std::invalid_argument("invalid axial patch vision grid");
        const std::int64_t count = std::int64_t(height) * width;
        const auto hidden = config.hidden, heads = config.heads, dim = config.head_dim();
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        if (patches.dim() == 4) patches = tyloi::ops::reshape(patches, {count, config.patch_width()});
#else
        if (patches.dim() == 4) patches = patches.reshape({count, config.patch_width()});
#endif
        if (patches.dim() != 2 || patches.size(0) != count || patches.size(1) != config.patch_width())
            throw std::invalid_argument("axial patch vision patch geometry mismatch");
        const auto host = loader.for_layer(loader.placement.device, true);
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
        patches = tyloi::ops::cast(patches, ::tyloi::kFloat16);
#else
        patches = patches.to(::tyloi::kFloat16);
#endif
        const auto activations = std::size_t(count) * std::size_t(std::max(6 * hidden, 2 * config.intermediate)) * sizeof(float)
            + (std::size_t(32) << 20);
        DenseTensor value;
        {
            auto patch = affine(host, "vision.patch_embedding", config.patch_width(), hidden);
            auto loan = loader.cache.borrow(fields::bytes(patch) + activations);
            DenseTensor output;
            {
                auto device = fields::to_cuda(std::move(patch));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                output = tyloi::ops::affine(patches, device.weight, device.bias);
#else
                output = tyloi::cuda::affine(projector, device, patches);
#endif
            }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            value = loan.finish(std::move(output));
#else
            value = loan.finish(std::move(output));
#endif
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
            auto loan = loader.cache.borrow(fields::bytes(block) + activations);
            DenseTensor output;
            {
                auto weights = fields::to_cuda(std::move(block));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto qkv = tyloi::ops::affine(weights.norm1(value), weights.qkv.weight, weights.qkv.bias);
                const auto rope = [&](int index) {
                    return tyloi::ops::axial_rope(tyloi::ops::reshape(
                        tyloi::ops::slice(qkv, -1, index * hidden, (index + 1) * hidden),
                        {count, heads, dim}), positions, config.rope_theta);
                };
                auto query = rope(0), key = rope(1);
                auto v = tyloi::ops::reshape(tyloi::ops::slice(qkv, -1, 2 * hidden, 3 * hidden), {count, heads, dim});
                auto attended = tyloi::ops::reshape(tyloi::ops::segmented_attention(query, key, v, {std::int32_t(count)}), {count, hidden});
                auto result = tyloi::ops::add(value, tyloi::ops::affine(std::move(attended), weights.output.weight, weights.output.bias));
                auto gate_up = tyloi::ops::affine(weights.norm2(result), weights.gate_up.weight, weights.gate_up.bias);
                output = tyloi::ops::add(result, tyloi::ops::affine(tyloi::ops::multiply(
                    tyloi::ops::silu(tyloi::ops::slice(gate_up, -1, 0, config.intermediate)),
                    tyloi::ops::slice(gate_up, -1, config.intermediate, 2 * config.intermediate)), weights.down.weight, weights.down.bias));
#else
                auto qkv = tyloi::cuda::affine(projector, weights.qkv, weights.norm1(value));
                const auto rope = [&](int index) {
                    return vision::apply_axial_rope(
                        qkv.narrow(-1, index * hidden, hidden).reshape({count, heads, dim}), positions, config.rope_theta);
                };
                auto query = rope(0), key = rope(1);
                auto v = qkv.narrow(-1, 2 * hidden, hidden).reshape({count, heads, dim});
                auto attended = vision::segmented_attention(query, key, v, {std::int32_t(count)}).reshape({count, hidden});
                auto result = value + tyloi::cuda::affine(projector, weights.output, std::move(attended));
                auto gate_up = tyloi::cuda::affine(projector, weights.gate_up, weights.norm2(result));
                output = result + tyloi::cuda::affine(projector, weights.down,
                    ::tyloi::cuda::silu(gate_up.narrow(-1, 0, config.intermediate)) *
                    gate_up.narrow(-1, config.intermediate, config.intermediate));
#endif
            }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            value = loan.finish(std::move(output));
#else
            value = loan.finish(std::move(output));
#endif
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
            auto loan = loader.cache.borrow(fields::bytes(weights) + activations);
            DenseTensor output;
            {
                auto device = fields::to_cuda(std::move(weights));
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto padded = tyloi::ops::cat({device.norm(value), tyloi::ops::zeros({1, hidden}, value.options())}, 0);
                DenseTensor index = tyloi::ops::tensor(indices, tyloi::Device{tyloi::DeviceType::cpu, 0});
                index = tyloi::ops::cast(index, ::tyloi::kInt64);
                index = tyloi::ops::cast(index, ::tyloi::kInt64, tyloi::Device{::tyloi::kCUDA, 0});
                auto gathered = tyloi::ops::reshape(tyloi::ops::contiguous(tyloi::ops::transpose(
                    tyloi::ops::reshape(tyloi::ops::index_select(padded, 0, index), {grid.count(), ratio * ratio, hidden}), 1, 2)),
                    {grid.count(), hidden * ratio * ratio});
#else
                auto padded = ::tyloi::cuda::cat({device.norm(value), ::tyloi::cuda::zeros({1, hidden}, value.options())}, 0);
                auto index = ::tyloi::cuda::tensor(indices,
                    ::tyloi::TensorOptions().dtype(::tyloi::kInt32).device(::tyloi::kCPU))
                    .to(::tyloi::kInt64).to(::tyloi::kCUDA);
                auto gathered = padded.index_select(0, index).reshape({grid.count(), ratio * ratio, hidden})
                    .transpose(1, 2).contiguous().reshape({grid.count(), hidden * ratio * ratio});
#endif
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
                auto projected = tyloi::ops::affine(std::move(gathered), device.input.weight, device.input.bias);
                output = tyloi::ops::affine(tyloi::ops::gelu(projected), device.output.weight, device.output.bias);
#else
                auto projected = tyloi::cuda::affine(projector, device.input, std::move(gathered));
                output = tyloi::cuda::affine(projector, device.output, ::tyloi::cuda::gelu(projected, "none"));
#endif
            }
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
            value = loan.finish(std::move(output));
#else
            value = loan.finish(std::move(output));
#endif
        }
        co_yield std::move(value);
    }
};

CudaAxialPatchVisionTower CudaAxialPatchVisionTower::load(
    const weight_loader::Loader& loader, const Config& config) {
    config.validate();
    for (const auto* name : {"vision.patch_embedding.weight", "vision.output_norm.weight",
                            "vision.aligner.input.weight", "vision.aligner.output.weight"})
        TYLOINFER_RUNTIME_CHECK(loader.has(name), std::string("axial patch vision tensor missing: ") + name);
    return CudaAxialPatchVisionTower(std::make_unique<Impl>(Impl{loader, config}));
}

CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
CudaAxialPatchVisionTower::CudaAxialPatchVisionTower(CudaAxialPatchVisionTower&&) noexcept = default;
CudaAxialPatchVisionTower& CudaAxialPatchVisionTower::operator=(CudaAxialPatchVisionTower&&) noexcept = default;
CudaAxialPatchVisionTower::~CudaAxialPatchVisionTower() = default;

tyloi::StepSequence<DenseTensor> CudaAxialPatchVisionTower::encode_steps(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    Projector projector,
#endif
    DenseTensor patches, int height, int width) const {
    return impl_->encode_steps(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        projector,
#endif
        std::move(patches), height, width);
}

DenseTensor CudaAxialPatchVisionTower::encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    Projector projector,
#endif
    DenseTensor patches, int height, int width) const {
    auto steps = encode_steps(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        projector,
#endif
        std::move(patches), height, width);
    DenseTensor output;
    while (auto step = steps.next()) if (step.value) output = std::move(*step.value);
    return output;
}

const Config& CudaAxialPatchVisionTower::config() const noexcept { return impl_->config; }
} // namespace tyloi::cuda::axial_patch_vision
