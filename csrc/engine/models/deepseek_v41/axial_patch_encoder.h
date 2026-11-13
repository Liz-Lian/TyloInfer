#pragma once

#include "engine/tensor.h"
#include "engine/models/deepseek_v41/axial_patch_vision.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>

namespace tyloi::weights { class Cache; struct Loader; }
namespace tyloi::cuda {
class Projector;
}

namespace tyloi::cuda::axial_patch_vision {

using Config = tyloi::models::axial_patch_vision::Config;

// CUDA implementation of the backend-neutral axial patch ViT contract shared
// with the Metal runtime. It owns only the vision tower; prompt-span semantics
// belong to the architecture adapter.
class CudaAxialPatchVisionTower {
  public:
    /** Load tower configuration and retain the source-backed weight loader. */
    static CudaAxialPatchVisionTower load(const ::tyloi::weights::Loader& loader,
        const Config& config);

    CudaAxialPatchVisionTower(CudaAxialPatchVisionTower&&) noexcept;
    CudaAxialPatchVisionTower& operator=(
        CudaAxialPatchVisionTower&&) noexcept;
    ~CudaAxialPatchVisionTower();

    CudaAxialPatchVisionTower(const CudaAxialPatchVisionTower&) = delete;
    CudaAxialPatchVisionTower& operator=(
        const CudaAxialPatchVisionTower&) = delete;

    // Patches accept [H*W,3,P,P] or [H*W,3*P*P]. The output is the
    // row-major, ceil(H/R)*ceil(W/R) aligned embedding grid.

    /** Load one block per step and request workspace around its execution. */
    tyloi::StepSequence<::tyloi::Tensor> encode_steps(::tyloi::Tensor patches,
        int height,
        int width) const;

    const Config& config() const noexcept;

  private:
    struct Impl;
    explicit CudaAxialPatchVisionTower(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace tyloi::cuda::axial_patch_vision
