#pragma once

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
#include "engine/tensor.h"
#else
#include "backend.h"
#endif
#include "engine/models/common/vision/axial_patch_vision.h"
#include "engine/generation/sequence.h"

#include <cstdint>
#include <memory>

namespace mfq::cuda {
class Projector;
namespace weight_loader { struct Loader; }
}

namespace mfq::cuda::axial_patch_vision {

using Config = mfq::models::axial_patch_vision::Config;
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using Tensor = mfq::Tensor;
#else
using Tensor = mfq_tensor_backend::Tensor;
#endif

// CUDA implementation of the backend-neutral axial patch ViT contract shared
// with the Metal runtime. It owns only the vision tower; prompt-span semantics
// belong to the architecture adapter.
class CudaAxialPatchVisionTower {
  public:
    static CudaAxialPatchVisionTower load(
        const weight_loader::Loader& loader,
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
    Tensor encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector,
#endif
        Tensor patches,
        int height,
        int width) const;

    // Load one block at a time and restore borrowed expert storage before
    // yielding, so image preparation can share the request scheduler.
    mfq::StepSequence<Tensor> encode_steps(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector,
#endif
        Tensor patches,
        int height,
        int width) const;

    const Config& config() const noexcept;

  private:
    struct Impl;
    explicit CudaAxialPatchVisionTower(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace mfq::cuda::axial_patch_vision
