#pragma once

#include "case.h"
#include "engine/backends/cuda/native/tensor/backend.h"

#include <memory>
#include <string>

namespace tyloi::diagnostics::numerics {
/** Own fixed input storage and CUDA tensors until native work has drained. */
class TensorWork {
  public:
    /** Create an executor for the currently selected CUDA device. */
    explicit TensorWork(Case& value);
    ~TensorWork() noexcept;
    TensorWork(const TensorWork&) = delete;
    TensorWork& operator=(const TensorWork&) = delete;
    /** Read and retain a named fixed input on the active CUDA device. */
    tyloi::DenseTensor input(const std::string& name);
    /** Retain a tensor used by submitted CUDA work. */
    tyloi::DenseTensor keep(tyloi::DenseTensor value);
    /** Export actual tensor values and their actual shape and dtype. */
    void output(const std::string& name, tyloi::DenseTensor value);
    /** Drain work and publish execution metadata after all observations. */
    void finish(const std::string& entrypoint, Json details = Json::object(),
                Json checks = Json::object());
    /** Return the selected CUDA device. */
    tyloi::Device device() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tyloi::diagnostics::numerics
