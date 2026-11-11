#pragma once

#include "backend.h"
#include "engine/backends/cuda/ops/ffn/options.h"
#include <functional>
#include <memory>
#include <utility>

#ifndef TYLOINFER_NATIVE_CUDA
struct CudaProfiler;
using GatedResidualMarker = std::function<void(const char*)>;

namespace tyloi::cuda {
class Projector;

class GatedResidual {
public:
    using DenseTensor = ::tyloi::DenseTensor;
    struct Result {
        DenseTensor mixed;
        DenseTensor residual;
        DenseTensor gate;
    };
    enum class Part { generic, final_down, final_up };
    struct Projection {
#ifdef TYLOINFER_NATIVE_CUDA
        ::tyloi::Tensor weight;
        DenseTensor right, promoted, prepared_right, up_group_coefficients;
        bool can_mix = false, can_activate = false;
        DenseTensor forward(const DenseTensor& input) const;
        DenseTensor mix(const DenseTensor& input, const DenseTensor& normalized, int64_t streams) const;
        DenseTensor activate(const DenseTensor& input, int64_t streams, bool injection) const;
        explicit operator bool() const { return weight.defined(); }
#else
        using Forward = std::function<DenseTensor(const DenseTensor&)>;
        using Mix = std::function<DenseTensor(const DenseTensor&, const DenseTensor&, int64_t)>;
        using Activation = std::function<DenseTensor(const DenseTensor&, int64_t, bool)>;
        Forward forward;
        Mix mixed;
        Activation activated;
        ::tyloi::Tensor weight;
        // F32 [column, stream], or vector layout [column/4, stream, 4].
        DenseTensor prepared_right;
        // Optional FP32 coefficients or packed row/group records.
        DenseTensor up_group_coefficients;
        explicit operator bool() const { return bool(forward); }
#endif
    };
#ifdef TYLOINFER_NATIVE_CUDA
    static Projection projection(::tyloi::Tensor, Part);
#else
    static Projection projection(::tyloi::Tensor, Projector,
                                 CudaProfiler&, const ResidualOptions&, Part);
#endif

    GatedResidual() = default;
#ifdef TYLOINFER_NATIVE_CUDA
    GatedResidual(::tyloi::Tensor norm, Projection down, Projection up, Projection injection,
                  int64_t hidden, int64_t streams, double eps)
        : norm(std::move(norm)), down(std::move(down)), up(std::move(up)),
          injection(std::move(injection)), hidden(hidden), streams(streams), eps(eps), initialized_(true) {}
#else
    GatedResidual(DenseTensor norm, Projection down, Projection up, Projection injection,
                  int64_t hidden, int64_t streams, double eps,
                  const ResidualOptions& options, const bool& cache_serial)
        : norm(std::move(norm)), down(std::move(down)), up(std::move(up)),
          injection(std::move(injection)), hidden(hidden), streams(streams), eps(eps),
          options_(&options), cache_serial_(&cache_serial) {}
#endif

    Result pre(const DenseTensor& input, const GatedResidualMarker& marker = {}) const;
    Result pre_after(const DenseTensor& input, const DenseTensor& residual, const DenseTensor& gate,
                     const GatedResidualMarker& marker = {}) const;
    DenseTensor post(const DenseTensor& input, const DenseTensor& residual, const DenseTensor& gate) const;
    bool can_fuse(const DenseTensor& input) const;

private:
#ifdef TYLOINFER_NATIVE_CUDA
    ::tyloi::Tensor norm;
#else
    DenseTensor norm;
#endif
    Projection down, up, injection;
    int64_t hidden = 0, streams = 0;
    double eps = 0;
#ifndef TYLOINFER_NATIVE_CUDA
    const ResidualOptions* options_ = nullptr;
    const bool* cache_serial_ = nullptr;
#else
    bool initialized_ = false;
#endif

    int64_t prefill_work_chunk(const DenseTensor& input) const;
    template<class Project>
    Result bounded_pre(const DenseTensor& input, int64_t chunk,
                       bool reuse_input, Project project) const;
    Result fused(const DenseTensor& input, const DenseTensor& residual,
                 const DenseTensor& gate, bool after,
                 const GatedResidualMarker& marker) const;
    Result prepare(const DenseTensor& input, const DenseTensor& prepared_norm,
                   const GatedResidualMarker& marker) const;
};

} // namespace tyloi::cuda
#endif
