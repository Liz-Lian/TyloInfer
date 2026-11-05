#pragma once

#include "backend.h"
#include <functional>
#include <memory>
#include <utility>

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
    struct Options {
        bool fused_projections = true;
        bool fused_activation = true;
        bool fused_post_norm = true;
        bool prepared_dense = false;
        bool native_input = true;
        bool fixed_group = true;
        bool compact_mix = true;
        bool two_stage = true;
        bool dense_injection = true;
    };
    struct Projection {
        using Forward = std::function<DenseTensor(const DenseTensor&)>;
        using Mix = std::function<DenseTensor(const DenseTensor&, const DenseTensor&, int64_t)>;
        using Activation = std::function<DenseTensor(const DenseTensor&, int64_t, bool)>;
        Forward forward;
        Mix mixed;
        Activation activated;
        std::shared_ptr<const ::tyloi::Tensor> weight;
        // F32 [column, stream], or vector layout [column/4, stream, 4].
        DenseTensor prepared_right;
        // Optional FP32 coefficients or packed row/group records.
        DenseTensor up_group_coefficients;
        explicit operator bool() const { return bool(forward); }
    };
    enum class Part { generic, final_down, final_up };
    // Borrowed execution resources and options outlive the prepared projections.
    static Projection projection(std::shared_ptr<const ::tyloi::Tensor>, Projector,
                                 CudaProfiler&, const Options&, Part);

    GatedResidual() = default;
    GatedResidual(DenseTensor norm, Projection down, Projection up, Projection injection,
                  int64_t hidden, int64_t streams, double eps,
                  const Options& options, const bool& cache_serial)
        : norm(std::move(norm)), down(std::move(down)), up(std::move(up)),
          injection(std::move(injection)), hidden(hidden), streams(streams), eps(eps),
          options_(&options), cache_serial_(&cache_serial) {}

    Result pre(const DenseTensor& input, const GatedResidualMarker& marker = {}) const;
    Result pre_after(const DenseTensor& input, const DenseTensor& residual, const DenseTensor& gate,
                     const GatedResidualMarker& marker = {}) const;
    DenseTensor post(const DenseTensor& input, const DenseTensor& residual, const DenseTensor& gate) const;
    bool can_fuse(const DenseTensor& input) const;

private:
    DenseTensor norm;
    Projection down, up, injection;
    int64_t hidden = 0, streams = 0;
    double eps = 0;
    // Borrow settings so diagnostics stay live; the engine drains before releasing them.
    const Options* options_ = nullptr;
    const bool* cache_serial_ = nullptr;

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
