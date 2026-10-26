#pragma once

#include "backend.h"
#include <functional>
#include <memory>
#include <utility>

struct QuantLinear;
using GatedResidualMarker = std::function<void(const char*)>;

namespace mfq::cuda {

class GatedResidual {
public:
    using Tensor = mfq_tensor_backend::Tensor;
    struct Result {
        Tensor mixed;
        Tensor residual;
        Tensor gate;
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
        using Forward = std::function<Tensor(const Tensor&)>;
        using Mix = std::function<Tensor(const Tensor&, const Tensor&, int64_t)>;
        using Activation = std::function<Tensor(const Tensor&, int64_t, bool)>;
        Forward forward;
        Mix mixed;
        Activation activated;
        std::shared_ptr<const QuantLinear> weight;
        // F32 [column, stream], or vector layout [column/4, stream, 4].
        Tensor prepared_right;
        // Optional FP32 coefficients or packed row/group records.
        Tensor up_group_coefficients;
        explicit operator bool() const { return bool(forward); }
    };

    GatedResidual() = default;
    GatedResidual(Tensor norm, Projection down, Projection up, Projection injection,
                  int64_t hidden, int64_t streams, double eps,
                  const Options& options, const bool& cache_serial)
        : norm(std::move(norm)), down(std::move(down)), up(std::move(up)),
          injection(std::move(injection)), hidden(hidden), streams(streams), eps(eps),
          options_(&options), cache_serial_(&cache_serial) {}

    Result pre(const Tensor& input, const GatedResidualMarker& marker = {}) const;
    Result pre_after(const Tensor& input, const Tensor& residual, const Tensor& gate,
                     const GatedResidualMarker& marker = {}) const;
    Tensor post(const Tensor& input, const Tensor& residual, const Tensor& gate) const;
    bool can_fuse(const Tensor& input) const;

private:
    Tensor norm;
    Projection down, up, injection;
    int64_t hidden = 0, streams = 0;
    double eps = 0;
    // Borrow settings so diagnostics stay live; the engine drains before releasing them.
    const Options* options_ = nullptr;
    const bool* cache_serial_ = nullptr;

    int64_t prefill_work_chunk(const Tensor& input) const;
    template<class Project>
    Result bounded_pre(const Tensor& input, int64_t chunk,
                       bool reuse_input, Project project) const;
    Result fused(const Tensor& input, const Tensor& residual,
                 const Tensor& gate, bool after,
                 const GatedResidualMarker& marker) const;
    Result prepare(const Tensor& input, const Tensor& prepared_norm,
                   const GatedResidualMarker& marker) const;
};

} // namespace mfq::cuda
