#pragma once
#include "engine/tensor.h"
#include <cstdint>
#include <memory>

namespace tyloi::ops {
/** Immutable projections and normalization for gated residual mixing. */
struct ResidualWeights { Tensor norm, down, up, injection; };
/** Geometry and numerical policy for gated residual mixing. */
struct ResidualConfig { std::int64_t hidden = 0, streams = 0; double eps = 0; bool final = false; };
/** Identify the numerical stage in graph profiling. */
enum class ResidualRole { none, attention, ffn, final };
/** Execute gated residual mixing with prepared immutable projections. */
class Residual {
public:
    /** Own the mixed output, residual stream and optional injection gate. */
    struct Result { Tensor mixed, residual, gate; };
    /** Create an empty slot that can later receive a loaded residual layer. */
    Residual();
    /** Prepare projection weights for this residual layer. */
    Residual(ResidualWeights weights, ResidualConfig config);
    ~Residual();
    Residual(Residual&&) noexcept;
    Residual& operator=(Residual&&) noexcept;
    Residual(const Residual&) = delete;
    Residual& operator=(const Residual&) = delete;
    /** Normalize and mix a residual stream before its branch. */
    Result pre(const Tensor& input, ResidualRole role = ResidualRole::none) const;
    /** Combine a previous branch and prepare the following residual stream. */
    Result pre_after(const Tensor& input, const Tensor& residual, const Tensor& gate,
                     ResidualRole role = ResidualRole::none) const;
    /** Add a computed branch to its residual stream. */
    Tensor post(const Tensor& input, const Tensor& residual, const Tensor& gate) const;
    /** Report whether the current backend can chain this input. */
    bool can_chain(const Tensor& input) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace tyloi::ops
