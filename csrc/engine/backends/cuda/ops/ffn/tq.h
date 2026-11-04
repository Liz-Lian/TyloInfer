#pragma once

#include "engine/backends/cuda/ops/projection/tq.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tyloi::cuda { class BlasResources; }

namespace tq_ops {

using IndexedTiles = std::array<std::vector<TrellisIndexedProjectionJob>, 4>;
using WideTiles = std::array<std::vector<TrellisIndexedWideProjectionJob>, 4>;

struct IndexedJobs {
    IndexedTiles gate_up, down;
    WideTiles wide;
    std::vector<TrellisIndexedWidePairJob> wide_gate_up;
};

IndexedJobs make_indexed_jobs(
    std::span<const TqWeight> gate,
    std::span<const TqWeight> up,
    std::span<const TqWeight> down,
    const std::vector<std::vector<std::int64_t>>& slots,
    std::size_t begin, std::int64_t routes, std::int64_t input,
    std::int64_t hidden, std::int64_t output, bool wide = false);

void routed_projection(
    const std::vector<TqWeight>& weights,
    ::tyloi::Tensor source,
    ::tyloi::Tensor output);

void run_long_expert(
    const TqWeight& gate, const TqWeight& up, const TqWeight& down,
    const std::vector<std::int64_t>& slots, std::int64_t routes,
    const ::tyloi::Tensor& source,
    ::tyloi::Tensor output,
    tyloi::cuda::BlasResources* blas, float limit);

void run_indexed(
    std::span<const TqWeight> gate,
    std::span<const TqWeight> up,
    std::span<const TqWeight> down,
    const std::vector<std::vector<std::int64_t>>& slots,
    std::size_t begin, std::int64_t routes,
    const ::tyloi::Tensor& source,
    ::tyloi::Tensor activated,
    ::tyloi::Tensor pairs, float limit, bool wide);

} // namespace tq_ops
