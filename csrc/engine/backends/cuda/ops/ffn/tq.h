#pragma once
#include "engine/weights/encoding/tq.h"

#include "engine/backends/cuda/ops/projection/tq.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace tyloi { class TqBank; }
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
    std::span<const ::tyloi::encoding::TqTensor> gate,
    std::span<const ::tyloi::encoding::TqTensor> up,
    std::span<const ::tyloi::encoding::TqTensor> down,
    const std::vector<std::vector<std::int64_t>>& slots,
    std::size_t begin, std::int64_t routes, std::int64_t input,
    std::int64_t hidden, std::int64_t output, bool wide = false);

::tyloi::DenseTensor routed_projection(
    const ::tyloi::TqBank& weight, ::tyloi::DenseTensor source, ::tyloi::DenseTensor ids);

void routed_projection(
    const std::vector<::tyloi::encoding::TqTensor>& weights,
    ::tyloi::DenseTensor source,
    ::tyloi::DenseTensor output);

void run_long_expert(
    const ::tyloi::encoding::TqTensor& gate, const ::tyloi::encoding::TqTensor& up, const ::tyloi::encoding::TqTensor& down,
    const std::vector<std::int64_t>& slots, std::int64_t routes,
    const ::tyloi::DenseTensor& source,
    ::tyloi::DenseTensor output,
    tyloi::cuda::BlasResources* blas, float limit);

void run_indexed(
    std::span<const ::tyloi::encoding::TqTensor> gate,
    std::span<const ::tyloi::encoding::TqTensor> up,
    std::span<const ::tyloi::encoding::TqTensor> down,
    const std::vector<std::vector<std::int64_t>>& slots,
    std::size_t begin, std::int64_t routes,
    const ::tyloi::DenseTensor& source,
    ::tyloi::DenseTensor activated,
    ::tyloi::DenseTensor pairs, float limit, bool wide);

} // namespace tq_ops
