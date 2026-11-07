#pragma once

#include "engine/requests/policy.h"

#include <cstddef>
#include <cstdint>
#include <memory>

class MoeExpertCache;

namespace tyloi::cuda {

engine::PrefillChunk budget_batch_prefill(
    std::int64_t hidden_size, std::int64_t hc_mult, int device,
    const std::shared_ptr<MoeExpertCache>&,
    engine::PrefillChunk, std::size_t growth = 0);

} // namespace tyloi::cuda
