#pragma once

#include "engine/tensor.h"

#include <cstdint>
#include <string_view>

namespace tyloi::source { class Source; }

namespace tyloi::weights {

/** Read a bounded matrix row range as owned CPU storage, preserving F16, BF16 or F32 source precision. */
Tensor read_rows(const source::Source& source, std::string_view name,
    std::int64_t first, std::int64_t count);

} // namespace tyloi::weights
