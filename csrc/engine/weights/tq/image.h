#pragma once

#include "engine/backends/cuda/ops/projection/tq.h"
#include "engine/weights/encoding/tq_rows.h"

// Shared TQ wire-to-device conversion used by ordinary loads, row lookups and
// the routed expert cache. Model-source loading does not participate.
std::vector<std::uint8_t> tq_image(
    const std::vector<std::uint8_t>& blob, const tyloi::TqLayout& layout);
TqWeight tq_view(::tyloi::Tensor image,
                 const tyloi::TqLayout& layout, std::int64_t columns);
TqWeight promote_tq(const TqWeight& weight,
                    const ::tyloi::Device& device);
TqWeight to_device_tq(const std::vector<std::uint8_t>&,
                      std::int64_t input_columns, int device, bool cuda = true);
