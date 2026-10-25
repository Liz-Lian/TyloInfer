#pragma once

#include "ops/tq/include/tq.h"
#include "formats/tq_rows.h"

// Shared TQ wire-to-device conversion used by ordinary loads, row lookups and
// the routed expert cache. Model-source loading does not participate.
std::vector<std::uint8_t> tq_image(
    const std::vector<std::uint8_t>& blob, const mfq::TqLayout& layout);
TqWeight tq_view(mfq_tensor_backend::Tensor image,
                 const mfq::TqLayout& layout, std::int64_t columns);
TqWeight to_device_tq(const std::vector<std::uint8_t>&,
                      std::int64_t input_columns, int device, bool cuda = true);
