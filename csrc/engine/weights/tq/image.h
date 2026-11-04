#pragma once

#include "engine/tensor.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/encoding/tq_rows.h"

// Shared TQ wire-to-device conversion used by ordinary loads, row lookups and
// the routed expert cache. Model-source loading does not participate.
std::vector<std::uint8_t> tq_image(
    const std::vector<std::uint8_t>& blob, const tyloi::TqLayout& layout);
::tyloi::encoding::TqTensor tq_view(::tyloi::DenseTensor image,
                 const tyloi::TqLayout& layout, std::int64_t columns);
::tyloi::encoding::TqTensor promote_tq(const ::tyloi::encoding::TqTensor& weight,
                    const ::tyloi::Device& device);
::tyloi::encoding::TqTensor to_device_tq(const std::vector<std::uint8_t>&,
                      std::int64_t input_columns, int device, bool cuda = true);
