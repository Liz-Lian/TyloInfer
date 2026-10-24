#pragma once

#include "model_source/model_source.h"
#include "tq.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Shared TQ wire-to-device conversion used by ordinary loads, row lookups and
// the routed expert cache. The format and view remain owned by the TQ loader.
std::vector<std::uint8_t> tq_image(
    const std::vector<std::uint8_t>& blob, const mfq::TqLayout& layout);
TqWeight tq_view(mfq_tensor_backend::Tensor image,
                 const mfq::TqLayout& layout, std::int64_t columns);
TqWeight to_device_tq(const std::vector<std::uint8_t>&,
                      std::int64_t input_columns, int device, bool cuda = true);

TqWeight load_tq_weight(
    const mfq::ModelSource&, const std::string&, int device, bool cuda = true);

std::shared_ptr<TqExpertCache> ensure_tq_expert_cache(CudaExecutionContext&);
std::uint64_t register_tq_expert_source(
    const std::shared_ptr<TqExpertCache>&, std::size_t storage_bytes);

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
    CudaExecutionContext&, const mfq::ModelSource&, const std::string&);
