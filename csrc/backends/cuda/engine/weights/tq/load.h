#pragma once

#include "model_source/model_source.h"
#include "tq.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

TqWeight load_tq_weight(
    const mfq::ModelSource&, const std::string&, int device, bool cuda = true);

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
    CudaExecutionContext&, const mfq::ModelSource&, const std::string&);
std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
    CudaExecutionContext&, const mfq::ModelSource&, const std::string&,
    std::int64_t experts, std::int64_t output, std::int64_t input);
