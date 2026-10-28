#pragma once

#include "source/model_source.h"
#include "backends/cuda/engine/weights/tq/routed.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mfq::cuda { struct RoutedCache; }
class TqExec;

TqWeight load_tq_weight(
    const mfq::ModelSource&, const std::string&, int device, bool cuda = true);

std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
    mfq::cuda::RoutedCache&, const std::shared_ptr<TqExec>&,
    const mfq::ModelSource&, const std::string&);
std::shared_ptr<TqRoutedWeight> load_tq_routed_weight(
    mfq::cuda::RoutedCache&, const std::shared_ptr<TqExec>&,
    const mfq::ModelSource&, const std::string&,
    std::int64_t experts, std::int64_t output, std::int64_t input);
