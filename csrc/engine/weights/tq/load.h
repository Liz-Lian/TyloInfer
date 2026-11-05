#pragma once

#include "engine/weights/source/source.h"
#include "engine/weights/encoding/tq.h"
#include "engine/weights/tq/routed.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace tyloi::cuda { struct RoutedCache; }
class TqExec;

::tyloi::encoding::TqTensor load_tq_weight(
    const tyloi::source::Source&, const std::string&, int device, bool cuda = true);

std::shared_ptr<tyloi::TqBank> load_tq_routed_weight(
    tyloi::cuda::RoutedCache&, const std::shared_ptr<TqExec>&,
    const tyloi::source::Source&, const std::string&);
std::shared_ptr<tyloi::TqBank> load_tq_routed_weight(
    tyloi::cuda::RoutedCache&, const std::shared_ptr<TqExec>&,
    const tyloi::source::Source&, const std::string&,
    std::int64_t experts, std::int64_t output, std::int64_t input);
