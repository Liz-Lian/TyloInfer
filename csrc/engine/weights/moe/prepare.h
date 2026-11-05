#pragma once

#include "engine/weights/moe/pool.h"

#include <cstdint>
#include <vector>

namespace tyloi { class MfeTensor; }

std::vector<MoeActivationGeometry> moe_activation_geometry(
    const std::vector<MixedMoePool>& pools);

void prepare_moe(::tyloi::MfeTensor& runtime,bool heterogeneous,bool dense_prefill=false,
    const std::vector<std::vector<int32_t>>* expert_local_host=nullptr);
