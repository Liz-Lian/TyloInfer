#pragma once

#include "engine/weights/moe/pool.h"

#include <cstdint>
#include <vector>

struct MixedMoeRuntime;

std::vector<MoeActivationGeometry> moe_activation_geometry(
    const std::vector<MixedMoePool>& pools);

void prepare_moe(MixedMoeRuntime& runtime,bool heterogeneous,bool dense_prefill=false,
    const std::vector<std::vector<int32_t>>* expert_local_host=nullptr);
