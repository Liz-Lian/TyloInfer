#pragma once

#include "moe.h"

void prepare_moe(MixedMoeRuntime& runtime,bool heterogeneous,bool dense_prefill=false,
    const std::vector<std::vector<int32_t>>* expert_local_host=nullptr);
