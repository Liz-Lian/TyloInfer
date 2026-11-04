#pragma once

#include "backend.h"

#include <vector>

enum class MixedNvqF16FormatGroup : int {
    All = 0,
    Standard = 1,
    Extended = 2,
    Legacy = 3,
};

struct MixedNvqDispatch {
    // Retain the packed device allocation and its asynchronous upload source.
    ::tyloi::Tensor metadata_device,metadata_host;
    ::tyloi::Tensor map_ptrs;
    ::tyloi::Tensor weight_ptrs;
    ::tyloi::Tensor weight_sizes;
    ::tyloi::Tensor pool_params;
    ::tyloi::Tensor initial_expert_pool;
    ::tyloi::Tensor initial_expert_local;
    int pool_count = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool masked_experts = false;
    bool dense_groups = false;
};

struct MixedNintDispatch {
    ::tyloi::Tensor metadata_device,metadata_host;
    ::tyloi::Tensor map_ptrs,initial_expert_pool,initial_expert_local,pool_params;
    std::vector<int> pools;
};
