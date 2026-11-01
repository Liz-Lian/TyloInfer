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
    mfq_tensor_backend::Tensor metadata_device,metadata_host;
    mfq_tensor_backend::Tensor map_ptrs;
    mfq_tensor_backend::Tensor weight_ptrs;
    mfq_tensor_backend::Tensor weight_sizes;
    mfq_tensor_backend::Tensor pool_params;
    mfq_tensor_backend::Tensor expert_pool;
    mfq_tensor_backend::Tensor expert_local;
    int pool_count = 0;
    MixedNvqF16FormatGroup f16_format_group =
        MixedNvqF16FormatGroup::All;
    bool masked_experts = false;
    bool dense_groups = false;
};

struct MixedNintDispatch {
    mfq_tensor_backend::Tensor metadata_device,metadata_host;
    mfq_tensor_backend::Tensor map_ptrs,expert_pool,expert_local,pool_params;
    std::vector<int> pools;
};
