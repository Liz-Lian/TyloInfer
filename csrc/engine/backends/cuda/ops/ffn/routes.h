#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mfq::cuda {
class Collectives;
}

struct MoeRoutePlan;

MoeRoutePlan build_moe_route_plan(mfq_tensor_backend::Tensor ids, int n_experts);

const MoeRoutePlan& moe_route_to_device(
    mfq::cuda::Collectives& collectives,
    const MoeRoutePlan& route,
    int device);
int select_nint_prefill_route_tile(
    const MoeRoutePlan& route, int tokens, int routes, int experts);
const mfq_tensor_backend::Tensor& nint_route_tile_bounds(
    const MoeRoutePlan& route, int tile_m);
const mfq_tensor_backend::Tensor& nint_route_tile_experts(
    const MoeRoutePlan& route, int tile_m);

struct MoeRoutePlan {
    mfq_tensor_backend::Tensor ids;
    mfq_tensor_backend::Tensor ids_dst;
    mfq_tensor_backend::Tensor expert_bounds;
    mfq_tensor_backend::Tensor tile_bounds;
    mfq_tensor_backend::Tensor tile_experts;
    mfq_tensor_backend::Tensor mma_tile_bounds;
    mfq_tensor_backend::Tensor mma_tile_experts;
    mfq_tensor_backend::Tensor wide_tile_bounds;
    mfq_tensor_backend::Tensor wide_tile_experts;
    mfq_tensor_backend::Tensor counts;
    mfq_tensor_backend::Tensor cursors;
    int n_experts = 0;
    int mma_tile_m = 8;
    int wide_tile_m = 8;
    bool map_ready = false;
    uint64_t generation = 0;
    mutable std::shared_ptr<std::vector<int32_t>>
        host_unique_experts;
    mutable std::shared_ptr<std::unordered_map<
        int, std::shared_ptr<MoeRoutePlan>>> device_replicas =
            std::make_shared<std::unordered_map<
                int, std::shared_ptr<MoeRoutePlan>>>();
};
