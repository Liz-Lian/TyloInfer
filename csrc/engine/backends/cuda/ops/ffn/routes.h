#pragma once

#include "backend.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace tyloi::cuda {
class Collectives;
}

struct MoeRoutePlan;

MoeRoutePlan build_moe_route_plan(::tyloi::Tensor ids, int n_experts);

const MoeRoutePlan& moe_route_to_device(
    tyloi::cuda::Collectives& collectives,
    const MoeRoutePlan& route,
    int device);
int select_nint_prefill_route_tile(
    const MoeRoutePlan& route, int tokens, int routes, int experts);
const ::tyloi::Tensor& nint_route_tile_bounds(
    const MoeRoutePlan& route, int tile_m);
const ::tyloi::Tensor& nint_route_tile_experts(
    const MoeRoutePlan& route, int tile_m);

struct MoeRoutePlan {
    ::tyloi::Tensor ids;
    ::tyloi::Tensor ids_dst;
    ::tyloi::Tensor expert_bounds;
    ::tyloi::Tensor tile_bounds;
    ::tyloi::Tensor tile_experts;
    ::tyloi::Tensor mma_tile_bounds;
    ::tyloi::Tensor mma_tile_experts;
    ::tyloi::Tensor wide_tile_bounds;
    ::tyloi::Tensor wide_tile_experts;
    ::tyloi::Tensor counts;
    ::tyloi::Tensor cursors;
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
