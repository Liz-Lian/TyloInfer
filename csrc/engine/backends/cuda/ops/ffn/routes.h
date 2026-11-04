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

MoeRoutePlan build_moe_route_plan(::tyloi::DenseTensor ids, int n_experts);

const MoeRoutePlan& moe_route_to_device(
    tyloi::cuda::Collectives& collectives,
    const MoeRoutePlan& route,
    int device);
int select_nint_prefill_route_tile(
    const MoeRoutePlan& route, int tokens, int routes, int experts);
const ::tyloi::DenseTensor& nint_route_tile_bounds(
    const MoeRoutePlan& route, int tile_m);
const ::tyloi::DenseTensor& nint_route_tile_experts(
    const MoeRoutePlan& route, int tile_m);

struct MoeRoutePlan {
    ::tyloi::DenseTensor ids;
    ::tyloi::DenseTensor ids_dst;
    ::tyloi::DenseTensor expert_bounds;
    ::tyloi::DenseTensor tile_bounds;
    ::tyloi::DenseTensor tile_experts;
    ::tyloi::DenseTensor mma_tile_bounds;
    ::tyloi::DenseTensor mma_tile_experts;
    ::tyloi::DenseTensor wide_tile_bounds;
    ::tyloi::DenseTensor wide_tile_experts;
    ::tyloi::DenseTensor counts;
    ::tyloi::DenseTensor cursors;
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
