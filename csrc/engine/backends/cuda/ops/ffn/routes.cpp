#include "routes.h"
#include "engine/backends/cuda/native/collectives.h"
#include "engine/backends/cuda/kernels/mfq_cuda_moe_ops.h"
#include "engine/weights/encoding/format_features.h"
#include <algorithm>
#include <atomic>
#include <stdexcept>
#include <utility>

static std::atomic<uint64_t> g_moe_route_generation{1};

#if TYLOINFER_ENABLE_MFQ_COMPAT
int select_nint_prefill_route_tile(
        const MoeRoutePlan & route,
        int tokens,
        int routes,
        int experts) {
    const int64_t pairs = static_cast<int64_t>(tokens) * routes;
    const int64_t rows_per_expert = std::max<int64_t>(
        1, (pairs + experts - 1) / experts);
    if (rows_per_expert < 4) {
        return 8;
    }
    if (rows_per_expert > 64 && route.wide_tile_m == 128) {
        return 128;
    }
    return route.mma_tile_m == 16 || route.mma_tile_m == 32 ||
            route.mma_tile_m == 64
        ? route.mma_tile_m
        : 8;
}

const ::tyloi::Tensor & nint_route_tile_bounds(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_bounds
        : (tile_m == 8 ? route.tile_bounds : route.mma_tile_bounds);
}

const ::tyloi::Tensor & nint_route_tile_experts(
        const MoeRoutePlan & route, int tile_m) {
    return tile_m == 128
        ? route.wide_tile_experts
        : (tile_m == 8 ? route.tile_experts : route.mma_tile_experts);
}
#endif

static void copy_route_tensor(
        tyloi::cuda::Collectives& collectives,
        ::tyloi::Tensor& destination,
        const ::tyloi::Tensor& source,
        int device) {
    if (!source.defined()) {
        destination = ::tyloi::Tensor();
    } else if (source.is_cuda() && source.get_device() == device) {
        destination = source.contiguous();
    } else {
        destination = collectives.copy(source, device, std::move(destination));
    }
}

const MoeRoutePlan& moe_route_to_device(
        tyloi::cuda::Collectives& collectives,
        const MoeRoutePlan& source,
        int device) {
    auto& replica = (*source.device_replicas)[device];
    if (!replica) replica = std::make_shared<MoeRoutePlan>();
    if (replica->generation == source.generation) return *replica;
    copy_route_tensor(collectives, replica->ids, source.ids, device);
    copy_route_tensor(collectives, replica->ids_dst, source.ids_dst, device);
    copy_route_tensor(collectives, replica->expert_bounds, source.expert_bounds, device);
    copy_route_tensor(collectives, replica->tile_bounds, source.tile_bounds, device);
    copy_route_tensor(collectives, replica->tile_experts, source.tile_experts, device);
    if (source.mma_tile_m == 8) {
        replica->mma_tile_bounds = replica->tile_bounds;
        replica->mma_tile_experts = replica->tile_experts;
    } else {
        copy_route_tensor(collectives, replica->mma_tile_bounds, source.mma_tile_bounds, device);
        copy_route_tensor(collectives, replica->mma_tile_experts, source.mma_tile_experts, device);
    }
    if (source.wide_tile_m == 8) {
        replica->wide_tile_bounds = replica->tile_bounds;
        replica->wide_tile_experts = replica->tile_experts;
    } else if (source.wide_tile_m == source.mma_tile_m) {
        replica->wide_tile_bounds = replica->mma_tile_bounds;
        replica->wide_tile_experts = replica->mma_tile_experts;
    } else {
        copy_route_tensor(collectives, replica->wide_tile_bounds, source.wide_tile_bounds, device);
        copy_route_tensor(collectives, replica->wide_tile_experts, source.wide_tile_experts, device);
    }
    copy_route_tensor(collectives, replica->counts, source.counts, device);
    copy_route_tensor(collectives, replica->cursors, source.cursors, device);
    replica->n_experts = source.n_experts;
    replica->mma_tile_m = source.mma_tile_m;
    replica->wide_tile_m = source.wide_tile_m;
    replica->map_ready = source.map_ready;
    replica->generation = source.generation;
    replica->host_unique_experts = source.host_unique_experts;
    return *replica;
}

MoeRoutePlan build_moe_route_plan(::tyloi::Tensor ids, int n_experts) {
    if (!ids.is_cuda() || ids.scalar_type() != ::tyloi::kInt32 || ids.dim() != 2) {
        throw std::runtime_error("MoE ids must be CUDA int32 [tokens, routes]");
    }
    MoeRoutePlan result;
    result.generation = g_moe_route_generation.fetch_add(
        1, std::memory_order_relaxed);
    result.ids = ids.contiguous();
    result.n_experts = n_experts;
    auto empty = ::tyloi::cuda::empty({0}, result.ids.options());
    result.ids_dst = empty;
    result.expert_bounds = empty;
    result.tile_bounds = empty;
    result.tile_experts = empty;
    result.mma_tile_bounds = empty;
    result.mma_tile_experts = empty;
    result.wide_tile_bounds = empty;
    result.wide_tile_experts = empty;
    result.counts = empty;
    result.cursors = empty;
    if (result.ids.size(0) > 8) {
        const int64_t rows_per_expert = std::max<int64_t>(
            1, (result.ids.numel() + n_experts - 1) / n_experts);
        const bool use_coarse_mma = rows_per_expert >= 4;
        const int mma_tile_m = rows_per_expert <= 16
            ? 16 : (rows_per_expert <= 32 ? 32 : 64);
        auto mapped = !use_coarse_mma
            ? moe_build_expert_map_cuda(result.ids, n_experts, 8)
            : (rows_per_expert > 64
                ? moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 128)
                : moe_build_expert_maps_cuda(
                    result.ids, n_experts, 8, mma_tile_m, 0));
        result.ids_dst = mapped.at(0);
        result.expert_bounds = mapped.at(1);
        result.tile_bounds = mapped.at(2);
        result.tile_experts = mapped.at(3);
        result.counts = mapped.at(4);
        if (use_coarse_mma) {
            result.mma_tile_bounds = mapped.at(5);
            result.mma_tile_experts = mapped.at(6);
            result.mma_tile_m = mma_tile_m;
            if (rows_per_expert > 64) {
                result.wide_tile_bounds = mapped.at(7);
                result.wide_tile_experts = mapped.at(8);
                result.wide_tile_m = 128;
            } else {
                result.wide_tile_bounds = result.mma_tile_bounds;
                result.wide_tile_experts = result.mma_tile_experts;
                result.wide_tile_m = result.mma_tile_m;
            }
        } else {
            result.mma_tile_bounds = result.tile_bounds;
            result.mma_tile_experts = result.tile_experts;
            result.mma_tile_m = 8;
            result.wide_tile_bounds = result.tile_bounds;
            result.wide_tile_experts = result.tile_experts;
            result.wide_tile_m = 8;
        }
    }
    result.map_ready = result.ids.size(0) <= 8 ||
        result.ids_dst.numel() == result.ids.numel();
    return result;
}
