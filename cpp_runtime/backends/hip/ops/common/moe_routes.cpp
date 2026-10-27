#include "moe.h"
#include "mfq_hip_moe_ops.h"
#include "quant_linear.h"
#include <algorithm>
#include <atomic>

static std::atomic<uint64_t> g_moe_route_generation{1};
#include <cstring>
#include <limits>

mfq_tensor_backend::Tensor moe_tensor_to_device(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor value, int device) {
    return value.defined()
        ? tensor_to_cuda_device(collectives, std::move(value), device)
        : value;
}

static void copy_route_tensor(
        ModelParallelCollectiveRuntime& collectives,
        mfq_tensor_backend::Tensor& destination,
        const mfq_tensor_backend::Tensor& source,
        int device) {
    if (!source.defined()) {
        destination = mfq_tensor_backend::Tensor();
    } else if (source.is_cuda() && source.get_device() == device) {
        destination = source.contiguous();
    } else {
        destination = tensor_to_cuda_device(
            collectives, source, device, std::move(destination));
    }
}

const MoeRoutePlan& moe_route_to_device(
        ModelParallelCollectiveRuntime& collectives,
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

MoeRoutePlan build_moe_route_plan(mfq_tensor_backend::Tensor ids, int n_experts) {
    if (!ids.is_cuda() || ids.scalar_type() != mfq_tensor_backend::kInt32 || ids.dim() != 2) {
        throw std::runtime_error("MoE ids must be CUDA int32 [tokens, routes]");
    }
    MoeRoutePlan result;
    result.generation = g_moe_route_generation.fetch_add(
        1, std::memory_order_relaxed);
    result.ids = ids.contiguous();
    result.n_experts = n_experts;
    auto empty = mfq_tensor_backend::empty({0}, result.ids.options());
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

mfq_tensor_backend::Tensor copy_cpu_weight_to_cuda(
        const mfq_tensor_backend::Tensor & source) {
    return source.defined()
        ? source.to(mfq_tensor_backend::Device(
            mfq_tensor_backend::kCUDA,
            mfq_current_cuda_device())).contiguous()
        : mfq_tensor_backend::Tensor();
}
