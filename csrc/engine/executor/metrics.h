#pragma once

#include "backends/cuda/engine/executor/options.h"
#include "backends/cuda/engine/state/kv_memory.h"
#include "backends/cuda/engine/state/kv_offload.h"
#include "backends/cuda/engine/state/session/text_cache.h"
#include "backends/cuda/engine/weights/moe/expert_cache.h"
#include "backends/cuda/engine/weights/placement.h"
#include "backends/cuda/engine/weights/routed_cache.h"
#include "engine/batch/batch.h"
#include "engine/executor/budget.h"
#include "engine/executor/engine.h"
#include "engine/executor/mtp_metrics.h"
#include "engine/weights/workspace.h"
#include "native/cuda/tensor/backend.h"
#include "engine/weights/memory.h"
#include "source/model_source.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <utility>

namespace mfq::cuda::engine_detail {

template <class Model, class Parts>
mfq::engine::Metrics runtime_metrics(
        const Model& language, const Parts& components, int graph_depth,
        const ExpertSpace* space, mfq::engine::Batch* batching,
        const mfq::ModelSource* source, WeightMemoryLedger& ledger,
        const RoutedCache& weights, KvMemoryManager* kv,
        KvOffloadStore* offload, const internal::TextSessionCache& cache,
        const Budget& budget, const LayerPlacementConfig& placement,
        const ParallelConfig& tp, const ParallelConfig& ep) {
    using mfq::engine::Metrics;

    const auto batch_metrics = batching ? batching->metrics() : Metrics{};
    const auto workspace_metrics = space ? space->metrics() : ExpertSpace::Metrics{};
    Metrics metrics{
        {"mtp_graph_prepared_depth", double(graph_depth)},
        {"mtp_flexible_expert_capacity_bytes", double(workspace_metrics.decode_capacity)},
        {"mtp_flexible_expert_used_bytes", double(workspace_metrics.decode_used)},
        {"prefill_workspace_expert_capacity_bytes", double(workspace_metrics.prefill_capacity)},
        {"prefill_workspace_expert_used_bytes", double(workspace_metrics.prefill_used)},
        {"prefill_workspace_expert_reclaimed_used_bytes", double(workspace_metrics.prefill_reclaimed_used)}};
    const auto component_metrics = components.metrics();
    metrics.insert(metrics.end(), component_metrics.begin(), component_metrics.end());
    auto experts = weights.metrics();
    metrics.insert(metrics.end(), experts.begin(), experts.end());
    if constexpr (requires { components.profile_metrics(language); }) {
        auto phases = components.profile_metrics(language);
        metrics.insert(metrics.end(), phases.begin(), phases.end());
    }
    if (source) {
        const auto io = source->file_read_stats();
        metrics.insert(metrics.end(), {
            {"model_io_direct", io.mode == mfq::FileReadMode::Direct ? 1.0 : 0.0},
            {"model_io_read_calls", double(io.calls)},
            {"model_io_logical_bytes", double(io.logical_bytes)},
            {"model_io_physical_bytes", double(io.physical_bytes)},
            {"model_io_read_ms", double(io.read_nanoseconds) / 1e6},
            {"model_io_staging_bytes", double(io.staging_bytes)},
            {"model_io_cached_staging_bytes", double(io.staging_cached_bytes)},
            {"model_io_staging_allocations", double(io.staging_allocations)}});
    }
    if (kv) {
        auto kv_metrics = kv->metrics();
        metrics.insert(metrics.end(), kv_metrics.begin(), kv_metrics.end());
    }
    std::size_t resident_weights = ledger.resident_bytes();
    for (const auto& metric : experts)
        if (metric.first == "resident_expert_cache_bytes")
            resident_weights += std::size_t(metric.second);
    const auto ple = ledger.ple_bytes();
    metrics.insert(metrics.end(), {
        {"resident_weight_bytes", double(resident_weights)},
        {"ssd_ple_enabled", ple > 0 ? 1.0 : 0.0},
        {"ssd_ple_payload_bytes", double(ple)}});
    std::optional<Metrics> memory;
    if constexpr (requires { components.memory_metrics(language); })
        memory = components.memory_metrics(language);
    if (memory) {
        for (auto& [name, value] : *memory)
            for (const auto& [batch_name, batch_value] : batch_metrics) {
                if (name == "kv_cache_bytes" && batch_name == "continuous_batching_request_kv_bytes")
                    value += batch_value;
                if (name == "kv_cache_contexts" && batch_name == "continuous_batching_request_contexts")
                    value += batch_value;
            }
        metrics.insert(metrics.end(), memory->begin(), memory->end());
    }
    const auto metric_bytes = [&](const std::string& key) {
        for (const auto& [name, value] : metrics)
            if (name == key)
                return std::size_t(value);
        return std::size_t(0);
    };
    std::set<int> devices(placement.devices.begin(), placement.devices.end());
    devices.insert(tp.devices.begin(), tp.devices.end());
    devices.insert(ep.devices.begin(), ep.devices.end());
    const auto primary = placement.primary_device();
    devices.insert(primary);
    auto prefix_bytes = cache.device_hot_bytes();
    std::size_t allocated = 0, reserved = 0, free_bytes = 0, total_bytes = 0, buffers = 0;
    for (const int device : devices) {
        MfqCudaGuard guard(device);
        size_t free = 0, total = 0;
        TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free, &total));
        const auto memory_stats = mfq_cuda_memory_stats(device);
        const auto external = device == primary ? metric_bytes("kv_memory_vmm_bytes") : 0;
        const auto resident = std::max(memory_stats.reserved_bytes, memory_stats.allocated_bytes) + external;
        const auto stem = "gpu_" + std::to_string(device) + "_";
        auto device_weights = ledger.resident_bytes(device);
        if (device == primary)
            device_weights += metric_bytes("resident_expert_cache_bytes");
        const auto hot = prefix_bytes[device];
        auto live = metric_bytes(stem + "live_kv_bytes");
        if (device == primary)
            for (const auto& [name, value] : batch_metrics)
                if (name == "continuous_batching_request_kv_bytes")
                    live += std::size_t(value);
        metrics.insert(metrics.end(), {
            {stem + "resident_bytes", double(resident)},
            {stem + "resident_weight_bytes", double(device_weights)},
            {stem + "prefix_cache_bytes", double(hot)},
            {stem + "memory_total_bytes", double(total)},
            {stem + "memory_available_bytes", double(free)}});
        if (memory) {
            const auto categorized = device_weights + live + hot;
            const auto buffer = resident > categorized ? resident - categorized : 0;
            metrics.emplace_back(stem + "resident_buffer_bytes", double(buffer));
            metrics.emplace_back(stem + "kv_bytes", double(live + hot));
            buffers += buffer;
        }
        allocated += memory_stats.allocated_bytes + external;
        reserved += memory_stats.reserved_bytes + external;
        free_bytes += free;
        total_bytes += total;
    }
    metrics.insert(metrics.end(), {
        {"device_free_bytes", double(free_bytes)},
        {"device_total_bytes", double(total_bytes)},
        {"cuda_allocated_bytes", double(allocated)},
        {"cuda_reserved_bytes", double(reserved)}});
    if (memory)
        metrics.emplace_back("resident_buffer_bytes", double(buffers));
    const auto expert_bytes = metric_bytes("resident_expert_cache_bytes");
    const auto minimum = moe_expert_cache_minimum_bytes(weights.moe);
    const bool online_budget_supported = placement.devices.size() <= 1 &&
        !tp.enabled() && !ep.enabled();
    metrics.insert(metrics.end(), {
        {"resident_memory_dynamic_budget", online_budget_supported ? 1.0 : 0.0},
        {"resident_weight_dynamic_budget", online_budget_supported ? 1.0 : 0.0},
        {"resident_memory_budget_bytes", double(budget.resident())},
        {"resident_weight_budget_bytes", double(budget.weights())},
        {"resident_memory_used_bytes", double(std::max(allocated, reserved))},
        {"resident_reclaimable_weight_bytes", double(expert_bytes > minimum ? expert_bytes - minimum : 0)}});
    metrics.emplace_back("qsa_kv_offload_supported", components.streamed_kv() ? 1.0 : 0.0);
    if (offload) {
        auto offload_metrics = offload->metrics();
        metrics.insert(metrics.end(), offload_metrics.begin(), offload_metrics.end());
    } else {
        metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
            {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0},
            {"qsa_kv_ssd_bytes", 0.0}});
    }
    if (components.mtp) {
        mfq::engine::mtp::append_generation_metrics(metrics, components.mtp->last_stats);
        metrics.emplace_back("mtp_predictor_graph_replays",
            double(components.mtp->graph_replay_count()));
    }
    if (batching)
        metrics.insert(metrics.end(), batch_metrics.begin(), batch_metrics.end());
    return metrics;
}

} // namespace mfq::cuda::engine_detail
