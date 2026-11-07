#pragma once

#include "engine/metrics.h"
#include "engine/executor.h"
#include "engine/budget.h"
#include "engine/state/session/sessions.h"
#include "engine/weights/cache.h"
#include "engine/weights/memory.h"
#include "engine/weights/source/source.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <utility>

namespace tyloi::engine {

inline Metrics runtime_metrics(
        const Metrics& component_metrics, const Metrics& profile_metrics,
        std::optional<Metrics> memory, bool streamed_kv, int graph_depth,
        const Metrics& batch_metrics, const Metrics& kv_metrics,
        std::optional<Metrics> offload_metrics, const Metrics& predictor_metrics,
        const source::Source* source, const weights::Memory& ledger,
        const weights::Cache& weights, const state::Sessions& cache,
        const Budget& budget, std::span<const tyloi::Executor::Memory> devices,
        Device primary, bool online_budget_supported) {
    Metrics metrics{{"mtp_graph_prepared_depth", double(graph_depth)}};
    metrics.insert(metrics.end(), component_metrics.begin(), component_metrics.end());
    auto experts = weights.metrics();
    metrics.insert(metrics.end(), experts.begin(), experts.end());
    metrics.insert(metrics.end(), profile_metrics.begin(), profile_metrics.end());
    if (source) {
        const auto io = source->file_read_stats();
        metrics.insert(metrics.end(), {
            {"model_io_direct", io.mode == tyloi::FileReadMode::Direct ? 1.0 : 0.0},
            {"model_io_read_calls", double(io.calls)},
            {"model_io_logical_bytes", double(io.logical_bytes)},
            {"model_io_physical_bytes", double(io.physical_bytes)},
            {"model_io_read_ms", double(io.read_nanoseconds) / 1e6},
            {"model_io_staging_bytes", double(io.staging_bytes)},
            {"model_io_cached_staging_bytes", double(io.staging_cached_bytes)},
            {"model_io_staging_allocations", double(io.staging_allocations)}});
    }
    metrics.insert(metrics.end(), kv_metrics.begin(), kv_metrics.end());
    std::size_t resident_weights = ledger.resident_bytes();
    for (const auto& metric : experts)
        if (metric.first == "resident_expert_cache_bytes")
            resident_weights += std::size_t(metric.second);
    const auto ple = ledger.ple_bytes();
    metrics.insert(metrics.end(), {
        {"resident_weight_bytes", double(resident_weights)},
        {"ssd_ple_enabled", ple > 0 ? 1.0 : 0.0},
        {"ssd_ple_payload_bytes", double(ple)}});
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
    auto prefix_bytes = cache.device_hot_bytes();
    std::size_t allocated = 0, reserved = 0, free_bytes = 0, total_bytes = 0, buffers = 0;
    for (const auto& snapshot : devices) {
        const auto device = snapshot.device.index;
        const auto external = snapshot.device == primary ? metric_bytes("kv_memory_vmm_bytes") : 0;
        const auto resident = std::max(snapshot.reserved, snapshot.allocated) + external;
        const auto stem = "gpu_" + std::to_string(device) + "_";
        auto device_weights = ledger.resident_bytes(device);
        if (snapshot.device == primary)
            device_weights += metric_bytes("resident_expert_cache_bytes");
        const auto hot = prefix_bytes[device];
        auto live = metric_bytes(stem + "live_kv_bytes");
        if (snapshot.device == primary)
            for (const auto& [name, value] : batch_metrics)
                if (name == "continuous_batching_request_kv_bytes")
                    live += std::size_t(value);
        metrics.insert(metrics.end(), {
            {stem + "resident_bytes", double(resident)},
            {stem + "resident_weight_bytes", double(device_weights)},
            {stem + "prefix_cache_bytes", double(hot)},
            {stem + "memory_total_bytes", double(snapshot.total)},
            {stem + "memory_available_bytes", double(snapshot.available)}});
        if (memory) {
            const auto categorized = device_weights + live + hot;
            const auto buffer = resident > categorized ? resident - categorized : 0;
            metrics.emplace_back(stem + "resident_buffer_bytes", double(buffer));
            metrics.emplace_back(stem + "kv_bytes", double(live + hot));
            buffers += buffer;
        }
        allocated += snapshot.allocated + external;
        reserved += snapshot.reserved + external;
        free_bytes += snapshot.available;
        total_bytes += snapshot.total;
    }
    metrics.insert(metrics.end(), {
        {"device_free_bytes", double(free_bytes)},
        {"device_total_bytes", double(total_bytes)},
        {"cuda_allocated_bytes", double(allocated)},
        {"cuda_reserved_bytes", double(reserved)}});
    if (memory)
        metrics.emplace_back("resident_buffer_bytes", double(buffers));
    const auto expert_bytes = metric_bytes("resident_expert_cache_bytes");
    const auto minimum = weights.minimum();
    metrics.insert(metrics.end(), {
        {"resident_memory_dynamic_budget", online_budget_supported ? 1.0 : 0.0},
        {"resident_weight_dynamic_budget", online_budget_supported ? 1.0 : 0.0},
        {"resident_memory_budget_bytes", double(budget.resident())},
        {"resident_weight_budget_bytes", double(budget.weights())},
        {"resident_memory_used_bytes", double(std::max(allocated, reserved))},
        {"resident_reclaimable_weight_bytes", double(expert_bytes > minimum ? expert_bytes - minimum : 0)}});
    metrics.emplace_back("qsa_kv_offload_supported", streamed_kv ? 1.0 : 0.0);
    if (offload_metrics) {
        metrics.insert(metrics.end(), offload_metrics->begin(), offload_metrics->end());
    } else {
        metrics.insert(metrics.end(), {{"qsa_kv_offload_enabled", 0.0},
            {"qsa_kv_ram_enabled", 0.0}, {"qsa_kv_ram_bytes", 0.0},
            {"qsa_kv_ssd_bytes", 0.0}});
    }
    metrics.insert(metrics.end(), predictor_metrics.begin(), predictor_metrics.end());
    metrics.insert(metrics.end(), batch_metrics.begin(), batch_metrics.end());
    return metrics;
}

} // namespace tyloi::engine
