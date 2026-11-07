#include "engine/weights/workspace.h"

#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/routed_cache.h"
#include "env.h"
#include "resident_allocations.h"
#include "engine/weights/source/source.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

namespace tyloi::cuda {
namespace {

bool is_memory_pressure(const std::exception& error) {
    const std::string message = error.what();
    return message.find("VRAM limit") != std::string::npos ||
        message.find("out of memory") != std::string::npos;
}

::tyloi::DenseTensor make_storage(std::size_t bytes) {
    return ::tyloi::cuda::empty({static_cast<std::int64_t>(bytes)},
        ::tyloi::TensorOptions().device(::tyloi::kCUDA)
            .dtype(::tyloi::kUInt8));
}

} // namespace

struct RoutedCache::Workspace {
    struct PrefillRegion {
        ::tyloi::DenseTensor storage;
        std::size_t region = 0;
    };

    int device = 0;
    std::shared_ptr<KvExpertLoan> loan;
    ::tyloi::DenseTensor decode_storage;
    std::optional<std::size_t> decode_region;
    std::vector<PrefillRegion> prefill_regions;
    std::uint64_t prefill_reclaimed_used_bytes = 0;
};

RoutedCache::RoutedCache() = default;
RoutedCache::~RoutedCache() { clear_workspace(); }

void RoutedCache::workspace(int device, std::shared_ptr<KvExpertLoan> loan) {
    clear_workspace();
    workspace_ = std::make_unique<Workspace>();
    workspace_->device = device;
    workspace_->loan = std::move(loan);
}

void RoutedCache::clear_workspace() noexcept {
    if (!workspace_) return;
    try {
        release_workspace();
    } catch (const std::exception& error) {
        std::cerr << "runtime_mtp_workspace_cache cleanup_error="
            << error.what() << '\n';
    } catch (...) {
        std::cerr << "runtime_mtp_workspace_cache cleanup_error=unknown\n";
    }
    workspace_.reset();
}

bool RoutedCache::has_decode_workspace() const noexcept {
    return workspace_ && workspace_->decode_region.has_value();
}

std::size_t RoutedCache::prefill_workspace_capacity() const noexcept {
    std::size_t bytes = 0;
    if (workspace_)
        for (const auto& part : workspace_->prefill_regions) bytes += part.storage.nbytes();
    return bytes;
}

void RoutedCache::release_workspace(std::size_t required_available, const char* reason) {
    if (!workspace_ || (!workspace_->decode_region && workspace_->prefill_regions.empty())) return;
    auto& w = *workspace_;
    if (reason) {
        const auto context = default_context(w.device);
        std::cerr << "runtime_workspace_cache_reclaim reason=" << reason
            << " requested_bytes=" << required_available
            << " available_bytes=" << context->available_memory() << '\n';
    }

    finish_moe_expert_exchanges(moe);
    TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto context = default_context(w.device);
    const bool all = required_available == std::numeric_limits<std::size_t>::max();
    while (!w.prefill_regions.empty() &&
            (all || context->available_memory() < required_available)) {
        const auto& part = w.prefill_regions.back();
        const auto bytes = part.storage.nbytes();
        const auto used = w.loan->region_bytes(part.region);
        w.loan->remove(part.region);
        w.prefill_reclaimed_used_bytes += used;
        w.prefill_regions.pop_back();
        context->reclaim_idle_memory();
        std::cerr << "runtime_prefill_workspace_cache action=reclaim bytes="
            << bytes << " used_bytes=" << used << '\n';
    }
    if (w.decode_region &&
            (all || context->available_memory() < required_available)) {
        const auto bytes = w.decode_storage.nbytes();
        w.loan->remove(*w.decode_region);
        w.decode_region.reset();
        w.decode_storage = {};
        std::cerr << "runtime_mtp_workspace_cache action=reclaim bytes="
            << bytes << '\n';
    }
    context->reclaim_idle_memory();
}

std::size_t RoutedCache::lend_decode(std::size_t maximum_minus_active) {
    if (!workspace_) return 0;
    auto& w = *workspace_;
    if (w.decode_region || !w.loan || !maximum_minus_active) return 0;

    const auto context = default_context(w.device);
    const auto available = context->available_memory();
    constexpr auto headroom = std::size_t(256) << 20;
    auto bytes = std::min(maximum_minus_active,
        available > headroom ? available - headroom : std::size_t(0));
    bytes = bytes / KvExpertLoan::page * KvExpertLoan::page;
    if (!bytes) return 0;

    ResidentAllocationScope resident;
    try {
        w.decode_storage = make_storage(bytes);
    } catch (const std::exception& error) {
        if (!is_memory_pressure(error)) throw;
        std::cerr << "runtime_mtp_workspace_cache action=skip reason=memory_pressure\n";
        return 0;
    }
    w.decode_region = w.loan->add(w.decode_storage, static_cast<std::int64_t>(bytes),
        1, true, 1, true, true);
    return bytes;
}

void RoutedCache::restore_workspace(std::size_t reserve_bytes, const tyloi::source::Source* source) {
    if (!workspace_) return;
    auto& w = *workspace_;
    if (!w.loan) return;

    bool ssd = false;
    std::size_t transfer = 0;
    std::size_t streamed = 0;
    const auto expert_metrics = metrics();
    for (const auto& [name, value] : expert_metrics) {
        if (name == "ssd_expert_enabled") ssd = value != 0;
        if (name == "expert_decode_stage_budget_bytes") transfer = std::size_t(value);
        if (name == "ram_expert_payload_bytes" || name == "ssd_expert_payload_bytes")
            streamed += std::size_t(value);
    }
    if (!ssd) return;

    if (const auto* profile = env_lookup("TYLOINFER_PROFILE_SSD_EXPERTS");
            profile && profile[0] == '1' && source) {
        const auto io = source->file_read_stats();
        std::cerr << "runtime_model_io_stats boundary=prefill_end calls=" << io.calls
            << " logical_bytes=" << io.logical_bytes
            << " physical_bytes=" << io.physical_bytes
            << " read_ms=" << double(io.read_nanoseconds) / 1e6 << '\n';
        std::cerr << "runtime_moe_pipeline_stats boundary=prefill_end";
        for (const auto& [name, value] : expert_metrics)
            if (name == "expert_route_wait_ms" || name == "expert_plan_ms" ||
                    name == "expert_host_fetch_ms" ||
                    name == "expert_logical_transfer_bytes" ||
                    name == "ssd_expert_read_wait_ms")
                std::cerr << ' ' << name << '=' << value;
        std::cerr << '\n';
    }

    const auto existing = prefill_workspace_capacity();
    const auto decode = (std::size_t(1280) << 20) + transfer;
    const auto target = std::max(streamed,
        reserve_bytes > decode ? reserve_bytes - decode : std::size_t(0));
    if (existing >= target) return;

    const auto context = default_context(w.device);
    const auto available = context->available_memory();
    constexpr auto headroom = std::size_t(256) << 20;
    auto bytes = std::min(target - existing,
        available > headroom ? available - headroom : std::size_t(0));
    bytes = bytes / KvExpertLoan::page * KvExpertLoan::page;
    if (!bytes) return;

    ResidentAllocationScope resident;
    while (bytes) {
        const auto chunk = std::min(bytes, std::size_t(256) << 20);
        ::tyloi::DenseTensor storage;
        try {
            storage = make_storage(chunk);
        } catch (const std::exception& error) {
            if (!is_memory_pressure(error)) throw;
            break;
        }
        w.prefill_regions.reserve(w.prefill_regions.size() + 1);
        const auto region = w.loan->add(storage, static_cast<std::int64_t>(chunk),
            1, true, 1, true, true);
        w.prefill_regions.push_back({std::move(storage), region});
        bytes -= chunk;
    }
    const auto capacity = prefill_workspace_capacity();
    std::cerr << "runtime_prefill_workspace_cache action=lend bytes="
        << capacity - existing
        << " capacity_bytes=" << capacity
        << " reserve_bytes=" << reserve_bytes << '\n';
}

void RoutedCache::append_workspace_metrics(
        std::vector<std::pair<std::string, double>>& metrics) const {
    std::size_t decode_capacity = 0, decode_used = 0, prefill_used = 0;
    std::uint64_t prefill_reclaimed_used = 0;
    if (workspace_) {
        const auto& w = *workspace_;
        decode_capacity = w.decode_storage.defined() ? w.decode_storage.nbytes() : 0;
        decode_used = w.decode_region && w.loan ? w.loan->region_bytes(*w.decode_region) : 0;
        for (const auto& part : w.prefill_regions)
            if (w.loan) prefill_used += w.loan->region_bytes(part.region);
        prefill_reclaimed_used = w.prefill_reclaimed_used_bytes;
    }
    metrics.insert(metrics.end(), {
        {"mtp_flexible_expert_capacity_bytes", double(decode_capacity)},
        {"mtp_flexible_expert_used_bytes", double(decode_used)},
        {"prefill_workspace_expert_capacity_bytes", double(prefill_workspace_capacity())},
        {"prefill_workspace_expert_used_bytes", double(prefill_used)},
        {"prefill_workspace_expert_reclaimed_used_bytes", double(prefill_reclaimed_used)}});
}

} // namespace tyloi::cuda

::tyloi::DenseTensor tyloi::cuda::WeightLoan::finish(::tyloi::DenseTensor output) {
    if (active_) {
        suspend();
        output = output.clone();
        finish();
    }
    return output;
}
