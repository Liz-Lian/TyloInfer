#include "engine/weights/workspace.h"

#include "engine/ops.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/cache_internal.h"
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

namespace tyloi::weights {
using namespace tyloi::cuda;
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

struct detail::Workspace {
    struct PrefillRegion {
        ::tyloi::DenseTensor storage;
        std::size_t region = 0;
    };

    int device = 0;
    std::vector<PrefillRegion> prefill_regions;
    std::uint64_t prefill_reclaimed_used_bytes = 0;
};

struct detail::KvOwner {
    std::shared_ptr<KvExpertLoan> loan = std::make_shared<KvExpertLoan>();
    std::weak_ptr<MoeExpertCache> moe;
};

struct KvRegion::Impl {
    std::weak_ptr<detail::KvOwner> owner;
    std::size_t region = 0;
};

KvRegion::KvRegion() noexcept = default;
KvRegion::KvRegion(KvRegion&&) noexcept = default;
KvRegion& KvRegion::operator=(KvRegion&& other) {
    if (this != &other) {
        reset();
        impl_ = std::move(other.impl_);
    }
    return *this;
}
KvRegion::~KvRegion() noexcept {
    try {
        reset();
    } catch (const std::exception& error) {
        std::cerr << "kv_region cleanup_error=" << error.what() << '\n';
    } catch (...) {
        std::cerr << "kv_region cleanup_error=unknown\n";
    }
}
void KvRegion::reset() {
    if (!impl_) return;
    if (auto owner = impl_->owner.lock()) {
        finish_moe_expert_exchanges(owner->moe.lock());
        owner->loan->remove(impl_->region);
    }
    impl_.reset();
}
void KvRegion::advance(std::int64_t end) {
    if (impl_) if (auto owner = impl_->owner.lock()) owner->loan->ensure(end);
}
void KvRegion::protect(std::int64_t end) {
    if (impl_) if (auto owner = impl_->owner.lock()) owner->loan->protect(impl_->region, end);
}
std::size_t KvRegion::bytes() const {
    if (impl_) if (auto owner = impl_->owner.lock()) return owner->loan->region_bytes(impl_->region);
    return 0;
}

detail::CacheData::CacheData() = default;
detail::CacheData::~CacheData() = default;
Cache::Cache() : data_(std::make_unique<detail::CacheData>()) {}
Cache::~Cache() { reset(); }

detail::CacheData& detail::cache(Cache& value) { return *value.data_; }
const detail::CacheData& detail::cache(const Cache& value) { return *value.data_; }

void Cache::prepare_kv() {
    if (!data_->kv) data_->kv = std::make_shared<detail::KvOwner>();
    data_->kv->moe = data_->moe;
}

KvRegion Cache::lend_kv(const Tensor& backing, std::int64_t capacity,
        std::size_t row_bytes, bool prefix, std::int64_t ratio) {
    KvRegion result;
    if (!data_->kv) return result;
    auto impl = std::make_unique<KvRegion::Impl>();
    impl->owner = data_->kv;
    impl->region = data_->kv->loan->add(dynamic_cast<const DenseTensor&>(backing),
        capacity, row_bytes, prefix, ratio);
    result.impl_ = std::move(impl);
    return result;
}

KvRegion Cache::lend_kv(const Tensor& arena) {
    KvRegion result;
    if (!data_->kv) return result;
    auto impl = std::make_unique<KvRegion::Impl>();
    impl->owner = data_->kv;
    const auto& dense = dynamic_cast<const DenseTensor&>(arena);
    impl->region = data_->kv->loan->add(dense, static_cast<std::int64_t>(dense.nbytes()),
        1, false, 1, true);
    result.impl_ = std::move(impl);
    return result;
}

void Cache::commit_kv() {
    if (data_->kv) data_->kv->moe = data_->moe;
    set_moe_kv_expert_loan(data_->moe, data_->kv ? data_->kv->loan : nullptr);
}
void Cache::protect_kv(std::int64_t end) {
    if (data_->kv) data_->kv->loan->ensure(end);
}
Cache::KvMetrics Cache::kv_metrics() const {
    if (!data_->kv) return {};
    const auto& loan = *data_->kv->loan;
    return {loan.bytes() - loan.external_bytes(), loan.hits(), loan.lookup_us()};
}
void Cache::bind(engine::Budget& budget, Device device, bool predictor) {
    if (!data_->moe) return;
    if (!data_->kv && (predictor || moe_expert_cache_ssd_experts(data_->moe))) {
        prepare_kv();
        commit_kv();
    }
    if (data_->kv) {
        clear_workspace();
        data_->workspace = std::make_unique<detail::Workspace>();
        data_->workspace->device = device.index;
    }
    bind_moe_budget(data_->moe, &budget);
}
void Cache::unbind() noexcept {
    bind_moe_budget(data_->moe, nullptr);
}
void Cache::retire_decode() {
    release_moe_decode_buffers(data_->moe);
}

void Cache::clear_workspace() noexcept {
    if (!data_->workspace) return;
    try {
        release_workspace();
    } catch (const std::exception& error) {
        std::cerr << "runtime_mtp_workspace_cache cleanup_error="
            << error.what() << '\n';
    } catch (...) {
        std::cerr << "runtime_mtp_workspace_cache cleanup_error=unknown\n";
    }
    data_->workspace.reset();
}

std::size_t Cache::prefill_workspace_capacity() const noexcept {
    std::size_t bytes = 0;
    if (data_->workspace)
        for (const auto& part : data_->workspace->prefill_regions) bytes += part.storage.nbytes();
    return bytes;
}

void Cache::release_workspace(std::size_t required_available, const char* reason) {
    if (!data_->workspace || data_->workspace->prefill_regions.empty()) return;
    auto& w = *data_->workspace;
    auto& loan = *data_->kv->loan;
    if (reason) {
        const auto context = default_context(w.device);
        std::cerr << "runtime_workspace_cache_reclaim reason=" << reason
            << " requested_bytes=" << required_available
            << " available_bytes=" << context->available_memory() << '\n';
    }

    finish_moe_expert_exchanges(data_->moe);
    TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto context = default_context(w.device);
    const bool all = required_available == std::numeric_limits<std::size_t>::max();
    while (!w.prefill_regions.empty() &&
            (all || context->available_memory() < required_available)) {
        const auto& part = w.prefill_regions.back();
        const auto bytes = part.storage.nbytes();
        const auto used = loan.region_bytes(part.region);
        loan.remove(part.region);
        w.prefill_reclaimed_used_bytes += used;
        w.prefill_regions.pop_back();
        context->reclaim_idle_memory();
        std::cerr << "runtime_prefill_workspace_cache action=reclaim bytes="
            << bytes << " used_bytes=" << used << '\n';
    }
    context->reclaim_idle_memory();
}

void Cache::restore_workspace(std::size_t reserve_bytes, const tyloi::source::Source* source) {
    if (!data_->workspace) return;
    auto& w = *data_->workspace;
    if (!data_->kv) return;

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
        const auto region = data_->kv->loan->add(storage, static_cast<std::int64_t>(chunk),
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

void Cache::append_workspace_metrics(
        std::vector<std::pair<std::string, double>>& metrics) const {
    std::size_t prefill_used = 0;
    std::uint64_t prefill_reclaimed_used = 0;
    if (data_->workspace) {
        const auto& w = *data_->workspace;
        for (const auto& part : w.prefill_regions)
            if (data_->kv) prefill_used += data_->kv->loan->region_bytes(part.region);
        prefill_reclaimed_used = w.prefill_reclaimed_used_bytes;
    }
    metrics.insert(metrics.end(), {
        {"prefill_workspace_expert_capacity_bytes", double(prefill_workspace_capacity())},
        {"prefill_workspace_expert_used_bytes", double(prefill_used)},
        {"prefill_workspace_expert_reclaimed_used_bytes", double(prefill_reclaimed_used)}});
}

} // namespace tyloi::weights

tyloi::weights::Loan::Loan(Cache& cache, std::size_t bytes) : Loan(cache) { begin(bytes); }

std::shared_ptr<tyloi::Tensor> tyloi::weights::Loan::finish(std::shared_ptr<Tensor> output) {
    if (active_) {
        suspend();
        output = ops::clone(*output);
        finish();
    }
    return output;
}
