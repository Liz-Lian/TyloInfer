#include "engine/weights/workspace.h"

#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/routed_cache.h"
#include "env.h"
#include "resident_allocations.h"
#include "source/model_source.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

namespace mfq::cuda {
namespace {

bool is_memory_pressure(const std::exception& error) {
    const std::string message = error.what();
    return message.find("VRAM limit") != std::string::npos ||
        message.find("out of memory") != std::string::npos;
}

mfq_tensor_backend::Tensor make_storage(std::size_t bytes) {
    return mfq_tensor_backend::empty({static_cast<std::int64_t>(bytes)},
        mfq_tensor_backend::TensorOptions().device(mfq_tensor_backend::kCUDA)
            .dtype(mfq_tensor_backend::kUInt8));
}

} // namespace

ExpertSpace::ExpertSpace(RoutedCache& weights, int device,
        std::shared_ptr<KvExpertLoan> loan)
    : weights_(weights), device_(device), loan_(std::move(loan)) {}

ExpertSpace::~ExpertSpace() noexcept {
    try {
        release();
    } catch (const std::exception& error) {
        std::cerr << "runtime_mtp_workspace_cache cleanup_error="
            << error.what() << '\n';
    } catch (...) {
        std::cerr << "runtime_mtp_workspace_cache cleanup_error=unknown\n";
    }
}

bool ExpertSpace::empty() const noexcept {
    return !decode_region_ && prefill_regions_.empty();
}

void ExpertSpace::release(std::size_t required_available) {
    if (empty()) return;

    finish_moe_expert_exchanges(weights_.moe);
    TYLOINFER_CUDA_CHECK(cudaDeviceSynchronize());
    const auto context = default_context(device_);
    const bool all = required_available == std::numeric_limits<std::size_t>::max();
    while (!prefill_regions_.empty() &&
            (all || context->available_memory() < required_available)) {
        const auto& part = prefill_regions_.back();
        const auto bytes = part.storage.nbytes();
        const auto used = loan_->region_bytes(part.region);
        loan_->remove(part.region);
        prefill_reclaimed_used_bytes_ += used;
        prefill_regions_.pop_back();
        context->reclaim_idle_memory();
        std::cerr << "runtime_prefill_workspace_cache action=reclaim bytes="
            << bytes << " used_bytes=" << used << '\n';
    }
    if (decode_region_ &&
            (all || context->available_memory() < required_available)) {
        const auto bytes = decode_storage_.nbytes();
        loan_->remove(*decode_region_);
        decode_region_.reset();
        decode_storage_ = {};
        std::cerr << "runtime_mtp_workspace_cache action=reclaim bytes="
            << bytes << '\n';
    }
    context->reclaim_idle_memory();
}

std::size_t ExpertSpace::lend_decode(std::size_t maximum_minus_active) {
    if (decode_region_ || !loan_ || !maximum_minus_active) return 0;

    const auto context = default_context(device_);
    const auto available = context->available_memory();
    constexpr auto headroom = std::size_t(256) << 20;
    auto bytes = std::min(maximum_minus_active,
        available > headroom ? available - headroom : std::size_t(0));
    bytes = bytes / KvExpertLoan::page * KvExpertLoan::page;
    if (!bytes) return 0;

    ResidentAllocationScope resident;
    try {
        decode_storage_ = make_storage(bytes);
    } catch (const std::exception& error) {
        if (!is_memory_pressure(error)) throw;
        std::cerr << "runtime_mtp_workspace_cache action=skip reason=memory_pressure\n";
        return 0;
    }
    decode_region_ = loan_->add(decode_storage_, static_cast<std::int64_t>(bytes),
        1, true, 1, true, true);
    return bytes;
}

void ExpertSpace::restore(std::size_t reserve_bytes, const ModelSource* source) {
    if (!loan_) return;

    bool ssd = false;
    std::size_t transfer = 0;
    std::size_t streamed = 0;
    const auto expert_metrics = weights_.metrics();
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

    const auto existing = prefill_capacity();
    const auto decode = (std::size_t(1280) << 20) + transfer;
    const auto target = std::max(streamed,
        reserve_bytes > decode ? reserve_bytes - decode : std::size_t(0));
    if (existing >= target) return;

    const auto context = default_context(device_);
    const auto available = context->available_memory();
    constexpr auto headroom = std::size_t(256) << 20;
    auto bytes = std::min(target - existing,
        available > headroom ? available - headroom : std::size_t(0));
    bytes = bytes / KvExpertLoan::page * KvExpertLoan::page;
    if (!bytes) return;

    ResidentAllocationScope resident;
    while (bytes) {
        const auto chunk = std::min(bytes, std::size_t(256) << 20);
        mfq_tensor_backend::Tensor storage;
        try {
            storage = make_storage(chunk);
        } catch (const std::exception& error) {
            if (!is_memory_pressure(error)) throw;
            break;
        }
        prefill_regions_.reserve(prefill_regions_.size() + 1);
        const auto region = loan_->add(storage, static_cast<std::int64_t>(chunk),
            1, true, 1, true, true);
        prefill_regions_.push_back({std::move(storage), region});
        bytes -= chunk;
    }
    const auto capacity = prefill_capacity();
    std::cerr << "runtime_prefill_workspace_cache action=lend bytes="
        << capacity - existing
        << " capacity_bytes=" << capacity
        << " reserve_bytes=" << reserve_bytes << '\n';
}

ExpertSpace::Metrics ExpertSpace::metrics() const {
    Metrics result;
    result.decode_capacity = decode_storage_.defined() ? decode_storage_.nbytes() : 0;
    result.decode_used = decode_region_ && loan_
        ? loan_->region_bytes(*decode_region_) : 0;
    result.prefill_capacity = prefill_capacity();
    for (const auto& part : prefill_regions_) {
        if (loan_) result.prefill_used += loan_->region_bytes(part.region);
    }
    result.prefill_reclaimed_used = prefill_reclaimed_used_bytes_;
    return result;
}

} // namespace mfq::cuda
