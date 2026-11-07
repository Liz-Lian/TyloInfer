#include "engine/budget.h"

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/generation/prefill_budget.h"
#include "engine/state/session/sessions.h"
#include "engine/requests/processor.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/moe/kv_expert_loan.h"
#include "engine/weights/cache_internal.h"
#include "engine/engine.h"
#include "engine/weights/memory.h"
#include "components/error.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <utility>

namespace tyloi::engine {
using namespace tyloi::cuda;

static void require_cuda_device(Device device) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("budget requires a CUDA device");
}

Budget::Budget(::tyloi::weights::Memory& weight_memory, ::tyloi::weights::Cache& weights,
        state::Sessions& cache, Device primary)
    : weight_memory_(weight_memory), weights_(weights), cache_(cache),
      primary_device_(primary.index), initial_expert_budget_(weights.budget()) {
    require_cuda_device(primary);
    allocation_ceiling_ = default_context(primary_device_)->memory_stats().limit;
}

Budget::~Budget() { unbind(); }

void Budget::bind() {
    auto& moe = ::tyloi::weights::detail::cache(weights_).moe;
    if (moe) {
        if (moe_expert_cache_ssd_experts(moe)) {
            auto loan = std::make_shared<KvExpertLoan>();
            set_moe_kv_expert_loan(moe, loan);
            ::tyloi::weights::detail::init_workspace(weights_, primary_device_, std::move(loan));
        }
        bind_moe_budget(moe, this);
    }
}

void Budget::unbind() noexcept {
    bind_moe_budget(::tyloi::weights::detail::cache(weights_).moe, nullptr);
}

void Budget::reclaim(std::size_t bytes) {
    auto context=default_context(primary_device_);
    const auto available=context->available_memory();
    if(available>=bytes)return;
    weights_.release_workspace(bytes, "workspace");
    if(context->available_memory()>=bytes)return;
    const auto released=trim_prefix(bytes, primary_device_);
    context->reclaim_idle_memory();
    if(released)std::cerr<<"runtime_workspace_reclaim requested_bytes="<<bytes
        <<" available_bytes="<<context->available_memory()<<" prefix_released_bytes="<<released<<'\n';
}

void Budget::reserve_kv(std::size_t bytes) {
    auto context = default_context(primary_device_);
    if (context->available_memory() >= bytes) return;
    weights_.release_workspace(bytes, "kv_growth");
    if (context->available_memory() >= bytes) return;
    trim_prefix(bytes, primary_device_);
    context->reclaim_idle_memory();
    if (context->available_memory() < bytes)
        throw engine::InferenceInputError(engine::InferenceInputErrorCode::ContextLength,
            "insufficient VRAM for context growth; reduce concurrent requests or prompt length");
}

void Budget::reserve_ram(std::size_t bytes) {
    cache_.reserve_ram(bytes);
    weights_.reserve(bytes);
}

std::size_t Budget::trim_prefix(std::size_t bytes, int device) {
    const auto context=default_context(device);
    std::size_t hot=0;
    for(const auto& [name,value]:cache_.metrics())
        if(name=="prefix_cache_hot_bytes")hot=std::size_t(value);
    const auto shortage=bytes-context->available_memory();
    return cache_.trim_hot(hot>shortage?hot-shortage:0);
}

void Budget::workspace(Device device, std::size_t bytes) {
    require_cuda_device(device);
    reserve_moe_workspace(tyloi::weights::detail::cache(weights_).moe, bytes);
    const auto context=default_context(device.index);
    if(context->available_memory()>=bytes)return;
    const auto released=trim_prefix(bytes, device.index);
    context->reclaim_idle_memory();
    std::cerr<<"runtime_workspace_budget requested_bytes="<<bytes
        <<" available_bytes="<<context->available_memory()<<" prefix_released_bytes="<<released<<'\n';
}

PrefillChunk Budget::batch(Device device, std::size_t row_bytes, PrefillChunk chunk,
        std::size_t growth) {
    require_cuda_device(device);
    const auto context=default_context(device.index);
    const auto transfer=moe_prefill_workspace_budget(tyloi::weights::detail::cache(weights_).moe,false).transfer_bytes;
    const auto fixed=engine::prefill_budget_add((std::size_t(1)<<30)+growth,transfer);
    const auto available=context->available_memory()+context->workspace_available();
    const auto selected=engine::select_prefill_chunk(chunk.count,chunk.count,available,fixed,row_bytes);
    if(!selected)throw engine::InferenceInputError(
        engine::InferenceInputErrorCode::ContextLength,
        "insufficient CUDA workspace for batched prefill; reduce concurrent requests");
    chunk.count=selected;
    return chunk;
}

void Budget::request(Device device, bool retire_decode) {
    require_cuda_device(device);
    const auto& moe = tyloi::weights::detail::cache(weights_).moe;
    finish_moe_expert_exchanges(moe);
    if (retire_decode) release_moe_decode_buffers(moe);
    else release_moe_prefill_buffers(moe);
    default_context(device.index)->reclaim_idle_memory();
}

Budget::Prefill Budget::prefill_mode(std::size_t tokens, bool resumed) const {
    const auto& moe = tyloi::weights::detail::cache(weights_).moe;
    if (resumed && tokens <= 128 && moe_expert_cache_ssd_experts(moe)) return Prefill::retained;
    if (tokens > 8 && moe_expert_cache_ssd_experts(moe)) return Prefill::streamed;
    return Prefill::regular;
}

void Budget::prepare_prefill(Device device, std::size_t tokens, std::int64_t limit,
        Prefill mode, bool retire_decode) {
    require_cuda_device(device);
    const auto& moe = tyloi::weights::detail::cache(weights_).moe;
    if (mode == Prefill::retained) {
        std::cerr << "runtime_streamed_prefill retained_decode=1 tokens=" << tokens
                  << " chunk=" << std::min<int64_t>(4, limit) << '\n';
        return;
    }
    if (mode == Prefill::streamed) {
        const bool retained = lend_moe_decode_cache_for_prefill(moe, runtime_options::workspace_reserve_bytes());
        if (!retained && retire_decode) {
            release_moe_decode_buffers(moe);
            default_context(device.index)->reclaim_idle_memory();
        }
    }
    if (tokens > 1024 || moe_expert_cache_ssd_experts(moe))
        workspace(device, runtime_options::workspace_reserve_bytes());
}

std::int64_t Budget::prefill(std::size_t tokens, std::int64_t limit,
        std::size_t row_bytes, bool resumed, bool automatic) {
    prepare_prefill(Device(kCUDA, primary_device_), tokens, limit, prefill_mode(tokens, resumed));
    return chunk(Device(kCUDA, primary_device_), tokens, limit, row_bytes, resumed, automatic);
}

std::int64_t Budget::chunk(Device device, std::size_t tokens, std::int64_t limit,
        std::size_t row_bytes, bool resumed, bool automatic, bool allow_borrow) {
    require_cuda_device(device);
    const auto& moe = tyloi::weights::detail::cache(weights_).moe;
    if (prefill_mode(tokens, resumed) == Prefill::retained)
        return std::min<int64_t>(4, limit);
    if (tokens <= 128) return std::min<int64_t>(std::max<std::size_t>(1, tokens), limit);

    auto budget = moe_prefill_workspace_budget(moe, allow_borrow);
    auto context = default_context(device.index);
    auto available = engine::prefill_budget_add(engine::prefill_budget_add(context->available_memory(), context->workspace_available()), budget.lendable_bytes);
    const auto sequence_bytes = allow_borrow
        ? std::size_t(tokens <= 16384 ? tokens : std::min<std::size_t>(tokens, limit)) * row_bytes * 3 / 8 : 0;
    const auto reserve = moe_expert_cache_ssd_experts(moe)
        ? std::max(std::size_t(1) << 30, runtime_options::workspace_reserve_bytes()) : std::size_t(1) << 30;
    const auto fixed = engine::prefill_budget_add(reserve + sequence_bytes, budget.transfer_bytes);
    const auto wanted = std::min<int64_t>({int64_t(tokens), limit, engine::max_prefill_compute_chunk});
    const auto required = engine::prefill_budget_add(fixed, std::size_t(wanted) * row_bytes);
    if (required > available) {
        workspace(device, required > budget.lendable_bytes ? required - budget.lendable_bytes : 0);
        budget = moe_prefill_workspace_budget(moe, allow_borrow);
        available = engine::prefill_budget_add(context->available_memory() + context->workspace_available(), budget.lendable_bytes);
    }
    const auto chunk_size = engine::select_prefill_chunk(tokens, limit, available, fixed, row_bytes);
    if (!chunk_size) throw engine::InferenceInputError(engine::InferenceInputErrorCode::ContextLength,
        "insufficient CUDA workspace for prefill computation and expert transfer buffers");
    std::cerr << "runtime_prefill_plan mode=" << (automatic ? "auto" : "limited")
              << " tokens=" << tokens << " limit=" << limit << " compute_chunk=" << chunk_size
              << " available_bytes=" << available << " lendable_bytes=" << budget.lendable_bytes
              << " transfer_bytes=" << budget.transfer_bytes << '\n';
    return chunk_size;
}

void Budget::configure_decode(std::size_t lendable_bytes, int graph_depth) {
    decode_lendable_bytes_ = lendable_bytes;
    decode_graph_depth_ = graph_depth;
}

void Budget::prepare(Stage stage, const source::Source* source) {
    const auto& data = tyloi::weights::detail::cache(weights_);
    if (stage == Stage::snapshot) {
        release_moe_prefill_buffers(data.moe);
        return;
    }
    if (data.tq || (stage == Stage::decode && !data.moe)) return;
    prepare_moe_decode_residency(data.moe);
    default_context(primary_device_)->reclaim_idle_memory();
    if (decode_lendable_bytes_) {
        const auto bytes = weights_.lend_decode(decode_lendable_bytes_);
        if (bytes) std::cerr << "runtime_mtp_workspace_cache action=lend depth="
            << decode_graph_depth_ << " bytes=" << bytes << '\n';
    }
    weights_.restore_workspace(runtime_options::workspace_reserve_bytes(), source);
}

bool Budget::snapshot(Device device, std::size_t bytes, std::size_t peak) {
    require_cuda_device(device);
    std::size_t hot = 0, limit = 0;
    for (const auto& metric : cache_.metrics()) {
        if (metric.first == "prefix_cache_hot_bytes") hot = std::size_t(metric.second);
        if (metric.first == "prefix_cache_max_bytes") limit = std::size_t(metric.second);
    }
    const auto target = limit > bytes ? limit - bytes : 0;
    if (hot > target) cache_.trim_hot(target);
    const auto needed = peak + (std::size_t(64) << 20);
    workspace(device, needed);
    const auto available = default_context(device.index)->available_memory();
    if (available < needed) {
        std::cerr << "runtime_session_cache action=skip reason=workspace_pressure"
            << " requested_bytes=" << needed << " available_bytes=" << available << '\n';
        return false;
    }
    return true;
}

Budget::Change Budget::change(const engine::SessionCommand& command) {
    return Change(*this, command);
}

Budget::Change::Change(Budget& budget, const engine::SessionCommand& command)
    : budget_(budget), bytes(command.bytes) {
    const auto context = default_context(budget.primary_device_);
    dense = budget.weight_memory_.resident_bytes();
    minimum = budget.weights_.minimum();
    requested_weights = command.weight_bytes.value_or(budget.weight_budget_);
    if (requested_weights && requested_weights < dense + minimum)
        throw std::invalid_argument("weight budget is below dense weights and minimum expert working set");
    limit = command.bytes ? budget.allocation_ceiling_
            ? std::min<std::size_t>(command.bytes, budget.allocation_ceiling_)
            : command.bytes
        : budget.allocation_ceiling_;
    if (limit) {
        std::size_t free = 0, total = 0;
        TYLOINFER_CUDA_CHECK(cudaMemGetInfo(&free, &total));
        limit = std::min(limit, std::min(total, free + context->resident_memory_usage()));
    }
    if (limit && limit < dense + minimum)
        throw std::invalid_argument("resident memory budget is below non-offloadable weights");
    previous_limit = context->memory_stats().limit;
    previous_experts = budget.weights_.budget();
    before = context->resident_memory_usage();
}

engine::SessionResult Budget::Change::apply() {
    auto& weights_ = budget_.weights_;
    auto& cache_ = budget_.cache_;
    const auto context = default_context(budget_.primary_device_);
    release_moe_decode_buffers(tyloi::weights::detail::cache(weights_).moe);
    context->reclaim_idle_memory();
    auto target = requested_weights ? requested_weights - dense : budget_.initial_expert_budget_;
    std::size_t hot = 0, expert_bytes = 0;
    for (const auto& [name, value] : cache_.metrics())
        if (name == "prefix_cache_hot_bytes") hot = std::size_t(value);
    for (const auto& [name, value] : weights_.metrics())
        if (name == "resident_expert_cache_bytes") expert_bytes = std::size_t(value);
    auto used = context->resident_memory_usage();
    if (limit && used > limit) {
        cache_.trim_hot(hot > used - limit ? hot - (used - limit) : 0);
        context->reclaim_idle_memory();
        used = context->resident_memory_usage();
    }
    const auto fixed = used > expert_bytes ? used - expert_bytes : 0;
    if (limit) {
        if (fixed > limit || minimum > limit - fixed)
            throw std::invalid_argument("resident memory budget is below live KV and workspace requirements");
        target = std::min(target, limit - fixed);
    }
    try {
        if (!limit || limit > previous_limit) context->set_memory_limit(limit);
        weights_.resize(std::max(target, minimum));
        context->set_memory_limit(limit);
    } catch (...) {
        context->set_memory_limit(0);
        weights_.resize(previous_experts);
        context->set_memory_limit(previous_limit);
        throw;
    }
    budget_.resident_budget_ = bytes;
    budget_.weight_budget_ = requested_weights;
    const auto after = context->resident_memory_usage();
    return {before > after ? before - after : 0, {}};
}

} // namespace tyloi::engine
