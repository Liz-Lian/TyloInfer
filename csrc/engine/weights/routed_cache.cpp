#include "engine/weights/cache_internal.h"

#include "engine/backends/cuda/native/context.h"
#include "engine/backends/cuda/executor/options.h"
#include "engine/weights/moe/expert_cache.h"
#include "engine/weights/source/source.h"
#include "engine/weights/tq/cache.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace tyloi::weights {

struct Cache::Routes::Impl {
    std::shared_ptr<MoeExpertCache> cache;
};

Cache::Routes::Routes() noexcept = default;
Cache::Routes::Routes(Routes&&) noexcept = default;
Cache::Routes::~Routes() {
    if (impl_ && impl_->cache) discard_moe_speculative_routes(impl_->cache);
}

void Cache::Routes::begin(Cache& owner, int rows) {
    if (!impl_) impl_ = std::make_unique<Impl>();
    auto cache = detail::cache(owner).moe;
    begin_moe_speculative_routes(cache, rows);
    impl_->cache = std::move(cache);
}

void Cache::Routes::resolve(int rows, bool cancelled) {
    if (!impl_ || !impl_->cache) return;
    if (cancelled) discard_moe_speculative_routes(impl_->cache);
    else commit_moe_speculative_routes(impl_->cache, rows);
    impl_->cache.reset();
}

struct Cache::Warmup::Impl {
    std::shared_ptr<MoeExpertCache> cache;
    bool suspended = false;
    bool previous_residency = false;
    bool previous_learning = false;
};

Cache::Warmup::Warmup(Cache& owner, bool suspend_residency) : impl_(std::make_unique<Impl>()) {
    impl_->cache = detail::cache(owner).moe;
    impl_->suspended = suspend_residency;
    if (suspend_residency) impl_->previous_residency = suspend_moe_residency(impl_->cache, true);
    impl_->previous_learning = set_moe_reuse_learning(impl_->cache, false);
}

Cache::Warmup::~Warmup() {
    set_moe_reuse_learning(impl_->cache, impl_->previous_learning);
    if (impl_->suspended) suspend_moe_residency(impl_->cache, impl_->previous_residency);
}

void Cache::Warmup::drain() {
    finish_moe_expert_exchanges(impl_->cache);
}

void Cache::Warmup::release(Device device, bool decode) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("warmup requires a CUDA device");
    if (decode) release_moe_decode_buffers(impl_->cache);
    else release_moe_prefill_buffers(impl_->cache);
    cuda::default_context(device.index)->reclaim_idle_memory();
}

Loader Cache::loader(const source::Source& source, Loader::Options options,
        Loader::Placement placement) {
    return Loader(source, std::move(options), std::move(placement), *this, data_->exec);
}

std::vector<std::pair<std::string, double>> Cache::metrics() const {
    auto result = data_->tq ? data_->tq->metrics() : moe_expert_memory_metrics(data_->moe);
    append_workspace_metrics(result);
    return result;
}

Cache::Activity Cache::activity() const {
    return {moe_expert_streamed_bytes(data_->moe), moe_expert_graph_captures(data_->moe)};
}

std::size_t Cache::budget() const {
    return data_->tq ? data_->tq->budget() : moe_expert_cache_budget_bytes(data_->moe);
}

std::size_t Cache::minimum() const {
    return moe_expert_cache_minimum_bytes(data_->moe);
}

bool Cache::graph_safe(Graph kind, bool parallel) const {
    return !data_->moe && !(data_->tq && (parallel || kind == Graph::legacy));
}

Cache::Prefill Cache::prefill_mode(std::size_t tokens, bool resumed) const {
    if (resumed && tokens <= 128 && moe_expert_cache_ssd_experts(data_->moe)) return Prefill::retained;
    if (tokens > 8 && moe_expert_cache_ssd_experts(data_->moe)) return Prefill::streamed;
    return Prefill::regular;
}

Cache::PrefillBudget Cache::prefill_budget(bool allow_borrow) const {
    const auto budget = moe_prefill_workspace_budget(data_->moe, allow_borrow);
    const auto reserve = moe_expert_cache_ssd_experts(data_->moe)
        ? std::max(std::size_t(1) << 30, cuda::options::workspace_reserve_bytes())
        : std::size_t(1) << 30;
    return {reserve, budget.lendable_bytes, budget.transfer_bytes};
}

void Cache::workspace(std::size_t bytes) {
    reserve_moe_workspace(data_->moe, bytes);
}

void Cache::request(bool retire_decode) {
    finish_moe_expert_exchanges(data_->moe);
    if (retire_decode) release_moe_decode_buffers(data_->moe);
    else release_moe_prefill_buffers(data_->moe);
}

std::optional<std::size_t> Cache::prepare_prefill(Device device, std::size_t tokens,
        std::int64_t limit, Prefill mode, bool retire_decode) {
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("budget requires a CUDA device");
    if (mode == Prefill::retained) {
        std::cerr << "runtime_streamed_prefill retained_decode=1 tokens=" << tokens
                  << " chunk=" << std::min<std::int64_t>(4, limit) << '\n';
        return std::nullopt;
    }
    if (mode == Prefill::streamed) {
        const bool retained = lend_moe_decode_cache_for_prefill(data_->moe, cuda::options::workspace_reserve_bytes());
        if (!retained && retire_decode) {
            release_moe_decode_buffers(data_->moe);
            cuda::default_context(device.index)->reclaim_idle_memory();
        }
    }
    if (tokens > 1024 || moe_expert_cache_ssd_experts(data_->moe))
        return cuda::options::workspace_reserve_bytes();
    return std::nullopt;
}

void Cache::configure_decode(std::size_t lendable_bytes, int graph_depth) {
    data_->decode_lendable_bytes = lendable_bytes;
    data_->decode_graph_depth = graph_depth;
}

void Cache::prepare(Stage stage, Device device, const source::Source* source) {
    if (stage == Stage::snapshot) {
        release_moe_prefill_buffers(data_->moe);
        return;
    }
    if (data_->tq || (stage == Stage::decode && !data_->moe)) return;
    if (!device.is_cuda() || device.index < 0)
        throw std::invalid_argument("budget requires a CUDA device");
    prepare_moe_decode_residency(data_->moe);
    cuda::default_context(device.index)->reclaim_idle_memory();
    if (data_->moe && data_->decode_lendable_bytes) {
        const auto bytes = lend_decode(data_->decode_lendable_bytes);
        if (bytes) std::cerr << "runtime_mtp_workspace_cache action=lend depth="
            << data_->decode_graph_depth << " bytes=" << bytes << '\n';
    }
    restore_workspace(cuda::options::workspace_reserve_bytes(), source);
}

void Cache::resize(std::size_t bytes) {
    if (data_->tq) data_->tq->resize(bytes);
    else resize_moe_expert_cache(data_->moe, bytes);
}

std::size_t Cache::reserve(std::size_t bytes) {
    return data_->tq ? data_->tq->reserve(bytes) : reclaim_moe_optional_host(data_->moe, bytes);
}

void Cache::drain() const {
    drain_moe_expert_cache(data_->moe);
}

void Cache::settle() {
    release_workspace();
    finish_moe_expert_exchanges(data_->moe);
}

void Cache::trace(const char* boundary, bool wait, const source::Source* source) {
    std::cerr << "moe_request_cache_stats boundary=" << boundary << '\n';
    if (wait) print_moe_expert_cache_stats(data_->moe, std::cerr);
    std::cerr << "moe_phase_metrics boundary=" << boundary;
    for (const auto& [name, value] : metrics())
        if (name == "ssd_expert_read_wait_ms" || name == "expert_route_wait_ms" ||
            name == "expert_host_fetch_ms" || name == "expert_host_copy_ms" ||
            name == "expert_transfer_cache_prepare_ms" || name == "ssd_expert_bytes_read" ||
            name == "ssd_expert_read_count" || name == "ram_expert_cache_hits" ||
            name == "ram_expert_cache_misses" || name == "ram_expert_pcie_read_bytes" ||
            name == "expert_logical_transfer_bytes") std::cerr << ' ' << name << '=' << value;
    std::cerr << '\n';
    if (source) {
        const auto io = source->file_read_stats();
        std::cerr << "runtime_model_io_stats boundary=" << boundary << " calls=" << io.calls
                  << " logical_bytes=" << io.logical_bytes << " physical_bytes=" << io.physical_bytes
                  << " read_ms=" << double(io.read_nanoseconds) / 1e6 << '\n';
    }
}

void Cache::reset() noexcept {
    clear_workspace();
    data_->moe.reset();
    data_->tq.reset();
    data_->device_bytes.reset();
    data_->exec.reset();
    data_->kv.reset();
    data_->decode_lendable_bytes = 0;
    data_->decode_graph_depth = 0;
}

} // namespace tyloi::weights
