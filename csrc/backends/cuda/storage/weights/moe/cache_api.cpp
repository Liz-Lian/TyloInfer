#include "storage/weights/moe/expert_cache.h"
#include "tyloinfer/format_features.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "mfq_cache_provider.h"
#endif

MoePrefillWorkspaceBudget moe_prefill_workspace_budget(const std::shared_ptr<MoeExpertCache>& cache, bool allow_borrow) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_prefill_workspace_budget(cache, allow_borrow);
#else
    return {};
#endif
}

void set_moe_kv_expert_loan(const std::shared_ptr<MoeExpertCache>& cache, std::shared_ptr<mfq::cuda::KvExpertLoan> loan) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_set_moe_kv_expert_loan(cache, std::move(loan));
#endif
}

std::size_t moe_prefill_transfer_bytes(const std::shared_ptr<MoeExpertCache>& cache, bool predictor) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_prefill_transfer_bytes(cache, predictor);
#else
    return {};
#endif
}

std::shared_ptr<MoeExpertCache> make_moe_expert_cache(std::int64_t bytes, const CudaExecutionConfig& config) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_make_moe_expert_cache(bytes, config);
#else
    return {};
#endif
}

bool moe_expert_cache_has_sources(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_has_sources(cache);
#else
    return {};
#endif
}

bool moe_expert_cache_finalized(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_finalized(cache);
#else
    return {};
#endif
}

bool moe_expert_cache_layer_prefill_ready(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_layer_prefill_ready(cache);
#else
    return {};
#endif
}

bool moe_expert_cache_ssd_experts(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_ssd_experts(cache);
#else
    return {};
#endif
}

bool moe_expert_cache_ssd_router_prefetch(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_ssd_router_prefetch(cache);
#else
    return {};
#endif
}

std::vector<std::pair<std::string, double>> moe_expert_memory_metrics(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_memory_metrics(cache);
#else
    return {};
#endif
}

void finalize_moe_expert_cache(const std::shared_ptr<MoeExpertCache>& cache, std::size_t kv_growth_reserve, std::size_t kv_budget_charge) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_finalize_moe_expert_cache(cache, kv_growth_reserve, kv_budget_charge);
#endif
}

std::size_t moe_expert_cache_minimum_bytes(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_minimum_bytes(cache);
#else
    return {};
#endif
}

std::size_t moe_expert_cache_budget_bytes(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_budget_bytes(cache);
#else
    return {};
#endif
}

void resize_moe_expert_cache(const std::shared_ptr<MoeExpertCache>& cache, std::size_t bytes) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_resize_moe_expert_cache(cache, bytes);
#endif
}

void set_moe_workspace_reservation(const std::shared_ptr<MoeExpertCache>& cache, std::function<void(std::size_t)> reserve, std::function<void(std::size_t)> reserve_ram) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_set_moe_workspace_reservation(cache, reserve, std::move(reserve_ram));
#endif
}

void reserve_moe_workspace(const std::shared_ptr<MoeExpertCache>& cache, std::size_t bytes) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_reserve_moe_workspace(cache, bytes);
#endif
}

std::size_t reclaim_moe_optional_host(const std::shared_ptr<MoeExpertCache>& cache, std::size_t allocation_bytes) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_reclaim_moe_optional_host(cache, allocation_bytes);
#else
    return {};
#endif
}

double moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_ram_pcie_fraction(cache);
#else
    return {};
#endif
}

std::uint64_t moe_expert_streamed_bytes(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_streamed_bytes(cache);
#else
    return {};
#endif
}

std::uint64_t moe_expert_graph_captures(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_graph_captures(cache);
#else
    return {};
#endif
}

void finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_finish_moe_expert_exchanges(cache);
#endif
}

void prepare_moe_predictor_exchange(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_prepare_moe_predictor_exchange(cache);
#endif
}

void begin_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache, int rows) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_begin_moe_speculative_routes(cache, rows);
#endif
}

void commit_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache, int rows) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_commit_moe_speculative_routes(cache, rows);
#endif
}

void discard_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache)noexcept {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_discard_moe_speculative_routes(cache);
#endif
}

void prepare_moe_decode_residency(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_prepare_moe_decode_residency(cache);
#endif
}

bool prepare_moe_graph_pool(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_prepare_moe_graph_pool(cache);
#else
    return {};
#endif
}

void release_moe_decode_buffers(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_release_moe_decode_buffers(cache);
#endif
}

bool lend_moe_decode_cache_for_prefill(const std::shared_ptr<MoeExpertCache>& cache, std::size_t workspace_bytes) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_lend_moe_decode_cache_for_prefill(cache, workspace_bytes);
#else
    return {};
#endif
}

bool suspend_moe_residency(const std::shared_ptr<MoeExpertCache>& cache, bool suspended) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_suspend_moe_residency(cache, suspended);
#else
    return {};
#endif
}

bool set_moe_reuse_learning(const std::shared_ptr<MoeExpertCache>& cache, bool enabled) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_set_moe_reuse_learning(cache, enabled);
#else
    return {};
#endif
}

void release_moe_prefill_buffers(MoeExpertCache* cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_release_moe_prefill_buffers(cache);
#endif
}

void release_moe_prefill_buffers(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_release_moe_prefill_buffers(cache);
#endif
}

void prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_prepare_moe_pipeline_comparison(cache, replay);
#endif
}

void prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay, bool transfer_budget) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_prepare_moe_cpu_budget_comparison(cache, replay, transfer_budget);
#endif
}

void prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_prepare_moe_cpu_calibration_comparison(cache, replay);
#endif
}

std::size_t finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_finish_moe_pipeline_comparison(cache);
#else
    return {};
#endif
}

void print_moe_expert_cache_stats(const std::shared_ptr<MoeExpertCache>& cache, std::ostream& output) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_print_moe_expert_cache_stats(cache, output);
#endif
}

void set_moe_expert_cache_profile(MoeExpertCache& cache, mfq::MoeCacheProfile profile) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_set_moe_expert_cache_profile(cache, std::move(profile));
#endif
}

#if !TYLOINFER_ENABLE_MFQ_COMPAT
MoePrefillWorkspaceLoan::~MoePrefillWorkspaceLoan() = default;
std::size_t MoePrefillWorkspaceLoan::begin(std::size_t) { return 0; }
void MoePrefillWorkspaceLoan::suspend() {}
void MoePrefillWorkspaceLoan::finish() {}
MoePrefillLayerPriority::MoePrefillLayerPriority(std::shared_ptr<MoeExpertCache> cache, int, int)
    : cache_(std::move(cache)) {}
MoePrefillLayerPriority::~MoePrefillLayerPriority() = default;
void MoePrefillLayerPriority::finish() {}
#endif
