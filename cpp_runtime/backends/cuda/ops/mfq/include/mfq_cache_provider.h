#pragma once
#include "moe_expert_cache.h"

MoePrefillWorkspaceBudget mfq_legacy_moe_prefill_workspace_budget(
    const std::shared_ptr<MoeExpertCache>& cache, bool allow_borrow);
void mfq_legacy_set_moe_kv_expert_loan(const std::shared_ptr<MoeExpertCache>& cache,std::shared_ptr<mfq::cuda::KvExpertLoan> loan);
std::size_t mfq_legacy_moe_prefill_transfer_bytes(const std::shared_ptr<MoeExpertCache>& cache,bool predictor=false);
std::shared_ptr<MoeExpertCache> mfq_legacy_make_moe_expert_cache(
    std::int64_t bytes,
    const CudaExecutionConfig& config);
bool mfq_legacy_moe_expert_cache_has_sources(
    const std::shared_ptr<MoeExpertCache>& cache);
bool mfq_legacy_moe_expert_cache_finalized(
    const std::shared_ptr<MoeExpertCache>& cache);
bool mfq_legacy_moe_expert_cache_layer_prefill_ready(const std::shared_ptr<MoeExpertCache>& cache);
bool mfq_legacy_moe_expert_cache_ssd_experts(const std::shared_ptr<MoeExpertCache>& cache);
bool mfq_legacy_moe_expert_cache_ssd_router_prefetch(const std::shared_ptr<MoeExpertCache>& cache);
std::vector<std::pair<std::string, double>> mfq_legacy_moe_expert_memory_metrics(
    const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_finalize_moe_expert_cache(
    const std::shared_ptr<MoeExpertCache>& cache, std::size_t kv_growth_reserve = 0, std::size_t kv_budget_charge = 0);
std::size_t mfq_legacy_moe_expert_cache_minimum_bytes(const std::shared_ptr<MoeExpertCache>& cache);
std::size_t mfq_legacy_moe_expert_cache_budget_bytes(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_resize_moe_expert_cache(const std::shared_ptr<MoeExpertCache>& cache, std::size_t bytes);
void mfq_legacy_set_moe_workspace_reservation(const std::shared_ptr<MoeExpertCache>& cache,
    std::function<void(std::size_t)> reserve,std::function<void(std::size_t)> reserve_ram={});
void mfq_legacy_reserve_moe_workspace(const std::shared_ptr<MoeExpertCache>& cache,std::size_t bytes);
std::size_t mfq_legacy_reclaim_moe_optional_host(const std::shared_ptr<MoeExpertCache>& cache,
    std::size_t allocation_bytes=0);
double mfq_legacy_moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache);
std::uint64_t mfq_legacy_moe_expert_streamed_bytes(const std::shared_ptr<MoeExpertCache>& cache);
std::uint64_t mfq_legacy_moe_expert_graph_captures(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_prepare_moe_predictor_exchange(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_begin_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache,int rows);
void mfq_legacy_commit_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache,int rows);
void mfq_legacy_discard_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache) noexcept;
void mfq_legacy_prepare_moe_decode_residency(const std::shared_ptr<MoeExpertCache>& cache);
bool mfq_legacy_prepare_moe_graph_pool(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_release_moe_decode_buffers(const std::shared_ptr<MoeExpertCache>& cache);
bool mfq_legacy_lend_moe_decode_cache_for_prefill(const std::shared_ptr<MoeExpertCache>& cache,
    std::size_t workspace_bytes);
bool mfq_legacy_suspend_moe_residency(const std::shared_ptr<MoeExpertCache>& cache,bool suspended);
bool mfq_legacy_set_moe_reuse_learning(const std::shared_ptr<MoeExpertCache>& cache,bool enabled);
void mfq_legacy_release_moe_prefill_buffers(MoeExpertCache* cache);
void mfq_legacy_release_moe_prefill_buffers(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay);
void mfq_legacy_prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>& cache,
    bool replay,bool transfer_budget);
void mfq_legacy_prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>& cache,bool replay);
std::size_t mfq_legacy_finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_print_moe_expert_cache_stats(
    const std::shared_ptr<MoeExpertCache>& cache,
    std::ostream& output);
void mfq_legacy_set_moe_expert_cache_profile(
    MoeExpertCache& cache,
    mfq::MoeCacheProfile profile);
