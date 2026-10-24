#pragma once
#include "moe_expert_cache.h"

std::shared_ptr<MoeExpertCache> mfq_legacy_make_moe_expert_cache(
    std::int64_t bytes,
    const CudaExecutionConfig& config);
bool mfq_legacy_moe_expert_cache_has_sources(
    const std::shared_ptr<MoeExpertCache>& cache);
bool mfq_legacy_moe_expert_cache_finalized(
    const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_finalize_moe_expert_cache(
    const std::shared_ptr<MoeExpertCache>& cache);
double mfq_legacy_moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache);
void mfq_legacy_finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache);
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
