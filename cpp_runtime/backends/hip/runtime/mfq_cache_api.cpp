#include "moe_expert_cache.h"
#include "tyloinfer/format_features.h"
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "mfq_cache_provider.h"
#endif

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

void finalize_moe_expert_cache(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_finalize_moe_expert_cache(cache);
#endif
}

double moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    return mfq_legacy_moe_expert_cache_ram_pcie_fraction(cache);
#else
    return {};
#endif
}

void finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache) {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    mfq_legacy_finish_moe_expert_exchanges(cache);
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
