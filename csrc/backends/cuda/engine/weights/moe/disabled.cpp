#include "backends/cuda/engine/weights/moe/expert_cache.h"

MoePrefillWorkspaceBudget moe_prefill_workspace_budget(const std::shared_ptr<MoeExpertCache>& cache, bool allow_borrow) { return {}; }

void set_moe_kv_expert_loan(const std::shared_ptr<MoeExpertCache>& cache, std::shared_ptr<mfq::cuda::KvExpertLoan> loan) {}

std::size_t moe_prefill_transfer_bytes(const std::shared_ptr<MoeExpertCache>& cache, bool predictor) { return {}; }

std::shared_ptr<MoeExpertCache> make_moe_expert_cache(std::int64_t bytes, const MoeCacheOptions& options) { return {}; }

bool moe_expert_cache_has_sources(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

bool moe_expert_cache_finalized(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

bool moe_expert_cache_layer_prefill_ready(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

bool moe_expert_cache_ssd_experts(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

bool moe_expert_cache_ssd_router_prefetch(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

std::vector<std::pair<std::string, double>> moe_expert_memory_metrics(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

void finalize_moe_expert_cache(const std::shared_ptr<MoeExpertCache>& cache, std::size_t kv_growth_reserve, std::size_t kv_budget_charge) {}

std::size_t moe_expert_cache_minimum_bytes(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

std::size_t moe_expert_cache_budget_bytes(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

void resize_moe_expert_cache(const std::shared_ptr<MoeExpertCache>& cache, std::size_t bytes) {}

void set_moe_workspace_reservation(const std::shared_ptr<MoeExpertCache>& cache, std::function<void(std::size_t)> reserve, std::function<void(std::size_t)> reserve_ram) {}

void reserve_moe_workspace(const std::shared_ptr<MoeExpertCache>& cache, std::size_t bytes) {}

std::size_t reclaim_moe_optional_host(const std::shared_ptr<MoeExpertCache>& cache, std::size_t allocation_bytes) { return {}; }

double moe_expert_cache_ram_pcie_fraction(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

std::uint64_t moe_expert_streamed_bytes(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

std::uint64_t moe_expert_graph_captures(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

void drain_moe_expert_cache(const std::shared_ptr<MoeExpertCache>&) {}

void finish_moe_expert_exchanges(const std::shared_ptr<MoeExpertCache>& cache) {}

void prepare_moe_predictor_exchange(const std::shared_ptr<MoeExpertCache>& cache) {}

void begin_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache, int rows) {}

void commit_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache, int rows) {}

void discard_moe_speculative_routes(const std::shared_ptr<MoeExpertCache>& cache)noexcept {}

void prepare_moe_decode_residency(const std::shared_ptr<MoeExpertCache>& cache) {}

bool prepare_moe_graph_pool(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

void release_moe_decode_buffers(const std::shared_ptr<MoeExpertCache>& cache) {}

bool lend_moe_decode_cache_for_prefill(const std::shared_ptr<MoeExpertCache>& cache, std::size_t workspace_bytes) { return {}; }

bool suspend_moe_residency(const std::shared_ptr<MoeExpertCache>& cache, bool suspended) { return {}; }

bool set_moe_reuse_learning(const std::shared_ptr<MoeExpertCache>& cache, bool enabled) { return {}; }

void release_moe_prefill_buffers(MoeExpertCache* cache) {}

void release_moe_prefill_buffers(const std::shared_ptr<MoeExpertCache>& cache) {}

void prepare_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay) {}

void prepare_moe_cpu_budget_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay, bool transfer_budget) {}

void prepare_moe_cpu_calibration_comparison(const std::shared_ptr<MoeExpertCache>& cache, bool replay) {}

std::size_t finish_moe_pipeline_comparison(const std::shared_ptr<MoeExpertCache>& cache) { return {}; }

void print_moe_expert_cache_stats(const std::shared_ptr<MoeExpertCache>& cache, std::ostream& output) {}

void set_moe_expert_cache_profile(MoeExpertCache& cache, mfq::MoeCacheProfile profile) {}

MoePrefillWorkspaceLoan::~MoePrefillWorkspaceLoan() = default;
std::size_t MoePrefillWorkspaceLoan::begin(std::size_t) { return 0; }
void MoePrefillWorkspaceLoan::suspend() {}
void MoePrefillWorkspaceLoan::finish() {}
MoePrefillLayerPriority::MoePrefillLayerPriority(std::shared_ptr<MoeExpertCache> cache, int, int)
    : cache_(std::move(cache)) {}
MoePrefillLayerPriority::~MoePrefillLayerPriority() = default;
void MoePrefillLayerPriority::finish() {}
