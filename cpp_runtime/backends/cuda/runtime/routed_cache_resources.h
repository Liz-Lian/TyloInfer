#pragma once
#include "tq.h"
#include "gguf_weight.h"
#include "storage/moe_expert_cache.h"

namespace mfq::cuda {
class RoutedPrefillWorkspaceLoan {
public:
    explicit RoutedPrefillWorkspaceLoan(CudaExecutionContext& execution)
        :gguf_(execution.gguf_expert_cache),legacy_(execution.moe_expert_cache),
         uses_gguf_(bool(execution.gguf_expert_cache)) {}
    std::size_t begin(std::size_t bytes) {return uses_gguf_?gguf_.begin(bytes):legacy_.begin(bytes);}
    void suspend() {if(uses_gguf_)gguf_.suspend();else legacy_.suspend();}
    void finish() {if(uses_gguf_)gguf_.finish();else legacy_.finish();}
private:
    GgufPrefillWorkspaceLoan gguf_;
    MoePrefillWorkspaceLoan legacy_;
    bool uses_gguf_;
};
inline MoePrefillWorkspaceBudget routed_prefill_workspace_budget(
        const CudaExecutionContext& execution,bool allow_borrow) {
    if(execution.gguf_expert_cache)
        return {allow_borrow?gguf_prefill_lendable_bytes(execution.gguf_expert_cache):0,0};
    return moe_prefill_workspace_budget(execution.moe_expert_cache,allow_borrow);
}
inline std::size_t routed_prefill_transfer_bytes(const CudaExecutionContext& execution) {
    return execution.gguf_expert_cache?0:moe_prefill_transfer_bytes(execution.moe_expert_cache);
}
inline std::uint64_t routed_expert_streamed_bytes(const CudaExecutionContext& execution) {
    if(execution.gguf_expert_cache)return gguf_expert_streamed_bytes(execution.gguf_expert_cache);
    if(execution.tq_expert_cache)return tq_expert_streamed_bytes(execution.tq_expert_cache);
    return moe_expert_streamed_bytes(execution.moe_expert_cache);
}
// Resource ownership follows the loaded bank implementation. The service uses
// these operations without interpreting a tensor's row precision or encoding.
inline std::vector<std::pair<std::string,double>> routed_expert_memory_metrics(
        const CudaExecutionContext& execution) {
    if(execution.gguf_expert_cache)return gguf_expert_memory_metrics(execution.gguf_expert_cache);
    return execution.tq_expert_cache?tq_expert_memory_metrics(execution.tq_expert_cache):
        moe_expert_memory_metrics(execution.moe_expert_cache);
}
inline std::size_t routed_expert_cache_budget_bytes(const CudaExecutionContext& execution) {
    if(execution.gguf_expert_cache)return gguf_expert_cache_budget_bytes(execution.gguf_expert_cache);
    return execution.tq_expert_cache?tq_expert_cache_budget_bytes(execution.tq_expert_cache):
        moe_expert_cache_budget_bytes(execution.moe_expert_cache);
}
inline void resize_routed_expert_cache(CudaExecutionContext& execution,std::size_t bytes) {
    if(execution.gguf_expert_cache)resize_gguf_expert_cache(execution.gguf_expert_cache,bytes);
    else if(execution.tq_expert_cache)resize_tq_expert_cache(execution.tq_expert_cache,bytes);
    else resize_moe_expert_cache(execution.moe_expert_cache,bytes);
}
inline void reserve_routed_workspace(CudaExecutionContext& execution,std::size_t bytes) {
    if(execution.gguf_expert_cache)reserve_gguf_expert_workspace(execution.gguf_expert_cache,bytes);
    else reserve_moe_workspace(execution.moe_expert_cache,bytes);
}
inline std::size_t reclaim_routed_optional_host(CudaExecutionContext& execution,std::size_t allocation_bytes) {
    if(execution.gguf_expert_cache)return reserve_gguf_expert_host(execution.gguf_expert_cache,allocation_bytes);
    return execution.tq_expert_cache?reserve_tq_expert_host(execution.tq_expert_cache,allocation_bytes):
        reclaim_moe_optional_host(execution.moe_expert_cache,allocation_bytes);
}
}
