#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/qsa_adaptive_cache.h"
#include "qsa_ram_cache.h"
#include <functional>
#include <memory>

namespace mfq::cuda {
struct QsaTqSession;
// Experimental adaptive QSA placement. Ordinary cache modes remain unchanged.
bool qsa_tq_cache_enabled();
class QsaTqCache {
public:
    using Tensor=mfq_tensor_backend::Tensor;
    using Pool=std::function<Tensor(const Tensor&,const Tensor&)>;
    QsaTqCache(int64_t maximum,int heads,int kv_heads,int width,int index_heads,int index_width,int budget,
               const mfq_tensor_backend::TensorOptions& options,bool ssd_cache=false);
    ~QsaTqCache();
    int64_t position() const;
    void reset();
    void truncate(int64_t keep);
    void wait();
    // Retire completed GPU append owners without joining background codecs.
    void collect();
    void release_idle();
    void set_budget(std::size_t bytes);
    // Attention scratch follows the live allocator budget independently of FFN.
    int64_t prefill_rows(int64_t requested) const;
    void release_prefill_workspace();
    // Low bits: partial M1, complete M1, or M2..8. Bit 2 requests host
    // staging boundaries; a model applies it to all its QSA layers so the
    // captured boundary count remains stable as individual layers spill.
    int device_graph_kind(int64_t tokens) const;
    bool pending_graph_enabled() const;
    void prepare_decode(bool allow_pending_graph=false);
    void prepare_decode_tokens(int64_t tokens,bool allow_pending_graph=false,int host_mode=0);
    bool graph_task_supported()const;
    void advance_graph(int64_t tokens);
    Tensor forward_pending_graph(const Tensor& query,const Tensor& key,const Tensor& value,const Tensor& iq,
        const Tensor& raw_index,const Tensor& cache_positions,const Tensor& gate,Tensor output,
        const Pool& pool={},const Tensor& absolute_positions={});
    Tensor forward(const Tensor& query,const Tensor& keys,const Tensor& values,const Tensor& iq,
                   const Tensor& raw_index,const Tensor& positions,const Pool& pool,bool history_only=false,
                   const Tensor& gate={},Tensor output={},bool fp16_math=true);
    tyloinfer::qsa::AdaptiveCacheStats stats() const;
    // Observe I/O without finishing pending history or changing cache policy.
    std::uint64_t ssd_read_bytes() const;
    std::optional<QsaRamMemory> ram_memory()const;
    std::shared_ptr<void> hold_host_workspace(std::size_t bytes);
    QsaRamWorkspaceProgress reclaim_shared_host_workspace();
    std::vector<Tensor> resident_tensors()const;
    bool supports_session()const;
    std::shared_ptr<QsaTqSession> capture_session(std::int64_t tokens);
    std::shared_ptr<void> reserve_session_restore(const QsaTqSession& state);
    void restore_session(const QsaTqSession& state,bool borrow_prefix);
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace mfq::cuda
