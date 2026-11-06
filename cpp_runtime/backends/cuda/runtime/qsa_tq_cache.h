#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/qsa_adaptive_cache.h"
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
    void set_budget(std::size_t bytes);
    // Attention scratch follows the live allocator budget independently of FFN.
    int64_t prefill_rows(int64_t requested) const;
    void release_prefill_workspace();
    int device_graph_kind(int64_t tokens) const;
    bool pending_graph_enabled() const;
    void prepare_decode(bool allow_pending_graph=false);
    void advance_graph(int64_t tokens);
    Tensor forward_pending_graph(const Tensor& query,const Tensor& key,const Tensor& value,const Tensor& iq,
        const Tensor& raw_index,const Tensor& cache_positions,const Tensor& gate,Tensor output,
        const Pool& pool={},const Tensor& absolute_positions={});
    Tensor forward(const Tensor& query,const Tensor& keys,const Tensor& values,const Tensor& iq,
                   const Tensor& raw_index,const Tensor& positions,const Pool& pool,bool history_only=false,
                   const Tensor& gate={},Tensor output={},bool fp16_math=true);
    tyloinfer::qsa::AdaptiveCacheStats stats() const;
    bool supports_session()const;
    std::shared_ptr<QsaTqSession> capture_session(std::int64_t tokens);
    void restore_session(const QsaTqSession& state,bool borrow_prefix);
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace mfq::cuda
