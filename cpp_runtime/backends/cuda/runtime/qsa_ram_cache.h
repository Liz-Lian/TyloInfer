#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/qsa_ram_store.h"
#include <cuda_runtime_api.h>
#include <functional>
#include <memory>

namespace mfq::cuda {
using QsaRamReclaimer=std::function<std::size_t(std::size_t)>;
// Registration is owned by the engine. Queries retain only the callback for
// their duration; it must in turn use weak ownership of disposable RAM data.
std::shared_ptr<void> register_qsa_ram_reclaimer(int device,cudaStream_t stream,QsaRamReclaimer reclaim);
std::size_t reclaim_qsa_ram(int device,cudaStream_t stream,std::size_t bytes);

class QsaRamCache {
public:
    using Tensor=mfq_tensor_backend::Tensor;
    using Encode=std::function<std::vector<tyloinfer::qsa::MicroblockImage>(
        const std::vector<tyloinfer::qsa::MicroblockImage>&,const std::vector<tyloinfer::qsa::Precision>&)>;
    QsaRamCache(int capacity,int rows,int selected,int bits,const mfq_tensor_backend::TensorOptions& options,const Tensor& live,
        bool ssd,const std::string& directory,Encode encode);
    ~QsaRamCache();
    void prepare();
    void append(std::vector<tyloinfer::qsa::MicroblockImage> images,std::uint64_t epoch,bool fresh=false);
    void drain();
    Tensor stage(const Tensor& selected,int queries,int count,const Tensor& live,
        const Tensor& position,std::int64_t host_position,int appended);
    void observe(const std::vector<double>& hits,std::uint64_t epoch);
    void wait();
    void truncate(std::size_t blocks);
    bool graph_safe()const;
    bool encoding_urgent()const;
    std::shared_ptr<tyloinfer::qsa::RamKvSnapshot> snapshot(std::size_t blocks);
    void restore(const tyloinfer::qsa::RamKvSnapshot& state,bool borrow_prefix);
    tyloinfer::qsa::MicroblockImage reconstruction_image(std::size_t id);
    tyloinfer::qsa::AdaptiveCacheStats stats()const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
} // namespace mfq::cuda
