#pragma once
#include "backend.h"
#include "tyloinfer/kernels/cuda/tq_ops.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <memory>

class TqExec;
class TqCache;
struct CudaProfiler;
struct CudaExecutionContext;

struct TqWeight {
    mfq_tensor_backend::Tensor image;
    TrellisProjectionView view{};
    std::int64_t out=0,neuron_len=0;
    mfq_tensor_backend::Tensor forward(TqExec&,mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor forward_groupwise(TqExec&,mfq_tensor_backend::Tensor,std::int64_t groups)const;
    mfq_tensor_backend::Tensor embedding(mfq_tensor_backend::Tensor)const;
};
// Device row plan only; compressed weights retain their original storage.
class TqProjectionGroup {
    std::vector<TqWeight> weights_;
    std::vector<TrellisProjectionJob> host_plan_;
    mfq_tensor_backend::Tensor plan_;
    std::int64_t input_=0,output_=0;
    std::size_t joined_prefix_=1;
    std::int64_t joined_rows_=0;
    int blocks_=0;
public:
    TqProjectionGroup()=default;
    explicit TqProjectionGroup(std::vector<TqWeight>);
    void join_prefix(std::size_t count);
    bool defined()const noexcept{return !weights_.empty();}
    std::vector<mfq_tensor_backend::Tensor> forward(TqExec&,mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor swiglu(TqExec&,mfq_tensor_backend::Tensor)const;
};

struct TqRoutedWeight {
    std::shared_ptr<TqCache> cache;
    std::uint64_t source_id=0;
    std::int64_t experts=0,out=0,input=0;
    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor,mfq_tensor_backend::Tensor)const;
};
bool tq_graph_supports_tokens(std::int64_t tokens)noexcept;

// Three ordinary [experts, rows, columns] TQ tensors. Expert ID selects
// a row range; every row retains its own bit width, scale and encoded data.
struct TqMoeBanks {
    std::shared_ptr<TqRoutedWeight> gate,up,down;
    mfq_tensor_backend::Tensor forward(TqExec&,mfq_tensor_backend::Tensor input,
        mfq_tensor_backend::Tensor ids,mfq_tensor_backend::Tensor weights,float swiglu_limit=0,
        CudaProfiler* profiler=nullptr,std::int64_t expert_begin=0,std::int64_t expert_end=-1)const;
    mfq_tensor_backend::Tensor forward_parallel(CudaExecutionContext&,
        mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor ids,
        mfq_tensor_backend::Tensor weights,float swiglu_limit=0)const;
};
