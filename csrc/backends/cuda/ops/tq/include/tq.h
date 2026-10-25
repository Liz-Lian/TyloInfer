#pragma once
#include "backend.h"
#include "tyloinfer/kernels/cuda/tq_ops.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mfq::cuda { class BlasResources; }

struct TqWeight {
    mfq_tensor_backend::Tensor image;
    TrellisProjectionView view{};
    std::int64_t out=0,neuron_len=0;
    mfq_tensor_backend::Tensor forward(mfq::cuda::BlasResources*,mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor forward_groupwise(mfq::cuda::BlasResources*,mfq_tensor_backend::Tensor,std::int64_t groups)const;
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
    std::vector<mfq_tensor_backend::Tensor> forward(mfq::cuda::BlasResources*,mfq_tensor_backend::Tensor)const;
    mfq_tensor_backend::Tensor swiglu(mfq::cuda::BlasResources*,mfq_tensor_backend::Tensor)const;
};
