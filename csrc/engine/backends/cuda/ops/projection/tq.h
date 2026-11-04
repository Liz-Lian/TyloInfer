#pragma once
#include "backend.h"
#include "engine/backends/cuda/kernels/tq_ops.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tyloi::cuda { class BlasResources; }

namespace tyloi::cuda::tq {
TrellisProjectionView view(const Tensor& weight);
Tensor project(const Tensor& weight, BlasResources*, Tensor input);
Tensor groupwise(const Tensor& weight, BlasResources*, Tensor input, std::int64_t groups);
Tensor embedding(const Tensor& weight, Tensor ids);
}
// Device row plan only; compressed weights retain their original storage.
class TqProjectionGroup {
    std::vector<::tyloi::Tensor> weights_;
    std::vector<TrellisProjectionJob> host_plan_;
    ::tyloi::Tensor plan_;
    std::int64_t input_=0,output_=0;
    std::size_t joined_prefix_=1;
    std::int64_t joined_rows_=0;
    int blocks_=0;
public:
    TqProjectionGroup()=default;
    explicit TqProjectionGroup(std::vector<::tyloi::Tensor>);
    void join_prefix(std::size_t count);
    bool defined()const noexcept{return !weights_.empty();}
    std::vector<::tyloi::Tensor> forward(tyloi::cuda::BlasResources*,::tyloi::Tensor)const;
    ::tyloi::Tensor swiglu(tyloi::cuda::BlasResources*,::tyloi::Tensor)const;
};
