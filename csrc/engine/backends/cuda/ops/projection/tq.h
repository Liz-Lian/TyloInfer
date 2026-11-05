#pragma once
#include "backend.h"
#include "engine/weights/encoding/tq.h"
#include "engine/backends/cuda/kernels/tq_ops.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace tyloi::cuda { class BlasResources; }

namespace tyloi::cuda::tq {
TrellisProjectionView view(const ::tyloi::encoding::TqTensor& weight);
DenseTensor project(const ::tyloi::encoding::TqTensor& weight, BlasResources*, DenseTensor input);
DenseTensor groupwise(const ::tyloi::encoding::TqTensor& weight, BlasResources*, DenseTensor input, std::int64_t groups);
DenseTensor embedding(const ::tyloi::encoding::TqTensor& weight, DenseTensor ids);
void validate_group(const std::vector<::tyloi::encoding::TqTensor>& weights);
}
// Device row plan only; compressed weights retain their original storage.
class TqProjectionGroup {
    std::vector<::tyloi::encoding::TqTensor> weights_;
    std::vector<TrellisProjectionJob> host_plan_;
    ::tyloi::DenseTensor plan_;
    std::int64_t input_=0,output_=0;
    std::size_t joined_prefix_=1;
    std::int64_t joined_rows_=0;
    int blocks_=0;
public:
    TqProjectionGroup()=default;
    explicit TqProjectionGroup(std::vector<::tyloi::encoding::TqTensor>);
    bool matches(const std::vector<::tyloi::encoding::TqTensor>& weights) const;
    std::vector<::tyloi::DenseTensor> buffers() const;
    void join_prefix(std::size_t count);
    bool defined()const noexcept{return !weights_.empty();}
    std::vector<::tyloi::DenseTensor> forward(tyloi::cuda::BlasResources*,::tyloi::DenseTensor)const;
    ::tyloi::DenseTensor swiglu(tyloi::cuda::BlasResources*,::tyloi::DenseTensor)const;
};
