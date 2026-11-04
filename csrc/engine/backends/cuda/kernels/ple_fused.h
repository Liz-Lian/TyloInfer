#pragma once
#include "backend.h"
#include <optional>
#include <vector>

namespace mfq_qwen4_exp {
using ::tyloi::DenseTensor;
std::vector<DenseTensor> ple_gate_norm_cuda(const DenseTensor& key,const DenseTensor& query,
    const DenseTensor& value,const DenseTensor& norm,int64_t hidden,int64_t streams,double eps);
std::vector<DenseTensor> ple_conv_add_cuda(const DenseTensor& normalized,const DenseTensor& weight,
    const std::optional<DenseTensor>& previous,int64_t dilation,const DenseTensor& gated,bool half_output);
}
