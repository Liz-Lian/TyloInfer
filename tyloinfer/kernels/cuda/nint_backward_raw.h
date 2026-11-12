#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace mfq_jit_nint {
/** Launch the original metadata-driven half input-gradient kernel. */
void backward(const std::uint8_t* bitstream,const std::uint8_t* row_bits,const std::int64_t* row_offsets,
    const std::uint8_t* subgroup_scale,const std::uint8_t* subgroup_minimum,
    const float* neuron_scale,const float* neuron_minimum,const void* gradient,void* result,
    int rows,int outputs,int groups,int width,int group_size,cudaStream_t stream);
}
