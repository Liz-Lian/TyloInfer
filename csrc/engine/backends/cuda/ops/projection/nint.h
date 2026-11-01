#pragma once

#include "backend.h"
#include "engine/weights/encoding/quant/nint_rows.h"
#include "engine/weights/nint.h"

#include <cstdint>
#include <string>
#include <vector>

struct CudaProfiler;

mfq_tensor_backend::Tensor nint_row_embedding_lookup(
    const mfq::NintRows& table, const mfq_tensor_backend::Tensor& ids);
mfq_tensor_backend::Tensor nint_sharded_embedding_lookup(
    const std::vector<std::shared_ptr<mfq::NintRows>>& tables,
    const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,
    const mfq_tensor_backend::Device& device);

mfq_tensor_backend::Tensor nint_matmul(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nint_matmul_input_mul(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);
mfq_tensor_backend::Tensor nint_matmul_input_mul_f32(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);
mfq_tensor_backend::Tensor nint_matmul_swiglu(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nint_matmul_geglu(
    CudaProfiler& profiler,
    const NintWeight& weight,
    mfq_tensor_backend::Tensor input);


struct CpuQuantizedActivation {
    int64_t rows = 0;
    int64_t groups = 0;
    int64_t group_size = 0;
    std::vector<int8_t> values;
    std::vector<float> scales;
    std::vector<int32_t> sums;
};

mfq_tensor_backend::Tensor pad_last(
    mfq_tensor_backend::Tensor input, int64_t target);
uint32_t cpu_load_packed_bits(
    const uint8_t* data, int64_t nbytes, int64_t bit, int bits);
int32_t cpu_dot_s8_s8_64(
    const int8_t* left, const int8_t* right, int32_t right_sum);
int32_t cpu_dot_u8_s8_64(
    const uint8_t* left, const int8_t* right);
void cpu_unpack_nint_group(
    const uint8_t* packed, int bits, int valid, uint8_t* values);
void cpu_unpack_nint_group_at_bit_offset(
    const uint8_t* packed, uint64_t bit_offset, int bits, int valid, uint8_t* values);
CpuQuantizedActivation cpu_quantize_activation(
    mfq_tensor_backend::Tensor input, int64_t width, int64_t group_size, bool with_sums);
float cpu_half_from_bytes(const int8_t* bytes, int64_t offset);
