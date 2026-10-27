#pragma once

#include "backend.h"
#include "formats/quant/nint_rows.h"
#include "formats/quant/nint_blob.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct CudaProfiler;

mfq_tensor_backend::Tensor nint_row_embedding_lookup(
    const mfq::NintRows& table, const mfq_tensor_backend::Tensor& ids);
mfq_tensor_backend::Tensor nint_sharded_embedding_lookup(
    const std::vector<std::shared_ptr<mfq::NintRows>>& tables,
    const std::vector<int64_t>& ids,const std::vector<int64_t>& shape,
    const mfq_tensor_backend::Device& device);

struct Workspace {
    int M = 0;
    int K_pad = 0;
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
};

struct NintWeight {
    mfq_tensor_backend::Tensor q_packed;
    mfq_tensor_backend::Tensor row_q_bits;
    mfq_tensor_backend::Tensor row_q_bit_offsets;
    mfq_tensor_backend::Tensor q8_zero_scale;
    mfq_tensor_backend::Tensor sub_scale;
    mfq_tensor_backend::Tensor sub_min;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor neuron_min;
    // Experimental routed row records; populated only by the comparison bench.
    mfq_tensor_backend::Tensor route_metadata;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t gs = 0;
    int64_t bits = 0;
    int64_t neuron_len = 0;
    int64_t q_expert_stride = 0;
    int format_version = 2;
    double aggregate_bpw = 0.0;
    double distribution_entropy = 0.0;
    bool q8_zero = false;
    // Proven once from CPU row metadata; false for unclassified/grouped weights.
    bool aligned_q8 = false;
    // Proven from immutable CPU row metadata; zero means mixed/unknown.
    int uniform_qbits = 0;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, Workspace> workspaces;

    Workspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        Workspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        const auto workspace_options = q_packed.options();
        ws.qx = mfq_tensor_backend::empty(
            {M, K_pad}, workspace_options.dtype(mfq_tensor_backend::kInt8));
        ws.xscale = mfq_tensor_backend::empty(
            {M, ng}, workspace_options.dtype(mfq_tensor_backend::kFloat32));
        auto res = workspaces.emplace(M, std::move(ws));
        return res.first->second;
    }

};

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

mfq_tensor_backend::Tensor dequant_nint8_zero_cpu(
    const mfq::nint8::Data& source);

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
