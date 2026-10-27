#pragma once

#include "native/cuda/profile.h"
#include "backend.h"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

struct NvqWorkspace {
    int M = 0;
    int K_pad = 0;
    mfq_tensor_backend::Tensor qx;
    mfq_tensor_backend::Tensor xscale;
    mfq_tensor_backend::Tensor swiglu_scratch;
};

struct NvqWeight {
    mfq_tensor_backend::Tensor indices_packed;
    mfq_tensor_backend::Tensor aux_packed;
    mfq_tensor_backend::Tensor sub_scale_packed;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor codebook;
    // Optional decode-only int8 digits: NVQ1-L 8*v+delta, S 32*v+5*delta.
    // Canonical books retain the dense/CPU and reference decode contract.
    mfq_tensor_backend::Tensor integer_codebook;
    // Complete NVQ1 GS24 records (S 4B, L 5B), aliasing indices_packed.
    mfq_tensor_backend::Tensor decode_records;
    // One byte-preserving group stream replaces indices/aux/states.
    bool dense_groups = false;
    uint32_t bank_map = 0;
    int64_t format = 0;
    int64_t kernel_format = 0;
    int64_t sign_mode = 0;
    int64_t sub_bits = 0;
    int64_t gs = 0;
    int64_t out = 0;
    int64_t ng = 0;
    int64_t neuron_len = 0;
    std::vector<int64_t> shape;
    mutable std::unordered_map<int, NvqWorkspace> workspaces;

    NvqWorkspace & workspace(int M) const {
        int K_pad = (int)(ng * gs);
        auto it = workspaces.find(M);
        if (it != workspaces.end() && it->second.K_pad == K_pad) return it->second;
        NvqWorkspace ws;
        ws.M = M;
        ws.K_pad = K_pad;
        const auto workspace_options = indices_packed.options();
        ws.qx = mfq_tensor_backend::empty(
            {M, K_pad}, workspace_options.dtype(mfq_tensor_backend::kInt8));
        ws.xscale = mfq_tensor_backend::empty(
            {M, ng}, workspace_options.dtype(mfq_tensor_backend::kFloat32));
        return workspaces.emplace(M, std::move(ws)).first->second;
    }
};

inline constexpr int64_t kNvq2JscXlGroupExecKernelFormat = 16;
inline constexpr int64_t kNvq3JscLGroupExecKernelFormat = 17;

mfq_tensor_backend::Tensor nvq_matmul_multi2(
    CudaProfiler& profiler,
    const NvqWeight& first,
    const NvqWeight& second,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_swiglu(
    CudaProfiler& profiler,
    const NvqWeight& gate,
    const NvqWeight& up,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul(
    CudaProfiler& profiler,
    const NvqWeight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor nvq_matmul_input_mul(
    CudaProfiler& profiler,
    const NvqWeight& weight,
    mfq_tensor_backend::Tensor input,
    mfq_tensor_backend::Tensor gate,
    int mode);

struct NepqWeight {
    mfq_tensor_backend::Tensor indices_packed;
    mfq_tensor_backend::Tensor aux_packed;
    mfq_tensor_backend::Tensor state_packed;
    mfq_tensor_backend::Tensor neuron_scale;
    mfq_tensor_backend::Tensor table_pool;
    mfq_tensor_backend::Tensor grouped_table_pool;
    mfq_tensor_backend::Tensor bank_ids;
    mfq_tensor_backend::Tensor rotation_signs;
    int format = 0;
    int state_bits = 0;
    int n_experts = 0;
    int out_per_expert = 0;
    int neuron_len = 0;
    int ng = 0;
    int rotation_block = 0;
    uint64_t rotation_seed = 0;
    bool residual = false;
    int residual_position_bits = 0;
    int residual_block_vectors = 0;
    mfq_tensor_backend::Tensor residual_codebook;
    mfq_tensor_backend::Tensor residual_first;
    mfq_tensor_backend::Tensor residual_second;
};

bool nvq_pair_compatible(const NvqWeight& first, const NvqWeight& second);
mfq_tensor_backend::Tensor nvq_dequant(const NvqWeight& weight);
mfq_tensor_backend::Tensor nvq_embedding(
    const NvqWeight& weight, mfq_tensor_backend::Tensor token_ids);
