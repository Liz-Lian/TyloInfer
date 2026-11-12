#include "mfq_sparse_attn_mma_f16.cuh"

#include <algorithm>
#include <climits>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>

namespace tyloi::cuda::sparse_mma_raw {
namespace {
void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

template<int D, bool hot>
Plan make_plan(int B, int M, int heads, int kv_heads, int max_seq, int selected) {
    constexpr int ncols = 16;
    constexpr int ncols1 = 1;
    constexpr int ncols2 = 16;
    constexpr bool view = D == 512;
    auto kernel = mfq_sparse_attn_mma::sparse_attention_kernel<D, D, ncols1, ncols2, view, hot>;
    int device = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device), "cudaGetDevice");
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    const int cc = properties.major * 100 + properties.minor * 10;
    if (!turing_mma_available(cc)) throw std::invalid_argument("attention requires Turing-class tensor cores or newer");
    const auto config = ampere_mma_available(cc)
        ? ggml_cuda_fattn_mma_get_config_ampere(D, D, ncols)
        : ggml_cuda_fattn_mma_get_config_turing(D, D, ncols);
    const int threads = config.nthreads;
    if (threads <= 0 || config.nbatch_fa <= 0 || config.nbatch_K2 <= 0 ||
        config.nbatch_V2 <= 0 || config.nbatch_combine <= 0)
        throw std::invalid_argument("unsupported MMA geometry");
    const int warps = threads / 32;
    const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));
    const size_t shared_kv = size_t(config.nbatch_fa) * std::max(config.nbatch_K2 + 4, config.nbatch_V2 + 4) * sizeof(half2);
    const size_t shared_q = size_t(ncols) * (D / 2 + 4) * sizeof(half2);
    const size_t shared_mask = size_t(ncols1) * (config.nbatch_fa / 2 + 4) * sizeof(half2);
    const size_t shared_combine = size_t(warps) * cols_per_warp * (config.nbatch_combine + 4) * sizeof(half2);
    const size_t shared = std::max(shared_combine, config.Q_in_reg ? std::max(shared_q, shared_kv + shared_mask) : shared_q + shared_kv + shared_mask);
    if (device < 0 || device >= 32) throw std::invalid_argument("unsupported CUDA device index");
    static bool shared_set[32] = {};
    if (!shared_set[device]) {
        check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)), "attention shared-memory attribute");
        shared_set[device] = true;
    }
    const int ratio = heads / kv_heads;
    const int iter_j = M;
    const int iter_z = int((int64_t(ratio) + ncols2 - 1) / ncols2);
    int64_t tiles64 = B;
    for (const int factor : {kv_heads, iter_z, iter_j}) {
        if (tiles64 > int64_t(INT_MAX) / factor)
            throw std::invalid_argument("attention CUDA grid is too large");
        tiles64 *= factor;
    }
    const int64_t ntiles_kv = (int64_t(selected) + config.nbatch_fa - 1) / config.nbatch_fa;
    const int tiles = int(tiles64);
    int max_blocks = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, threads, shared), "attention occupancy");
    if (max_blocks <= 0) throw std::runtime_error("attention occupancy is zero");
    const int64_t resident = int64_t(max_blocks) * properties.multiProcessorCount;
    const int64_t total_tiles = tiles64 * ntiles_kv;
    const int64_t raw = std::min(resident, total_tiles);
    const int64_t blocks64 = std::max(tiles64, (raw / tiles64) * tiles64);
    if (blocks64 > INT_MAX) throw std::invalid_argument("attention CUDA grid is too large");
    const int blocks = int(blocks64);
    const int parts = blocks / tiles;
    size_t required = 1;
    if (parts > 1) {
        required = size_t(blocks);
        for (const size_t factor : {size_t(ncols), size_t(2 + D / 2), size_t(2)}) {
            if (required > SIZE_MAX / factor) throw std::invalid_argument("attention workspace is too large");
            required *= factor;
        }
        if (required > size_t(std::numeric_limits<int64_t>::max()) || required > SIZE_MAX / sizeof(float))
            throw std::invalid_argument("attention workspace is too large");
    }
    return {B, M, heads, kv_heads, max_seq, selected, D, blocks, parts, tiles,
        iter_j, iter_z, threads, device, shared, required * sizeof(float), hot};
}

template<int D, bool hot>
void run(const Plan& p, const void* q, const void* k, const void* v,
    const int32_t* indices, const void* mask, const float* sinks, const void* positions,
    const int64_t strides[4], bool half_query, bool query_contiguous, bool wide_positions, int causal_columns,
    Hot cache, float scale, float* out, void* workspace, bool gated, cudaStream_t stream) {
    constexpr bool view = D == 512;
    constexpr int ncols1 = 1, ncols2 = 16;
    const mfq_sparse_attn_mma::RuntimeInput input{positions, strides[0], strides[1], strides[2], strides[3], causal_columns,
        half_query, positions && wide_positions,
        positions || half_query || !query_contiguous, cache};
    mfq_sparse_attn_mma::sparse_attention_kernel<D, D, ncols1, ncols2, view, hot>
        <<<p.blocks, dim3(32, p.threads / 32, 1), p.shared_bytes, stream>>>(
            q, static_cast<const half*>(k), static_cast<const half*>(v), indices,
            static_cast<const half*>(mask), sinks, out, static_cast<float2*>(workspace), scale,
            p.batch, p.tokens, p.heads, p.kv_heads, p.max_seq, p.selected,
            mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(p.tokens)), input);
    check(cudaGetLastError(), "attention launch");
    if (!gated && p.parts > 1) {
        const auto fd0 = mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(int64_t(p.iter_j) * p.iter_z_gqa * p.kv_heads));
        const auto fd1 = mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(int64_t(p.iter_j) * p.iter_z_gqa));
        const auto fd2 = mfq_sparse_attn_mma::sparse_fastdiv_values(uint32_t(p.iter_j));
        flash_attn_stream_k_fixup_uniform<D, ncols1, ncols2>
            <<<dim3(p.tiles, ncols1, ncols2), D, 0, stream>>>(out, static_cast<const float2*>(workspace),
                p.tokens, p.heads, p.kv_heads, p.blocks, p.heads / p.kv_heads, p.parts, fd0, fd1, fd2);
        check(cudaGetLastError(), "attention fixup");
    }
}
}

Plan plan(int B, int M, int heads, int kv_heads, int max_seq, int selected, int width, bool hot) {
    if (B <= 0 || M <= 0 || heads <= 0 || kv_heads <= 0 || heads % kv_heads || max_seq <= 0 || selected <= 0)
        throw std::invalid_argument("invalid attention geometry");
    if (width == 256) return hot ? make_plan<256, true>(B, M, heads, kv_heads, max_seq, selected) : make_plan<256, false>(B, M, heads, kv_heads, max_seq, selected);
    if (width == 512 && !hot) return make_plan<512, false>(B, M, heads, kv_heads, max_seq, selected);
    throw std::invalid_argument("unsupported attention MMA variant");
}

void launch(const Plan& p, const void* q, const void* k, const void* v, const int32_t* indices,
    const void* mask, const float* sinks, const void* positions, const int64_t strides[4], bool half_query, bool query_contiguous,
    bool wide_positions, int causal_columns, Hot hot, float scale, float* out, void* workspace,
    std::size_t bytes, bool gated, cudaStream_t stream) {
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    if (device != p.device || !q || !k || !v || !strides || !out || !workspace || bytes < p.workspace_bytes ||
        p.hot != (hot.mapping != nullptr) || (positions ? (causal_columns <= 0 || indices || mask) : !indices))
        throw std::invalid_argument("invalid attention MMA launch inputs");
    if (p.width == 256 && p.hot) run<256, true>(p, q, k, v, indices, mask, sinks, positions, strides, half_query, query_contiguous, wide_positions, causal_columns, hot, scale, out, workspace, gated, stream);
    else if (p.width == 256) run<256, false>(p, q, k, v, indices, mask, sinks, positions, strides, half_query, query_contiguous, wide_positions, causal_columns, hot, scale, out, workspace, gated, stream);
    else if (p.width == 512 && !p.hot) run<512, false>(p, q, k, v, indices, mask, sinks, positions, strides, half_query, query_contiguous, wide_positions, causal_columns, hot, scale, out, workspace, gated, stream);
    else throw std::invalid_argument("unsupported attention MMA variant");
}
}
