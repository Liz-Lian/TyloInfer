#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kv_cache_raw.h"
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
tyloi::cuda::NumericType raw_type(at::ScalarType type) {
    if (type == at::kFloat) return tyloi::cuda::NumericType::f32;
    if (type == at::kHalf) return tyloi::cuda::NumericType::f16;
    if (type == at::kBFloat16) return tyloi::cuda::NumericType::bf16;
    throw std::invalid_argument("unsupported KV dtype");
}
}
std::vector<torch::Tensor> kv_cache_write_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor positions)
{
    TORCH_CHECK(k_cache.is_cuda() && k_cache.is_contiguous(), "kv_cache_write: k_cache must be cuda contiguous");
    TORCH_CHECK(v_cache.is_cuda() && v_cache.is_contiguous(), "kv_cache_write: v_cache must be cuda contiguous");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous(), "kv_cache_write: k must be cuda contiguous");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous(), "kv_cache_write: v must be cuda contiguous");
    TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type() && k_cache.scalar_type() == k.scalar_type()
                && k_cache.scalar_type() == v.scalar_type(),
                "kv_cache_write: k/v/cache dtype mismatch");
    TORCH_CHECK(k_cache.scalar_type() == ::at::kHalf ||
                k_cache.scalar_type() == ::at::kBFloat16 ||
                k_cache.scalar_type() == ::at::kFloat,
                "kv_cache_write: dtype must be f16, bf16 or f32");
    TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() && positions.scalar_type() == at::kLong,
                "kv_cache_write: positions must be cuda contiguous int64");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "kv_cache_write: all k/v tensors must be 4D");
    TORCH_CHECK(k.sizes() == v.sizes(), "kv_cache_write: k/v input shapes must match");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "kv_cache_write: k/v cache shapes must match");

    int B = (int)k.size(0);
    int H = (int)k.size(1);
    int T = (int)k.size(2);
    int D = (int)k.size(3);
    int max_seq = (int)k_cache.size(2);
    TORCH_CHECK(k_cache.size(0) == B && k_cache.size(1) == H && k_cache.size(3) == D,
                "kv_cache_write: cache shape must be [B,H,max_seq,D]");
    if (T == 0) return {k_cache, v_cache};

    int pos_dim = positions.dim();
    TORCH_CHECK(pos_dim == 1 || pos_dim == 2, "kv_cache_write: positions must be [T] or [B,T]");
    if (pos_dim == 1) {
        TORCH_CHECK(positions.size(0) == T, "kv_cache_write: positions [T] length mismatch");
    } else {
        TORCH_CHECK(positions.size(0) == B && positions.size(1) == T,
                    "kv_cache_write: positions [B,T] shape mismatch");
    }

    tyloi::cuda::kv_cache_raw::write(k_cache.data_ptr(), v_cache.data_ptr(), k.data_ptr(), v.data_ptr(), positions.data_ptr<int64_t>(), B, H, T, D, max_seq, pos_dim, raw_type(k_cache.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return {k_cache, v_cache};
}

std::vector<torch::Tensor> kv_cache_write_ring_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k,
    torch::Tensor v,
    int64_t position_start)
{
    TORCH_CHECK(position_start >= 0, "kv_cache_write_ring: position_start must be nonnegative");
    TORCH_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                v_cache.is_cuda() && v_cache.is_contiguous(),
                "kv_cache_write_ring: caches must be cuda contiguous");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && v.is_cuda() && v.is_contiguous(),
                "kv_cache_write_ring: inputs must be cuda contiguous");
    TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type() &&
                k_cache.scalar_type() == k.scalar_type() && k_cache.scalar_type() == v.scalar_type(),
                "kv_cache_write_ring: k/v/cache dtype mismatch");
    TORCH_CHECK(k_cache.scalar_type() == ::at::kHalf ||
                k_cache.scalar_type() == ::at::kBFloat16 ||
                k_cache.scalar_type() == ::at::kFloat,
                "kv_cache_write_ring: dtype must be f16, bf16 or f32");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "kv_cache_write_ring: all tensors must be rank 4");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes() && k.sizes() == v.sizes(),
                "kv_cache_write_ring: k/v shapes must match");

    const int B = (int)k.size(0);
    const int H = (int)k.size(1);
    const int T = (int)k.size(2);
    const int D = (int)k.size(3);
    const int capacity = (int)k_cache.size(2);
    TORCH_CHECK(capacity > 0 && k_cache.size(0) == B && k_cache.size(1) == H && k_cache.size(3) == D,
                "kv_cache_write_ring: cache shape mismatch");
    if (T == 0) return {k_cache, v_cache};

    tyloi::cuda::kv_cache_raw::ring(k_cache.data_ptr(), v_cache.data_ptr(), k.data_ptr(), v.data_ptr(), position_start, B, H, T, D, capacity, raw_type(k_cache.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return {k_cache, v_cache};
}

std::vector<torch::Tensor> kv_cache_write_ring_positions_cuda(
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor positions)
{
    TORCH_CHECK(k_cache.is_cuda() && k_cache.is_contiguous() &&
                v_cache.is_cuda() && v_cache.is_contiguous(),
                "kv_cache_write_ring_positions: caches must be cuda contiguous");
    TORCH_CHECK(k.is_cuda() && k.is_contiguous() && v.is_cuda() && v.is_contiguous(),
                "kv_cache_write_ring_positions: inputs must be cuda contiguous");
    TORCH_CHECK(positions.is_cuda() && positions.is_contiguous() &&
                positions.scalar_type() == at::kLong && positions.dim() == 1,
                "kv_cache_write_ring_positions: positions must be cuda contiguous int64[T]");
    TORCH_CHECK(k_cache.scalar_type() == v_cache.scalar_type() &&
                k_cache.scalar_type() == k.scalar_type() && k_cache.scalar_type() == v.scalar_type(),
                "kv_cache_write_ring_positions: k/v/cache dtype mismatch");
    TORCH_CHECK(k_cache.scalar_type() == ::at::kHalf ||
                k_cache.scalar_type() == ::at::kBFloat16 ||
                k_cache.scalar_type() == ::at::kFloat,
                "kv_cache_write_ring_positions: dtype must be f16, bf16 or f32");
    TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "kv_cache_write_ring_positions: all tensors must be rank 4");
    TORCH_CHECK(k_cache.sizes() == v_cache.sizes() && k.sizes() == v.sizes(),
                "kv_cache_write_ring_positions: k/v shapes must match");

    const int B = (int)k.size(0);
    const int H = (int)k.size(1);
    const int T = (int)k.size(2);
    const int D = (int)k.size(3);
    const int capacity = (int)k_cache.size(2);
    TORCH_CHECK(capacity > 0 && k_cache.size(0) == B && k_cache.size(1) == H && k_cache.size(3) == D,
                "kv_cache_write_ring_positions: cache shape mismatch");
    TORCH_CHECK(positions.size(0) == T,
                "kv_cache_write_ring_positions: positions length mismatch");
    if (T == 0) return {k_cache, v_cache};

    tyloi::cuda::kv_cache_raw::ring_positions(k_cache.data_ptr(), v_cache.data_ptr(), k.data_ptr(), v.data_ptr(), positions.data_ptr<int64_t>(), B, H, T, D, capacity, raw_type(k_cache.scalar_type()), at::cuda::getCurrentCUDAStream().stream());
    return {k_cache, v_cache};
}
