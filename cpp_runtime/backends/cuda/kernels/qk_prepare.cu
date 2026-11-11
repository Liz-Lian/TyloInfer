#include "kernel_dispatch_trace.h"
#include "mfq_cuda_attention_ops.h"
#include "reduce.cuh"
#include <climits>
#include <cmath>

namespace {
struct HeadLayout {
    int64_t batch, head, token;
};

struct FlatCacheWrite {
    __half *key;
    __half *value;
    int64_t capacity;

    __device__ void resolve(int batch, int head, int64_t slot, int heads, int width,
                            __half *&key_out, __half *&value_out) const {
        if (slot < 0 || slot >= capacity)
            return;
        const auto offset = ((int64_t(batch) * heads + head) * capacity + slot) * width;
        key_out = key + offset;
        value_out = value + offset;
    }
};

struct PagedCacheWrite {
    const int64_t *key_chunks;
    const int64_t *value_chunks;
    const int32_t *pages;
    int chunks;
    int logical_pages;
    int page_size;
    int pages_per_chunk;

    __device__ void resolve(int batch, int head, int64_t slot, int heads, int width,
                            __half *&key_out, __half *&value_out) const {
        if (slot < 0)
            return;
        const int64_t logical = slot / page_size;
        if (logical >= logical_pages)
            return;
        const int physical = pages[int64_t(batch) * logical_pages + logical];
        if (physical < 0)
            return;
        const int chunk = physical / pages_per_chunk;
        if (chunk >= chunks || !key_chunks[chunk] || !value_chunks[chunk])
            return;
        const int local_page = physical % pages_per_chunk;
        const int64_t offset =
            ((int64_t(local_page) * heads + head) * page_size + slot % page_size) * width;
        key_out = reinterpret_cast<__half *>(uintptr_t(key_chunks[chunk])) + offset;
        value_out = reinterpret_cast<__half *>(uintptr_t(value_chunks[chunk])) + offset;
    }
};

// Match FP16/FP32->FP32 RMSNorm and the FP32 table RoPE arithmetic.
// Q may be a view of interleaved query/gate projection rows.
template <class Projection, class CacheWrite>
__global__ void qk_norm_rope_cache_kernel(
    const Projection *query, const Projection *key, const Projection *value,
    const float *query_weight, const float *key_weight, const int64_t *positions,
    const int64_t *write_positions, const float *cosine, const float *sine, float *query_out,
    CacheWrite cache, HeadLayout q_layout, HeadLayout k_layout, HeadLayout v_layout,
    int query_heads, int key_heads, int tokens, int width, int pairs, int table_length,
    int64_t position_batch_stride, int64_t write_batch_stride, float eps, float weight_offset) {
    const int combined = int(blockIdx.x), heads = query_heads + key_heads;
    const int batch = combined / (heads * tokens), token = (combined / heads) % tokens,
              index = combined % heads;
    const bool is_key = index >= query_heads;
    const int head = is_key ? index - query_heads : index;
    const auto layout = is_key ? k_layout : q_layout;
    const auto *source = (is_key ? key : query) + int64_t(batch) * layout.batch +
                         int64_t(head) * layout.head + int64_t(token) * layout.token;
    const auto *weight = is_key ? key_weight : query_weight;
    float sum = 0.f;
    for (int column = threadIdx.x; column < width; column += 256) {
        const float x = float(source[column]);
        sum += x * x;
    }
    sum = block_sum<8>(sum);
    const float inverse = rsqrtf(sum / float(width) + eps);
    const auto normalized = [&](int column) {
        return __fmul_rn(__fmul_rn(float(source[column]), inverse), weight[column] + weight_offset);
    };
    const int64_t position =
        min(max(int64_t(0), positions[int64_t(batch) * position_batch_stride + token]),
            int64_t(table_length - 1));
    const int64_t slot = write_positions[int64_t(batch) * write_batch_stride + token];
    __half *key_out = nullptr;
    __half *value_out = nullptr;
    if (is_key)
        cache.resolve(batch, head, slot, key_heads, width, key_out, value_out);
    const bool valid = key_out != nullptr;
    const auto store = [&](int column, float x) {
        if (is_key) {
            if (valid)
                key_out[column] = __float2half_rn(x);
        } else
            query_out[((int64_t(batch) * query_heads + head) * tokens + token) * width + column] =
                x;
    };
    for (int column = threadIdx.x; column < width; column += 256) {
        if (column < pairs) {
            const float x = normalized(column), y = normalized(column + pairs);
            const float c = cosine[position * pairs + column], s = sine[position * pairs + column];
            store(column, __fmaf_rn(x, c, -__fmul_rn(y, s)));
            store(column + pairs, __fmaf_rn(x, s, __fmul_rn(y, c)));
        } else if (column >= 2 * pairs)
            store(column, normalized(column));
        if (is_key && valid) {
            const auto at = int64_t(batch) * v_layout.batch + int64_t(head) * v_layout.head +
                            int64_t(token) * v_layout.token + column;
            value_out[column] = __float2half_rn(float(value[at]));
        }
    }
}

template <class CacheWrite>
mfq_tensor_backend::Tensor
prepare_qk(mfq_tensor_backend::Tensor query, mfq_tensor_backend::Tensor key,
           mfq_tensor_backend::Tensor value, mfq_tensor_backend::Tensor query_weight,
           mfq_tensor_backend::Tensor key_weight, mfq_tensor_backend::Tensor positions,
           mfq_tensor_backend::Tensor write_positions, mfq_tensor_backend::Tensor cosine,
           mfq_tensor_backend::Tensor sine, CacheWrite cache, const char *dispatch_name,
           int64_t rotary_dim, double eps, double weight_offset) {
    namespace tb = mfq_tensor_backend;
    const auto device = query.device();
    const auto input = [&](const tb::Tensor &x) {
        return x.is_cuda() && x.device() == device &&
               (x.scalar_type() == tb::kFloat16 || x.scalar_type() == tb::kFloat32) &&
               x.dim() == 4 && x.size(0) > 0 && x.size(1) > 0 && x.size(2) > 0 && x.stride(3) == 1;
    };
    TYLOINFER_RUNTIME_CHECK(input(query) && input(key) && input(value) &&
                                query.size(0) == key.size(0) && query.size(2) == key.size(2) &&
                                query.size(3) == key.size(3) && key.sizes() == value.sizes() &&
                                query.scalar_type() == key.scalar_type() &&
                                key.scalar_type() == value.scalar_type(),
                            "QK preparation requires matching CUDA FP16/FP32 head views");
    const int64_t width = query.size(3);
    TYLOINFER_RUNTIME_CHECK(width > 0 && width <= INT_MAX && query.size(1) <= INT_MAX &&
                                key.size(1) <= INT_MAX && query.size(1) + key.size(1) <= INT_MAX &&
                                query.size(2) <= INT_MAX &&
                                query.numel() / width <= INT_MAX - key.numel() / width &&
                                rotary_dim > 0 && rotary_dim <= width && rotary_dim % 2 == 0 &&
                                std::isfinite(eps) && eps >= 0 && std::isfinite(weight_offset),
                            "QK preparation geometry/epsilon differs");
    const auto f32 = [&](const tb::Tensor &x) {
        return x.is_cuda() && x.device() == device && x.scalar_type() == tb::kFloat32 &&
               x.is_contiguous();
    };
    TYLOINFER_RUNTIME_CHECK(
        f32(query_weight) && f32(key_weight) && query_weight.dim() == 1 && key_weight.dim() == 1 &&
            query_weight.numel() == width && key_weight.numel() == width && f32(cosine) &&
            f32(sine) && cosine.dim() == 2 && cosine.size(0) > 0 && cosine.size(0) <= INT_MAX &&
            cosine.size(1) == rotary_dim / 2 && sine.sizes() == cosine.sizes(),
        "QK preparation norm/rotary storage differs");
    const auto pos = [&](const tb::Tensor &x) {
        return x.is_cuda() && x.device() == device && x.scalar_type() == tb::kInt64 &&
               x.is_contiguous() &&
               ((x.dim() == 1 && x.size(0) == query.size(2)) ||
                (x.dim() == 2 && x.size(0) == query.size(0) && x.size(1) == query.size(2)));
    };
    TYLOINFER_RUNTIME_CHECK(pos(positions) && pos(write_positions),
                            "QK preparation positions differ");
    MfqCudaGuard guard(device);
    auto output = tb::empty(query.sizes(), query.options().dtype(tb::kFloat32));
    const auto layout = [](const tb::Tensor &x) {
        return HeadLayout{x.stride(0), x.stride(1), x.stride(2)};
    };
    const auto rows = query.numel() / width + key.numel() / width;
    const auto stream = mfq_current_cuda_stream();
    const auto launch = [&]<class Projection>() {
        const auto kernel = qk_norm_rope_cache_kernel<Projection, CacheWrite>;
        kernel<<<unsigned(rows), 256, 0, stream>>>(
            reinterpret_cast<const Projection *>(query.data_ptr()),
            reinterpret_cast<const Projection *>(key.data_ptr()),
            reinterpret_cast<const Projection *>(value.data_ptr()), query_weight.data_ptr<float>(),
            key_weight.data_ptr<float>(), positions.data_ptr<int64_t>(),
            write_positions.data_ptr<int64_t>(), cosine.data_ptr<float>(), sine.data_ptr<float>(),
            output.template data_ptr<float>(), cache, layout(query), layout(key), layout(value),
            int(query.size(1)), int(key.size(1)), int(query.size(2)), int(width),
            int(rotary_dim / 2), int(cosine.size(0)),
            positions.dim() == 2 ? positions.stride(0) : 0,
            write_positions.dim() == 2 ? write_positions.stride(0) : 0, float(eps),
            float(weight_offset));
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        if (mfq::cuda::dispatch_trace::enabled())
            mfq::cuda::dispatch_trace::record(
                stream, dispatch_name, reinterpret_cast<const void *>(kernel),
                "tokens=" + std::to_string(query.size(2)) + " batch=" + std::to_string(query.size(0)) +
                    " strided_heads=1 -> FP32 Q + FP16 KV");
    };
    if (query.scalar_type() == tb::kFloat16)
        launch.template operator()<__half>();
    else
        launch.template operator()<float>();
    return output;
}
} // namespace

mfq_tensor_backend::Tensor qk_norm_rope_cache_cuda(
    mfq_tensor_backend::Tensor query, mfq_tensor_backend::Tensor key,
    mfq_tensor_backend::Tensor value, mfq_tensor_backend::Tensor query_weight,
    mfq_tensor_backend::Tensor key_weight, mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor write_positions, mfq_tensor_backend::Tensor cosine,
    mfq_tensor_backend::Tensor sine, mfq_tensor_backend::Tensor key_cache,
    mfq_tensor_backend::Tensor value_cache, int64_t rotary_dim, double eps, double weight_offset) {
    namespace tb = mfq_tensor_backend;
    TYLOINFER_RUNTIME_CHECK(
        key.dim() == 4 && key_cache.is_cuda() && key_cache.device() == query.device() &&
            key_cache.scalar_type() == tb::kFloat16 && key_cache.dim() == 4 &&
            key_cache.is_contiguous() && value_cache.is_cuda() &&
            value_cache.device() == query.device() && value_cache.is_contiguous() &&
            value_cache.scalar_type() == tb::kFloat16 && value_cache.sizes() == key_cache.sizes() &&
            key_cache.size(0) == key.size(0) && key_cache.size(1) == key.size(1) &&
            key_cache.size(2) > 0 && key_cache.size(3) == key.size(3),
        "QK preparation cache storage differs");
    FlatCacheWrite cache{reinterpret_cast<__half *>(key_cache.data_ptr()),
                         reinterpret_cast<__half *>(value_cache.data_ptr()), key_cache.size(2)};
    return prepare_qk(query, key, value, query_weight, key_weight, positions, write_positions,
                      cosine, sine, cache, "qk_norm_rope_cache", rotary_dim, eps, weight_offset);
}

mfq_tensor_backend::Tensor qk_norm_rope_paged_cache_cuda(
    mfq_tensor_backend::Tensor query, mfq_tensor_backend::Tensor key,
    mfq_tensor_backend::Tensor value, mfq_tensor_backend::Tensor query_weight,
    mfq_tensor_backend::Tensor key_weight, mfq_tensor_backend::Tensor positions,
    mfq_tensor_backend::Tensor write_positions, mfq_tensor_backend::Tensor cosine,
    mfq_tensor_backend::Tensor sine, mfq_tensor_backend::Tensor key_chunks,
    mfq_tensor_backend::Tensor value_chunks, mfq_tensor_backend::Tensor pages, int64_t page_size,
    int64_t pages_per_chunk, int64_t rotary_dim, double eps, double weight_offset) {
    namespace tb = mfq_tensor_backend;
    const auto pointers = [&](const tb::Tensor &x) {
        return x.is_cuda() && x.device() == query.device() && x.is_contiguous() &&
               x.scalar_type() == tb::kInt64 && x.dim() == 1 && x.numel() > 0 &&
               x.numel() <= INT_MAX;
    };
    TYLOINFER_RUNTIME_CHECK(pointers(key_chunks) && pointers(value_chunks) &&
                                key_chunks.sizes() == value_chunks.sizes(),
                            "QK preparation paged chunk tables differ");
    TYLOINFER_RUNTIME_CHECK(key.dim() == 4 && pages.is_cuda() && pages.device() == query.device() &&
                                pages.is_contiguous() && pages.scalar_type() == tb::kInt32 &&
                                pages.dim() == 2 && pages.size(0) >= key.size(0) &&
                                pages.size(1) > 0 && pages.size(1) <= INT_MAX && page_size > 0 &&
                                page_size <= INT_MAX && pages_per_chunk > 0 &&
                                pages_per_chunk <= INT_MAX,
                            "QK preparation page geometry differs");
    // Page addresses use signed 64-bit element offsets. Reject genuinely
    // unrepresentable chunks without constraining any device or model geometry.
    const auto width = key.size(3), heads = key.size(1);
    TYLOINFER_RUNTIME_CHECK(width > 0 && heads > 0 && heads <= INT64_MAX / width &&
                                page_size <= INT64_MAX / (heads * width) &&
                                pages_per_chunk <= INT64_MAX / (heads * width * page_size),
                            "QK preparation paged chunk offset overflows");
    PagedCacheWrite cache{key_chunks.data_ptr<int64_t>(),
                          value_chunks.data_ptr<int64_t>(),
                          pages.data_ptr<int32_t>(),
                          int(key_chunks.numel()),
                          int(pages.size(1)),
                          int(page_size),
                          int(pages_per_chunk)};
    return prepare_qk(query, key, value, query_weight, key_weight, positions, write_positions,
                      cosine, sine, cache, "qk_norm_rope_paged_cache", rotary_dim, eps,
                      weight_offset);
}
