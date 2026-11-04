#pragma once

#include "backend.h"
#include "engine/state/kv_view.h"
#include "mfq_cuda_kv_hot_cache.h"
#include <string>

namespace tyloi::cuda {
using KvQuantization = tyloi::state::KvQuantization;
KvQuantization cuda_kv_quantization();
int cuda_kv_packed_width(int dimension, int bits);
std::string cuda_kv_quantization_tag();
::tyloi::DenseTensor cuda_kv_encode(const ::tyloi::DenseTensor&, int bits, bool value);
::tyloi::DenseTensor cuda_kv_decode(const ::tyloi::DenseTensor&, int dimension, int bits,
    bool value, ::tyloi::ScalarType dtype = ::tyloi::kFloat16);

// Packed rows contain the FP32 norm followed by bit-packed centroid indices.
// Both resident and paged attention consume these rows without a floating KV copy.
struct QuantizedKvPages {
    ::tyloi::DenseTensor keys, values, table;
    int page_size = 0, pages_per_chunk = 0, heads = 0;
    bool enabled() const noexcept { return page_size > 0; }
};
void cuda_kv_write_paged(const QuantizedKvPages&, const ::tyloi::DenseTensor& keys,
    const ::tyloi::DenseTensor& values, const ::tyloi::DenseTensor& positions);
// Query [B,H,T,D], packed caches [B,KVH,L,P], selected [B,T,N] or empty for dense.
// Positions [T] or [B,T] are device-side causal boundaries, including graph replay.
// Output [B,T,H,D], FP32. Indexer and recurrent state are never quantized.
::tyloi::DenseTensor cuda_kv_attention(const ::tyloi::DenseTensor& query,
    const ::tyloi::DenseTensor& keys, const ::tyloi::DenseTensor& values,
    KvQuantization quantization, const ::tyloi::DenseTensor& selected = {},
    const ::tyloi::DenseTensor& positions = {}, int64_t query_offset = 0,
    const QuantizedKvPages& pages = {}, int64_t capacity = 0,
    const ::tyloi::DenseTensor& lengths = {}, const ::tyloi::DenseTensor& mask = {},
    double scale = 0, HotKvView hot = {});
}
