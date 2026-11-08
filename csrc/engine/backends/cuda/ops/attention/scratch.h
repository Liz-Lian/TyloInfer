#pragma once

#include "backend.h"
#include "engine/backends/cuda/ops/ffn/routes.h"

#include <cstdint>
#include <map>
#include <utility>

namespace tyloi::cuda {

struct MlaScratch {
    std::map<int64_t, MoeRoutePlan> head_routes;
    MoeRoutePlan transient_head_route;
    int64_t transient_head_route_key = -1;
    ::tyloi::DenseTensor dense_mask, decode_mask, kv_max, attention_meta, hadamard_signs;

    const MoeRoutePlan& head_route(int64_t rows, int heads);
    void ensure_meta();
    void ensure_compressed();
    void ensure_dense_workspace(int64_t batch, int64_t rows, int64_t logical_len);
    void ensure_decode_workspace(int64_t batch, int64_t planned_len);
};

struct AttentionScratch {
    static constexpr std::int64_t kMaxParts = 16;
    ::tyloi::DenseTensor partial_o, partial_m, partial_l;
    ::tyloi::DenseTensor mask, kv_max, meta;

    // Executor::storage() retains this per-stream arena for graph capture and replay.
    void ensure_meta(std::int64_t minimum, ::tyloi::Device device) {
        if (!meta.defined() || meta.device() != device ||
            meta.scalar_type() != ::tyloi::kFloat32 || meta.numel() < minimum)
            meta = ::tyloi::cuda::empty(
                {minimum}, ::tyloi::TensorOptions().device(device).dtype(::tyloi::kFloat32));
    }

    void split(std::int64_t rows, std::int64_t parts, std::int64_t width,
               ::tyloi::Device device) {
        if (partial_o.defined() && partial_o.device() == device && partial_o.size(0) >= rows &&
                partial_o.size(1) == parts && partial_o.size(2) == width) return;
        const auto options = ::tyloi::TensorOptions().device(device).dtype(::tyloi::kFloat32);
        auto o = ::tyloi::cuda::empty({rows, parts, width}, options);
        auto m = ::tyloi::cuda::empty({rows, parts}, options);
        auto l = ::tyloi::cuda::empty({rows, parts}, options);
        partial_o = std::move(o);
        partial_m = std::move(m);
        partial_l = std::move(l);
    }

    void mma(std::int64_t batch, std::int64_t kv_heads, std::int64_t width,
             std::int64_t visible, std::int64_t tile,
             ::tyloi::Device device) {
        const auto tiles = (visible + tile - 1) / tile;
        const auto stride = tiles * tile;
        const auto blocks = batch * kv_heads * tiles;
        const auto floats = 2 * blocks * 8 * (2 + width / 2);
        const auto options = ::tyloi::TensorOptions().device(device);
        if (!mask.defined() || mask.device() != device || mask.size(0) != batch || mask.size(1) < stride)
            mask = ::tyloi::cuda::empty({batch, stride}, options.dtype(::tyloi::kFloat16));
        if (!kv_max.defined() || kv_max.device() != device || kv_max.numel() < batch)
            kv_max = ::tyloi::cuda::empty({batch}, options.dtype(::tyloi::kInt32));
        if (!meta.defined() || meta.device() != device || meta.numel() < floats)
            meta = ::tyloi::cuda::empty({floats}, options.dtype(::tyloi::kFloat32));
    }
};

} // namespace tyloi::cuda
