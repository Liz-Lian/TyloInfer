#pragma once

#include "backend.h"

#include <cstdint>
#include <utility>

namespace mfq::cuda {

struct AttentionScratch {
    mfq_tensor_backend::Tensor partial_o, partial_m, partial_l;
    mfq_tensor_backend::Tensor mask, kv_max, meta;

    void split(std::int64_t rows, std::int64_t parts, std::int64_t width,
               mfq_tensor_backend::Device device) {
        namespace tb = mfq_tensor_backend;
        if (partial_o.defined() && partial_o.device() == device && partial_o.size(0) >= rows &&
                partial_o.size(1) == parts && partial_o.size(2) == width) return;
        const auto options = tb::TensorOptions().device(device).dtype(tb::kFloat32);
        auto o = tb::empty({rows, parts, width}, options);
        auto m = tb::empty({rows, parts}, options);
        auto l = tb::empty({rows, parts}, options);
        partial_o = std::move(o);
        partial_m = std::move(m);
        partial_l = std::move(l);
    }

    void mma(std::int64_t batch, std::int64_t kv_heads, std::int64_t width,
             std::int64_t visible, std::int64_t tile,
             mfq_tensor_backend::Device device) {
        namespace tb = mfq_tensor_backend;
        const auto tiles = (visible + tile - 1) / tile;
        const auto stride = tiles * tile;
        const auto blocks = batch * kv_heads * tiles;
        const auto floats = 2 * blocks * 8 * (2 + width / 2);
        const auto options = tb::TensorOptions().device(device);
        if (!mask.defined() || mask.device() != device || mask.size(0) != batch || mask.size(1) < stride)
            mask = tb::empty({batch, stride}, options.dtype(tb::kFloat16));
        if (!kv_max.defined() || kv_max.device() != device || kv_max.numel() < batch)
            kv_max = tb::empty({batch}, options.dtype(tb::kInt32));
        if (!meta.defined() || meta.device() != device || meta.numel() < floats)
            meta = tb::empty({floats}, options.dtype(tb::kFloat32));
    }
};

} // namespace mfq::cuda
