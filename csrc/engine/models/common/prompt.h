#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/prepared.h"
#include "engine/state/media.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/ops.h"
#endif

#include <cstdint>
#include <memory>
#include <string>
#include <stdexcept>
#include <vector>

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA
using MediaCache = tyloi::cache::Media<std::shared_ptr<tyloi::Tensor>>;
using PreparedPrompt = tyloi::models::Prepared;
#else
using MediaCache = tyloi::cache::Media<::tyloi::DenseTensor>;
struct PreparedPrompt {
    std::vector<std::int64_t> token_ids;
    ::tyloi::DenseTensor embeddings;
    ::tyloi::DenseTensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    ::tyloi::DenseTensor image_mask;

    bool transformed() const noexcept {
        return embeddings.defined() || positions.defined() ||
            decode_position_delta != 0;
    }
};
#endif

// Image participation is scoped to one prepared prefill chunk. It must not
// survive cancellation, a failed forward, ordinary decode or graph capture.
template<class Model> class ImageMaskScope {
    Model& model_;
    bool active_=false;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const Tensor> mask_;
#endif
public:
    ImageMaskScope(Model& model,const PreparedPrompt* prepared,
                   int64_t offset,int64_t count):model_(model) {
#ifdef TYLOINFER_NATIVE_CUDA
        if(!prepared || !prepared->image_mask || !prepared->image_mask->defined())return;
        if constexpr(requires {model_.adapter_set_image_mask(mask_);}) {
            auto slice=tyloi::ops::slice(*prepared->image_mask,1,offset,offset+count);
            mask_=tyloi::ops::contiguous(*slice);
            model_.adapter_set_image_mask(mask_);
            active_=true;
        } else throw std::invalid_argument("model cannot apply an image participation mask");
#else
        if(!prepared || !prepared->image_mask.defined())return;
        if constexpr(requires {model_.adapter_set_image_mask(prepared->image_mask);}) {
            model_.adapter_set_image_mask(prepared->image_mask.narrow(1,offset,count).contiguous());
            active_=true;
        } else throw std::invalid_argument("model cannot apply an image participation mask");
#endif
    }
    ~ImageMaskScope() {
#ifdef TYLOINFER_NATIVE_CUDA
        if constexpr(requires {model_.adapter_set_image_mask(mask_);})
            if(active_)model_.adapter_set_image_mask(std::shared_ptr<const Tensor>{});
#else
        if constexpr(requires {model_.adapter_set_image_mask(::tyloi::DenseTensor{});})
            if(active_)model_.adapter_set_image_mask({});
#endif
    }
    ImageMaskScope(const ImageMaskScope&)=delete;
    ImageMaskScope& operator=(const ImageMaskScope&)=delete;
};

} // namespace tyloi::cuda
