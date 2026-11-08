#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/input.h"
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
#else
using MediaCache = tyloi::cache::Media<::tyloi::DenseTensor>;
#endif

// Image participation is scoped to one prepared prefill chunk. It must not
// survive cancellation, a failed forward, ordinary decode or graph capture.
template<class State> class ImageMaskScope {
    State& state_;
    bool active_=false;
#ifdef TYLOINFER_NATIVE_CUDA
    std::shared_ptr<const Tensor> mask_;
#endif
public:
    ImageMaskScope(State& state,const tyloi::models::Prepared* prepared,
                   int64_t offset,int64_t count):state_(state) {
#ifdef TYLOINFER_NATIVE_CUDA
        if(!prepared || !prepared->image_mask || !prepared->image_mask.defined())return;
        if constexpr(requires {state_.set_image_mask(mask_);}) {
            auto slice=tyloi::ops::slice(prepared->image_mask,1,offset,offset+count);
            mask_=tyloi::ops::contiguous(slice);
            state_.set_image_mask(mask_);
            active_=true;
        } else throw std::invalid_argument("model cannot apply an image participation mask");
#else
        if(!prepared || !prepared->image_mask.defined())return;
        if constexpr(requires {state_.set_image_mask(prepared->image_mask);}) {
            state_.set_image_mask(prepared->image_mask.narrow(1,offset,count).contiguous());
            active_=true;
        } else throw std::invalid_argument("model cannot apply an image participation mask");
#endif
    }
    ~ImageMaskScope() {
#ifdef TYLOINFER_NATIVE_CUDA
        if constexpr(requires {state_.set_image_mask(mask_);})
            if(active_)state_.set_image_mask(Tensor{});
#else
        if constexpr(requires {state_.set_image_mask(::tyloi::DenseTensor{});})
            if(active_)state_.set_image_mask({});
#endif
    }
    ImageMaskScope(const ImageMaskScope&)=delete;
    ImageMaskScope& operator=(const ImageMaskScope&)=delete;
};

} // namespace tyloi::cuda
