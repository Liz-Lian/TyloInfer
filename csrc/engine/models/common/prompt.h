#pragma once

#include "engine/models/input.h"
#include "engine/ops.h"

#include <cstdint>
#include <stdexcept>

namespace tyloi::cuda {

// Image participation is scoped to one prepared prefill chunk. It must not
// survive cancellation, a failed forward, ordinary decode or graph capture.
template<class State> class ImageMaskScope {
    State& state_;
    bool active_=false;
    ::tyloi::Tensor mask_;
public:
    ImageMaskScope(State& state,const tyloi::models::Prepared* prepared,
                   int64_t offset,int64_t count):state_(state) {
        if(!prepared || !prepared->image_mask || !prepared->image_mask.defined())return;
        if constexpr(requires {state_.set_image_mask(mask_);}) {
            auto slice=tyloi::ops::slice(prepared->image_mask,1,offset,offset+count);
            mask_=tyloi::ops::contiguous(slice);
            state_.set_image_mask(mask_);
            active_=true;
        } else throw std::invalid_argument("model cannot apply an image participation mask");
    }
    ~ImageMaskScope() {
        if constexpr(requires {state_.set_image_mask(mask_);})
            if(active_)state_.set_image_mask(::tyloi::Tensor{});
    }
    ImageMaskScope(const ImageMaskScope&)=delete;
    ImageMaskScope& operator=(const ImageMaskScope&)=delete;
};

} // namespace tyloi::cuda
