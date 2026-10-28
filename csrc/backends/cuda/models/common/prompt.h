#pragma once

#include "backends/cuda/engine/requests/prompt.h"

#include <cstdint>
#include <stdexcept>

namespace mfq::cuda {

// Image participation is scoped to one prepared prefill chunk. It must not
// survive cancellation, a failed forward, ordinary decode or graph capture.
template<class Model> class ImageMaskScope {
    Model& model_;
    bool active_=false;
public:
    ImageMaskScope(Model& model,const PreparedPrompt* prepared,
                   int64_t offset,int64_t count):model_(model) {
        if(!prepared || !prepared->image_mask.defined())return;
        if constexpr(requires {model_.adapter_set_image_mask(prepared->image_mask);}) {
            model_.adapter_set_image_mask(prepared->image_mask.narrow(1,offset,count).contiguous());
            active_=true;
        } else throw std::invalid_argument("model cannot apply an image participation mask");
    }
    ~ImageMaskScope() {
        if constexpr(requires {model_.adapter_set_image_mask(mfq_tensor_backend::Tensor{});})
            if(active_)model_.adapter_set_image_mask({});
    }
    ImageMaskScope(const ImageMaskScope&)=delete;
    ImageMaskScope& operator=(const ImageMaskScope&)=delete;
};

} // namespace mfq::cuda
