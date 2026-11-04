#pragma once

#include "engine/backends/cuda/native/tensor/backend.h"
#include "engine/models/prepared.h"
#include "engine/state/media.h"

#include <cstdint>
#include <string>
#include <stdexcept>
#include <vector>

namespace tyloi::cuda {

#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
using MediaCache = tyloi::cache::Media<tyloi::Tensor>;
using PreparedPrompt = tyloi::models::Prepared;
#else
using MediaCache = tyloi::cache::Media<::tyloi::Tensor>;
struct PreparedPrompt {
    std::vector<std::int64_t> token_ids;
    ::tyloi::Tensor embeddings;
    ::tyloi::Tensor positions;
    int decode_position_delta = 0;
    std::string cache_key;
    ::tyloi::Tensor image_mask;

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
        if constexpr(requires {model_.adapter_set_image_mask(::tyloi::Tensor{});})
            if(active_)model_.adapter_set_image_mask({});
    }
    ImageMaskScope(const ImageMaskScope&)=delete;
    ImageMaskScope& operator=(const ImageMaskScope&)=delete;
};

} // namespace tyloi::cuda
