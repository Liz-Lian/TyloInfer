#pragma once

#include "engine/generation/sequence.h"
#include "engine/tensor.h"

#include <memory>

namespace tyloi::models {

/** Hold one media encoding request, including explicit batch-indexed bounds. */
struct MediaInput {
    Tensor ids, pixels, patch_mask, target_sizes, image_bounds;
    Tensor audio_features, audio_lengths, audio_bounds;
};

/** Own the observable outputs of media encoding and prompt assembly. */
struct MediaResult {
    Tensor vision_states, image_embeddings, audio_embeddings, input_embeddings;
};

/** Encode media using model-owned weights in the caller's execution scope. */
class Media {
public:
    /** Own request inputs and state until the caller drains and retires it. */
    class Request {
    public:
        virtual ~Request() = default;
        /** Advance encoding or report completion. */
        virtual StepResult<MediaResult> next() = 0;
    };
    virtual ~Media() = default;
    /** Create a request borrowing this capability and retaining its inputs. */
    virtual std::unique_ptr<Request> encode(MediaInput input) const = 0;
    /** Encode and resample a batch of images without assembling a prompt. */
    virtual std::unique_ptr<Request> images(Tensor pixels, Tensor mask, Tensor sizes) const = 0;
    /** Encode a batch of audio features without assembling a prompt. */
    virtual std::unique_ptr<Request> audios(Tensor features, Tensor lengths) const = 0;
};

} // namespace tyloi::models
