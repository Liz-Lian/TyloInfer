#pragma once

#include "engine/generation/sequence.h"
#include "engine/models/common/prompt.h"
#include "engine/requests/media.h"
#include "engine/requests/result.h"
#include "engine/models/duplex.h"
#include "engine/models/media.h"
#include "engine/models/speech.h"

#include <memory>

namespace tyloi::weights { struct Loader; }
namespace tyloi::cuda {
class Executor;
namespace minicpmo45 {
namespace text { struct Model; class State; }

class Components
    : public tyloi::models::Speech
{
  public:
    /** Load immutable media and speech weights for the selected device. */
    static std::unique_ptr<Components> load(const tyloi::weights::Loader&, int primary);
    ~Components();
    /** Prepare one prompt's image and audio embeddings. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const text::Model&,
        const std::vector<int64_t>& prompt, const MfqMultimodalInput& media) const;
    /** Encode explicit media tensors and preserve observable intermediate outputs. */
    std::unique_ptr<tyloi::models::Media::Request> encode(const text::Model&,
        tyloi::models::MediaInput input) const;
    /** Encode one media batch without prompt assembly. */
    std::unique_ptr<tyloi::models::Media::Request> images(const text::Model&,
        tyloi::Tensor pixels, tyloi::Tensor mask, tyloi::Tensor sizes) const;
    /** Encode one audio batch without prompt assembly. */
    std::unique_ptr<tyloi::models::Media::Request> audios(const text::Model&,
        tyloi::Tensor features, tyloi::Tensor lengths) const;
    /** Build official speech conditioning. */
    tyloi::Tensor condition(const tyloi::Tensor& ids, const tyloi::Tensor& hidden) const override;
    /** Run one official speech request with optional raw logits capture. */
    std::unique_ptr<tyloi::models::Speech::Request> generate(tyloi::Tensor condition,
        tyloi::models::SpeechOptions options, bool capture_logits, std::mt19937* evaluator_rng) const override;
    /** Create a cold duplex session borrowing the given language request. */
    std::unique_ptr<tyloi::models::Duplex> duplex(const text::Model&, text::State&) const;

  private:
    struct State;
    explicit Components(std::unique_ptr<State>);
    enum class MediaMode { prompt, images, audios };
    std::unique_ptr<tyloi::models::Media::Request> media_request(const text::Model&,
        tyloi::models::MediaInput, MediaMode) const;
    std::unique_ptr<State> state_;
};

} // namespace minicpmo45
} // namespace tyloi::cuda
