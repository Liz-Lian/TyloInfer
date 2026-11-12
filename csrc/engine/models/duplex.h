#pragma once

#include "engine/tensor.h"
#include "engine/requests/media.h"
#include "engine/requests/result.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace tyloi::models {

struct ForwardOptions;

/** Own one duplex session while borrowing its model and language state. */
class Duplex {
public:
    /** Hold exact tensor inputs for one duplex step. */
    struct StepInput {
        /** Preserve the full-width audio context and generation limits. */
        std::int64_t audio_prefix_extra_frames = 0, audio_suffix_extra_frames = 0;
        std::int64_t max_new_speak_tokens = 20;
        bool force_listen = false, force_speak = false;
        Tensor pixels, patch_mask, target_sizes, image_slice_counts, audio_features, text_ids;
    };
    /** Own the product response and original tensor observations. */
    struct StepOutput {
        MfqDuplexStepResult result;
        Tensor decision_logits, audio_embeddings, generated_ids, tts_codes;
    };
    virtual ~Duplex() = default;
    /** Plan persistent allocations for starting a session. */
    virtual std::vector<std::pair<Device, std::size_t>> growth(const MfqDuplexSessionParams&, const ForwardOptions&) const = 0;
    /** Plan persistent allocations for the next session step. */
    virtual std::vector<std::pair<Device, std::size_t>> growth(const MfqDuplexStepInput&, const ForwardOptions&) const = 0;
    /** Start the session after its allocations are admitted. */
    virtual void start(const MfqDuplexSessionParams&, const ForwardOptions&) = 0;
    /** Process the next audio or text input after admission. */
    virtual MfqDuplexStepResult step(const MfqDuplexStepInput&, const ForwardOptions&) = 0;
    /** Process exact tensor inputs; drain before replacing or retiring retained values. */
    virtual StepOutput step(StepInput, const ForwardOptions&) = 0;
    /** Reset to the original system inputs without reseeding sampling. */
    virtual void reset(const ForwardOptions&) = 0;
    /** Retire audio, speech and session state. */
    virtual void stop() = 0;
};

} // namespace tyloi::models
