#pragma once

#include "engine/generation/sequence.h"
#include "engine/models/common/prompt.h"
#include "engine/requests/media.h"
#include "engine/requests/result.h"
#ifdef TYLOINFER_NATIVE_CUDA
#include "engine/models/duplex.h"
#endif

#include <memory>

#ifndef TYLOINFER_NATIVE_CUDA
struct CudaExecutionContext;
#endif
namespace tyloi::weights { struct Loader; }
namespace tyloi::cuda {
class Executor;
namespace minicpmo45 {
namespace text { struct Model; class State; }

class Components {
  public:
#ifdef TYLOINFER_NATIVE_CUDA
    /** Load immutable media and speech weights for the selected device. */
    static std::unique_ptr<Components> load(const tyloi::weights::Loader&, int primary);
    ~Components();
    /** Prepare one prompt's image and audio embeddings. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(const text::Model&,
        const std::vector<int64_t>& prompt, const MfqMultimodalInput& media) const;
    /** Create a cold duplex session borrowing the given language request. */
    std::unique_ptr<tyloi::models::Duplex> duplex(const text::Model&, text::State&) const;
#else
    Components(const text::Model& language, text::State& language_state,
               Executor& language_executor, const tyloi::weights::Loader& weights,
               CudaExecutionContext& execution);
    ~Components();

    tyloi::StepSequence<tyloi::models::Prepared> prepare(const std::vector<int64_t>& prompt,
                                               const MfqMultimodalInput& media);
    void start(const MfqDuplexSessionParams& parameters);
    MfqDuplexStepResult step(const MfqDuplexStepInput& input);
    void stop();
#endif

  private:
    struct State;
#ifdef TYLOINFER_NATIVE_CUDA
    explicit Components(std::unique_ptr<State>);
#endif
    std::unique_ptr<State> state_;
};

#ifndef TYLOINFER_NATIVE_CUDA
std::unique_ptr<Components> load_components(text::Model& model, text::State& state,
    Executor& executor, const tyloi::weights::Loader& weights,
    bool load_optional_components, CudaExecutionContext& execution);
#endif
} // namespace minicpmo45
} // namespace tyloi::cuda
