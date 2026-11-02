#pragma once

#include "engine/generation/sequence.h"
#include "engine/models/common/prompt.h"
#include "engine/requests/media.h"
#include "engine/requests/result.h"

#include <memory>

struct CudaExecutionContext;
namespace mfq::cuda {
class Executor;
namespace weight_loader { struct Loader; }
namespace minicpmo45 {
namespace text { struct Model; class State; }

class Components {
  public:
    Components(const text::Model& language, text::State& language_state,
               Executor& language_executor, const weight_loader::Loader& weights,
               CudaExecutionContext& execution);
    ~Components();

    mfq::StepSequence<PreparedPrompt> prepare(const std::vector<int64_t>& prompt,
                                               const MfqMultimodalInput& media);
    void start(const MfqDuplexSessionParams& parameters);
    MfqDuplexStepResult step(const MfqDuplexStepInput& input);
    void stop();

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace minicpmo45
} // namespace mfq::cuda
