#pragma once

#include "engine/generation/sequence.h"
#include "engine/models/common/prompt.h"
#include "engine/requests/media.h"
#include "engine/requests/result.h"

#include <memory>

struct CudaExecutionContext;
namespace tyloi::weights { struct Loader; }
namespace tyloi::cuda {
class Executor;
namespace minicpmo45 {
namespace text { struct Model; class State; }

class Components {
  public:
    Components(const text::Model& language, text::State& language_state,
               Executor& language_executor, const tyloi::weights::Loader& weights,
               CudaExecutionContext& execution);
    ~Components();

    tyloi::StepSequence<PreparedPrompt> prepare(const std::vector<int64_t>& prompt,
                                               const MfqMultimodalInput& media);
    void start(const MfqDuplexSessionParams& parameters);
    MfqDuplexStepResult step(const MfqDuplexStepInput& input);
    void stop();

  private:
    struct State;
    std::unique_ptr<State> state_;
};

std::unique_ptr<Components> load_components(text::Model& model, text::State& state,
    Executor& executor, const tyloi::weights::Loader& weights,
    bool load_optional_components, CudaExecutionContext& execution);
} // namespace minicpmo45
} // namespace tyloi::cuda
