#pragma once
#include "engine/models/source.h"
#include "engine/models/common/prompt.h"
#include "ops.h"
#include "engine/requests/media.h"
#include "engine/generation/sequence.h"
#include "engine/state/media.h"

namespace tyloi::weights { class Cache; }

namespace tyloi::cuda::deepseek_v41_runtime {

class Vision {
public:
    /** Load the streamed vision adapter from its model source. */
    static Vision load(std::shared_ptr<const tyloi::ModelSource> source, tyloi::weights::Loader weights,
        const CommonConfig& config);
    /** Prepare vision embeddings and participation masks using the owning request's media cache. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
        const deepseek_v41::Model& language,
        tyloi::cache::Media& images,
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const;
private:
    Vision(std::shared_ptr<const tyloi::ModelSource> source, tyloi::weights::Loader weights,
        CommonConfig config)
        : source_(std::move(source)), weights_(std::move(weights)),
          config_(std::move(config)) {}
    tyloi::StepSequence<::tyloi::Tensor> encode(
        ::tyloi::Tensor patches,int height,int width) const;
    std::shared_ptr<const tyloi::ModelSource> source_;
    tyloi::weights::Loader weights_;
    CommonConfig config_;
};

} // namespace tyloi::cuda::deepseek_v41_runtime
