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
#ifndef TYLOINFER_NATIVE_CUDA
    using DenseTensor = ::tyloi::DenseTensor;
#endif
    /** Load the streamed vision adapter from its model source. */
    static Vision load(std::shared_ptr<const tyloi::ModelSource> source, tyloi::weights::Loader weights,
#ifndef TYLOINFER_NATIVE_CUDA
        ::tyloi::weights::Cache& cache,
#endif
        const CommonConfig& config);
    /** Prepare vision embeddings and participation masks using the owning request's media cache. */
    tyloi::StepSequence<tyloi::models::Prepared> prepare(
#ifdef TYLOINFER_NATIVE_CUDA
        const deepseek_v41::Model& language,
#else
        deepseek_v41::Model& language,
#endif
        tyloi::cache::Media& images,
#ifndef TYLOINFER_NATIVE_CUDA
        Projector projector,
#endif
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const;
private:
    Vision(std::shared_ptr<const tyloi::ModelSource> source, tyloi::weights::Loader weights,
#ifndef TYLOINFER_NATIVE_CUDA
        ::tyloi::weights::Cache& cache,
#endif
        CommonConfig config)
        : source_(std::move(source)), weights_(std::move(weights)),
#ifndef TYLOINFER_NATIVE_CUDA
          cache_(cache),
#endif
          config_(std::move(config)) {}
#ifdef TYLOINFER_NATIVE_CUDA
    tyloi::StepSequence<::tyloi::Tensor> encode(
        ::tyloi::Tensor patches,int height,int width) const;
#else
    tyloi::StepSequence<DenseTensor> encode(Projector projector, DenseTensor patches,int height,int width) const;
#endif
    std::shared_ptr<const tyloi::ModelSource> source_;
    tyloi::weights::Loader weights_;
#ifndef TYLOINFER_NATIVE_CUDA
    ::tyloi::weights::Cache& cache_;
#endif
    CommonConfig config_;
};

} // namespace tyloi::cuda::deepseek_v41_runtime
