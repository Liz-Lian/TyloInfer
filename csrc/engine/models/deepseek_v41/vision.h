#pragma once
#include "engine/models/source.h"
#include "engine/models/common/prompt.h"
#include "ops.h"
#include "engine/requests/media.h"
#include "engine/generation/sequence.h"

namespace tyloi::weights { class Cache; }

namespace tyloi::cuda::deepseek_v41_runtime {

class Vision {
public:
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
    using DenseTensor = ::tyloi::DenseTensor;
#endif
    static Vision load(std::shared_ptr<const tyloi::ModelSource> source,tyloi::weights::Loader weights,::tyloi::weights::Cache& cache,const CommonConfig& config);
    tyloi::StepSequence<PreparedPrompt> prepare(deepseek_v41::Model& language, MediaCache& images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector,
#endif
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const;
private:
    Vision(std::shared_ptr<const tyloi::ModelSource> source,tyloi::weights::Loader weights,::tyloi::weights::Cache& cache,CommonConfig config)
        :source_(std::move(source)),weights_(std::move(weights)),cache_(cache),config_(std::move(config)) {}
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    tyloi::StepSequence<std::shared_ptr<tyloi::Tensor>> encode(
        std::shared_ptr<const tyloi::Tensor> patches,int height,int width) const;
#else
    tyloi::StepSequence<DenseTensor> encode(Projector projector, DenseTensor patches,int height,int width) const;
#endif
    std::shared_ptr<const tyloi::ModelSource> source_;
    tyloi::weights::Loader weights_;
    ::tyloi::weights::Cache& cache_;
    CommonConfig config_;
};

} // namespace tyloi::cuda::deepseek_v41_runtime
