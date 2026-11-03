#pragma once
#include "engine/models/source.h"
#include "engine/models/common/prompt.h"
#include "ops.h"
#include "engine/requests/media.h"
#include "engine/generation/sequence.h"

namespace mfq::cuda::deepseek_v41_runtime {

class Vision {
public:
#ifdef TYLOINFER_NATIVE_CUDA_RUNTIME
    using Tensor = mfq::Tensor;
#else
    using Tensor = mfq_tensor_backend::Tensor;
#endif
    static Vision load(std::shared_ptr<const mfq::ModelSource> source,weight_loader::Loader weights,const CommonConfig& config);
    mfq::StepSequence<PreparedPrompt> prepare(DeepseekV41CausalLm& language, MediaCache& images,
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector,
#endif
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const;
private:
    Vision(std::shared_ptr<const mfq::ModelSource> source,weight_loader::Loader weights,CommonConfig config)
        :source_(std::move(source)),weights_(std::move(weights)),config_(std::move(config)) {}
    mfq::StepSequence<Tensor> encode(
#ifndef TYLOINFER_NATIVE_CUDA_RUNTIME
        Projector projector,
#endif
        Tensor patches,int height,int width) const;
    std::shared_ptr<const mfq::ModelSource> source_;
    weight_loader::Loader weights_;
    CommonConfig config_;
};

} // namespace mfq::cuda::deepseek_v41_runtime
