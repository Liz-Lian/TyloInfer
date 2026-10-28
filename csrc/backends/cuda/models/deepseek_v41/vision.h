#pragma once
#include "backends/cuda/engine/requests/prompt.h"
#include "ops.h"
#include "engine/requests/media.h"
#include "engine/executor/sequence.h"

namespace mfq::cuda::deepseek_v41_runtime {

class Vision {
public:
    static Vision load(std::shared_ptr<const mfq::ModelSource> source,weight_loader::Loader weights,const CommonConfig& config);
    mfq::StepSequence<PreparedPrompt> prepare(DeepseekV41CausalLm& language, Projector projector,
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const;
private:
    Vision(std::shared_ptr<const mfq::ModelSource> source,weight_loader::Loader weights,CommonConfig config)
        :source_(std::move(source)),weights_(std::move(weights)),config_(std::move(config)) {}
    mfq::StepSequence<Tensor> encode(Projector projector,Tensor patches,int height,int width) const;
    std::shared_ptr<const mfq::ModelSource> source_;
    weight_loader::Loader weights_;
    CommonConfig config_;
    mutable std::string cached_key_;
    mutable Tensor cached_image_;
};

} // namespace mfq::cuda::deepseek_v41_runtime
