#pragma once
#include "backends/cuda/engine/requests/prompt.h"
#include "ops.h"
#include "engine/requests/media.h"
#include "engine/executor/sequence.h"

namespace mfq::cuda::deepseek_v41_runtime {

class Vision {
public:
    static Vision load(std::shared_ptr<const mfq::ModelSource> source,const CommonConfig& config);
    mfq::StepSequence<PreparedPrompt> prepare(DeepseekV41CausalLm& language,
        const std::vector<int64_t>& tokens,const MfqMultimodalInput& media) const;
private:
    Vision(std::shared_ptr<const mfq::ModelSource> source,CommonConfig config)
        :source_(std::move(source)),config_(std::move(config)) {}
    mfq::StepSequence<Tensor> encode(CudaExecutionContext& execution,Tensor patches,int height,int width) const;
    std::shared_ptr<const mfq::ModelSource> source_;
    CommonConfig config_;
    mutable std::string cached_key_;
    mutable Tensor cached_image_;
};

} // namespace mfq::cuda::deepseek_v41_runtime
