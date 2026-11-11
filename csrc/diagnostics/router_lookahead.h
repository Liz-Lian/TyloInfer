#pragma once
#include "engine/ops.h"
#include "engine/tensor.h"
#include <memory>
#include <stdexcept>
#include <vector>

namespace mfq::cuda {
// Diagnostic snapshots are copied after a completed decode window. They retain
// the inputs available before each FFN without changing routing or cache policy.
class RouterLookaheadInputs {
public:
    RouterLookaheadInputs(int layers,int hidden,int samples)
        :layers_(layers),hidden_(hidden),limit_(samples) {
        if(layers<=0||hidden<=0||samples<=0)
            throw std::invalid_argument("invalid router lookahead audit geometry");
        latest_=tyloi::ops::zeros({layers,hidden},tyloi::TensorOptions()
            .device(tyloi::Device(tyloi::kCUDA,0)).dtype(tyloi::kFloat16)).dense();
        samples_.reserve(samples);
    }
    void input(int layer,const tyloi::DenseTensor& value) {
        if(layer<0||layer>=layers_||!value.device().is_cuda()||value.numel()!=hidden_)
            throw std::invalid_argument("router lookahead audit requires a single CUDA input row");
        tyloi::ops::copy_into(latest_.narrow(0,layer,1),
            tyloi::ops::cast(value.reshape({1,hidden_}),tyloi::kFloat16,value.device()));
    }
    void finish_sample() {
        if(samples_.size()>=static_cast<std::size_t>(limit_))
            throw std::runtime_error("router lookahead audit sample limit exceeded");
        samples_.push_back(latest_.to(tyloi::Device(tyloi::kCPU,0)).contiguous());
    }
    int samples() const {return static_cast<int>(samples_.size());}
    tyloi::DenseTensor rows(int layer) const {
        if(layer<0||layer>=layers_||samples_.empty())
            throw std::invalid_argument("router lookahead audit has no input rows");
        std::vector<tyloi::Tensor> values;values.reserve(samples_.size());
        for(const auto& sample:samples_)values.push_back(sample.narrow(0,layer,1));
        return tyloi::ops::contiguous(tyloi::ops::cat(values,0)).dense();
    }
private:
    int layers_,hidden_,limit_;
    tyloi::DenseTensor latest_;
    std::vector<tyloi::DenseTensor> samples_;
};
}
