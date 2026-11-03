#pragma once
#include "core/attention.h"
#include "core/rope.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace qsa_compact_pool_test {
inline void check() {
    namespace tb=mfq_tensor_backend;
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    int cases=0;
    for(int pools:{1,3,64})for(int pool:{4,8})for(int width:{32,96,128,256})
    for(bool half:{false,true})for(int axes:{1,3}) {
        const int dimension=width/2,pairs=dimension/2;
        mfq::cuda::RotaryEmbedding rotary(dimension,65536,10000.,
            axes==3?std::vector<int64_t>{pairs/2,pairs/4,pairs/4}:std::vector<int64_t>{},axes==3,true);
        std::vector<float> input(pool*pools*(width+8)),coefficients(width);
        for(std::size_t i=0;i<input.size();++i)input[i]=.7f*std::sin(float(i)*.173f)+.1f*std::cos(float(i)*.31f);
        for(int i=0;i<width;++i)coefficients[i]=.13f*std::sin(float(i)*.119f);
        auto raw=tb::tensor(input,options).reshape({1,pool*pools,width+8})
            .to(half?tb::kFloat16:tb::kFloat32).narrow(-1,0,width);
        auto weight=tb::tensor(coefficients,options);
        std::vector<int64_t> positions(axes*pools);
        for(int axis=0;axis<axes;++axis)for(int i=0;i<pools;++i)
            positions[axis*pools+i]=i%3==0?-7:i%3==1?int64_t(1)<<40:1234+i*pool+axis*31;
        if(axes==3)for(auto& position:positions)if(position>INT32_MAX)position=INT32_MAX;
        auto starts=tb::tensor(positions,options.dtype(tb::kInt64)).reshape({axes,1,pools})
            .to(axes==3?tb::kInt32:tb::kInt64);
        const auto reference=[&] {
            auto mean=raw.reshape({1,pools,pool,width}).to(tb::kFloat32).mean(2).to(raw.scalar_type());
            mean=mfq::cuda::attention_ops::rms_norm(mean,weight+1,1e-6);
            return rotary.forward(mean.unsqueeze(1),starts).squeeze(1);
        };
        for(bool zero:{false,true}) {
            if(zero)raw.zero_();
            auto expected=reference().to(tb::kFloat32).contiguous().cpu();
            auto result=rotary.pool_normalized(raw,weight,starts,pool,1e-6);
            if(!result.defined() || result.sizes()!=expected.sizes() || result.scalar_type()!=raw.scalar_type())
                throw std::runtime_error("compact QSA pooling did not dispatch");
            auto actual=result.to(tb::kFloat32).contiguous().cpu();
            for(int64_t i=0;i<actual.numel();++i) {
                const float r=expected.data_ptr<float>()[i],a=actual.data_ptr<float>()[i];
                if(!std::isfinite(a) || std::abs(a-r)>(half?2e-3f:2e-6f)+2e-6f*std::abs(r))
                    throw std::runtime_error("compact QSA pool/RMS/RoPE differs");
            }
            ++cases;
        }
    }
    std::cout<<"compact QSA pooling: strided FP16/FP32, complete pools, MRoPE, clamped positions, zero input PASS cases="<<cases<<std::endl;
}
}
