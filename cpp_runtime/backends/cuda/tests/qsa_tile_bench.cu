#include "mfq_cuda_context.h"
#include "mfq_sparse_attn_mma_f16.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// Explicit geometry; no development-card identity or model-size requirement.
void qsa_tile_pair_benchmark(int rows,int heads,int kv_heads,int capacity,int selected) {
    namespace tb=mfq_tensor_backend;
    using namespace mfq::cuda;
    constexpr int width=256;
    if(rows<=0 || heads<=0 || kv_heads<=0 || heads%kv_heads || capacity<=0 || selected<=0)
        throw std::invalid_argument("tile benchmark needs positive, compatible geometry");
    const auto data=[](std::vector<std::int64_t> shape,float phase,float scale) {
        std::int64_t count=1;for(auto n:shape)count*=n;
        std::vector<float> values(count);
        for(std::int64_t i=0;i<count;++i)values[i]=scale*std::sin(float(i)*.173f+phase);
        return tb::tensor(values).reshape(shape).to(tb::kCUDA);
    };
    const auto query=data({1,rows,heads,2*width},1.,.3f).to(tb::kFloat16).narrow(-1,0,width).permute({0,2,1,3});
    const auto gate=data({1,rows,heads,2*width},2.,.7f).narrow(-1,width,width);
    const auto key=data({1,kv_heads,capacity,width},3.,.17f).to(tb::kFloat16);
    const auto value=data({1,kv_heads,capacity,width},4.,.19f).to(tb::kFloat16);
    const int padded=(selected+63)/64*64;
    std::vector<std::int32_t> descriptors(std::size_t(rows)*padded);
    std::vector<float> masks(descriptors.size());
    for(int t=0;t<rows;++t)for(int i=0;i<padded;++i) {
        const auto at=std::size_t(t)*padded+i;
        descriptors[at]=int((std::int64_t(i)*37+t*13)%capacity);
        if(i>=selected)masks[at]=-std::numeric_limits<float>::infinity();
    }
    const auto ids=tb::tensor(descriptors).reshape({1,rows,padded}).to(tb::kCUDA);
    const auto mask=tb::tensor(masks).reshape({1,rows,padded}).to(tb::kCUDA).to(tb::kFloat16);
    const auto mapping=tb::arange(capacity,ids.options());
    const HotKvView hot{key.data_ptr(),value.data_ptr(),mapping.data_ptr<std::int32_t>(),capacity};
    std::array<tb::Tensor,2> outputs;
    const auto compute=[&](int which) {
        if(which==0)outputs[0]=mfq_sparse_attn_mma::launch<256,256,1,8,false,true,true,false>(
            query,key,value,ids,mask,{}, {},1./16.,"qsa_tile_pair8",gate,true,{},0,hot);
        else outputs[1]=mfq_sparse_attn_mma::launch<256,256,1,16,false,true,true,false>(
            query,key,value,ids,mask,{}, {},1./16.,"qsa_tile_pair16",gate,true,{},0,hot);
    };
    const auto stream=current_stream().stream();
    std::array<Graph,2> graphs;
    for(int which=0;which<2;++which) {
        {GraphWarmupScope warm(default_context(query.get_device()),stream);compute(which);}
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));outputs[which]={};
        graphs[which].capture_begin_shared();compute(which);graphs[which].capture_end();
    }
    graphs[0].replay();graphs[1].replay();
    const auto error=(outputs[0].to(tb::kFloat32)-outputs[1].to(tb::kFloat32)).abs().max().item<float>();
    if(!std::isfinite(error) || error>.002f)throw std::runtime_error("head tile attention output differs");
    // Warm and interleave both graphs on the same stream and allocations;
    // separately timed batches reverse order to reduce clock/order bias.
    for(int i=0;i<512;++i){graphs[0].replay();graphs[1].replay();}
    std::array<std::vector<float>,2> samples;
    constexpr int replays=512,trials=9;
    for(int trial=0;trial<trials;++trial)for(int order=0;order<2;++order) {
        const int which=(trial+order)&1;Event begin(0),end(0);
        begin.record(stream);for(int i=0;i<replays;++i)graphs[which].replay();end.record(stream);end.synchronize();
        float ms=0;TYLOINFER_NATIVE_CUDA_CHECK(cudaEventElapsedTime(&ms,begin.get(),end.get()));
        samples[which].push_back(ms*1000/replays);
    }
    for(auto& sample:samples)std::sort(sample.begin(),sample.end());
    const auto a=samples[0][trials/2],b=samples[1][trials/2];
    std::cout<<"qsa_tile_pair rows="<<rows<<" heads="<<heads<<" kv_heads="<<kv_heads<<" capacity="<<capacity
        <<" selected="<<selected<<" fp16_query=1 resolved_hot=1 head8_us="<<a<<" head16_us="<<b
        <<" ratio="<<a/b<<" max_error="<<error<<'\n';
}
