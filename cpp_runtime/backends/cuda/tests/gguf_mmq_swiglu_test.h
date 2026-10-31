#pragma once

void run_mmq_swiglu_quantize(bool benchmark=false) {
    namespace mmq=mfq::cuda::gguf::mmq;
    const auto stream=mfq_current_cuda_stream();
    const auto options=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
    const auto same=[](const tb::Tensor& a,const tb::Tensor& b) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        if(x.nbytes()!=y.nbytes() || std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))
            throw std::runtime_error("MMQ fused SwiGLU changed Q8 payload/scales/sums/zero guard");
    };
    const auto input=[&](int rows,int width,int salt) {
        std::vector<float> values(size_t(rows)*width*2);
        for(size_t i=0;i<values.size();++i)values[i]=i%53==0?0.f:std::sin(float(i)*.073f+salt)*(i%47==0?90.f:3.f);
        return tb::tensor(values).reshape({rows,2*width}).to(tb::kCUDA);
    };
    int cases=0;
    for(auto type:{GGML_TYPE_Q2_0,GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,
            GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ4_XS,GGML_TYPE_Q8_0,
            GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q5_1})
    for(int width:{256,640,768,1024})for(int rows:{1,3,17,129})for(float limit:{0.f,1.5f}) {
        if(width%ggml_blck_size(type))continue;
        auto gu=input(rows,width,701);
        if(rows==3)gu.zero_();
        auto hidden=tb::empty({rows,width},options);
        const auto bytes=mmq::activation_bytes(rows,width);
        auto expected=tb::empty({int64_t(bytes)},options.dtype(tb::kUInt8)),actual=tb::empty_like(expected);
        TYLOINFER_CUDA_CHECK(cudaMemsetAsync(expected.data_ptr(),0xa5,bytes,stream));
        TYLOINFER_CUDA_CHECK(cudaMemsetAsync(actual.data_ptr(),0x5a,bytes,stream));
        mmq::swiglu(gu.data_ptr<float>(),hidden.data_ptr<float>(),rows,width,limit,stream);
        mmq::quantize(hidden.data_ptr<float>(),nullptr,expected.data_ptr(),int(type),width,rows,stream);
        mmq::swiglu_quantize(gu.data_ptr<float>(),actual.data_ptr(),int(type),width,rows,limit,stream);
        same(actual,expected);++cases;
    }
    {
        constexpr int rows=5,width=256;const auto type=GGML_TYPE_Q4_K;
        auto gu=input(rows,width,702),hidden=tb::empty({rows,width},options);
        auto output=tb::empty({int64_t(mmq::activation_bytes(rows,width))},options.dtype(tb::kUInt8));
        auto expected=tb::empty_like(output);
        const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
        context->begin_graph_pool(stream);
        struct EndPool {
            std::shared_ptr<mfq::cuda::Context> context;cudaStream_t stream;
            ~EndPool(){context->end_graph_pool(stream);}
        } pool{context,stream};
        mfq::cuda::DecodeWindow graph(stream);
        graph.capture([&]{mmq::swiglu_quantize(gu.data_ptr<float>(),output.data_ptr(),int(type),width,rows,1.5f,stream);},[]{},[]{});
        for(int replay=0;replay<3;++replay) {
            gu.copy_(replay==1?tb::zeros_like(gu):input(rows,width,703+replay));
            graph.run();
            mmq::swiglu(gu.data_ptr<float>(),hidden.data_ptr<float>(),rows,width,1.5f,stream);
            mmq::quantize(hidden.data_ptr<float>(),nullptr,expected.data_ptr(),int(type),width,rows,stream);
            same(output,expected);++cases;
        }
    }
    if(benchmark) {
        constexpr int rows=1280,width=1536;const auto type=GGML_TYPE_Q4_K;
        auto gu=input(rows,width,711),hidden=tb::empty({rows,width},options);
        auto output=tb::empty({int64_t(mmq::activation_bytes(rows,width))},options.dtype(tb::kUInt8));
        const auto execute=[&](int mode) {
            if(mode)mmq::swiglu_quantize(gu.data_ptr<float>(),output.data_ptr(),int(type),width,rows,0.f,stream);
            else {
                mmq::swiglu(gu.data_ptr<float>(),hidden.data_ptr<float>(),rows,width,0.f,stream);
                mmq::quantize(hidden.data_ptr<float>(),nullptr,output.data_ptr(),int(type),width,rows,stream);
            }
        };
        cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
        std::vector<float> elapsed[2];
        for(int round=0;round<5;++round)for(int j=0;j<2;++j) {
            const int mode=(round+j)&1;execute(mode);TYLOINFER_CUDA_CHECK(cudaStreamSynchronize(stream));
            TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));for(int repeat=0;repeat<32;++repeat)execute(mode);
            TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));elapsed[mode].push_back(ms/32);
        }
        for(auto& e:elapsed)std::sort(e.begin(),e.end());
        std::cout<<"MMQ SwiGLU quantize rows="<<rows<<" width="<<width<<" old_ms="<<elapsed[0][2]<<" fused_ms="<<elapsed[1][2]<<std::endl;
        cudaEventDestroy(begin);cudaEventDestroy(end);
    }
    std::cout<<"MMQ SwiGLU quantize packet/scales/sums/guard/changing graph cases="<<cases<<" PASS"<<std::endl;
}
