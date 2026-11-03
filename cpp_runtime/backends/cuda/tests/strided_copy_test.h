void check_strided_inner_copy() {
    using namespace mfq::cuda;
    int cases=0;
    for(const auto type:{kFloat32,kFloat16,kBFloat16,kInt64})
    for(const int width:{16,20,128,132})for(const bool broadcast:{false,true}) {
        const int storage_width=width+16;
        auto host=arange(2*3*5*storage_width,TensorOptions{}.dtype(kFloat32).device(kCPU)).to(type).reshape({2,3,5,storage_width});
        auto gpu=host.to(kCUDA);
        auto expected_view=host.narrow(3,0,width).permute({0,2,1,3});
        auto source=gpu.narrow(3,0,width).permute({0,2,1,3});
        if(broadcast) {
            expected_view=expected_view.narrow(0,0,1).expand({2,5,3,width});
            source=source.narrow(0,0,1).expand({2,5,3,width});
        }
        const auto result=source.contiguous().to(kCPU);
        const auto bytes=host.element_size();
        for(int64_t linear=0;linear<result.numel();++linear) {
            int64_t offset=0,index=linear;
            for(int d=expected_view.dim()-1;d>=0;--d) {
                offset+=(index%expected_view.size(d))*expected_view.stride(d);
                index/=expected_view.size(d);
            }
            require(std::memcmp(static_cast<const char*>(result.data_ptr())+linear*bytes,
                static_cast<const char*>(expected_view.data_ptr())+offset*bytes,bytes)==0,
                "strided copy changed an element or outer-axis order");
        }
        ++cases;
    }
    auto large=empty({2,3,5,16},TensorOptions{}.dtype(kInt64).device(kCPU));
    for(int64_t i=0;i<large.numel();++i)large.data_ptr<int64_t>()[i]=(INT64_C(1)<<60)+i;
    const auto copied=large.to(kCUDA).transpose(1,2).contiguous().to(kCPU);
    for(int b=0;b<2;++b)for(int t=0;t<5;++t)for(int h=0;h<3;++h)for(int d=0;d<16;++d)
        require(copied.data_ptr<int64_t>()[((b*5+t)*3+h)*16+d]==large.data_ptr<int64_t>()[((b*3+h)*5+t)*16+d],
            "strided same-type copy rounded a large integer");
    auto values=arange(2*3*5*16,TensorOptions{}.dtype(kFloat32).device(kCPU)).reshape({2,3,5,16}).to(kCUDA);
    auto source=values.transpose(1,2),output=empty(source.sizes(),source.options());
    output.copy_(source);
    Graph graph;graph.prepare_memory();
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(current_stream().stream()));
    graph.capture_begin();output.copy_(source);graph.capture_end();
    for(int step=1;step<=3;++step) {
        values.fill_(step);graph.replay();
        for(const auto actual:host_values(output))require(actual==float(step),"strided graph copy replay used stale data");
    }
    std::cout<<"strided inner copy: permutations, slices, broadcasts, FP16/BF16/FP32/int64, tails and graph replay PASS cases="<<cases<<std::endl;
}
