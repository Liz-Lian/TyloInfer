#pragma once

void run_quad_token_vectors() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    const auto exact=[](const tb::Tensor& actual,const tb::Tensor& reference) {
        const auto a=actual.cpu(),b=reference.cpu();
        require(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() &&
            std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,
            "IQ2 token-vector projection changed the native Q8 result");
    };
    int cases=0,graphs=0;
    for(auto type:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S})
    for(int columns:{256,768,4096})for(int rows:{4,16}) {
        Source source;source.add("w",{rows,columns},type);
        auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());weight.repack_inplace();
        require(weight.quad_projection_ready(),"Token-vector fixture lacks compressed rows");
        for(int tokens:{3,4,5,6,7,8})for(auto input_type:{tb::kFloat16,tb::kFloat32}) {
            auto host=tb::empty({tokens,columns},cpu);
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.13f*std::sin(float(i)*.173f);
            auto x=host.to(tb::kCUDA,input_type);
            auto native=tb::empty({int64_t(tokens*gg::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8));
            const auto quantize_native=[&] {
                if(input_type==tb::kFloat16)gg::quantize_input_f16(x.data_ptr(),tokens,columns,native.data_ptr(),stream);
                else gg::quantize_input(x.data_ptr<float>(),tokens,columns,native.data_ptr(),stream);
            };
            quantize_native();
            require(weight.prefers_position_major_q8(tokens),"Prepared IQ2 input did not select token-vector storage");
            const auto prepared=prepare_gguf_group_input({weight},x);
            require(prepared.interleaved.defined(),"Prepared IQ2 input omitted its selected packet");
            exact(prepared.project(weight),weight.forward_q8_base(native,tokens,input_type,false));
            for(auto output_type:{tb::kFloat16,tb::kFloat32})for(int offset:(tokens>=5?std::vector<int>{0,16,32}:std::vector<int>{0,4,12})) {
                const auto bytes=gg::interleaved_activation_bytes(tokens,columns);
                auto storage=tb::full({int64_t(bytes+64)},165.,native.options());
                auto packet=storage.narrow(0,offset,int64_t(bytes));
                const auto quantize_packet=[&] {
                    gg::quantize_input_interleaved(x.data_ptr(),input_type==tb::kFloat16,tokens,columns,
                        packet.data_ptr(),nullptr,stream);
                };
                quantize_packet();
                auto expected=weight.forward_q8_base(native,tokens,output_type,false);
                auto actual=weight.forward_q8_base(packet,tokens,output_type,true);
                exact(actual,expected);
                const auto guarded=storage.cpu();
                for(int64_t i=0;i<guarded.numel();++i)if(i<offset || i>=offset+int64_t(bytes))
                    require(guarded.data_ptr<uint8_t>()[i]==165,"Token-vector packet overwrote its byte guards");
                ++cases;
                if(columns==768 && rows==16 && offset==0 && input_type==tb::kFloat32 && output_type==tb::kFloat32) {
                    struct Pool {
                        std::shared_ptr<mfq::cuda::Context> context=mfq::cuda::default_context(mfq_current_cuda_device());
                        cudaStream_t stream=mfq_current_cuda_stream();
                        bool owned=!context->has_graph_pool(stream);
                        Pool(){if(owned)context->begin_graph_pool(stream);}
                        ~Pool(){if(owned)context->end_graph_pool(stream);}
                    } pool;
                    mfq::cuda::DecodeWindow graph(stream);
                    graph.capture([&] {
                        quantize_native();quantize_packet();
                        expected=weight.forward_q8_base(native,tokens,output_type,false);
                        actual=weight.forward_q8_base(packet,tokens,output_type,true);
                    },[]{});
                    for(int step=0;step<3;++step) {
                        x.copy_(x*(step==2?0.f:.91f));graph.run();exact(actual,expected);++graphs;
                    }
                }
            }
        }
    }
    std::cout<<"GGUF IQ2 token-vector/native exact, F16/F32, aligned/offset guards cases="<<cases
        <<" changing_graphs="<<graphs<<" PASS\n";
}
