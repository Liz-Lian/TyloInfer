#pragma once

void run_q2_packed_affine() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    const auto exact=[](const tb::Tensor& actual,const tb::Tensor& reference) {
        const auto a=actual.cpu(),b=reference.cpu();
        require(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type() &&
            std::memcmp(a.data_ptr(),b.data_ptr(),a.nbytes())==0,
            "Q2 packed affine projection changed the native Q8 result");
    };
    int cases=0,graphs=0;
    for(int columns:{256,768,4096})for(int rows:{4,20}) {
        Source source;source.add("w",{rows,columns},GGML_TYPE_Q2_K);source.full_q2_codes("w");
        auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());weight.repack_inplace();
        require(weight.quad_projection_ready() && weight.prefers_position_major_q8(8),
            "Q2 affine fixture omitted its compressed rows or selected split packet");
        for(auto input_type:{tb::kFloat16,tb::kFloat32})for(int pattern=0;pattern<8;++pattern) {
            auto host=tb::empty({8,columns},cpu);
            for(int64_t i=0;i<host.numel();++i) {
                const int lane=int(i%32);
                float value=0.f;
                if(pattern==1)value=1.f;
                else if(pattern==2)value=-1.f;
                else if(pattern==3)value=lane<16?1.f:-1.f;
                else if(pattern==4)value=lane<16?-1.f:1.f;
                else if(pattern==5)value=.3f*std::sin(float(i)*.173f);
                else if(pattern==6)value=lane==0?100.f:0.f;
                else if(pattern==7)value=(lane<16?1.f:-1.f)/4096;
                host.data_ptr<float>()[i]=value;
            }
            auto x=host.to(tb::kCUDA,input_type);
            auto native=tb::empty({int64_t(8*gg::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8));
            auto packet=tb::empty({int64_t(gg::interleaved_activation_bytes(8,columns))},native.options());
            const auto quantize=[&] {
                if(input_type==tb::kFloat16)gg::quantize_input_f16(x.data_ptr(),8,columns,native.data_ptr(),stream);
                else gg::quantize_input(x.data_ptr<float>(),8,columns,native.data_ptr(),stream);
                gg::quantize_input_interleaved(x.data_ptr(),input_type==tb::kFloat16,8,columns,packet.data_ptr(),nullptr,stream);
            };
            quantize();
            const auto prepared=prepare_gguf_group_input({weight},x);
            require(prepared.interleaved.defined(),"Prepared Q2 omitted the selected packed affine input");
            exact(prepared.project(weight),weight.forward_q8_base(native,8,input_type,false));
            for(auto output_type:{tb::kFloat16,tb::kFloat32}) {
                auto expected=weight.forward_q8_base(native,8,output_type,false);
                auto actual=weight.forward_q8_base(packet,8,output_type,true);
                exact(actual,expected);++cases;
                if(columns==768 && rows==20 && pattern==5 && input_type==tb::kFloat32 && output_type==tb::kFloat32) {
                    struct Pool {
                        std::shared_ptr<mfq::cuda::Context> context=mfq::cuda::default_context(mfq_current_cuda_device());
                        cudaStream_t stream=mfq_current_cuda_stream();
                        bool owned=!context->has_graph_pool(stream);
                        Pool(){if(owned)context->begin_graph_pool(stream);}
                        ~Pool(){if(owned)context->end_graph_pool(stream);}
                    } pool;
                    mfq::cuda::DecodeWindow graph(stream);
                    graph.capture([&] {
                        quantize();
                        expected=weight.forward_q8_base(native,8,output_type,false);
                        actual=weight.forward_q8_base(packet,8,output_type,true);
                    },[]{});
                    const auto original=x.clone();
                    for(float scale:{0.f,-.3f,1.2f}) {
                        x.copy_(original*scale);graph.run();exact(actual,expected);++graphs;
                    }
                }
            }
        }
    }
    std::cout<<"Q2 packed affine all minima, signed/zero scales, extreme sums, F16/F32 cases="<<cases
        <<" changing_graphs="<<graphs<<" PASS\n";
}
