#pragma once

class SmallWindowCorrection final:public GgufProjectionCorrection {
public:
    mutable int prepares=0;
    void prepare(const Tensor&)const override{++prepares;}
    Tensor apply(const Tensor&,Tensor output)const override{return output;}
};
static void same_packets(const GgufProjectionInput& actual,const GgufProjectionInput& expected) {
    require(actual.native.defined()==expected.native.defined(),"RMS group allocated the wrong native packet");
    require(actual.interleaved.defined()==expected.interleaved.defined(),"RMS group allocated the wrong position-major packet");
    if(expected.native.defined())same(actual.native,expected.native,"RMS group native packet differs");
    if(expected.interleaved.defined())same(actual.interleaved,expected.interleaved,"RMS group position-major packet differs");
}
static void small_window_rms_checks() {
    const auto stream=mfq_get_current_cuda_stream(0);int packets=0,projections=0;
    for(int rows:{2,3,4})for(int width:{32,256,2016,2048,4096,5120,17408})
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(float offset:{0.f,1.f})for(bool native:{false,true}) {
        auto input=values({rows,width},dtype),norm=values({width},tb::kFloat32,.8f);
        input.narrow(0,0,1).fill_(0);
        auto expected_norm=rms_norm_offset_cuda(input,norm,1e-5,offset);
        auto expected=tb::empty({int64_t(gg::interleaved_activation_bytes(rows,width))},input.options().dtype(tb::kUInt8));
        auto expected_native=tb::empty({int64_t(rows*gg::q8_row_bytes(width))},expected.options());
        auto storage=tb::empty({expected.numel()+32},expected.options());storage.fill_(165);auto actual=storage.narrow(0,16,expected.numel());
        auto actual_native=tb::empty_like(expected_native),actual_norm=tb::empty_like(expected_norm);
        gg::quantize_input_interleaved(expected_norm.data_ptr(),false,rows,width,expected.data_ptr(),expected_native.data_ptr(),stream.stream());
        gg::quantize_rms_input_interleaved(input.data_ptr(),dtype==tb::kFloat16,norm.data_ptr<float>(),rows,width,
            1e-5,offset,actual_norm.data_ptr<float>(),actual.data_ptr(),native?actual_native.data_ptr():nullptr,stream.stream());
        same(actual_norm,expected_norm,"position-major RMS changed normalization reduction or output");
        same(actual,expected,"position-major RMS packet differs");if(native)same(actual_native,expected_native,"dual-output RMS native packet differs");
        auto sentry=storage.cpu();for(int i=0;i<16;++i)require(sentry.data_ptr<uint8_t>()[i]==165 &&
            sentry.data_ptr<uint8_t>()[sentry.numel()-16+i]==165,"RMS packet overwrote guard bytes");++packets;

        auto residual=values({rows,width},tb::kFloat16,.3f),projection=values({rows,width},dtype,.6f);
        const auto reference=acc_rms_norm_f16_cuda(residual,projection,norm,1e-5,offset);
        auto summed=tb::empty_like(residual),normalized=tb::empty_like(residual);
        gg::quantize_input_interleaved(reference[1].data_ptr(),true,rows,width,expected.data_ptr(),expected_native.data_ptr(),stream.stream());
        gg::quantize_acc_rms_input_interleaved(residual.data_ptr(),projection.data_ptr(),dtype==tb::kFloat32,norm.data_ptr<float>(),
            rows,width,1e-5,offset,summed.data_ptr(),normalized.data_ptr(),actual.data_ptr(),native?actual_native.data_ptr():nullptr,stream.stream());
        same(summed,reference[0],"position-major residual/RMS changed the residual rounding");
        try{same(normalized,reference[1],"position-major residual/RMS changed normalization rounding");}
        catch(...) {std::cerr<<"rows="<<rows<<" width="<<width<<" projection_float="<<(dtype==tb::kFloat32)
            <<" offset="<<offset<<" native="<<native<<std::endl;throw;}
        same(actual,expected,"position-major residual/RMS packet differs");
        if(native)same(actual_native,expected_native,"dual-output residual/RMS native packet differs");++packets;
    }
    // A contiguous float view need not satisfy the standalone norm's float2
    // alignment. Its scalar reduction must keep the same rounded FP16 output.
    for(int width:{2048,5120,17408})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        const int rows=3;auto norm=values({width+1},tb::kFloat32,.8f).narrow(0,1,width);
        auto residual=values({rows,width},tb::kFloat16,.3f),projection=values({rows,width},dtype,.6f);
        const auto reference=acc_rms_norm_f16_cuda(residual,projection,norm,1e-5,0);
        auto sum=tb::empty_like(residual),normalized=tb::empty_like(residual);
        auto packet=tb::empty({int64_t(gg::interleaved_activation_bytes(rows,width))},residual.options().dtype(tb::kUInt8));
        gg::quantize_acc_rms_input_interleaved(residual.data_ptr(),projection.data_ptr(),dtype==tb::kFloat32,norm.data_ptr<float>(),
            rows,width,1e-5,0,sum.data_ptr(),normalized.data_ptr(),packet.data_ptr(),nullptr,stream.stream());
        same(sum,reference[0],"unaligned norm view changed residual rounding");
        same(normalized,reference[1],"unaligned norm view selected the wrong RMS reduction");++packets;
    }
    CudaExecutionContext execution;
    for(const auto& types:std::vector<std::vector<ggml_type>>{{GGML_TYPE_IQ3_XXS},{GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS},
            {GGML_TYPE_IQ3_XXS,GGML_TYPE_Q8_0},{GGML_TYPE_IQ4_XS,GGML_TYPE_IQ1_S},{GGML_TYPE_Q8_0}}) {
        const int width=512,out=36;QuantLinearGroup group;std::vector<GgufWeight> weights;
        auto source=values({out,width},tb::kFloat32).cpu(),importance=tb::ones({width},source.options());
        for(auto type:types) {
            const auto stride=ggml_row_size(type,width);auto bytes=tb::empty({int64_t(out*stride)},source.options().dtype(tb::kUInt8));
            require(ggml_quantize_chunk(type,source.data_ptr<float>(),bytes.data_ptr(),0,out,width,importance.data_ptr<float>())==bytes.nbytes(),"RMS group weight quantization failed");
            QuantLinear linear;linear.kind=QuantLinearKind::Gguf;linear.gguf={bytes.to(tb::kCUDA),type,out,width,uint64_t(stride)};
            linear.gguf.repack_inplace();group.layers.push_back(linear);group.outs.push_back(out);
        }
        auto correction=std::make_shared<SmallWindowCorrection>();group.layers[0].gguf.projection_correction=correction;
        for(const auto& layer:group.layers)weights.push_back(layer.gguf);
        for(int rows:{1,2,3,4,5,32})for(auto dtype:{tb::kFloat16,tb::kFloat32})for(float offset:{0.f,1.f}) {
            auto input=values({rows,width},dtype),norm=values({width},tb::kFloat32,.8f);tb::Tensor normalized;
            correction->prepares=0;const auto actual=group.prepare_rms_norm_input(execution,input,norm,1e-5,offset,normalized);
            require(!actual.shape.empty() && correction->prepares==1,"RMS group fusion or neuron correction preparation was skipped");
            auto expected_norm=rms_norm_offset_cuda(input,norm,1e-5,offset);same(normalized,expected_norm,"RMS group normalization differs");
            require(actual.original.defined(),"RMS group did not retain normalization for its neuron correction");
            same(actual.original,expected_norm,"RMS group neuron correction received the wrong normalization");
            const auto reference=prepare_gguf_group_input(weights,expected_norm);same_packets(actual,reference);
            const auto outputs=group.forward_prepared_rms(actual),expected_outputs=group.forward_prepared_rms(reference);
            for(size_t i=0;i<outputs.size();++i){same(outputs[i],expected_outputs[i],"RMS group projection differs");++projections;}
            auto residual=values({rows,width},tb::kFloat16,.4f),projection=values({rows,width},dtype,.6f);tb::Tensor sum,acc_norm;
            correction->prepares=0;const auto acc=group.prepare_acc_rms_norm_input(execution,residual,projection,norm,1e-5,offset,sum,acc_norm);
            require(!acc.shape.empty() && correction->prepares==1,"residual/RMS group fusion or correction preparation was skipped");
            const auto expected_acc=acc_rms_norm_f16_cuda(residual,projection,norm,1e-5,offset);
            same(sum,expected_acc[0],"residual/RMS group sum differs");same(acc_norm,expected_acc[1],"residual/RMS group normalization differs");
            const auto acc_reference=prepare_gguf_group_input(weights,expected_acc[1]);same_packets(acc,acc_reference);
            require(acc.original.defined(),"residual/RMS group dropped correction input");same(acc.original,expected_acc[1],"residual/RMS correction input differs");
            const auto acc_outputs=group.forward_prepared_rms(acc),expected_acc_outputs=group.forward_prepared_rms(acc_reference);
            for(size_t i=0;i<acc_outputs.size();++i){same(acc_outputs[i],expected_acc_outputs[i],"residual/RMS group projection differs");++projections;}
            if(rows==3 && dtype==tb::kFloat16 && offset==0 && types.size()==2) {
                for(bool with_residual:{false,true}) {
                    const auto run=[&] {tb::Tensor sum,n;
                        const auto packet=with_residual?group.prepare_acc_rms_norm_input(execution,residual,projection,norm,1e-5,offset,sum,n):
                            group.prepare_rms_norm_input(execution,input,norm,1e-5,offset,n);
                        return group.forward_prepared_rms(packet);};
                    mfq::cuda::Graph graph;graph.prepare_memory();{auto warm=run();}
                    graph.capture_begin();auto captured=run();graph.capture_end();
                    for(int iteration=0;iteration<3;++iteration) {
                        input.copy_(values({rows,width},dtype,float(iteration)*.3f));
                        projection.copy_(values({rows,width},dtype,float(iteration)*.4f));graph.replay();
                        auto expected=with_residual?acc_rms_norm_f16_cuda(residual,projection,norm,1e-5,offset)[1]:
                            rms_norm_offset_cuda(input,norm,1e-5,offset);
                        const auto packet=prepare_gguf_group_input(weights,expected);const auto outputs=group.forward_prepared_rms(packet);
                        for(size_t i=0;i<outputs.size();++i)same(captured[i],outputs[i],"changing-input mixed RMS Graph replay differs");
                    }
                }
            }
        }
    }
    auto input=values({2,32},tb::kFloat32),norm=values({32},tb::kFloat32),output=tb::empty({2048},input.options().dtype(tb::kUInt8));
    bool invalid=false;try{gg::quantize_rms_input_interleaved(input.data_ptr(),false,norm.data_ptr<float>(),2,32,1e-5,0,
        input.data_ptr<float>(),output.data_ptr(),nullptr,stream.stream());}catch(const std::invalid_argument&){invalid=true;}
    require(invalid,"replicated RMS accepted an aliased input/output");
    std::cout<<"Small-window RMS dual-layout PASS packets="<<packets<<" exact_projections="<<projections
        <<"; native/position-major/mixed layouts, correction input and changing-input Graph replay PASS"<<std::endl;
}
