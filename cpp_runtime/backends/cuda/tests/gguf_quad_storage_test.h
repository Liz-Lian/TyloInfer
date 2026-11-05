#pragma once

#include "gguf_split_activation_test.h"

void run_quad_decode_views() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream_handle=mfq_get_current_cuda_stream(mfq_current_cuda_device());
    const ggml_type formats[]={GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_Q2_K};
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    const auto stream=stream_handle.stream();
    int cases=0;
    for(auto type:formats)for(int rows:{1,3,4,7,17})for(int columns:{256,768}) {
        Source source;source.add("w",{rows,columns},type);
        const auto native=load_gguf_weight(source,"w",mfq_current_cuda_device());
        auto packed=tb::empty_like(native.packed);
        gg::pack_quad_storage(type,native.packed.data_ptr(),native.row_bytes,rows,columns,packed.data_ptr(),stream);
        const auto reference=native.dequantize_stored().to(tb::kCPU,tb::kFloat32);
        for(int begin:{0,rows-1})for(auto dtype:{tb::kFloat16,tb::kFloat32})
        for(bool gather:{false,true})for(int offset:{0,1}) {
            const int view=rows-begin;
            const std::vector<int64_t> ids={view-1,0,-1,view,view/2,
                std::numeric_limits<int64_t>::min(),std::numeric_limits<int64_t>::max()};
            const int count=gather?int(ids.size()):view,bytes=dtype==tb::kFloat16?2:4;
            auto indices=tb::tensor(ids,cpu.dtype(tb::kInt64)).to(tb::kCUDA);
            auto expected=tb::full({count*columns+2},-13.,cpu.dtype(dtype));
            const auto decoded=reference.to(dtype);
            auto* destination=static_cast<uint8_t*>(expected.data_ptr())+offset*bytes;
            for(int row=0;row<count;++row) {
                const auto selected=gather?ids[row]:int64_t(row);
                if(selected>=0 && selected<view)
                    std::memcpy(destination+row*columns*bytes,
                        static_cast<const uint8_t*>(decoded.data_ptr())+(begin+selected)*columns*bytes,columns*bytes);
                else std::memset(destination+row*columns*bytes,0,columns*bytes);
            }
            auto actual=tb::full({count*columns+2},-13.,cpu.device(tb::kCUDA).dtype(dtype));
            gg::decode_quad_rows(type,packed.data_ptr(),rows,columns,begin,view,
                gather?indices.data_ptr<int64_t>():nullptr,count,
                static_cast<uint8_t*>(actual.data_ptr())+offset*bytes,dtype==tb::kFloat16,stream);
            const auto result=actual.to(tb::kCPU);
            require(std::memcmp(result.data_ptr(),expected.data_ptr(),result.nbytes())==0,
                "GGUF block decoder changed row views, gather zeros, alignment or boundary guards");
            ++cases;
        }
        for(int begin:{0,rows-1})for(int offset:{0,1}) {
            const int view=rows-begin;
            auto exponents=tb::empty({view},cpu.dtype(tb::kInt32));
            auto restored=tb::empty({view,columns},cpu);
            for(int row=0;row<view;++row) {
                const int exponent=row%13-6;exponents.data_ptr<int32_t>()[row]=exponent;
                for(int col=0;col<columns;++col)
                    restored.data_ptr<float>()[row*columns+col]=std::ldexp(
                        reference.data_ptr<float>()[(begin+row)*columns+col],-exponent);
            }
            const auto half=restored.to(tb::kFloat16);
            auto expected=tb::full({view*columns+2},-13.,cpu.dtype(tb::kFloat16));
            std::memcpy(static_cast<uint16_t*>(expected.data_ptr())+offset,half.data_ptr(),half.nbytes());
            auto actual=tb::full({view*columns+2},-13.,cpu.device(tb::kCUDA).dtype(tb::kFloat16));
            const auto device_exponents=exponents.to(tb::kCUDA);
            gg::decode_quad_rows_scaled(type,packed.data_ptr(),rows,columns,begin,view,
                device_exponents.data_ptr<int32_t>(),static_cast<uint16_t*>(actual.data_ptr())+offset,stream);
            const auto result=actual.to(tb::kCPU);
            require(std::memcmp(result.data_ptr(),expected.data_ptr(),result.nbytes())==0,
                "GGUF scaled block decoder changed row exponents or alignment");
            ++cases;
        }
    }
    std::cout<<"GGUF warp block decoder, gather, row tails, alignment and scales: "<<cases<<" cases PASS\n";
}

// Storage is lossless; projections are checked against the existing canonical
// kernels, including small decode windows, prefill, aliases and graph replay.
void run_quad_compute_storage() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const ggml_type formats[]={GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ3_XXS,GGML_TYPE_IQ3_S,GGML_TYPE_Q2_K};
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    {
        mfq::cuda::WeightMemoryLedger ledger;
        mfq::cuda::WeightMemoryLoadScope track(ledger);
        Source source;source.add("w",{16,4096},GGML_TYPE_IQ2_XXS);
        auto weight=load_gguf_weight(source,"w",mfq_current_cuda_device());
        const auto bytes=ledger.resident_bytes();
        require(bytes==weight.packed.nbytes(),"GGUF resident-weight ledger omitted its packed image");
        weight.repack_inplace();
        require(ledger.resident_bytes()==bytes,"GGUF in-place layout retained a second resident weight image");
    }
    const auto close=[](const tb::Tensor& actual,const tb::Tensor& reference,bool exact) {
        auto a=actual.to(tb::kCPU,tb::kFloat32).contiguous();
        auto b=reference.to(tb::kCPU,tb::kFloat32).contiguous();
        require(a.sizes()==b.sizes(),"GGUF compute-layout result shape changed");
        for(int64_t i=0;i<a.numel();++i) {
            const auto x=a.data_ptr<float>()[i],y=b.data_ptr<float>()[i];
            if(!std::isfinite(x) || std::abs(x-y)>(exact?0.f:1.e-5f+2.e-3f*std::abs(y)))
                std::cerr<<"GGUF compute-layout mismatch index="<<i<<" actual="<<x<<" reference="<<y
                    <<" abs_error="<<std::abs(x-y)<<" exact="<<exact<<'\n';
            require(std::isfinite(x) && std::abs(x-y)<=(exact?0.f:1.e-5f+2.e-3f*std::abs(y)),
                "GGUF compute-layout result differs from canonical projection");
        }
    };
    int cases=0;
    for(auto type:formats)for(int columns:{256,768,4096,6144,16640})for(int rows:{4,7,16}) {
        Source source;source.add("w",{rows,columns},type);
        const auto native=load_gguf_weight(source,"w",mfq_current_cuda_device());
        auto weight=native;weight.packed=native.packed.clone();
        const auto address=weight.packed.data_ptr();const auto bytes=weight.packed.nbytes();
        weight.repack_inplace();weight.repack_inplace();
        require(weight.quad_storage_rows==rows && weight.packed.data_ptr()==address &&
            weight.packed.nbytes()==bytes,"GGUF in-place conversion changed its allocation or byte count");
        const auto original=native.packed.cpu(),restored=weight.canonical_packed().cpu();
        require(std::memcmp(original.data_ptr(),restored.data_ptr(),bytes)==0,
            "GGUF compute storage did not restore the original encoded bytes");
        close(weight.dequantize_stored(),native.dequantize_stored(),true);
        const auto ids=tb::tensor(std::vector<int64_t>{rows-1,0,-1,rows,rows/2,
            std::numeric_limits<int64_t>::min(),std::numeric_limits<int64_t>::max()},cpu.dtype(tb::kInt64));
        close(weight.embedding(ids),native.embedding(ids),true);
        for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int tokens:{1,2,3,4,5,6,7,8,64}) {
            if((tokens==5 || tokens==7) && (columns!=768 || rows!=16))continue;
            auto host=tb::empty({tokens,columns},cpu);
            for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.13f*std::sin(float(i)*.173f);
            auto x=host.to(tb::kCUDA,dtype);
            // Q2_K joins subscale sums in integer arithmetic before FP32
            // scaling, as its M=1 kernel does; the final rounding tree differs.
            const bool same_tree=tokens!=1 && !(type==GGML_TYPE_Q2_K && tokens<=8 && weight.quad_projection_ready());
            close(weight.forward(x),native.forward(x),same_tree);
            if(rows==16 && (tokens==1 || (tokens<=8 && columns==768))) {
                tb::Tensor output;
                struct Pool {
                    std::shared_ptr<mfq::cuda::Context> context=mfq::cuda::default_context(mfq_current_cuda_device());
                    cudaStream_t stream=mfq_current_cuda_stream();
                    bool owned=!context->has_graph_pool(stream);
                    Pool(){if(owned)context->begin_graph_pool(stream);}
                    ~Pool(){if(owned)context->end_graph_pool(stream);}
                } pool;
                mfq::cuda::DecodeWindow graph(mfq_current_cuda_stream());
                graph.capture([&]{output=weight.forward(x);},[]{});
                for(int step=0;step<3;++step){x.copy_(x*.91f);graph.run();close(output,weight.forward(x),true);}
            }
            ++cases;
        }
    }
    // Join odd-sized matrices, then use both the joined image and unaligned
    // component views. They must retain one allocation and their own rows.
    for(auto type:formats) {
        Source source;source.add("a",{7,768},type);source.add("b",{9,768},type);
        auto a=load_gguf_weight(source,"a",0),b=load_gguf_weight(source,"b",0);
        auto x=tb::ones({3,768},cpu.device(tb::kCUDA));
        auto expected=tb::cat({a.forward(x),b.forward(x)},-1);
        a.repack_inplace();b.repack_inplace();
        const auto joined=combine_gguf_rows({&a,&b});
        require(a.packed.data_ptr()==joined.packed.data_ptr() && b.packed.data_ptr()==joined.packed.data_ptr() &&
            a.quad_row_begin==0 && b.quad_row_begin==7 && joined.packed.nbytes()==16*a.row_bytes,
            "GGUF combined compute-layout views do not share one exact-size allocation");
        close(joined.forward(x),expected,type!=GGML_TYPE_Q2_K);
        close(tb::cat({a.forward(x),b.forward(x)},-1),expected,true);
        ++cases;
    }
    for(auto first:formats)for(auto second:formats) {
        Source source;source.add("a",{16,4096},first);source.add("b",{8,4096},second);
        auto a=load_gguf_weight(source,"a",0),b=load_gguf_weight(source,"b",0);
        auto x=tb::ones({1,4096},cpu.device(tb::kCUDA).dtype(tb::kFloat16));
        auto multi=tb::ones({3,4096},x.options());
        auto expected=project_gguf_group({a,b},x),expected_multi=project_gguf_group({a,b},multi);
        a.repack_inplace();b.repack_inplace();
        auto actual=project_gguf_group({a,b},x);
        close(actual[0],expected[0],false);close(actual[1],expected[1],false);++cases;
        auto actual_multi=project_gguf_group({a,b},multi);
        close(actual_multi[0],expected_multi[0],first!=GGML_TYPE_Q2_K);
        close(actual_multi[1],expected_multi[1],second!=GGML_TYPE_Q2_K);++cases;
    }
    for(auto native_type:{GGML_TYPE_Q2_K,GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS,GGML_TYPE_Q4_0}) {
        Source source;source.add("q",{24,4096},native_type);source.add("k",{8,4096},GGML_TYPE_IQ1_M);
        source.add("v",{8,4096},GGML_TYPE_Q4_K);
        auto q=load_gguf_weight(source,"q",0),k=load_gguf_weight(source,"k",0),v=load_gguf_weight(source,"v",0);
        auto x=tb::ones({1,4096},cpu.device(tb::kCUDA));
        auto expected=project_gguf_group({q,k,v},x);
        k.repack_inplace();auto actual=project_gguf_group({q,k,v},x);
        for(size_t i=0;i<actual.size();++i)close(actual[i],expected[i],false);
        ++cases;
    }
    // An odd or tiny Q2 row view remains canonical beside an encoded peer.
    // Exercise both orders, activation layouts and three reduction lengths.
    for(auto peer:{GGML_TYPE_IQ1_M,GGML_TYPE_IQ2_XS})for(int columns:{256,4096,6144})
    for(int rows:{1,3,7,9})for(auto dtype:{tb::kFloat16,tb::kFloat32}){
        Source source;source.add("a",{rows,columns},GGML_TYPE_Q2_K);source.add("b",{16,columns},peer);
        auto a=load_gguf_weight(source,"a",0),b=load_gguf_weight(source,"b",0);
        auto host=tb::empty({1,columns},cpu);
        for(int i=0;i<columns;++i)host.data_ptr<float>()[i]=.13f*std::sin(float(i)*.173f);
        auto x=host.to(tb::kCUDA,dtype);
        auto expected=project_gguf_group({a,b},x),reverse=project_gguf_group({b,a},x);
        b.repack_inplace();
        auto actual=project_gguf_group({a,b},x),actual_reverse=project_gguf_group({b,a},x);
        close(actual[0],expected[0],false);close(actual[1],expected[1],false);
        close(actual_reverse[0],reverse[0],false);close(actual_reverse[1],reverse[1],false);
        cases+=2;
    }
    // Neuron corrections use logical output rows and the original activation,
    // independently of the compressed base's physical storage layout.
    struct NeuronCorrection final:GgufProjectionCorrection {
        Tensor pattern;mutable int prepared=0,applied=0;
        explicit NeuronCorrection(Tensor p):pattern(std::move(p)){}
        void prepare(const Tensor&)const override{++prepared;}
        Tensor apply(const Tensor& input,Tensor output)const override{
            ++applied;return output+input.sum(-1,true)*pattern;
        }
    };
    for(auto type:formats)for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int tokens:{1,2,3,4,6,8}) {
        Source source;source.add("w",{16,768},type);
        auto native=load_gguf_weight(source,"w",mfq_current_cuda_device());
        auto weight=native;weight.packed=native.packed.clone();weight.repack_inplace();
        auto host=tb::empty({tokens,768},cpu),mask=tb::zeros({1,16},cpu);
        for(int64_t i=0;i<host.numel();++i)host.data_ptr<float>()[i]=.13f*std::sin(float(i)*.173f);
        for(int row:{1,5,11,15})mask.data_ptr<float>()[row]=float(row+1)*.007f;
        auto x=host.to(tb::kCUDA,dtype),pattern=mask.to(x.device(),dtype);
        const auto expected=native.forward(x)+x.sum(-1,true)*pattern;
        auto correction=std::make_shared<NeuronCorrection>(pattern);weight.projection_correction=correction;
        const bool same_tree=tokens!=1 && !(type==GGML_TYPE_Q2_K && tokens<=8);
        close(weight.forward(x),expected,same_tree);
        require(weight.quad_storage_rows==16 && correction->prepared==1 && correction->applied==1,
            "GGUF in-place base skipped or repeated its neuron correction");
        auto grouped=project_gguf_group({weight},x);
        close(grouped.at(0),expected,same_tree);
        require(correction->prepared==2 && correction->applied==2,
            "GGUF grouped in-place base skipped or repeated its neuron correction");
        ++cases;
    }
    cases+=run_quad_split_activation_checks();
    std::cout<<"GGUF exact-size in-place storage, projections, prefill, views, neuron corrections and changing graphs: "
        <<cases<<" cases PASS\n";
}
