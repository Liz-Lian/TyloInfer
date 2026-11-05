#pragma once
int run_iq2_fractional_cpu_reference() {
    namespace gg=mfq::cuda::gguf;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());
    const auto stream=mfq_current_cuda_stream();
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    int cases=0,sparse_cases=0;
    for(auto type:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S})
        for(int columns:{256,768,4096})for(int rows:{4,7,16}) {
            Source source;source.add("w",{rows,columns},type);
            auto canonical=load_gguf_weight(source,"w",mfq_current_cuda_device());
            auto quad=canonical;quad.packed=canonical.packed.clone();quad.repack_inplace();
            const auto encoded=canonical.packed.cpu();
            std::vector<float> decoded(rows*columns);
            const auto codec=ggml_get_type_traits(type)->to_float;
            for(int r=0;r<rows;++r)codec(static_cast<const uint8_t*>(encoded.data_ptr())+
                uint64_t(r)*canonical.row_bytes,decoded.data()+r*columns,columns);
            for(int tokens:{1,2,3,4,5,6,7,8,9,32}) {
                for(int sparse=0;sparse<(columns==256 && rows==4?2:1);++sparse) {
                    auto host=tb::empty({tokens,columns},cpu);
                    for(int m=0;m<tokens;++m)for(int k=0;k<columns;++k)
                        host.data_ptr<float>()[m*columns+k]=sparse
                            ?(k==(m*13)%columns?1.f:0.f)
                            :.13f*std::sin(float(m*columns+k)*.173f);
                    auto x=host.to(tb::kCUDA);
                    auto q8=tb::empty({int64_t(tokens*gg::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8));
                    gg::quantize_input(x.data_ptr<float>(),tokens,columns,q8.data_ptr(),stream);
                    const auto packed=q8.cpu();
                    const auto* activations=static_cast<const block_q8_1*>(packed.data_ptr());
                    for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
                        auto a=canonical.forward_q8_base(q8,tokens,dtype,false).cpu().to(tb::kFloat32);
                        auto b=quad.forward_q8_base(q8,tokens,dtype,false).cpu().to(tb::kFloat32);
                        tb::Tensor prepared_output;
                        if(tokens<=8) {
                            const auto prepared=prepare_gguf_group_input({quad,canonical},x);
                            prepared_output=prepared.project(quad).cpu().to(tb::kFloat32);
                        }
                        for(int m=0;m<tokens;++m)for(int r=0;r<rows;++r) {
                            double reference=0,magnitude=0;
                            for(int k=0;k<columns;++k) {
                                const auto& activation=activations[m*(columns/32)+k/32];
                                const double term=double(decoded[r*columns+k])*
                                    ggml_fp16_to_fp32(activation.d)*activation.qs[k%32];
                                reference+=term;magnitude+=std::abs(term);
                            }
                            const double limit=3e-6*magnitude+1e-7+
                                (dtype==tb::kFloat16?.00051*std::abs(reference)+1e-6:0);
                            const double actual[]={a.data_ptr<float>()[m*rows+r],b.data_ptr<float>()[m*rows+r]};
                            for(double value:actual)
                                require(std::isfinite(value) && std::abs(value-reference)<=limit,
                                    "IQ2 fractional canonical/quad projection differs from independent decoded-weight FP64 dot");
                            if(prepared_output.defined()) {
                                const double value=prepared_output.data_ptr<float>()[m*rows+r];
                                require(std::isfinite(value) && std::abs(value-reference)<=3e-6*magnitude+1e-7,
                                    "Prepared IQ2 split/native group differs from independent FP64 dot");
                            }
                        }
                        ++cases;sparse_cases+=sparse;
                    }
                }
            }
        }
    std::cout<<"IQ2 canonical-fractional CPU FP64 reference cases="<<cases
        <<" sparse-basis="<<sparse_cases
        <<" M1..9/32, K256/768/4096, rows4/7/16, F16/F32, canonical/quad/prepared PASS\n";
    return cases;
}
