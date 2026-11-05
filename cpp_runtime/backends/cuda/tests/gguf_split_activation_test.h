#pragma once

// The full verification window keeps the native Q8 codes and half2 metadata;
// only its two 16-byte code planes differ. In particular, choosing this packet
// must preserve the native fused residual/RMS reduction, not the short-window
// standalone RMS reduction.
static int run_quad_split_activation_checks() {
    namespace gg=mfq::cuda::gguf;
    const auto device=mfq_current_cuda_device();
    const auto stream=mfq_get_current_cuda_stream(device).stream();
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kFloat32);
    const auto same=[&](const tb::Tensor& a,const tb::Tensor& b) {
        const auto x=a.cpu(),y=b.cpu();
        require(x.nbytes()==y.nbytes() && !std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()),
            "Full-window split activation changed native arithmetic");
    };
    const auto packets=[&](const tb::Tensor& alternate,const tb::Tensor& native,int columns,int tokens) {
        const auto a=alternate.cpu(),n=native.cpu();
        const auto* p=static_cast<const uint8_t*>(a.data_ptr());
        const auto* q=static_cast<const uint8_t*>(n.data_ptr());
        for(int i=0;i<tokens*columns/32;++i) {
            require(!std::memcmp(p+tokens*columns+i*4,q+i*36,4),"Split Q8 scale/sum differs");
            for(int lane=0;lane<32;++lane)
                require(p[(lane/16)*(tokens*columns/2)+i*16+lane%16]==q[i*36+4+lane],
                    "Split Q8 coordinates differ");
        }
    };
    int cases=0;
    for(int tokens:{5,6,7,8})for(int columns:{256,4096,5120,17408})for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto host=tb::empty({tokens,columns},cpu);
        for(int64_t i=0;i<host.numel();++i) {
            unsigned v=unsigned(i)+174139u;v=(v^(v>>16))*0x45d9f3bu;v=(v^(v>>16))*0x45d9f3bu;v^=v>>16;
            host.data_ptr<float>()[i]=float(int(v&65535)-32768)/8192.f*((v&31)==0?128.f:1.f);
        }
        auto x=host.to(tb::Device(tb::kCUDA,device),dtype);
        auto native=tb::empty({int64_t(tokens*gg::q8_row_bytes(columns))},x.options().dtype(tb::kUInt8));
        auto reference=tb::empty_like(native);
        auto alternate=tb::empty({int64_t(gg::interleaved_activation_bytes(tokens,columns))},native.options());
        if(dtype==tb::kFloat16)gg::quantize_input_f16(x.data_ptr(),tokens,columns,reference.data_ptr(),stream);
        else gg::quantize_input(x.data_ptr<float>(),tokens,columns,reference.data_ptr(),stream);
        gg::quantize_input_interleaved(x.data_ptr(),dtype==tb::kFloat16,tokens,columns,
            alternate.data_ptr(),native.data_ptr(),stream);
        same(native,reference);packets(alternate,reference,columns,tokens);++cases;

        auto residual=x.to(tb::kFloat16),norm=tb::ones({columns},x.options().dtype(tb::kFloat32));
        auto sum=tb::empty_like(residual),old_sum=tb::empty_like(residual);
        auto normalized=tb::empty_like(residual),old_normalized=tb::empty_like(residual);
        gg::quantize_acc_rms_input(residual.data_ptr(),x.data_ptr(),dtype==tb::kFloat32,norm.data_ptr<float>(),
            tokens,columns,1e-6f,.01f,old_sum.data_ptr(),old_normalized.data_ptr(),reference.data_ptr(),stream);
        gg::quantize_acc_rms_input_interleaved(residual.data_ptr(),x.data_ptr(),dtype==tb::kFloat32,
            norm.data_ptr<float>(),tokens,columns,1e-6f,.01f,sum.data_ptr(),normalized.data_ptr(),
            alternate.data_ptr(),native.data_ptr(),stream);
        same(sum,old_sum);same(normalized,old_normalized);same(native,reference);
        packets(alternate,reference,columns,tokens);++cases;
    }
    for(auto type:{GGML_TYPE_IQ2_XXS,GGML_TYPE_IQ2_XS,GGML_TYPE_IQ2_S,GGML_TYPE_IQ1_S,GGML_TYPE_IQ1_M,GGML_TYPE_Q2_K})
    for(int tokens:{5,6,7,8})for(int rows:{4,12,20})for(int columns:{256,5120}) {
        Source source;source.add("w",{rows,columns},type);
        auto canonical=load_gguf_weight(source,"w",device),weight=canonical;
        weight.packed=canonical.packed.clone();weight.repack_inplace();
        require(weight.prefers_position_major_q8(tokens) && !canonical.prefers_position_major_q8(tokens),
            "Split activation selection must follow the actual compute layout");
        auto x=tb::ones({tokens,columns},cpu.device(tb::Device(tb::kCUDA,device)));
        const auto prepared=prepare_gguf_group_input({weight,canonical},x);
        require(prepared.native.defined() && prepared.interleaved.defined(),
            "Mixed full-window projections need both packets from one producer");
        packets(prepared.interleaved,prepared.native,columns,tokens);
        same(prepared.project(weight),weight.forward_q8_base(prepared.native,tokens,tb::kFloat32,false));++cases;
    }
    return cases;
}
