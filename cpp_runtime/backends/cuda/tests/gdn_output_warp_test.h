#pragma once

void gdn_output_warp_case(bool benchmark) {
    GdnColumnsScope option(0,"TYLOINFER_GDN_NORM_WARP_REDUCTION");
    int sms=0;TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,mfq_current_cuda_device()));
    const auto bits=[](const Tensor& a,const Tensor& b) {
        auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        if(x.scalar_type()!=y.scalar_type() || x.nbytes()!=y.nbytes() || std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))
            throw std::runtime_error("GDN row-warp norm differs bitwise from original block tree");
    };
    const auto key="TYLOINFER_GDN_NORM_WARP_REDUCTION";
    int cases=0;
    for(int width:{17,32,64,128,129,256,320,513})for(int heads:{3,48})
    for(bool half_gate:{false,true})for(bool half_output:{false,true})for(bool silu:{false,true}) {
        const int batch=heads==3?2:1,tokens=(4*sms+heads-1)/heads+7;
        auto attended=data({batch,heads,tokens,width},581,.17f).to(tb::kFloat32);
        auto gate=data({batch,tokens,heads*width+17},582,14.f)
            .to(half_gate?tb::kFloat16:tb::kFloat32).narrow(2,0,heads*width);
        auto weight=data({width},583,1.1f).to(tb::kFloat32);
        const double eps=silu?1e-6:.013456789123;
        GdnColumnsScope::set(0,key);
        auto expected=gdn_rms_norm_gate_cuda(attended,gate,weight,eps,silu,half_output);
        GdnColumnsScope::set(-1,key);
        auto actual=gdn_rms_norm_gate_cuda(attended,gate,weight,eps,silu,half_output);
        bits(actual,expected);++cases;
    }
    {
        const int heads=3,tokens=(4*sms+heads-1)/heads+7,width=129;
        auto attended=data({1,heads,tokens,width},584,.00013f).to(tb::kFloat32);
        auto gate=data({1,tokens,heads*width+7},585,14.f).to(tb::kFloat16).narrow(2,0,heads*width);
        auto weight=data({width},586,1.1f).to(tb::kFloat32);
        Tensor output;DecodeWindow window(mfq_current_cuda_stream());
        GdnColumnsScope::set(-1,key);
        window.capture([&]{output=gdn_rms_norm_gate_cuda(attended,gate,weight,1e-6,true,true);},[]{},[&]{output={};});
        for(int replay=0;replay<3;++replay) {
            attended.copy_(replay==1?tb::zeros_like(attended):data(attended.sizes().vec(),587+replay,.00013f).to(tb::kFloat32));
            gate.copy_(data(gate.sizes().vec(),591+replay,88.f).to(tb::kFloat16));
            window.run();
            GdnColumnsScope::set(0,key);
            auto expected=gdn_rms_norm_gate_cuda(attended,gate,weight,1e-6,true,true);
            bits(output,expected);GdnColumnsScope::set(-1,key);++cases;
        }
    }
    if(benchmark) {
        constexpr int heads=48,tokens=4132,width=128;
        auto attended=data({1,heads,tokens,width},595,.17f).to(tb::kFloat32);
        auto gate=data({1,tokens,heads*width},596,14.f).to(tb::kFloat16);
        auto weight=data({width},597,1.1f).to(tb::kFloat32);
        std::vector<float> elapsed[2];
        Tensor output;cudaEvent_t begin,end;
        TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
        for(int round=0;round<5;++round)for(int j=0;j<2;++j) {
            const int mode=(round+j)&1;GdnColumnsScope::set(mode? -1:0,key);
            output=gdn_rms_norm_gate_cuda(attended,gate,weight,1e-6,true,true);mfq_cuda_synchronize();
            TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,mfq_current_cuda_stream()));
            for(int repeat=0;repeat<16;++repeat)output=gdn_rms_norm_gate_cuda(attended,gate,weight,1e-6,true,true);
            TYLOINFER_CUDA_CHECK(cudaEventRecord(end,mfq_current_cuda_stream()));
            TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
            float ms=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));elapsed[mode].push_back(ms/16);
        }
        for(auto& e:elapsed)std::sort(e.begin(),e.end());
        std::cout<<"gdn_output_warp_bench tokens="<<tokens<<" old_ms="<<elapsed[0][2]<<" warp_ms="<<elapsed[1][2]<<std::endl;
        cudaEventDestroy(begin);cudaEventDestroy(end);
    }
    std::cout<<"GDN row-warp norm bitwise original-tree cases="<<cases<<" PASS"<<std::endl;
}
