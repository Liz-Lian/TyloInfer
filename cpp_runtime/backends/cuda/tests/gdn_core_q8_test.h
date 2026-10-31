#pragma once
#include "../kernels/gguf.h"

inline void gdn_core_q8_case() {
    const auto set_tiles=[](const char* value) {
#ifdef _WIN32
        _putenv_s("TYLOINFER_GDN_DECODE_TILES",value?value:"");
#else
        if(value)setenv("TYLOINFER_GDN_DECODE_TILES",value,1);
        else unsetenv("TYLOINFER_GDN_DECODE_TILES");
#endif
    };
    const auto* old_tiles=std::getenv("TYLOINFER_GDN_DECODE_TILES");
    const std::optional<std::string> saved_tiles=old_tiles?std::optional<std::string>(old_tiles):std::nullopt;
    struct RestoreTiles {
        const decltype(set_tiles)& set;const std::optional<std::string>& saved;
        ~RestoreTiles(){set(saved?saved->c_str():nullptr);}
    } restore_tiles{set_tiles,saved_tiles};
    const auto bits=[](const Tensor& a,const Tensor& b,const char* label) {
        const auto x=a.cpu().contiguous(),y=b.cpu().contiguous();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() ||
                std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))throw std::runtime_error(label);
    };
    int checked=0;
    for(const char* tiles:{"1","2","4"})for(int width:{32,64,128})
    for(bool half:{false,true})for(bool output_half:{false,true})
    for(bool transpose:{false,true})for(bool snapshot:{false,true}) {
        set_tiles(tiles);
        const int batch=width==64?2:1,nk=2,nv=6,channels=(2*nk+nv)*width;
        const bool tiled=half,silu=output_half,inplace=*tiles!='1';
        const auto dtype=half?tb::kFloat16:tb::kFloat32;
        auto qkv=data({batch,1,channels+17},491,.4f).to(dtype).narrow(2,0,channels);
        auto gate=data({batch,1,nv*width+13},492,2.f).to(half?tb::kFloat32:tb::kFloat16).narrow(2,0,nv*width);
        auto alpha=data({batch,1,nv+3},493,.3f).to(gate.scalar_type()).narrow(2,0,nv);
        auto beta=data({batch,1,nv+3},494,.3f).to(gate.scalar_type()).narrow(2,0,nv);
        auto initial_conv=data({batch,3,channels},495,.3f),initial_state=data({batch,nv,width,width},496,.1f);
        auto conv=initial_conv.clone(),state=initial_state.clone();
        auto reference_conv=initial_conv.clone(),reference_state=initial_state.clone();
        auto weight=data({channels,1,4},497,.2f),bias=data({nv},498,.1f),a=data({nv},499,.2f),norm=data({width},500,1.3f);
        auto workspace=tb::zeros({batch,nv,width+1},state.options()),reference_workspace=tb::zeros_like(workspace);
        auto saved_conv=tb::empty_like(conv),saved_state=tb::empty_like(state);
        auto reference_saved_conv=tb::empty_like(conv),reference_saved_state=tb::empty_like(state);
        const auto decode=[&](bool prepared) {
            auto c=prepared?conv:reference_conv,s=prepared?state:reference_state;
            auto w=prepared?workspace:reference_workspace;
            if(snapshot)return gdn_decode_core_checkpoint_cuda(qkv,gate,alpha,beta,c,s,weight,bias,a,norm,
                nk,nv,width,1e-6,.013456789123,silu,output_half,transpose,w,inplace,
                prepared?saved_conv:reference_saved_conv,prepared?saved_state:reference_saved_state,tiled,prepared);
            return gdn_decode_core_cuda(qkv,gate,alpha,beta,c,s,weight,bias,a,norm,
                nk,nv,width,1e-6,.013456789123,silu,output_half,transpose,w,inplace,{},tiled,prepared);
        };
        const auto reset=[&] {
            conv.copy_(initial_conv);state.copy_(initial_state);workspace.zero_();
            reference_conv.copy_(initial_conv);reference_state.copy_(initial_state);reference_workspace.zero_();
        };
        std::vector<Tensor> actual;DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&]{actual=decode(true);},reset,[&]{actual.clear();});
        const int repeats=width==128 && *tiles=='4'?3:1;
        for(int replay=0;replay<repeats;++replay) {
            reset();qkv.copy_(data(qkv.sizes().vec(),501+replay,.3f).to(dtype));
            const auto expected=decode(false);
            auto prepared=tb::empty({int64_t(batch*mfq::cuda::gguf::q8_row_bytes(nv*width))},qkv.options().dtype(tb::kUInt8));
            if(output_half)mfq::cuda::gguf::quantize_input_f16(expected[0].data_ptr(),batch,nv*width,prepared.data_ptr(),mfq_current_cuda_stream());
            else mfq::cuda::gguf::quantize_input(expected[0].data_ptr<float>(),batch,nv*width,prepared.data_ptr(),mfq_current_cuda_stream());
            graph.run();
            bits(actual[0],prepared,"GDN core prepared output changed Q8 bytes");
            bits(actual[1],expected[1],"GDN core prepared output changed convolution");
            bits(actual[2],expected[2],"GDN core prepared output changed recurrence");
            if(snapshot) {
                bits(saved_conv,reference_saved_conv,"GDN prepared output changed convolution snapshot");
                bits(saved_state,reference_saved_state,"GDN prepared output changed recurrent snapshot");
            }
            ++checked;
        }
    }
    std::cout<<"GDN core prepared output Q8/state/snapshot changing-graph bitwise cases="<<checked<<" PASS\n";
}

inline void gdn_core_q8_benchmark(int layers) {
    if(layers<1)throw std::invalid_argument("positive layer count required");
    constexpr int nk=16,nv=48,width=128,channels=(2*nk+nv)*width;
    const auto stream=mfq_current_cuda_stream();
    for(bool half:{false,true}) {
        const auto dtype=half?tb::kFloat16:tb::kFloat32;
        auto qkv=data({1,1,channels},521,.3f).to(dtype),gate=data({1,1,nv*width},522,.3f).to(dtype);
        auto alpha=data({1,1,nv},523,.3f).to(dtype),beta=data({1,1,nv},524,.3f).to(dtype);
        auto weight=data({channels,1,4},525,.2f),bias=data({nv},526,.1f),a=data({nv},527,.2f),norm=data({width},528,1.1f);
        auto initial_state=data({1,nv,width,width},529,.1f),initial_conv=data({1,3,channels},530,.2f);
        struct Layer {Tensor state,conv,workspace;std::vector<Tensor> decoded;Tensor prepared;};
        std::vector<Layer> work[2];
        for(auto& group:work)for(int i=0;i<layers;++i)group.push_back({initial_state.clone(),initial_conv.clone(),
            tb::zeros({1,nv,width+1},initial_state.options()),{},
            tb::empty({int64_t(mfq::cuda::gguf::q8_row_bytes(nv*width))},qkv.options().dtype(tb::kUInt8))});
        const auto reset=[&]{for(auto& group:work)for(auto& l:group) {
            l.state.copy_(initial_state);l.conv.copy_(initial_conv);l.workspace.zero_();
        }};
        DecodeWindow old_graph(stream),new_graph(stream);
        const auto capture=[&](DecodeWindow& graph,int mode) {
            graph.capture([&] {
                for(auto& l:work[mode]) {
                    l.decoded=gdn_decode_core_cuda(qkv,gate,alpha,beta,l.conv,l.state,weight,bias,a,norm,
                        nk,nv,width,1e-6,1e-6,true,half,true,l.workspace,true,{},true,mode==1);
                    if(mode)l.prepared=l.decoded[0];
                    else if(half)mfq::cuda::gguf::quantize_input_f16(l.decoded[0].data_ptr(),1,nv*width,l.prepared.data_ptr(),stream);
                    else mfq::cuda::gguf::quantize_input(l.decoded[0].data_ptr<float>(),1,nv*width,l.prepared.data_ptr(),stream);
                }
            },reset,[&]{for(auto& l:work[mode])l.decoded.clear();});
        };
        capture(old_graph,0);capture(new_graph,1);reset();old_graph.run();new_graph.run();
        for(int i=0;i<layers;++i) {
            const auto x=work[0][i].prepared.cpu(),y=work[1][i].prepared.cpu();
            if(std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))throw std::runtime_error("GDN prepared timing output differs");
        }
        for(int i=0;i<8;++i){old_graph.run();new_graph.run();}
        cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
        std::vector<float> samples[2];
        for(int trial=0;trial<7;++trial)for(int pass=0;pass<2;++pass) {
            const int mode=(trial+pass)%2;
            TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));
            for(int repeat=0;repeat<64;++repeat)
                if(!(mode?new_graph:old_graph).enqueue_if_device_only())throw std::runtime_error("GDN timing graph has host work");
            TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
            float elapsed=0;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&elapsed,begin,end));samples[mode].push_back(elapsed/64);
        }
        for(auto& times:samples)std::sort(times.begin(),times.end());
        std::printf("gdn_core_q8 layers=%d half=%d old_us=%.6f fused_us=%.6f bitwise=1 median7=1 gpu_queued=1\n",
            layers,int(half),1000.*samples[0][3],1000.*samples[1][3]);
        TYLOINFER_CUDA_CHECK(cudaEventDestroy(begin));TYLOINFER_CUDA_CHECK(cudaEventDestroy(end));
    }
}
