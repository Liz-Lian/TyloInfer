#pragma once
#include "../kernels/gguf.h"
inline void gdn_replay_batch_benchmark(int layers) {
    if(layers<1)throw std::invalid_argument("layer count must be positive");
    constexpr int nk=16,nv=48,width=128,tokens=4,channels=(2*nk+nv)*width;
    const auto stream=mfq_current_cuda_stream();
    for(auto dtype:{tb::kFloat16,tb::kFloat32}) {
        auto qkv=data({1,tokens,channels},81,.3f).to(dtype),alpha=data({1,tokens,nv},82,.2f).to(dtype),beta=data({1,tokens,nv},83,.4f).to(dtype);
        auto initial_state=data({1,nv,width,width},84,.2f),initial_conv=data({1,3,channels},85,.3f);
        auto weight=data({channels,1,4},86,.1f),bias=data({nv},87,.1f),a=data({nv},88,.2f);
        auto gate=tb::zeros({1,tokens,nv*width},qkv.options()),norm=tb::ones({width},initial_state.options());
        std::vector<Tensor> state,conv,next;std::vector<GdnReplayLayer> meta;
        for(int i=0;i<layers;++i) {
            state.push_back(initial_state.clone());conv.push_back(initial_conv.clone());next.push_back(tb::empty_like(initial_conv));
            meta.push_back({qkv.data_ptr(),alpha.data_ptr(),beta.data_ptr(),conv.back().data_ptr<float>(),state.back().data_ptr<float>(),next.back().data_ptr<float>(),
                weight.data_ptr<float>(),bias.data_ptr<float>(),a.data_ptr<float>(),qkv.stride(1),alpha.stride(1),alpha.stride(2),beta.stride(1),beta.stride(2),
                nk,nv,channels,tokens,0,1,1,int(dtype==tb::kFloat32),int(dtype==tb::kFloat32),1e-6f});
        }
        auto host=tb::empty({int64_t(meta.size()*sizeof(GdnReplayLayer))},initial_conv.options().device(tb::kCPU).dtype(tb::kUInt8));
        std::memcpy(host.data_ptr(),meta.data(),meta.size()*sizeof(GdnReplayLayer));auto descriptors=host.to(tb::kCUDA);
        const auto reset=[&]{for(int i=0;i<layers;++i){state[i].copy_(initial_state);conv[i].copy_(initial_conv);}};
        for(int keep=1;keep<=tokens;++keep) {
            cudaGraph_t graphs[2]{};cudaGraphExec_t exec[2]{};
            for(int kind=0;kind<2;++kind) {
                TYLOINFER_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
                if(kind)gdn_replay_commit_layers_cuda(descriptors,layers,nv,channels,keep);
                else for(int i=0;i<layers;++i)
                    gdn_replay_commit_layers_cuda(descriptors.narrow(0,int64_t(i)*sizeof(GdnReplayLayer),sizeof(GdnReplayLayer)),1,nv,channels,keep);
                TYLOINFER_CUDA_CHECK(cudaStreamEndCapture(stream,&graphs[kind]));
                TYLOINFER_CUDA_CHECK(cudaGraphInstantiate(&exec[kind],graphs[kind],0));
            }
            reset();
            // The unpartitioned typed kernel is independent of the batched
            // column tiles. Compare every accepted prefix before timing.
            const auto reference=gdn_replay_window_cuda(qkv.narrow(1,0,keep),gate.narrow(1,0,keep),
                alpha.narrow(1,0,keep),beta.narrow(1,0,keep),conv[0],state[0],weight,bias,a,norm,
                nk,nv,1e-6,1e-6,false,dtype==tb::kFloat16,true,true,true);
            const auto expected_state=state[0].cpu(),expected_conv=reference[1].cpu();
            reset();TYLOINFER_CUDA_CHECK(cudaGraphLaunch(exec[1],stream));
            for(int i=0;i<layers;++i) {
                const auto actual_state=state[i].cpu(),actual_conv=conv[i].cpu();
                if(std::memcmp(expected_state.data_ptr(),actual_state.data_ptr(),expected_state.nbytes()) ||
                        std::memcmp(expected_conv.data_ptr(),actual_conv.data_ptr(),expected_conv.nbytes()))
                    throw std::runtime_error("GDN cross-layer commit changed a state or convolution history");
            }
            float median[2]{};
            for(int kind=0;kind<2;++kind) {
                for(int warm=0;warm<8;++warm)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(exec[kind],stream));
                std::vector<float> samples(5);
                for(auto& sample:samples) {
                    cudaEvent_t begin,end;TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));for(int r=0;r<100;++r)TYLOINFER_CUDA_CHECK(cudaGraphLaunch(exec[kind],stream));
                    TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
                    TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&sample,begin,end));sample*=10;cudaEventDestroy(begin);cudaEventDestroy(end);
                }
                std::sort(samples.begin(),samples.end());median[kind]=samples[2];
            }
            std::cout<<"gdn_commit_layers layers="<<layers<<" keep="<<keep<<" float="<<int(dtype==tb::kFloat32)
                <<" serial_us="<<median[0]<<" batch_us="<<median[1]<<" exact=1\n";
            for(int kind=0;kind<2;++kind){cudaGraphExecDestroy(exec[kind]);cudaGraphDestroy(graphs[kind]);}
        }
    }
}
inline void gdn_recorded_update_benchmark(int layers) {
    if(layers<1)throw std::invalid_argument("layer count must be positive");
    constexpr int nk=16,nv=48,width=128,channels=(2*nk+nv)*width;
    const auto context=mfq::cuda::default_context(mfq_current_cuda_device());const auto stream=mfq_current_cuda_stream();
    for(auto dtype:{tb::kFloat16,tb::kFloat32})for(int tokens:{2,3,4}) {
        auto qkv=data({1,tokens,channels},81,.3f).to(dtype),alpha=data({1,tokens,nv},82,.2f).to(dtype),beta=data({1,tokens,nv},83,.4f).to(dtype);
        auto initial_state=data({1,nv,width,width},84,.2f),initial_conv=data({1,3,channels},85,.3f);
        auto weight=data({channels,1,4},86,.1f),bias=data({nv},87,.1f),a=data({nv},88,.2f);
        auto gate=tb::zeros({1,tokens,nv*width},qkv.options()),norm=tb::ones({width},initial_state.options());
        for(int keep:{1,tokens}) {
            std::vector<Tensor> state[2],conv[2],next[2],initial_records;
            std::vector<std::vector<Tensor>> owned[2];std::vector<GdnReplayLayer> meta[2];
            Tensor host[2],descriptors[2];
            std::vector<Tensor> old_quantized;
            for(int i=0;i<layers;++i)old_quantized.push_back(tb::empty({int64_t(tokens*mfq::cuda::gguf::q8_row_bytes(nv*width))},qkv.options().dtype(tb::kUInt8)));
            for(int mode=0;mode<2;++mode) {
                owned[mode].resize(layers);
                for(int i=0;i<layers;++i) {
                    state[mode].push_back(initial_state.clone());conv[mode].push_back(initial_conv.clone());next[mode].push_back(tb::empty_like(initial_conv));
                    const float* updates=nullptr;
                    if(mode) {
                        auto record=gdn_replay_window_cuda(qkv,gate,alpha,beta,conv[mode].back(),state[mode].back(),weight,bias,a,norm,
                            nk,nv,1e-6,1e-6,false,dtype==tb::kFloat16,true,false,true,true);
                        updates=record[3].data_ptr<float>();initial_records.push_back(record[3]);
                    }
                    meta[mode].push_back({qkv.data_ptr(),alpha.data_ptr(),beta.data_ptr(),conv[mode].back().data_ptr<float>(),state[mode].back().data_ptr<float>(),next[mode].back().data_ptr<float>(),
                        weight.data_ptr<float>(),bias.data_ptr<float>(),a.data_ptr<float>(),qkv.stride(1),alpha.stride(1),alpha.stride(2),beta.stride(1),beta.stride(2),
                        nk,nv,channels,tokens,0,1,1,int(dtype==tb::kFloat32),int(dtype==tb::kFloat32),1e-6f,updates});
                }
                host[mode]=tb::empty({int64_t(layers)*int64_t(sizeof(GdnReplayLayer))},initial_conv.options().device(tb::kCPU).dtype(tb::kUInt8));
                std::memcpy(host[mode].data_ptr(),meta[mode].data(),host[mode].nbytes());descriptors[mode]=host[mode].to(tb::kCUDA);
            }
            const auto reset=[&](int mode){for(int i=0;i<layers;++i){state[mode][i].copy_(initial_state);conv[mode][i].copy_(initial_conv);}};
            const auto execute=[&](int mode) {
                for(int i=0;i<layers;++i) {
                    owned[mode][i]=gdn_replay_window_cuda(qkv,gate,alpha,beta,conv[mode][i],state[mode][i],weight,bias,a,norm,
                        nk,nv,1e-6,1e-6,false,dtype==tb::kFloat16,true,false,true,mode!=0,mode!=0);
                    if(!mode) {
                        if(dtype==tb::kFloat16)mfq::cuda::gguf::quantize_input_f16(owned[0][i][0].data_ptr(),tokens,nv*width,old_quantized[i].data_ptr(),stream);
                        else mfq::cuda::gguf::quantize_input(owned[0][i][0].data_ptr<float>(),tokens,nv*width,old_quantized[i].data_ptr(),stream);
                    }
                }
                gdn_replay_commit_layers_cuda(descriptors[mode],layers,nv,channels,keep,mode!=0);
            };
            mfq::cuda::Graph graphs[2];
            {mfq::cuda::GraphWarmupScope warm(context,stream);execute(0);mfq_cuda_synchronize();owned[0].assign(layers,{});reset(0);}
            graphs[0].capture_begin_shared();execute(0);graphs[0].capture_end();
            {mfq::cuda::GraphWarmupScope warm(context,stream);execute(1);mfq_cuda_synchronize();owned[1].assign(layers,{});reset(1);}
            graphs[1].capture_begin_shared();execute(1);graphs[1].capture_end();
            // A captured record owns its actual output buffer. Bind that
            // address after capture; device descriptors are read on replay.
            for(int i=0;i<layers;++i)meta[1][i].updates=owned[1][i][3].data_ptr<float>();
            std::memcpy(host[1].data_ptr(),meta[1].data(),host[1].nbytes());descriptors[1].copy_(host[1]);
            reset(0);reset(1);graphs[0].replay();graphs[1].replay();mfq_cuda_synchronize();
            for(int i=0;i<layers;++i)for(const auto& pair:{std::pair{state[0][i],state[1][i]},std::pair{conv[0][i],conv[1][i]},std::pair{old_quantized[i],owned[1][i][0]}}) {
                auto expected=pair.first.cpu(),actual=pair.second.cpu();
                if(std::memcmp(expected.data_ptr(),actual.data_ptr(),expected.nbytes()))throw std::runtime_error("GDN recorded-update graph differs from legacy replay");
            }
            std::vector<float> samples[2];cudaEvent_t begin,end;
            TYLOINFER_CUDA_CHECK(cudaEventCreate(&begin));TYLOINFER_CUDA_CHECK(cudaEventCreate(&end));
            for(int trial=0;trial<5;++trial)for(int pass=0;pass<2;++pass) {
                const int mode=(trial+pass)%2;reset(mode);mfq_cuda_synchronize();
                TYLOINFER_CUDA_CHECK(cudaEventRecord(begin,stream));for(int r=0;r<100;++r)graphs[mode].replay();
                TYLOINFER_CUDA_CHECK(cudaEventRecord(end,stream));TYLOINFER_CUDA_CHECK(cudaEventSynchronize(end));
                float ms;TYLOINFER_CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));samples[mode].push_back(ms*10);
            }
            for(auto& sample:samples)std::sort(sample.begin(),sample.end());
            std::cout<<"gdn_recorded_updates layers="<<layers<<" tokens="<<tokens<<" keep="<<keep<<" float="<<int(dtype==tb::kFloat32)
                <<" legacy_read_quant_commit_us="<<samples[0][2]<<" recorded_read_quant_commit_us="<<samples[1][2]
                <<" old_nodes="<<graphs[0].nodes()<<" new_nodes="<<graphs[1].nodes()<<" exact=1 quantized_output=1\n";
            cudaEventDestroy(begin);cudaEventDestroy(end);
        }
    }
}
inline void gdn_replay_checkpoint_case() {
    using namespace mfq::cuda;
    using namespace mfq::cuda::qwen4_exp;
    const auto data=[](std::vector<int64_t> shape,int salt,float scale=.03f){return ::data(std::move(shape),salt,scale).to(tb::kFloat16);};
    const auto linear=[&](int in,int out,int salt) -> Linear {
        auto w=data({out,in},salt);
        return [w](CudaExecutionContext&,const Tensor& x){return tb::matmul(x.to(w.scalar_type()),w.transpose(-1,-2));};
    };
    struct ReplaySetting {
        bool present=std::getenv("TYLOINFER_MTP_GDN_REPLAY")!=nullptr;
        std::string old=present?std::getenv("TYLOINFER_MTP_GDN_REPLAY"):"";
        static void set(const char* value) {
#ifdef _WIN32
            _putenv_s("TYLOINFER_MTP_GDN_REPLAY",value?value:"");
#else
            if(value)setenv("TYLOINFER_MTP_GDN_REPLAY",value,1);else unsetenv("TYLOINFER_MTP_GDN_REPLAY");
#endif
        }
        ReplaySetting(){set("1");}~ReplaySetting(){set(present?old.c_str():nullptr);}
    } setting;
    CudaExecutionContext execution;
    const auto bits=[](const Tensor& a,const Tensor& b,const char* label) {
        const auto x=a.contiguous().cpu(),y=b.contiguous().cpu();
        if(x.sizes()!=y.sizes() || x.scalar_type()!=y.scalar_type() || std::memcmp(x.data_ptr(),y.data_ptr(),x.nbytes()))
            throw std::runtime_error(std::string(label)+" changed bytes");
    };
    int checked=0;
    for(bool tiled:{false,true})for(int tokens:{2,3,4}) {
        const int hidden=32,nk=16,nv=48,width=128,channels=(2*nk+nv)*width;
        const auto gpu=tb::TensorOptions().device(tb::kCUDA).dtype(tb::kFloat32);
        GdnWeights weights{linear(hidden,channels,11),linear(hidden,nv*width,12),linear(hidden,nv,13),linear(hidden,nv,14),
            linear(nv*width,hidden,15),data({channels,4},16),tb::zeros({nv},gpu),tb::zeros({nv},gpu),tb::ones({width},gpu)};
        const auto separate=weights;
        weights.input_projection=[separate](CudaExecutionContext& e,const Tensor& x) {
            return std::vector<Tensor>{separate.qkv(e,x),separate.gate(e,x),separate.alpha(e,x),separate.beta(e,x)};
        };
        Gdn actual(weights,nk,nv,width,4,1e-6,false,true,true,true,true,true,true,tiled);
        Gdn reference(weights,nk,nv,width,4,1e-6,false,true,true,true,true,true,true,tiled);
        auto history=data({1,3,hidden},17,.4f),input=data({1,tokens,hidden},18,.3f);
        actual.forward_chunk(execution,history,true);
        auto initial_conv=actual.conv_state().clone(),initial_state=actual.graph_state()[1]->clone();
        reference.restore_session(initial_conv,initial_state);
        std::vector<Tensor> expected_conv,expected_state;
        for(int i=1;i<tokens;++i){expected_conv.push_back(tb::empty_like(initial_conv));expected_state.push_back(tb::empty_like(initial_state));}
        auto expected=reference.forward_graph(execution,input,1,expected_conv,expected_state).clone();
        std::vector<Tensor> empty_conv,empty_state,inputs;Tensor output;
        DecodeWindow graph(mfq_current_cuda_stream());
        graph.capture([&]{output=actual.forward_graph(execution,input,1,empty_conv,empty_state,&inputs);},[&]{
            actual.graph_state()[0]->copy_(initial_conv);actual.graph_state()[1]->copy_(initial_state);
        },[&]{output={};});
        for(int keep=1;keep<=tokens;++keep) {
            graph.run();bits(output,expected,"GDN read-only projected output");
            bits(actual.conv_state(),initial_conv,"GDN read-only conv");bits(*actual.graph_state()[1],initial_state,"GDN read-only recurrence");
            actual.bind_graph_replay(inputs,17);
            if(keep==tokens)actual.commit();else actual.rollback(17+keep);
            bits(actual.conv_state(),keep==tokens?reference.conv_state():expected_conv[keep-1],"GDN replay committed conv");
            bits(*actual.graph_state()[1],keep==tokens?*reference.graph_state()[1]:expected_state[keep-1],"GDN replay committed recurrence");
            actual.restore_session(initial_conv,initial_state,true);++checked;
        }
        auto eager=actual.forward(execution,input,true,1,17);
        bits(eager,expected,"GDN eager replay projected output");
        actual.rollback(18);
        bits(actual.conv_state(),expected_conv[0],"GDN eager rejected prefix conv");
        bits(*actual.graph_state()[1],expected_state[0],"GDN eager rejected prefix recurrence");
        actual.restore_session(initial_conv,initial_state,true);
        if(tiled && tokens==4) {
            auto second_weights=weights;
            second_weights.input_projection=[separate](CudaExecutionContext& e,const Tensor& x) {
                return std::vector<Tensor>{separate.qkv(e,x).to(tb::kFloat32),separate.gate(e,x).to(tb::kFloat32),
                    separate.alpha(e,x).to(tb::kFloat32),separate.beta(e,x).to(tb::kFloat32)};
            };
            Gdn second(second_weights,nk,nv,width,4,1e-6,false,true,true,true,true,true,true,tiled);
            Gdn second_reference(second_weights,nk,nv,width,4,1e-6,false,true,true,true,true,true,true,tiled);
            second.restore_session(initial_conv,initial_state);second_reference.restore_session(initial_conv,initial_state);
            auto other_input=input*.7;
            std::vector<Tensor> other_conv,other_state;
            for(int i=1;i<tokens;++i){other_conv.push_back(tb::empty_like(initial_conv));other_state.push_back(tb::empty_like(initial_state));}
            (void)second_reference.forward_graph(execution,other_input,1,other_conv,other_state);
            std::vector<Tensor> second_inputs;
            (void)second.forward_graph(execution,other_input,1,empty_conv,empty_state,&second_inputs);
            graph.run();
            struct CommitGroup final:GdnReplayCommitGroup {
                Tensor meta;int heads,channels,launches=0,kept=-1;
                CommitGroup(Tensor value,int maximum_heads,int maximum_channels):meta(std::move(value)),heads(maximum_heads),channels(maximum_channels){}
                void commit(int64_t keep) override {
                    if(kept>=0){if(kept!=keep)throw std::runtime_error("layer replay verdict changed");return;}
                    gdn_replay_commit_layers_cuda(meta,2,heads,channels,int(keep),true);++launches;kept=int(keep);
                }
            };
            std::vector<Tensor> retained;Tensor scratch[2];
            GdnReplayLayer layers[]{actual.replay_layer(inputs,scratch[0],retained),second.replay_layer(second_inputs,scratch[1],retained)};
            if(!layers[0].updates || !layers[1].updates)throw std::runtime_error("GDN model did not bind recorded updates");
            auto host=tb::empty({int64_t(sizeof(layers))},gpu.device(tb::kCPU).dtype(tb::kUInt8));
            std::memcpy(host.data_ptr(),layers,sizeof(layers));auto group=std::make_shared<CommitGroup>(host.to(tb::kCUDA),nv+1,channels);
            actual.bind_graph_replay(inputs,17,group);second.bind_graph_replay(second_inputs,17,group);
            actual.rollback(19);second.rollback(19);
            bits(actual.conv_state(),expected_conv[1],"batched GDN half convolution");
            bits(*actual.graph_state()[1],expected_state[1],"batched GDN half state");
            bits(second.conv_state(),other_conv[1],"batched GDN float convolution");
            bits(*second.graph_state()[1],other_state[1],"batched GDN float state");
            if(group->launches!=1)throw std::runtime_error("GDN layer group committed more than once");
            actual.restore_session(initial_conv,initial_state,true);
        }
        (void)actual.forward(execution,input,true,1,17);actual.reset(true);
        if(actual.conv_state().abs().max().item<float>()!=0 || actual.graph_state()[1]->abs().max().item<float>()!=0)
            throw std::runtime_error("GDN reset replayed a discarded speculative window");
    }
    std::cout<<"GDN read-only/replay projection + graph/eager commit/partial reject/reset checks="<<checked<<" PASS\n";
}
