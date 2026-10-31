// Included after the runtime fixtures: test real fixed-slot exchanges without
// joining the preceding admission job before the next captured window.
void run_layer_admission_snapshot() {
    constexpr int experts=11,slots=259;
    const auto offset=(3*experts*sizeof(std::uint64_t)+63)&~std::size_t(63);
    mfq::cuda::HostBuffer host(offset+64,true);
    void* mapped=nullptr;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaHostGetDevicePointer(&mapped,host.data(),0));
    auto* ready=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(host.data())+offset);
    auto* device_ready=reinterpret_cast<std::uint32_t*>(static_cast<std::uint8_t*>(mapped)+offset);
    const auto cpu=tb::TensorOptions().device(tb::kCPU).dtype(tb::kInt64);
    auto ids=tb::empty({slots},cpu);
    for(int i=0;i<slots;++i)ids.data_ptr<std::int64_t>()[i]=i%13-1;
    const auto device_ids=ids.to(tb::kCUDA);
    auto result=tb::empty({3*experts+2},device_ids.options());
    const auto producer=mfq_get_stream_from_pool(false,0);
    const auto stream=mfq_current_cuda_stream();
    auto one=tb::full({1},1,device_ids.options().dtype(tb::kInt32));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(ready,one.data_ptr(),sizeof(std::uint32_t),cudaMemcpyDeviceToHost,producer.stream()));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(producer.stream()));
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    mfq::cuda::gguf::snapshot_routed_addresses(static_cast<const std::uint64_t*>(mapped),
        reinterpret_cast<std::uint64_t*>(result.data_ptr<std::int64_t>()+1),
        device_ids.data_ptr<std::int64_t>(),slots,experts,stream,device_ready);
    TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamEndCapture(stream,&graph));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    for(int replay=0;replay<12;++replay) {
        result.fill_(-77);
        *reinterpret_cast<volatile std::uint32_t*>(ready)=0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphLaunch(executable,stream));
        // Flush a possible WDDM command batch while the old table is held.
        const auto until=std::chrono::steady_clock::now()+std::chrono::milliseconds(1);
        while(std::chrono::steady_clock::now()<until) {}
        const auto progress=cudaStreamQuery(stream);
        require(progress==cudaErrorNotReady,"GGUF resident snapshot ignored its closed admission gate");
        auto* table=static_cast<std::uint64_t*>(host.data());
        for(int i=0;i<3*experts;++i)table[i]=std::uint64_t(1000+replay*100+i);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        std::atomic<bool> watchdog_done{false},watchdog_released{false};
        std::thread watchdog([&] {
            const auto until=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            while(!watchdog_done.load() && std::chrono::steady_clock::now()<until)std::this_thread::yield();
            if(!watchdog_done.load()) {
                std::cerr<<"GGUF admission watchdog host_ready="<<*reinterpret_cast<volatile std::uint32_t*>(ready)<<'\n';
                watchdog_released.store(true);mfq::cuda::publish_mapped_flag(ready);
            }
        });
        TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(ready,one.data_ptr(),sizeof(std::uint32_t),cudaMemcpyDeviceToHost,producer.stream()));
        const auto submitted=cudaStreamQuery(producer.stream());
        if(submitted!=cudaErrorNotReady)TYLOINFER_NATIVE_CUDA_CHECK(submitted);
        TYLOINFER_NATIVE_CUDA_CHECK(cudaStreamSynchronize(stream));
        watchdog_done.store(true);watchdog.join();
        require(!watchdog_released.load(),"GGUF admission signal could not progress beside the waiting graph");
        const auto actual=result.cpu();
        require(actual.data_ptr<std::int64_t>()[0]==-77 && actual.data_ptr<std::int64_t>()[3*experts+1]==-77,
            "GGUF gated resident snapshot overwrote a canary");
        for(int i=0;i<3*experts;++i)require(std::uint64_t(actual.data_ptr<std::int64_t>()[i+1])==table[i],
            "GGUF gated resident snapshot consumed a stale table");
    }
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphExecDestroy(executable));
    TYLOINFER_NATIVE_CUDA_CHECK(cudaGraphDestroy(graph));
    std::cout<<"GGUF admission gated snapshot repeated/invalid routes, multi-block capture/replay and canaries PASS\n";
}

void run_layer_admission_windows(bool quick=false) {
    ScopedEnvironment cpu("TYLOINFER_GGUF_CPU_EXPERTS","0");
    ScopedEnvironment slots("TYLOINFER_GGUF_STRATA_CACHE","1");
    ScopedEnvironment asynchronous("TYLOINFER_GGUF_ASYNC_ADMISSION","1");
    int cases=0;
    for(bool enabled:{false,true})for(bool half:{false,true})for(int tokens:{1,2,3,4}) {
        if(quick && !(enabled && half && tokens==4))continue;
        std::cerr<<"GGUF admission check enabled="<<enabled<<" half="<<half<<" rows="<<tokens<<'\n';
        ScopedEnvironment admission("TYLOINFER_GGUF_LAYER_ADMISSION",enabled?"1":"0");
        Source source;
        for(int layer=0;layer<3;++layer)for(const auto* role:{"gate","up","down"})
            source.add(std::string(role)+std::to_string(layer),{8,256,256});
        CudaExecutionContext execution;
        execution.config.moe_host_cache_bytes=1;
        execution.config.moe_device_cache_bytes=6*3*source.find_tensor("gate0")->nbytes/8;
        std::vector<GgufMoeBanks> banks;
        for(int layer=0;layer<3;++layer)banks.push_back({
            std::make_shared<GgufRoutedWeight>(execution,source,"gate"+std::to_string(layer)),
            std::make_shared<GgufRoutedWeight>(execution,source,"up"+std::to_string(layer)),
            std::make_shared<GgufRoutedWeight>(execution,source,"down"+std::to_string(layer))});
        // A reference sharing these fixed slots could repeat an early-read or
        // wrong-slot error. Use separate banks whose budget cannot retain even
        // one image, so the plain projections upload the source bytes again.
        CudaExecutionContext reference_execution;
        reference_execution.config.moe_host_cache_bytes=2ULL<<20;
        reference_execution.config.moe_device_cache_bytes=1;
        std::vector<GgufMoeBanks> reference;
        for(int layer=0;layer<3;++layer)reference.push_back({
            std::make_shared<GgufRoutedWeight>(reference_execution,source,"gate"+std::to_string(layer)),
            std::make_shared<GgufRoutedWeight>(reference_execution,source,"up"+std::to_string(layer)),
            std::make_shared<GgufRoutedWeight>(reference_execution,source,"down"+std::to_string(layer))});
        const auto cache=execution.gguf_expert_cache;
        auto x=input(64,1,false).to(tb::kFloat16);
        auto ids=tb::arange(64,x.options().dtype(tb::kInt64)).remainder(2).reshape({64,1});
        auto weights=tb::full({64,1},.3,x.options().dtype(tb::kFloat32));
        for(auto& bank:banks)(void)bank.forward(x,ids,weights).cpu();
        restore_gguf_decode_workspace(cache);
        require(gguf_prefill_lendable_bytes(cache)>0,"GGUF layer admission fixture did not create a fixed tier");
        prepare_gguf_graph_pool(cache);
        x=x.narrow(0,0,tokens).to(half?tb::kFloat16:tb::kFloat32);ids=ids.narrow(0,0,tokens).contiguous();weights=weights.narrow(0,0,tokens);
        std::array<tb::Tensor,3> outputs;
        std::unique_ptr<mfq::cuda::DecodeWindow> graph;
        const auto capture=[&] {
            graph.reset();
            graph=std::make_unique<mfq::cuda::DecodeWindow>(mfq_current_cuda_stream());
            begin_gguf_speculative_routes(cache,tokens);
            graph->capture([&]{for(int layer=0;layer<3;++layer)outputs[layer]=banks[layer].forward(x,ids,weights);},[]{});
            discard_gguf_speculative_routes(cache);
        };
        capture();
        const auto metric=[&](const char* name) {
            for(const auto& [key,value]:gguf_expert_memory_metrics(cache))if(key==name)return value;
            throw std::runtime_error("GGUF layer admission metric missing");
        };
        for(int step=0;step<(quick?6:20);++step) {
            ids.fill_(2+(step/4)%6);
            begin_gguf_speculative_routes(cache,tokens);graph->run();
            for(int layer=0;layer<3;++layer)check_moe(outputs[layer],reference[layer],x,ids,weights);
            commit_gguf_speculative_routes(cache,1);
            if(quick)std::cerr<<"GGUF admission exchange checked step="<<step<<'\n';
        }
        wait_gguf_expert_admission(cache);
        require((metric("gguf_layer_admission_epochs")>0)==enabled,"GGUF layer admission ignored its default/override");
        if(enabled)require(metric("gguf_layer_admission_signals")>0 && metric("gguf_layer_admission_served")>0 && metric("gguf_layer_admission_ready_updates")>0,
            "GGUF next-window fixture did not exercise layer readiness");
        // Force a source error in an exchange, propagate it through the next
        // graph, then recapture on the same cache after its gates are released.
        bool failed=false;
        for(int step=0;step<24 && !failed;++step) {
            ids.fill_(2+(step/4)%6);
            begin_gguf_speculative_routes(cache,tokens);graph->run();
            source.fail_next_read(true);
            try {
                commit_gguf_speculative_routes(cache,1);
                begin_gguf_speculative_routes(cache,tokens);graph->run();
                wait_gguf_expert_admission(cache);
            }catch(const std::runtime_error& error) {
                failed=std::string(error.what())=="fixture single read failure";
                if(!failed)throw;
                try{wait_gguf_expert_admission(cache);}catch(...){}
            }
            source.clear_read_failure();
            discard_gguf_speculative_routes(cache);
        }
        require(failed,"GGUF fixed-tier source failure fixture did not exchange an expert");
        capture();
        begin_gguf_speculative_routes(cache,tokens);graph->run();
        for(int layer=0;layer<3;++layer)check_moe(outputs[layer],reference[layer],x,ids,weights);
        for(const auto& [key,value]:gguf_expert_memory_metrics(reference_execution.gguf_expert_cache))
            if(key=="resident_expert_cache_bytes")require(value==0,"GGUF admission reference reused resident expert images");
        discard_gguf_speculative_routes(cache);
        ++cases;
    }
    std::cout<<"GGUF layer admission M1..4 f16/f32 three-layer exchanges, override, source failure/retry cases="<<cases<<" PASS\n";
}
