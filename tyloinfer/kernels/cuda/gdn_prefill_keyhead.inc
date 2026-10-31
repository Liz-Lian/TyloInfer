// Strata gdn_rec_kh_kernel (MIT), adapted to native separate Q/K/V arrays,
// arbitrary head groups/batches and either recurrent-state layout.
template<int Values,bool Transposed,bool Tiled,bool Prepared,bool FullHeads=false>
__global__ __launch_bounds__(128) void gdn_prefill_keyhead_kernel(
    const float* __restrict__ q,const float* __restrict__ k,const float* __restrict__ v,
    const float* __restrict__ gate,const float* __restrict__ beta,const float* __restrict__ input_state,
    float* __restrict__ output,float* __restrict__ output_state,int Hq,int Hv,int T) {
    constexpr int D=128,CB=32,RG=4,RPG=D/RG,TB=8,NT=CB*RG;
    __shared__ __align__(16) float sq[2][TB][D],sk[2][TB][D],sv[2][TB][Values][CB];
    __shared__ float sg[2][TB][Values],sb[2][TB][Values],rkv[Values][RG][CB],ro[Values][RG][CB];
    const int ratio=Hv/Hq,groups=(ratio+Values-1)/Values;
    const int group=blockIdx.x%groups,bh=blockIdx.x/groups,hq=bh%Hq,b=bh/Hq;
    const int c=threadIdx.x,rg=threadIdx.y,tid=rg*CB+c,col=blockIdx.y*CB+c;
    int head[Values];float state[Values][RPG];
    #pragma unroll
    for(int j=0;j<Values;++j) {
        const int vi=group*Values+j;
        head[j]=(FullHeads || vi<ratio)?(Tiled?hq+vi*Hq:hq*ratio+vi):-1;
        #pragma unroll
        for(int r=0;r<RPG;++r) {
            const int row=rg*RPG+r;
            const auto index=std::size_t(b*Hv+head[j])*D*D+(Transposed?col*D+row:row*D+col);
            state[j][r]=(FullHeads || head[j]>=0) && input_state?input_state[index]:0.f;
        }
    }
    const auto stage=[&](int tile) {
        const int bb=tile&1,t0=tile*TB;
        for(int p=tid;p<TB*2*(D/4);p+=NT) {
            const int i=p/(2*(D/4)),w=p%(2*(D/4)),key=w/(D/4),row=w%(D/4)*4;
            if(t0+i<T)gdn_prefill_cp16(key?&sk[bb][i][row]:&sq[bb][i][row],
                (key?k:q)+(std::size_t(b*Hq+hq)*T+t0+i)*D+row);
        }
        for(int p=tid;p<TB*Values*(CB/4);p+=NT) {
            const int i=p/(Values*(CB/4)),j=p/(CB/4)%Values,cc=p%(CB/4)*4;
            if(t0+i<T && (FullHeads || head[j]>=0))gdn_prefill_cp16(&sv[bb][i][j][cc],
                v+(std::size_t(b*Hv+head[j])*T+t0+i)*D+blockIdx.y*CB+cc);
        }
        for(int p=tid;p<2*TB*Values;p+=NT) {
            const int is_beta=p/(TB*Values),i=p/Values%TB,j=p%Values;
            if(t0+i<T && (FullHeads || head[j]>=0))gdn_prefill_cp4(is_beta?&sb[bb][i][j]:&sg[bb][i][j],
                (is_beta?beta:gate)+std::size_t(b*Hv+head[j])*T+t0+i);
        }
    };
    const int tiles=(T+TB-1)/TB;
    if(tiles)stage(0);gdn_prefill_cp_commit();
    for(int tile=0;tile<tiles;++tile) {
        if(tile+1<tiles)stage(tile+1);
        gdn_prefill_cp_commit();gdn_prefill_cp_wait();__syncthreads();
        const int bb=tile&1,count=min(TB,T-tile*TB);
        for(int i=0;i<count;++i) {
            float key[RPG],decay[Values],kv[Values]={},delta[Values],out[Values]={};
            #pragma unroll
            for(int r=0;r<RPG;++r)key[r]=sk[bb][i][rg*RPG+r];
            #pragma unroll
            for(int j=0;j<Values;++j)decay[j]=(!FullHeads && head[j]<0)?0.f:Prepared?sg[bb][i][j]:expf(sg[bb][i][j]);
            #pragma unroll
            for(int r=0;r<RPG;++r)
                #pragma unroll
                for(int j=0;j<Values;++j)kv[j]=fmaf(state[j][r],key[r],kv[j]);
            #pragma unroll
            for(int j=0;j<Values;++j)rkv[j][rg][c]=kv[j];
            __syncthreads();
            #pragma unroll
            for(int j=0;j<Values;++j) {
                const float dot=rkv[j][0][c]+rkv[j][1][c]+rkv[j][2][c]+rkv[j][3][c];
                delta[j]=(!FullHeads && head[j]<0)?0.f:(sv[bb][i][j][c]-decay[j]*dot)*sb[bb][i][j];
            }
            #pragma unroll
            for(int r=0;r<RPG;++r) {
                const float query=sq[bb][i][rg*RPG+r];
                #pragma unroll
                for(int j=0;j<Values;++j) {
                    state[j][r]=fmaf(decay[j],state[j][r],key[r]*delta[j]);
                    out[j]=fmaf(state[j][r],query,out[j]);
                }
            }
            #pragma unroll
            for(int j=0;j<Values;++j)ro[j][rg][c]=out[j];
            __syncthreads();
            #pragma unroll
            for(int j=0;j<Values;++j)if(rg==j && (FullHeads || head[j]>=0))
                output[(std::size_t(b*Hv+head[j])*T+tile*TB+i)*D+col]=
                    (ro[j][0][c]+ro[j][1][c]+ro[j][2][c]+ro[j][3][c])*(1.f/sqrtf(float(D)));
        }
    }
    #pragma unroll
    for(int j=0;j<Values;++j)if(FullHeads || head[j]>=0) {
        #pragma unroll
        for(int r=0;r<RPG;++r) {
            const int row=rg*RPG+r;
            output_state[std::size_t(b*Hv+head[j])*D*D+(Transposed?col*D+row:row*D+col)]=state[j][r];
        }
    }
}

template<bool Transposed,bool Tiled,bool Prepared>
bool launch_gdn_prefill_keyhead(int B,int Hq,int Hv,int T,const float* q,const float* k,const float* v,
        const float* g,const float* beta,const float* state,float* out,float* next,cudaStream_t stream) {
    const auto* setting=std::getenv("TYLOINFER_GDN_PREFILL_KEYHEAD");
    if((setting && setting[0]=='0') || T<128)return false;
    static thread_local int cached_device=-1,cached_major=0,cached_sms=0;
    const int device=mfq_current_cuda_device();
    if(cached_device!=device) {
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&cached_major,cudaDevAttrComputeCapabilityMajor,device));
        TYLOINFER_CUDA_CHECK(cudaDeviceGetAttribute(&cached_sms,cudaDevAttrMultiProcessorCount,device));
        cached_device=device;
    }
    if(cached_major<8)return false; // cp.async begins at compute capability 8.0.
    const auto launch=[&]<int Values>() {
        const int groups=(Hv/Hq+Values-1)/Values;
        // Complete groups need no per-head predicates in the recurrent loop.
        // This is derived from the actual ratio; incomplete groups retain masks.
        auto kernel=(Hv/Hq)%Values==0
            ?gdn_prefill_keyhead_kernel<Values,Transposed,Tiled,Prepared,true>
            :gdn_prefill_keyhead_kernel<Values,Transposed,Tiled,Prepared,false>;
        int resident=0;TYLOINFER_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,kernel,128,0));
        if(resident<=0)return false;
        kernel<<<dim3(B*Hq*groups,4),dim3(32,4),0,stream>>>(q,k,v,g,beta,state,out,next,Hq,Hv,T);
        if(const auto* trace=std::getenv("TYLOINFER_TRACE_KERNEL_DISPATCH");trace && trace[0]=='1') {
            static std::atomic<bool> reported{false};
            if(!reported.exchange(true)) {
                cudaFuncAttributes attributes{};TYLOINFER_CUDA_CHECK(cudaFuncGetAttributes(&attributes,kernel));
                std::fprintf(stderr,"native_gdn_prefill_keyhead actual_match=1 values_per_block=%d heads=%d tokens=%d registers=%d local_bytes=%zu blocks=%lld sms=%d\n",
                    Values,Hv,T,attributes.numRegs,attributes.localSizeBytes,4ll*B*Hq*groups,cached_sms);
            }
        }
        return true;
    };
    const int ratio=Hv/Hq;
    int values=std::min(ratio,4);
    // Sharing Q/K across heads is useful only if its grid can fill the device.
    // Split an undersized grid into complete smaller groups. This derives from
    // the actual head/batch geometry and SM count, without a device whitelist.
    while(values>1 && 4ll*B*Hq*((ratio+values-1)/values)<cached_sms) {
        do {--values;}while(values>1 && ratio%values!=0);
    }
    switch(values) {
        case 1:return launch.template operator()<1>();
        case 2:return launch.template operator()<2>();
        case 3:return launch.template operator()<3>();
        default:return launch.template operator()<4>();
    }
}
