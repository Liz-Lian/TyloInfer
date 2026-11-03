#pragma once

// Host-side cuBLASLt metadata only. Scratch belongs to each caller's tensor
// allocator, so cached plans neither reserve GPU workspace nor retain weights.
struct PrefillLtDevice {
    int device=0;cublasLtHandle_t handle=nullptr;
    ~PrefillLtDevice() {
        int previous=device;const auto status=cudaGetDevice(&previous);
        if(status==cudaSuccess && previous!=device)(void)cudaSetDevice(device);
        if(handle)(void)cublasLtDestroy(handle);
        if(status==cudaSuccess && previous!=device)(void)cudaSetDevice(previous);
    }
};
struct PrefillLtResidual {
    std::shared_ptr<PrefillLtDevice> owner;
    cublasLtMatmulDesc_t operation=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulAlgo_t algorithm{};
    int64_t tokens=0,rows=0,columns=0,stride=0;size_t workspace=0;
    ~PrefillLtResidual() {
        if(a)(void)cublasLtMatrixLayoutDestroy(a);
        if(b)(void)cublasLtMatrixLayoutDestroy(b);
        if(c)(void)cublasLtMatrixLayoutDestroy(c);
        if(operation)(void)cublasLtMatmulDescDestroy(operation);
    }
};
void prefill_lt_checked(cublasStatus_t status) {
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("GGUF residual cuBLASLt failed: "+std::to_string(int(status)));
}
unsigned prefill_lt_alignment(const void* pointer) {
    unsigned alignment=256;
    while(reinterpret_cast<std::uintptr_t>(pointer)%alignment)alignment/=2;
    return alignment;
}
std::shared_ptr<PrefillLtResidual> prefill_lt_residual_plan(const void* input,const void* weight,
        const void* residual,void* output,int64_t tokens,int64_t rows,int64_t columns,int64_t stride,cudaStream_t stream) {
    int device=0;checked(cudaGetDevice(&device));
    const unsigned aa=prefill_lt_alignment(weight),ab=prefill_lt_alignment(input),ac=prefill_lt_alignment(residual),ad=prefill_lt_alignment(output);
    using Key=std::tuple<int,int64_t,int64_t,int64_t,int64_t,unsigned,unsigned,unsigned,unsigned>;
    struct Entry {Key key;std::shared_ptr<PrefillLtResidual> plan;};
    static thread_local std::list<Entry> cache;
    static thread_local std::unordered_map<int,std::weak_ptr<PrefillLtDevice>> devices;
    const Key key{device,tokens,rows,columns,stride,aa,ab,ac,ad};
    for(auto i=cache.begin();i!=cache.end();++i)if(i->key==key) {
        auto result=i->plan;cache.splice(cache.begin(),cache,i);return result;
    }
    cudaStreamCaptureStatus capturing=cudaStreamCaptureStatusNone;
    checked(cudaStreamIsCapturing(stream,&capturing));
    // Prepared graph workloads use their warmup plan. Planning a new library
    // algorithm may initialize CUDA resources, so an unprepared capture keeps
    // the ordinary GEMM/add operation.
    if(capturing!=cudaStreamCaptureStatusNone)return {};
    const auto remember=[&](std::shared_ptr<PrefillLtResidual> plan) {
        cache.push_front({key,plan});
        if(cache.size()>64)cache.pop_back(); // Bound host descriptor metadata.
        return plan;
    };
    auto owner=devices[device].lock();
    if(!owner) {
        owner=std::make_shared<PrefillLtDevice>();owner->device=device;
        const auto status=cublasLtCreate(&owner->handle);
        if(status==CUBLAS_STATUS_ALLOC_FAILED)return {};
        if(status==CUBLAS_STATUS_NOT_SUPPORTED)return remember({});
        prefill_lt_checked(status);devices[device]=owner;
    }
    auto plan=std::make_shared<PrefillLtResidual>();plan->owner=std::move(owner);
    plan->tokens=tokens;plan->rows=rows;plan->columns=columns;plan->stride=stride;
    prefill_lt_checked(cublasLtMatmulDescCreate(&plan->operation,CUBLAS_COMPUTE_16F,CUDA_R_16F));
    const cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
    prefill_lt_checked(cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
    prefill_lt_checked(cublasLtMatmulDescSetAttribute(plan->operation,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
    prefill_lt_checked(cublasLtMatrixLayoutCreate(&plan->a,CUDA_R_16F,columns,rows,columns));
    prefill_lt_checked(cublasLtMatrixLayoutCreate(&plan->b,CUDA_R_16F,columns,tokens,columns));
    prefill_lt_checked(cublasLtMatrixLayoutCreate(&plan->c,CUDA_R_16F,rows,tokens,stride));
    struct Preference {
        cublasLtMatmulPreference_t value=nullptr;
        ~Preference(){if(value)(void)cublasLtMatmulPreferenceDestroy(value);}
    } preference;
    prefill_lt_checked(cublasLtMatmulPreferenceCreate(&preference.value));
    // This bounds a library algorithm's optional scratch, independently of
    // GPU capacity. Only its actual requested bytes are allocated by callers.
    const size_t limit=32ULL<<20;
    prefill_lt_checked(cublasLtMatmulPreferenceSetAttribute(preference.value,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&limit,sizeof(limit)));
    for(auto item:{std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,aa},std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,ab},
            std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,ac},std::pair{CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,ad}})
        prefill_lt_checked(cublasLtMatmulPreferenceSetAttribute(preference.value,item.first,&item.second,sizeof(item.second)));
    cublasLtMatmulHeuristicResult_t algorithm{};int count=0;
    const auto status=cublasLtMatmulAlgoGetHeuristic(plan->owner->handle,plan->operation,plan->a,plan->b,plan->c,plan->c,
        preference.value,1,&algorithm,&count);
    if(status==CUBLAS_STATUS_ALLOC_FAILED)return {};
    if(status==CUBLAS_STATUS_NOT_SUPPORTED)return remember({});
    prefill_lt_checked(status);
    if(!count)return remember({});
    if(algorithm.state==CUBLAS_STATUS_ALLOC_FAILED)return {};
    if(algorithm.state==CUBLAS_STATUS_NOT_SUPPORTED)return remember({});
    prefill_lt_checked(algorithm.state);
    if(algorithm.workspaceSize>limit)return remember({});
    plan->algorithm=algorithm.algo;plan->workspace=algorithm.workspaceSize;
    return remember(std::move(plan));
}
__global__ void prefill_residual_add(half* output,const half* residual,int64_t count,int64_t rows,int64_t stride) {
    for(auto i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
        const auto at=i/rows*stride+i%rows;
        output[at]=__float2half_rn(__half2float(output[at])+__half2float(residual[at]));
    }
}
