#include "../kernels/qsa_tq.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using tyloinfer::qsa::indexer_affine16_scores;
void check(int value) {
    if(value)throw std::runtime_error(cudaGetErrorString(static_cast<cudaError_t>(value)));
}
struct Buffer {
    void* data=nullptr;
    explicit Buffer(std::size_t bytes){check(cudaMalloc(&data,bytes));}
    ~Buffer(){cudaFree(data);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    template<class T>T* as(){return static_cast<T*>(data);}
};
struct Fixture {
    int queries,heads,width,blocks;
    std::vector<float> query;
    std::vector<std::int16_t> keys,zero;
    std::vector<std::uint16_t> scales;
    std::vector<std::int64_t> positions;
    Buffer q,k,z,s,p,out;
    Fixture(int m,int h,int w,int n):queries(m),heads(h),width(w),blocks(n),
        query(std::size_t(m)*h*w),keys(std::size_t(n)*w),zero(n),scales(n),positions(m),
        q(query.size()*4),k(keys.size()*2),z(n*2),s(n*2),p(m*8),out(std::size_t(m)*n*4) {
        std::uint32_t state=0x419f3a72;
        const auto next=[&] {state^=state<<13;state^=state>>17;state^=state<<5;return state;};
        for(auto& x:query)x=(float(next()&65535)/32768.f-1.f)*.1f;
        for(auto& x:keys)x=std::int16_t(int(next()%60001)-30000);
        for(int i=0;i<n;++i) {
            zero[i]=std::int16_t(int(next()%8001)-4000);
            scales[i]=__half_as_ushort(__float2half_rn(3.e-6f*(1.f+(next()%9)/8.f)));
        }
        for(auto& x:positions)x=std::int64_t(next()%(n*4+13))-5;
        upload();
    }
    void upload(cudaStream_t stream=nullptr) {
        check(cudaMemcpyAsync(q.data,query.data(),query.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(k.data,keys.data(),keys.size()*2,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(z.data,zero.data(),zero.size()*2,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(s.data,scales.data(),scales.size()*2,cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(p.data,positions.data(),positions.size()*8,cudaMemcpyHostToDevice,stream));
    }
    void run(cudaStream_t stream=nullptr) {
        check(indexer_affine16_scores(q.as<float>(),k.as<std::int16_t>(),s.as<std::uint16_t>(),
            z.as<std::int16_t>(),p.as<std::int64_t>(),out.as<float>(),queries,heads,width,blocks,stream));
    }
    void verify(cudaStream_t stream=nullptr) {
        check(cudaStreamSynchronize(stream));
        std::vector<float> actual(std::size_t(queries)*blocks);
        check(cudaMemcpy(actual.data(),out.data,actual.size()*4,cudaMemcpyDeviceToHost));
        for(int row=0;row<queries;++row)for(int block=0;block<blocks;++block) {
            const float value=actual[std::size_t(row)*blocks+block];
            if(block>=(positions[row]+1)/4) {
                if(value!=-INFINITY)throw std::runtime_error("Indexer causal mask mismatch");
                continue;
            }
            double expected=0;
            const float scale=__half2float(__ushort_as_half(scales[block]));
            for(int head=0;head<heads;++head) {
                double dot=0;
                for(int d=0;d<width;++d) {
                    // Reconstruct the stored INT16 affine key in FP32, then
                    // use an independent FP64 dot product as the reference.
                    const float key=float(int(keys[std::size_t(block)*width+d])-int(zero[block]))*scale;
                    dot+=double(query[(std::size_t(row)*heads+head)*width+d])*key;
                }
                expected+=std::max(dot,0.0);
            }
            if(!std::isfinite(value) || std::abs(value-expected)>3.e-6+std::abs(expected)*3.e-5)
                throw std::runtime_error("Indexer score differs from independent FP64 reference");
        }
    }
};
void graph_case() {
    Fixture f(39,3,128,65);f.run();f.verify();
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));f.run(stream);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
    for(int pass=0;pass<3;++pass) {
        for(int i=0;i<f.queries;++i)f.positions[i]=pass==0?-1:pass==1?i*4:259-i*3;
        if(pass==2)std::fill(f.query.begin(),f.query.end(),0.f);
        f.upload(stream);check(cudaGraphLaunch(executable,stream));f.verify(stream);
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
} // namespace
int main()try {
    int devices=0;const auto available=cudaGetDeviceCount(&devices);
    if(available==cudaErrorNoDevice || available==cudaErrorInsufficientDriver || !devices) {
        std::cout<<"SKIP: CUDA device unavailable\n";return 77;
    }
    check(available);int cases=0;
    for(const int width:{17,64,128,256})for(const int heads:{1,3,7})for(const int rows:{2,39}) {
        Fixture f(rows,heads,width,65);f.run();f.verify();++cases;
    }
    for(const int rows:{1,3,7,16,17,128}) {
        Fixture f(rows,4,96,33);f.run();f.verify();++cases;
    }
    int device=0,ordinary=0,optin=0;check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&ordinary,cudaDevAttrMaxSharedMemoryPerBlock,device));
    check(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    // Exercise the fallback from actual resource requirements, on any GPU.
    {Fixture f(17,3,std::max(ordinary,optin)/196+2,33);f.run();f.verify();++cases;}
    {Fixture f(17,3,128,7);f.run();f.verify();++cases;}
    if(indexer_affine16_scores(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1,1,nullptr)!=int(cudaErrorInvalidValue))
        throw std::runtime_error("Indexer accepted invalid pointers");
    graph_case();
    std::cout<<"QSA affine INT16 Indexer: "<<cases<<" CPU-reference cases, resource fallback and 3 graph replays PASS\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
