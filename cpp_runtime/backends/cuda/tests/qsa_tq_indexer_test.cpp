#include "../kernels/qsa_tq.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <cstring>
#include <limits>
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
int paired_layout_cases() {
    int cases=0;
    for(int width:{2,6,66,130})for(int heads:{1,4,19}) {
        Fixture f(1,heads,width,33);f.positions[0]=131;f.upload();f.run();f.verify();
        std::fill(f.query.begin(),f.query.end(),0.f);f.upload();f.run();f.verify();
        cases+=2;
    }
    for(int shifted=0;shifted<4;++shifted) {
        Fixture f(1,4,128,65);f.positions[0]=259;
        Buffer query(f.query.size()*4+4),keys(f.keys.size()*2+2);
        auto* q=query.as<float>()+((shifted&1)?1:0);
        auto* k=keys.as<std::int16_t>()+((shifted&2)?1:0);
        check(cudaMemcpy(q,f.query.data(),f.query.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(k,f.keys.data(),f.keys.size()*2,cudaMemcpyHostToDevice));
        f.upload();
        check(indexer_affine16_scores(q,k,f.s.as<std::uint16_t>(),f.z.as<std::int16_t>(),
            f.p.as<std::int64_t>(),f.out.as<float>(),1,f.heads,f.width,f.blocks,nullptr));
        f.verify();
        std::vector<float> plain(f.blocks);
        check(cudaMemcpy(plain.data(),f.out.data,plain.size()*4,cudaMemcpyDeviceToHost));
        // The packed row adds its affine scale and zero after the coordinates.
        std::vector<std::int16_t> packed(std::size_t(f.blocks)*(f.width+2));
        for(int row=0;row<f.blocks;++row) {
            auto* record=packed.data()+std::size_t(row)*(f.width+2);
            std::copy_n(f.keys.data()+std::size_t(row)*f.width,f.width,record);
            record[f.width]=static_cast<std::int16_t>(f.scales[row]);
            record[f.width+1]=f.zero[row];
        }
        Buffer records(packed.size()*2+2);
        auto* data=records.as<std::int16_t>()+((shifted&2)?1:0);
        check(cudaMemcpy(data,packed.data(),packed.size()*2,cudaMemcpyHostToDevice));
        check(tyloinfer::qsa::indexer_affine16_record_scores(q,data,nullptr,f.out.as<float>(),
            1,f.heads,f.width,f.blocks,4,nullptr));
        std::vector<float> actual(f.blocks);
        check(cudaMemcpy(actual.data(),f.out.data,actual.size()*4,cudaMemcpyDeviceToHost));
        const auto inverse=1.f/std::sqrt(float(f.width));
        for(int row=0;row<f.blocks;++row) {
            const auto expected=plain[row]*inverse;
            if(!std::isfinite(actual[row]) || std::abs(actual[row]-expected)>3.e-6+std::abs(expected)*3.e-5)
                throw std::runtime_error("Packed Indexer alignment or normalization mismatch");
        }
        cases+=2;
    }
    return cases;
}
void graph_case(bool integer_workspace=false) {
    Fixture f(39,3,128,65);f.run();f.verify();
    const auto bytes=tyloinfer::qsa::indexer_affine16_score_scratch(f.queries,f.heads,f.width);
    Buffer workspace(std::max<std::size_t>(1,bytes));
    const auto run=[&](cudaStream_t stream) {
        if(integer_workspace)check(indexer_affine16_scores(f.q.as<float>(),f.k.as<std::int16_t>(),
            f.s.as<std::uint16_t>(),f.z.as<std::int16_t>(),f.p.as<std::int64_t>(),f.out.as<float>(),
            f.queries,f.heads,f.width,f.blocks,stream,workspace.data,bytes));
        else f.run(stream);
    };
    cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));run(stream);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
    for(int pass=0;pass<3;++pass) {
        for(int i=0;i<f.queries;++i)f.positions[i]=pass==0?-1:pass==1?i*4:259-i*3;
        if(pass==2)std::fill(f.query.begin(),f.query.end(),0.f);
        f.upload(stream);check(cudaGraphLaunch(executable,stream));f.verify(stream);
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
int integer_workspace_cases() {
    int cases=0;
    for(int width:{1,17,128,257})for(int heads:{1,3,4,19})for(int rows:{1,2,3,4,8,17}) {
        Fixture f(rows,heads,width,65);
        const auto bytes=tyloinfer::qsa::indexer_affine16_score_scratch(rows,heads,width);
        Buffer workspace(bytes);
        check(indexer_affine16_scores(f.q.as<float>(),f.k.as<std::int16_t>(),f.s.as<std::uint16_t>(),
            f.z.as<std::int16_t>(),f.p.as<std::int64_t>(),f.out.as<float>(),rows,heads,width,f.blocks,
            nullptr,workspace.data,bytes));
        f.verify();++cases;
        std::fill(f.keys.begin(),f.keys.end(),12345);std::fill(f.zero.begin(),f.zero.end(),12345);f.upload();
        check(indexer_affine16_scores(f.q.as<float>(),f.k.as<std::int16_t>(),f.s.as<std::uint16_t>(),
            f.z.as<std::int16_t>(),f.p.as<std::int64_t>(),f.out.as<float>(),rows,heads,width,f.blocks,
            nullptr,workspace.data,bytes));
        f.verify();++cases;
    }
    {
        Fixture f(3,4,128,1027);std::fill(f.positions.begin(),f.positions.end(),4107);f.upload();f.run();
        std::vector<float> baseline(std::size_t(f.queries)*f.blocks),actual(baseline.size());
        check(cudaMemcpy(baseline.data(),f.out.data,baseline.size()*4,cudaMemcpyDeviceToHost));
        const auto bytes=tyloinfer::qsa::indexer_affine16_score_scratch(f.queries,f.heads,f.width);
        Buffer workspace(bytes+16);
        check(indexer_affine16_scores(f.q.as<float>(),f.k.as<std::int16_t>(),f.s.as<std::uint16_t>(),
            f.z.as<std::int16_t>(),f.p.as<std::int64_t>(),f.out.as<float>(),f.queries,f.heads,f.width,f.blocks,
            nullptr,workspace.data,bytes));
        f.verify();
        check(cudaMemcpy(actual.data(),f.out.data,actual.size()*4,cudaMemcpyDeviceToHost));
        for(int q=0;q<f.queries;++q) {
            std::vector<int> a(f.blocks),b(f.blocks);std::iota(a.begin(),a.end(),0);b=a;
            const auto order=[&](const auto& scores,int x,int y) {
                const auto offset=std::size_t(q)*f.blocks;
                return scores[offset+x]!=scores[offset+y]?scores[offset+x]>scores[offset+y]:x<y;
            };
            std::partial_sort(a.begin(),a.begin()+512,a.end(),[&](int x,int y){return order(baseline,x,y);});
            std::partial_sort(b.begin(),b.begin()+512,b.end(),[&](int x,int y){return order(actual,x,y);});
            a.resize(512);b.resize(512);std::sort(a.begin(),a.end());std::sort(b.begin(),b.end());
            if(a!=b)throw std::runtime_error("Integer Indexer changed the selected block set");
        }
        for(int mode=0;mode<3;++mode) {
            void* data=mode==0?nullptr:mode==1?workspace.as<std::uint8_t>()+1:workspace.data;
            const auto size=mode==2?bytes-1:bytes;
            check(indexer_affine16_scores(f.q.as<float>(),f.k.as<std::int16_t>(),f.s.as<std::uint16_t>(),
                f.z.as<std::int16_t>(),f.p.as<std::int64_t>(),f.out.as<float>(),f.queries,f.heads,f.width,f.blocks,
                nullptr,data,size));
            f.verify();
        }
        ++cases;
    }
    if(tyloinfer::qsa::indexer_affine16_score_scratch(0,1,128)!=0 ||
       tyloinfer::qsa::indexer_affine16_score_scratch(3,4,65537)!=0)
        throw std::runtime_error("Integer Indexer scratch bound mismatch");
    return cases;
}

int integer_pooling_cases() {
    Fixture f(7,3,17,65);
    std::vector<std::int16_t> records(std::size_t(f.blocks)*(f.width+2));
    for(int b=0;b<f.blocks;++b) {
        auto* row=records.data()+std::size_t(b)*(f.width+2);
        std::memcpy(row,f.keys.data()+std::size_t(b)*f.width,f.width*2);
        std::memcpy(row+f.width,&f.scales[b],2);row[f.width+1]=f.zero[b];
    }
    Buffer packed(records.size()*2);
    check(cudaMemcpy(packed.data,records.data(),records.size()*2,cudaMemcpyHostToDevice));
    const auto bytes=tyloinfer::qsa::indexer_affine16_score_scratch(f.queries,f.heads,f.width);
    Buffer workspace(bytes);int cases=0;
    for(int pooling:{1,3,4,16,257,std::numeric_limits<int>::max()})for(bool unmasked:{false,true}) {
        cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t executable;
        check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        check(tyloinfer::qsa::indexer_affine16_record_scores(f.q.as<float>(),packed.as<std::int16_t>(),
            unmasked?nullptr:f.p.as<std::int64_t>(),f.out.as<float>(),f.queries,f.heads,f.width,f.blocks,
            pooling,stream,workspace.data,bytes));
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
        for(int pass=0;pass<3;++pass) {
            f.positions={std::numeric_limits<std::int64_t>::min(),-5,-1,0,
                std::int64_t(pooling)-1,std::int64_t(63+pass)*pooling-2,
                std::numeric_limits<std::int64_t>::max()-1};
            f.upload(stream);check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
            std::vector<float> actual(std::size_t(f.queries)*f.blocks);
            check(cudaMemcpy(actual.data(),f.out.data,actual.size()*4,cudaMemcpyDeviceToHost));
            for(int q=0;q<f.queries;++q)for(int b=0;b<f.blocks;++b) {
                const auto value=actual[std::size_t(q)*f.blocks+b];
                // Compare complete-block end positions rather than repeat the
                // device's visible-count division in the reference.
                if(!unmasked && f.positions[q]<(std::int64_t(b)+1)*pooling-1) {
                    if(value!=-INFINITY)throw std::runtime_error("Pooled Indexer causal bound mismatch");
                    continue;
                }
                double expected=0;
                for(int head=0;head<f.heads;++head) {
                    double dot=0;
                    for(int d=0;d<f.width;++d) {
                        const auto key=float(int(f.keys[std::size_t(b)*f.width+d])-int(f.zero[b]))*
                            __half2float(__ushort_as_half(f.scales[b]));
                        dot+=double(f.query[(std::size_t(q)*f.heads+head)*f.width+d])*key;
                    }
                    expected+=std::max(dot,0.0);
                }
                expected/=std::sqrt(double(f.width));
                if(!std::isfinite(value) || std::abs(value-expected)>3e-6+std::abs(expected)*3e-5)
                    throw std::runtime_error("Pooled Indexer FP64 score mismatch");
            }
            ++cases;
        }
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    }
    return cases;
}
} // namespace
int main()try {
    int devices=0;const auto available=cudaGetDeviceCount(&devices);
    if(available==cudaErrorNoDevice || available==cudaErrorInsufficientDriver || !devices) {
        std::cout<<"SKIP: CUDA device unavailable\n";return 77;
    }
    check(available);int cases=0;
    for(const int width:{17,64,128,256,257})for(const int heads:{1,3,7,19})for(const int rows:{1,2,3,4,39}) {
        Fixture f(rows,heads,width,65);f.run();f.verify();++cases;
    }
    for(const int rows:{1,3,4,5,6,7,8,16,17,128}) {
        Fixture f(rows,4,96,33);f.run();f.verify();++cases;
    }
    int device=0,ordinary=0,optin=0;check(cudaGetDevice(&device));
    check(cudaDeviceGetAttribute(&ordinary,cudaDevAttrMaxSharedMemoryPerBlock,device));
    check(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,device));
    // Exercise wide rows beyond the shared-memory tile on any GPU.
    {Fixture f(17,3,std::max(ordinary,optin)/196+2,33);f.run();f.verify();++cases;}
    {Fixture f(17,3,128,7);f.run();f.verify();++cases;}
    if(indexer_affine16_scores(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,1,1,1,1,nullptr)!=int(cudaErrorInvalidValue))
        throw std::runtime_error("Indexer accepted invalid pointers");
    cases+=paired_layout_cases();
    cases+=integer_workspace_cases();
    cases+=integer_pooling_cases();
    graph_case();
    graph_case(true);
    std::cout<<"QSA affine INT16 Indexer: "<<cases<<" CPU-reference cases, wide rows and 3 graph replays PASS\n";
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
