#include "tyloinfer/row_residual.h"
#include "../kernels/gguf_row_residual.h"
#include "ggml.h"
#include "../../../tests/row_residual_fixture.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(cudaError_t e) { if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
    void* data=nullptr;
    explicit Buffer(std::size_t size) { check(cudaMalloc(&data,std::max<std::size_t>(1,size))); }
    ~Buffer() { cudaFree(data); }
    template<class T>T* as() { return static_cast<T*>(data); }
};
void near(const std::vector<float>& actual,const std::vector<float>& expected) {
    for(std::size_t i=0;i<actual.size();++i)
        if(std::abs(actual[i]-expected[i])>2e-4f*(1+std::abs(expected[i])))
            throw std::runtime_error("row residual output differs from decoded reference");
}
}
int main(int argc,char** argv) try {
    if(argc>2)throw std::invalid_argument("expected at most one residual fixture path");
    int devices=0;check(cudaGetDeviceCount(&devices));
    if(!devices) {std::cout<<"SKIP: CUDA device unavailable\n";return 77;}
    std::unique_ptr<RowResidualFixture> generated;
    if(argc==1)generated=std::make_unique<RowResidualFixture>(10,137);
    mfq::RowResidualFile file(argc==2?std::filesystem::path(argv[1]):generated->path());
    const auto rows=std::int64_t(file.row_count()),columns=std::int64_t(file.columns());
    if(rows%2)throw std::runtime_error("fixture needs an even number of rows for the two-expert test");
    std::vector<std::uint64_t> offsets;
    std::vector<std::int32_t> types;
    std::vector<std::uint8_t> bytes(std::max<std::uint64_t>(1,file.payload_bytes()));
    file.read_payload(0,bytes.data(),std::size_t(file.payload_bytes()));
    std::vector<float> decoded(rows*columns),temporary((columns+127)/128*128);
    for(std::int64_t row=0;row<rows;++row) {
        const auto d=file.rows()[row];offsets.push_back(d.offset);types.push_back(d.type);
        if(d.type) {
            ggml_get_type_traits(ggml_type(d.type))->to_float(bytes.data()+d.offset,temporary.data(),temporary.size());
            std::copy_n(temporary.data(),columns,decoded.data()+row*columns);
        }
    }
    Buffer q(bytes.size()),o(offsets.size()*8),t(types.size()*4);
    check(cudaMemcpy(q.data,bytes.data(),bytes.size(),cudaMemcpyHostToDevice));
    check(cudaMemcpy(o.data,offsets.data(),offsets.size()*8,cudaMemcpyHostToDevice));
    check(cudaMemcpy(t.data,types.data(),types.size()*4,cudaMemcpyHostToDevice));
    for(const auto tokens:{1,2,3,4,33,129}) {
        std::vector<float> x(tokens*columns),expected(tokens*rows,.25f),actual(expected.size());
        for(std::size_t i=0;i<x.size();++i)x[i]=std::sin(float(i)*.17f);
        for(int token=0;token<tokens;++token)for(std::int64_t row=0;row<rows;++row)
            for(std::int64_t c=0;c<columns;++c)expected[token*rows+row]+=x[token*columns+c]*decoded[row*columns+c];
        std::vector<float> initial(expected.size(),.25f);
        Buffer dx(x.size()*4),dy(actual.size()*4);
        check(cudaMemcpy(dx.data,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dy.data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));
        mfq::cuda::gguf::add_row_residual(q.data,o.as<std::uint64_t>(),t.as<std::int32_t>(),
            0,rows,0,columns,dx.as<float>(),dy.as<float>(),tokens,nullptr,1,1,false,nullptr);
        check(cudaMemcpy(actual.data(),dy.data,actual.size()*4,cudaMemcpyDeviceToHost));near(actual,expected);
        cudaStream_t stream;check(cudaStreamCreate(&stream));
        cudaGraph_t graph;cudaGraphExec_t executable;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        mfq::cuda::gguf::add_row_residual(q.data,o.as<std::uint64_t>(),t.as<std::int32_t>(),
            0,rows,0,columns,dx.as<float>(),dy.as<float>(),tokens,nullptr,1,1,false,stream);
        check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
        check(cudaMemcpyAsync(dy.data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        check(cudaMemcpy(actual.data(),dy.data,actual.size()*4,cudaMemcpyDeviceToHost));near(actual,expected);
        check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    }
    for(bool routed:{false,true}) {
        const std::int64_t tokens=3,routes=2,out=rows/2;
        const std::vector<std::int64_t> ids={1,0,0,1,1,1};
        std::vector<float> x(tokens*(routed?routes:1)*columns),expected(tokens*routes*out,.5f),actual(expected.size());
        for(std::size_t i=0;i<x.size();++i)x[i]=std::cos(float(i)*.11f);
        for(std::int64_t slot=0;slot<tokens*routes;++slot)for(std::int64_t row=0;row<out;++row)
            for(std::int64_t c=0;c<columns;++c)
                expected[slot*out+row]+=x[(routed?slot:slot/routes)*columns+c]*decoded[(ids[slot]*out+row)*columns+c];
        std::vector<float> initial(expected.size(),.5f);
        Buffer dx(x.size()*4),dy(actual.size()*4),di(ids.size()*8);
        check(cudaMemcpy(dx.data,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(dy.data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice));
        check(cudaMemcpy(di.data,ids.data(),ids.size()*8,cudaMemcpyHostToDevice));
        mfq::cuda::gguf::add_row_residual(q.data,o.as<std::uint64_t>(),t.as<std::int32_t>(),0,out,0,columns,
            dx.as<float>(),dy.as<float>(),tokens,di.as<std::int64_t>(),routes,2,routed,nullptr);
        check(cudaMemcpy(actual.data(),dy.data,actual.size()*4,cudaMemcpyDeviceToHost));near(actual,expected);
    }
    std::cout<<"PASS: heterogeneous GGUF residuals, decode/prefill, routed experts and CUDA Graph\n";
    return 0;
} catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
