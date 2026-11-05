#include "../kernels/gguf.h"
#include "ggml.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace gg=mfq::cuda::gguf;
void check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Device {
    T* data=nullptr;
    explicit Device(std::size_t count){check(cudaMalloc(&data,count*sizeof(T)));}
    ~Device(){cudaFree(data);}
    Device(const Device&)=delete;
    Device& operator=(const Device&)=delete;
};
struct alignas(4) Q8 {__half d,s;std::int8_t codes[32];};
static_assert(sizeof(Q8)==36);
struct Graph {
    cudaStream_t stream=nullptr;
    cudaGraph_t graph=nullptr;
    cudaGraphExec_t exec=nullptr;
    Graph(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Graph(){if(exec)cudaGraphExecDestroy(exec);if(graph)cudaGraphDestroy(graph);cudaStreamDestroy(stream);}
};

void run(int rows,int columns,int tokens,bool half_output,int& cases,bool unaligned=false) {
    const auto* traits=ggml_get_type_traits(GGML_TYPE_Q5_K);
    if(traits->blck_size!=256 || traits->type_size!=176 || !traits->to_float)
        throw std::runtime_error("Q5_K block traits differ from the affine-group decoder");
    const auto stride=std::size_t(ggml_row_size(GGML_TYPE_Q5_K,columns));
    std::vector<std::uint8_t> weights(rows*stride);
    for(std::size_t i=0;i<weights.size();++i)weights[i]=std::uint8_t(i*37+i/11+23);
    for(std::size_t block=0;block<weights.size()/176;++block) {
        const auto d=__float2half((block%2?-.00031f:.00031f)*float(block%17));
        const auto minimum=__float2half((block%3?-.00017f:.00017f)*float(block%19));
        std::memcpy(weights.data()+block*176,&d,2);
        std::memcpy(weights.data()+block*176+2,&minimum,2);
    }
    std::vector<float> input(std::size_t(tokens)*columns),decoded(std::size_t(rows)*columns);
    std::vector<Q8> packed(std::size_t(tokens)*(columns/32));
    const auto output_bytes=(std::size_t(tokens)*rows+2)*(half_output?2:4);
    std::vector<std::uint8_t> output(output_bytes);
    const int prefix=unaligned?4:0;
    Device<std::uint8_t> w(weights.size()+prefix),y(output_bytes);
    Device<float> x(input.size());Device<Q8> q8(packed.size());Graph graph;
    const auto launch=[&] {
        gg::quantize_input(x.data,tokens,columns,q8.data,graph.stream);
        if(half_output)gg::matvec_q8_f16(GGML_TYPE_Q5_K,w.data+prefix,stride,rows,columns,
            q8.data,tokens,y.data+2,graph.stream);
        else gg::matvec_q8(GGML_TYPE_Q5_K,w.data+prefix,stride,rows,columns,
            q8.data,tokens,reinterpret_cast<float*>(y.data+4),graph.stream);
    };
    const int replays=rows==33 && columns==1280 && (tokens==3 || tokens==8)?3:1;
    for(int replay=0;replay<replays;++replay) {
        for(std::size_t i=0;i<input.size();++i)
            input[i]=std::sin(float(i)*(.013f+.003f*replay))*(replay+1)*.17f;
        if(replay)for(std::size_t block=0;block<weights.size()/176;++block)
            weights[block*176+16+block%32]^=std::uint8_t(0x5a+replay);
        for(int row=0;row<rows;++row)
            traits->to_float(weights.data()+row*stride,decoded.data()+row*columns,columns);
        // Uploads and graph execution share one stream; no default-stream ordering is assumed.
        check(cudaMemcpyAsync(w.data+prefix,weights.data(),weights.size(),cudaMemcpyHostToDevice,graph.stream));
        check(cudaMemcpyAsync(x.data,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,graph.stream));
        check(cudaMemsetAsync(y.data,0x5a,output_bytes,graph.stream));
        if(replays>1) {
            if(!replay) {
                check(cudaStreamSynchronize(graph.stream));
                check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));launch();
                check(cudaStreamEndCapture(graph.stream,&graph.graph));
                check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
            }
            check(cudaGraphLaunch(graph.exec,graph.stream));
        } else launch();
        check(cudaMemcpyAsync(packed.data(),q8.data,packed.size()*sizeof(Q8),cudaMemcpyDeviceToHost,graph.stream));
        check(cudaMemcpyAsync(output.data(),y.data,output_bytes,cudaMemcpyDeviceToHost,graph.stream));
        check(cudaStreamSynchronize(graph.stream));
        const int element_bytes=half_output?2:4;
        for(int byte=0;byte<element_bytes;++byte)
            if(output[byte]!=0x5a || output[output_bytes-element_bytes+byte]!=0x5a)
                throw std::runtime_error("Q5 affine projection overwrote an output guard");
        for(int token=0;token<tokens;++token)for(int row=0;row<rows;++row) {
            double expected=0,absolute=0;
            for(int k=0;k<columns;++k) {
                const auto& a=packed[std::size_t(token)*(columns/32)+k/32];
                const double value=double(decoded[row*columns+k])*(float(a.codes[k%32])*__half2float(a.d));
                expected+=value;absolute+=std::abs(value);
            }
            const auto offset=(std::size_t(token)*rows+row+1)*element_bytes;
            float actual;
            if(half_output){__half value;std::memcpy(&value,output.data()+offset,2);actual=__half2float(value);}
            else std::memcpy(&actual,output.data()+offset,4);
            const double tolerance=3e-6*absolute+1e-6+(half_output?.0005*std::abs(expected)+1e-6:0);
            if(!std::isfinite(actual) || std::abs(actual-expected)>tolerance)
                throw std::runtime_error("Q5 affine projection differs from double scalar reference: K="+
                    std::to_string(columns)+" M="+std::to_string(tokens)+" row="+std::to_string(row)+
                    " expected="+std::to_string(expected)+" actual="+std::to_string(actual));
        }
        ++cases;
    }
}
}

int main()try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::puts("SKIP: CUDA unavailable");return 77;}
    int cases=0;
    // Complete eight-lane groups cover short, partial-warp and long reductions.
    for(int columns:{256,512,768,1024,1280,2048,2560,4096,6144,10240})
        for(int rows:{1,33})for(int tokens:{2,3,4,5,6,7,8})for(bool half:{false,true})
            run(rows,columns,tokens,half,cases);
    for(int tokens:{2,3,4,8})for(bool half:{false,true})run(33,1280,tokens,half,cases,true);
    for(bool half:{false,true})run(33,1280,32,half,cases);
    std::printf("Q5 affine groups scalar-double/F16-F32/M2..8/tails/changed-input-and-weight-graph cases=%d PASS\n",cases);
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
