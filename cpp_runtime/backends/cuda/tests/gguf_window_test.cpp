#include "../kernels/gguf.h"
#include "../kernels/gguf_window.h"
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

void run(ggml_type format,int rows,int columns,int tokens,bool half_output,int& cases,
         bool unaligned=false,bool split=false) {
    const auto* traits=ggml_get_type_traits(format);
    if(traits->blck_size!=256 || !traits->to_float)
        throw std::runtime_error("Window block traits differ from the decoder");
    const auto block_bytes=traits->type_size;
    const auto stride=std::size_t(ggml_row_size(format,columns));
    std::vector<std::uint8_t> weights(rows*stride);
    for(std::size_t i=0;i<weights.size();++i)weights[i]=std::uint8_t(i*37+i/11+23);
    for(std::size_t block=0;block<weights.size()/block_bytes;++block) {
        const auto d=__float2half((block%2?-.00031f:.00031f)*float(block%17));
        const auto minimum=__float2half((block%3?-.00017f:.00017f)*float(block%19));
        const auto scale_offset=format==GGML_TYPE_Q6_K?block_bytes-2:0;
        std::memcpy(weights.data()+block*block_bytes+scale_offset,&d,2);
        if(format!=GGML_TYPE_Q6_K)std::memcpy(weights.data()+block*block_bytes+2,&minimum,2);
    }
    std::vector<float> input(std::size_t(tokens)*columns),decoded(std::size_t(rows)*columns);
    std::vector<Q8> packed(std::size_t(tokens)*(columns/32));
    const auto output_bytes=(std::size_t(tokens)*rows+2)*(half_output?2:4);
    std::vector<std::uint8_t> output(output_bytes);
    const int prefix=unaligned?4:0;
    Device<std::uint8_t> w(weights.size()+prefix),y(output_bytes);
    Device<float> x(input.size());Device<Q8> q8(packed.size());Graph graph;
    Device<std::uint8_t> split_q8(split?gg::interleaved_activation_bytes(tokens,columns):1);
    const auto launch=[&] {
        if(split) {
            gg::quantize_input_interleaved(x.data,false,tokens,columns,split_q8.data,q8.data,graph.stream);
            if(!gg::project_dense_split_window_q8(format,w.data+prefix,stride,rows,columns,
                    split_q8.data,tokens,y.data+(half_output?2:4),half_output,graph.stream))
                throw std::runtime_error("Supported split window did not dispatch");
            return;
        }
        gg::quantize_input(x.data,tokens,columns,q8.data,graph.stream);
        if(half_output)gg::matvec_q8_f16(format,w.data+prefix,stride,rows,columns,
            q8.data,tokens,y.data+2,graph.stream);
        else gg::matvec_q8(format,w.data+prefix,stride,rows,columns,
            q8.data,tokens,reinterpret_cast<float*>(y.data+4),graph.stream);
    };
    const int replays=rows==33 && columns==1280 && (tokens==5 || tokens==8)?3:1;
    for(int replay=0;replay<replays;++replay) {
        for(std::size_t i=0;i<input.size();++i)
            input[i]=std::sin(float(i)*(.013f+.003f*replay))*(replay+1)*.17f;
        if(replay)for(std::size_t block=0;block<weights.size()/block_bytes;++block)
            weights[block*block_bytes+16+block%32]^=std::uint8_t(0x5a+replay);
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
                throw std::runtime_error("Window projection overwrote an output guard");
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
                throw std::runtime_error("Window projection differs from double scalar reference: format="+
                    std::to_string(format)+" K="+
                    std::to_string(columns)+" M="+std::to_string(tokens)+" row="+std::to_string(row)+
                    " expected="+std::to_string(expected)+" actual="+std::to_string(actual));
        }
        ++cases;
    }
}
}

int main(int argc,char** argv)try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::puts("SKIP: CUDA unavailable");return 77;}
    int cases=0;
    const bool affine_only=argc==2 && std::string(argv[1])=="--q4-affine-check";
    for(int columns:{256,512,768,1280,2560,4096,6144,10240})
        for(int rows:{1,33})for(int tokens:{2,3,4,5,6,7,8})for(bool half:{false,true})
            run(GGML_TYPE_Q4_K,rows,columns,tokens,half,cases);
    for(int tokens:{1,9})
        if(gg::project_dense_window_q8(GGML_TYPE_Q4_K,nullptr,0,1,256,nullptr,tokens,nullptr,false,nullptr))
            throw std::runtime_error("Unsupported Q4 affine window consumed ordinary projection");
    if(gg::project_dense_window_q8(GGML_TYPE_Q4_K,reinterpret_cast<const void*>(2),ggml_row_size(GGML_TYPE_Q4_K,256),
            1,256,reinterpret_cast<const void*>(16),2,reinterpret_cast<void*>(16),false,nullptr))
        throw std::runtime_error("Unaligned Q4 weights selected u32 gathers");
    if(affine_only) {
        std::printf("Q4_K whole-Q8-group M2..8 independent FP64/F16-F32/short-long-K/tails/changed-input-weight-Graph cases=%d PASS\n",cases);
        return 0;
    }
    // Complete eight-lane groups cover short, partial-warp and long reductions.
    for(auto format:{GGML_TYPE_IQ4_XS,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K}) {
    for(int columns:{256,512,768,1024,1280,2048,2560,4096,6144,10240})
        for(int rows:{1,33})for(int tokens:{5,6,7,8})for(bool half:{false,true})
            run(format,rows,columns,tokens,half,cases);
    for(int tokens:{5,6,7,8})for(bool half:{false,true})run(format,33,1280,tokens,half,cases,true);
    }
    for(auto format:{GGML_TYPE_Q5_K,GGML_TYPE_Q6_K}) {
        for(int columns:{256,512,768,1024,1280,2048,2560,4096,6144,10240})
            for(int rows:{1,33})for(bool half:{false,true})
                run(format,rows,columns,8,half,cases,false,true);
        for(int tokens:{1,2,3,4,5,6,7,9})
            if(gg::supports_dense_split_window_q8(format,reinterpret_cast<const void*>(16),
                    ggml_row_size(format,256),1,256,tokens))
                throw std::runtime_error("Incomplete window selected split activation storage");
        if(gg::supports_dense_split_window_q8(format,nullptr,ggml_row_size(format,256),1,256,8))
            throw std::runtime_error("Missing weights selected split activation storage");
    }
    if(gg::supports_dense_split_window_q8(GGML_TYPE_Q5_K,reinterpret_cast<const void*>(4),
            ggml_row_size(GGML_TYPE_Q5_K,256),1,256,8))
        throw std::runtime_error("Unaligned Q5 weights selected vector loads");
    for(int tokens:{1,2,3,4,9})
        if(gg::project_dense_window_q8(GGML_TYPE_IQ4_XS,nullptr,0,1,256,nullptr,tokens,nullptr,false,nullptr))
            throw std::runtime_error("Unsupported window consumed the ordinary projection");
    if(gg::project_dense_window_q8(GGML_TYPE_Q2_K,nullptr,0,1,256,nullptr,8,nullptr,false,nullptr))
        throw std::runtime_error("Unsupported format consumed the ordinary projection");
    std::printf("Q4/IQ4/Q5/Q6 native/split window groups scalar-double/F16-F32/M2..8/tails/changed-input-and-weight-graph cases=%d PASS\n",cases);
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
