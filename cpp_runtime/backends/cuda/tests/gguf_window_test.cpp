#include "../kernels/gguf.h"
#include "../kernels/gguf_window.h"
#include "ggml.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
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
         int prefix=0,bool split=false,bool grouped=false,bool position_major=false) {
    const auto* traits=ggml_get_type_traits(format);
    if(traits->blck_size!=((format==GGML_TYPE_Q8_0 || format==GGML_TYPE_IQ4_NL)?32:256) || !traits->to_float)
        throw std::runtime_error("Window block traits differ from the decoder");
    const auto block_bytes=traits->type_size;
    const auto stride=std::size_t(ggml_row_size(format,columns));
    std::vector<std::uint8_t> weights(rows*stride);
    for(std::size_t i=0;i<weights.size();++i)weights[i]=std::uint8_t(i*37+i/11+23);
    for(std::size_t block=0;block<weights.size()/block_bytes;++block) {
        const auto d=__float2half((block%2?-.00031f:.00031f)*float(block%17));
        const auto minimum=__float2half((block%3?-.00017f:.00017f)*float(block%19));
        const auto scale_offset=format==GGML_TYPE_Q6_K || format==GGML_TYPE_Q3_K?block_bytes-2:0;
        std::memcpy(weights.data()+block*block_bytes+scale_offset,&d,2);
        if(format==GGML_TYPE_Q4_K || format==GGML_TYPE_Q5_K)
            std::memcpy(weights.data()+block*block_bytes+2,&minimum,2);
    }
    std::vector<float> input(std::size_t(tokens)*columns),decoded(std::size_t(rows)*columns);
    std::vector<Q8> packed(std::size_t(tokens)*(columns/32));
    const auto output_bytes=(std::size_t(tokens)*rows+2)*(half_output?2:4);
    std::vector<std::uint8_t> output(output_bytes);
    Device<std::uint8_t> w(weights.size()+prefix),y(output_bytes);
    Device<float> x(input.size());Device<Q8> q8(packed.size());Graph graph;
    Device<std::uint8_t> split_q8(split || grouped || position_major?gg::interleaved_activation_bytes(tokens,columns):1);
    const auto launch=[&] {
        if(position_major) {
            gg::quantize_input_interleaved(x.data,false,tokens,columns,split_q8.data,q8.data,graph.stream);
            gg::matvec_interleaved(format,w.data+prefix,stride,rows,columns,split_q8.data,tokens,
                y.data+(half_output?2:4),half_output,graph.stream);
            return;
        }
        if(split) {
            gg::quantize_input_interleaved(x.data,false,tokens,columns,split_q8.data,q8.data,graph.stream);
            if(!gg::project_dense_split_window_q8(format,w.data+prefix,stride,rows,columns,
                    split_q8.data,tokens,y.data+(half_output?2:4),half_output,graph.stream))
                throw std::runtime_error("Supported split window did not dispatch");
            return;
        }
        if(grouped) {
            gg::quantize_input_interleaved(x.data,false,tokens,columns,split_q8.data,q8.data,graph.stream);
            gg::DenseProjection projections[3];
            const int lengths[]={1,13,rows-14};int begin=0;
            for(int i=0;i<3;++i) {
                projections[i]={w.data+prefix+begin*stride,stride,lengths[i],
                    y.data+(std::size_t(begin)*tokens+1)*(half_output?2:4),format};
                begin+=lengths[i];
            }
            gg::matvec_group(projections,3,columns,q8.data,split_q8.data,tokens,half_output,graph.stream);
            return;
        }
        gg::quantize_input(x.data,tokens,columns,q8.data,graph.stream);
        if(half_output)gg::matvec_q8_f16(format,w.data+prefix,stride,rows,columns,
            q8.data,tokens,y.data+2,graph.stream);
        else gg::matvec_q8(format,w.data+prefix,stride,rows,columns,
            q8.data,tokens,reinterpret_cast<float*>(y.data+4),graph.stream);
    };
    const int replays=rows==33 && (position_major ||
        (columns==1280 && (tokens==1 || tokens==3 || tokens==5 || tokens==8)))?3:1;
    for(int replay=0;replay<replays;++replay) {
        for(std::size_t i=0;i<input.size();++i)
            input[i]=std::sin(float(i)*(.013f+.003f*replay))*(replay+1)*.17f;
        if(replay)for(std::size_t block=0;block<weights.size()/block_bytes;++block)
            weights[block*block_bytes+((format==GGML_TYPE_Q8_0 || format==GGML_TYPE_IQ4_NL)?2:16)+block%(format==GGML_TYPE_IQ4_NL?16:32)]^=std::uint8_t(0x5a+replay);
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
            std::size_t at=std::size_t(token)*rows+row;
            if(grouped) {
                const int begin=row<1?0:row<14?1:14;
                const int length=row<1?1:row<14?13:rows-14;
                at=std::size_t(begin)*tokens+token*length+row-begin;
            }
            const auto offset=(at+1)*element_bytes;
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

void run_mixed(const ggml_type (&formats)[3],int columns,int tokens,bool half_output,int& cases) {
    Graph graph;const int lengths[]={1,13,19},element_bytes=half_output?2:4;
    std::vector<float> input(std::size_t(tokens)*columns);
    for(std::size_t i=0;i<input.size();++i)input[i]=std::sin(float(i)*.027f)*.23f;
    Device<float> x(input.size());Device<Q8> q8(std::size_t(tokens)*(columns/32));
    Device<std::uint8_t> position_major(gg::interleaved_activation_bytes(tokens,columns));
    std::vector<Q8> packed(std::size_t(tokens)*(columns/32));
    std::vector<std::vector<float>> decoded;
    std::vector<std::vector<std::uint8_t>> weights,outputs;
    std::vector<Device<std::uint8_t>*> gpu_weights,gpu_outputs;
    gg::DenseProjection projections[3];
    // Own the allocations through unique_ptr so a failed reference check
    // releases storage just like a successful run.
    std::vector<std::unique_ptr<Device<std::uint8_t>>> allocations;
    for(int i=0;i<3;++i) {
        const auto format=formats[i];const auto* traits=ggml_get_type_traits(format);
        const auto stride=ggml_row_size(format,columns);
        const auto block_bytes=traits->type_size;
        weights.emplace_back(std::size_t(lengths[i])*stride);auto& w=weights.back();
        for(std::size_t j=0;j<w.size();++j)w[j]=std::uint8_t(j*37+j/11+23);
        for(std::size_t block=0;block<w.size()/block_bytes;++block) {
            const auto d=__float2half((block%2?-.00031f:.00031f)*float(block%17));
            const auto minimum=__float2half((block%3?-.00017f:.00017f)*float(block%19));
            std::memcpy(w.data()+block*block_bytes+(format==GGML_TYPE_Q6_K?block_bytes-2:0),&d,2);
            if(format==GGML_TYPE_Q4_K || format==GGML_TYPE_Q5_K)
                std::memcpy(w.data()+block*block_bytes+2,&minimum,2);
        }
        decoded.emplace_back(std::size_t(lengths[i])*columns);
        for(int row=0;row<lengths[i];++row)traits->to_float(w.data()+row*stride,decoded.back().data()+row*columns,columns);
        const int prefix=format==GGML_TYPE_Q6_K?2:format==GGML_TYPE_Q5_K && i==2?4:0;
        allocations.push_back(std::make_unique<Device<std::uint8_t>>(w.size()+prefix));gpu_weights.push_back(allocations.back().get());
        outputs.emplace_back((std::size_t(lengths[i])*tokens+2)*element_bytes);
        allocations.push_back(std::make_unique<Device<std::uint8_t>>(outputs.back().size()));gpu_outputs.push_back(allocations.back().get());
        projections[i]={gpu_weights.back()->data+prefix,std::uint64_t(stride),lengths[i],gpu_outputs.back()->data+element_bytes,format};
        check(cudaMemcpyAsync(gpu_weights.back()->data+prefix,w.data(),w.size(),cudaMemcpyHostToDevice,graph.stream));
        check(cudaMemsetAsync(gpu_outputs.back()->data,0x5a,outputs.back().size(),graph.stream));
    }
    check(cudaMemcpyAsync(x.data,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice,graph.stream));
    check(cudaStreamSynchronize(graph.stream));
    check(cudaStreamBeginCapture(graph.stream,cudaStreamCaptureModeThreadLocal));
    gg::quantize_input_interleaved(x.data,false,tokens,columns,position_major.data,q8.data,graph.stream);
    gg::matvec_group(projections,3,columns,q8.data,position_major.data,tokens,half_output,graph.stream);
    check(cudaStreamEndCapture(graph.stream,&graph.graph));check(cudaGraphInstantiate(&graph.exec,graph.graph,0));
    check(cudaGraphLaunch(graph.exec,graph.stream));
    check(cudaMemcpyAsync(packed.data(),q8.data,packed.size()*sizeof(Q8),cudaMemcpyDeviceToHost,graph.stream));
    for(int i=0;i<3;++i)check(cudaMemcpyAsync(outputs[i].data(),gpu_outputs[i]->data,outputs[i].size(),cudaMemcpyDeviceToHost,graph.stream));
    check(cudaStreamSynchronize(graph.stream));
    for(int i=0;i<3;++i) {
        const auto& y=outputs[i];
        for(int byte=0;byte<element_bytes;++byte)
            if(y[byte]!=0x5a || y[y.size()-element_bytes+byte]!=0x5a)
                throw std::runtime_error("Mixed window group overwrote an output guard");
        for(int t=0;t<tokens;++t)for(int row=0;row<lengths[i];++row) {
            double expected=0,absolute=0;
            for(int k=0;k<columns;++k) {
                const auto& a=packed[std::size_t(t)*(columns/32)+k/32];
                const double value=double(decoded[i][row*columns+k])*(float(a.codes[k%32])*__half2float(a.d));
                expected+=value;absolute+=std::abs(value);
            }
            const auto offset=(std::size_t(t)*lengths[i]+row+1)*element_bytes;float actual;
            if(half_output){__half value;std::memcpy(&value,y.data()+offset,2);actual=__half2float(value);}
            else std::memcpy(&actual,y.data()+offset,4);
            const double tolerance=3e-6*absolute+1e-6+(half_output?.0005*std::abs(expected)+1e-6:0);
            if(!std::isfinite(actual) || std::abs(actual-expected)>tolerance)
                throw std::runtime_error("Mixed window group differs from independent double reference: format="+
                    std::to_string(formats[i])+" K="+std::to_string(columns)+" M="+std::to_string(tokens));
        }
    }
    ++cases;
}
}

int main(int argc,char** argv)try {
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices){std::puts("SKIP: CUDA unavailable");return 77;}
    int cases=0;
    const bool affine_only=argc==2 && std::string(argv[1])=="--q4-affine-check";
    if(!affine_only) {
        for(auto format:{GGML_TYPE_Q8_0,GGML_TYPE_Q3_K,GGML_TYPE_IQ4_NL}) {
        for(int columns:{256,512,768,1280,4096,6144,8192,10240})for(int rows:{1,33})
            for(int tokens:{1,2,3,4,5,6,7,8})for(bool half:{false,true})for(int prefix:{0,2})
                run(format,rows,columns,tokens,half,cases,prefix);
        for(int columns:{256,512,768,1280,4096,6144,8192,10240})for(int tokens:{2,3,4})
            for(bool half:{false,true})for(int prefix:{0,2})
                run(format,33,columns,tokens,half,cases,prefix,false,true);
        }
        for(auto format:{GGML_TYPE_Q8_0,GGML_TYPE_IQ4_NL})for(int columns:{32,128})
            for(int tokens:{1,2,3,4,5,6,7,8})for(bool half:{false,true})
                run(format,33,columns,tokens,half,cases,2);
    }
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
    for(int columns:{256,512,768,1280,2560,4096,6144,10240})
        for(int rows:{1,33})for(bool half:{false,true})
            run(GGML_TYPE_IQ4_XS,rows,columns,1,half,cases);
    for(bool half:{false,true})run(GGML_TYPE_IQ4_XS,33,1280,1,half,cases,4);
    for(int columns:{256,1280,4096,6144,10240})for(int rows:{1,33})
        for(int tokens:{2,3,4})for(bool half:{false,true})for(int prefix:{0,4})
            run(GGML_TYPE_IQ4_XS,rows,columns,tokens,half,cases,prefix,false,false,true);
    // Exercise both Q6 block alignments, including one final block with no
    // trailing padding, partial output tiles and changed-input graph replay.
    for(int prefix:{0,2})for(int columns:{256,512,768,1280,2560,4096,6144,10240})
        for(int rows:{1,33})for(int tokens:{1,2,3,4})for(bool half:{false,true})
            run(GGML_TYPE_Q6_K,rows,columns,tokens,half,cases,prefix);
    for(auto format:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS})
        for(int columns:{256,768,4096,6144})for(int tokens:{2,3,4})for(bool half:{false,true})
            run(format,33,columns,tokens,half,cases,0,false,true);
    for(int tokens:{2,3,4})for(bool half:{false,true}) {
        run(GGML_TYPE_Q6_K,33,1280,tokens,half,cases,2,false,true);
        run(GGML_TYPE_Q5_K,33,1280,tokens,half,cases,4,false,true);
    }
    const ggml_type groups[][3]={{GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_Q4_K},
        {GGML_TYPE_IQ4_XS,GGML_TYPE_Q6_K,GGML_TYPE_Q5_K}};
    for(const auto& formats:groups)for(int columns:{256,768,1280,4096,6144})
        for(int tokens:{2,3,4})for(bool half:{false,true})run_mixed(formats,columns,tokens,half,cases);
    for(int columns:{256,512,768,1024,1280,2048,2560,4096,6144,10240})
        for(int rows:{1,33})for(int tokens:{2,3,4})for(bool half:{false,true})
            run(GGML_TYPE_Q5_K,rows,columns,tokens,half,cases);
    for(auto format:{GGML_TYPE_IQ4_XS,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K}) {
    for(int columns:{256,512,768,1024,1280,2048,2560,4096,6144,10240})
        for(int rows:{1,33})for(int tokens:{5,6,7,8})for(bool half:{false,true})
            run(format,rows,columns,tokens,half,cases);
    for(int tokens:{5,6,7,8})for(bool half:{false,true})run(format,33,1280,tokens,half,cases,4);
    }
    for(auto format:{GGML_TYPE_Q4_K,GGML_TYPE_Q5_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ4_XS,
            GGML_TYPE_Q8_0,GGML_TYPE_Q3_K,GGML_TYPE_IQ4_NL}) {
        for(int columns:{256,512,768,1024,1280,2048,2560,4096,6144,10240})
            for(int rows:{1,33})for(int tokens:{5,6,7,8})for(bool half:{false,true})
                run(format,rows,columns,tokens,half,cases,false,true);
        if(format==GGML_TYPE_Q8_0 || format==GGML_TYPE_Q3_K || format==GGML_TYPE_IQ4_NL)
            for(int tokens:{5,6,7,8})for(bool half:{false,true})
                run(format,33,1280,tokens,half,cases,2,true);
        if(format==GGML_TYPE_Q8_0 || format==GGML_TYPE_IQ4_NL)
            for(int columns:{32,128})for(int tokens:{5,6,7,8})for(bool half:{false,true}) {
                const auto row_stride=ggml_row_size(format,columns);
                const bool supported=gg::supports_dense_split_window_q8(format,reinterpret_cast<const void*>(2),
                    row_stride,33,columns,tokens);
                if(supported!=(columns*tokens%64==0))
                    throw std::runtime_error("Small split-window helper alignment differs");
                if(supported)run(format,33,columns,tokens,half,cases,2,true);
            }
        for(int tokens:{1,2,3,4,9})
            if(gg::supports_dense_split_window_q8(format,reinterpret_cast<const void*>(16),
                    ggml_row_size(format,256),1,256,tokens))
                throw std::runtime_error("Incomplete window selected split activation storage");
        if(gg::supports_dense_split_window_q8(format,nullptr,ggml_row_size(format,256),1,256,8))
            throw std::runtime_error("Missing weights selected split activation storage");
    }
    if(gg::supports_dense_split_window_q8(GGML_TYPE_Q5_K,reinterpret_cast<const void*>(4),
            ggml_row_size(GGML_TYPE_Q5_K,256),1,256,8))
        throw std::runtime_error("Unaligned Q5 weights selected vector loads");
    for(auto format:{GGML_TYPE_Q4_K,GGML_TYPE_IQ4_XS})
        if(gg::supports_dense_split_window_q8(format,reinterpret_cast<const void*>(2),
                ggml_row_size(format,256),1,256,6))
            throw std::runtime_error("Unaligned split-window weights selected u32 loads");
    for(int tokens:{1,2,3,4,9})
        if(gg::project_dense_window_q8(GGML_TYPE_IQ4_XS,nullptr,0,1,256,nullptr,tokens,nullptr,false,nullptr))
            throw std::runtime_error("Unsupported window consumed the ordinary projection");
    if(gg::project_dense_window_q8(GGML_TYPE_Q2_K,nullptr,0,1,256,nullptr,8,nullptr,false,nullptr))
        throw std::runtime_error("Unsupported format consumed the ordinary projection");
    std::printf("Q3/Q4/IQ4/Q5/Q6/Q8 native/split window groups scalar-double/F16-F32/M1..8/tails/changed-input-and-weight-graph cases=%d PASS\n",cases);
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
