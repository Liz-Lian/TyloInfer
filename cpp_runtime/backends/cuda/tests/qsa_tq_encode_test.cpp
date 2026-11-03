#include "../kernels/qsa_tq_encode.h"
#include "../kernels/qsa_tq.h"
#include "tyloinfer/qsa_tq_cpu.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace tyloinfer::qsa;
void check(int result) {
    if(result)throw std::runtime_error(cudaGetErrorString(static_cast<cudaError_t>(result)));
}
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
struct Buffer {
    void* data=nullptr;
    explicit Buffer(std::size_t bytes){check(cudaMalloc(&data,bytes));}
    ~Buffer(){cudaFree(data);}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    template<class T>T* as(){return static_cast<T*>(data);}
};
void run(TqEncodeProfile profile,int rows) {
    constexpr int blocks=4;
    TqEncodePlan plan;
    check(tq_make_encode_plan(&plan,blocks,rows,profile));
    const std::size_t stride=std::size_t(rows)*260+32;
    std::vector<float> input(blocks*rows*256);
    for(int b=0;b<blocks;++b)for(int r=0;r<rows;++r)for(int d=0;d<256;++d)
        input[(b*rows+r)*256+d]=r?(r==1?8.f:1.f)*std::sin(d*.37f+r+b*.13f):0;
    std::vector<int> budgets{rows,rows*2,rows*4,rows*8};
    Buffer x(input.size()*4),targets(blocks*4),work(plan.workspace_bytes),images(blocks*stride),sizes(blocks*4),flags(blocks*4),status(4);
    check(cudaMemcpy(x.data,input.data(),input.size()*4,cudaMemcpyHostToDevice));
    check(cudaMemcpy(targets.data,budgets.data(),blocks*4,cudaMemcpyHostToDevice));
    const auto encode=[&] {
        check(tq_encode_modes(plan,x.as<float>(),targets.as<int>(),0,1.1f,images.as<std::uint8_t>(),stride,
            sizes.as<std::uint32_t>(),flags.as<std::uint32_t>(),status.as<int>(),work.data,plan.workspace_bytes,nullptr));
    };
    const auto verify=[&] {
        check(cudaDeviceSynchronize());int error=0;
        check(cudaMemcpy(&error,status.data,4,cudaMemcpyDeviceToHost));require(!error,"native encode device status");
        std::vector<std::uint8_t> bytes(blocks*stride);
        std::vector<std::uint32_t> lengths(blocks),tags(blocks);
        check(cudaMemcpy(bytes.data(),images.data,bytes.size(),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(lengths.data(),sizes.data,blocks*4,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(tags.data(),flags.data,blocks*4,cudaMemcpyDeviceToHost));
        std::vector<float> expected(input.size());
        for(int block=0;block<blocks;++block) {
            const auto* src=bytes.data()+block*stride;
            const int tag=tags[block]&255,width=tags[block]>>8?tags[block]>>8:16;
            require(width==int(profile),"trellis profile not preserved");
            const int map=2*rows+(rows+1)/2,header=(map+(tag==255?(3*rows+7)/8:0)+3)&~3;
            int cursor=header,budget=0;
            for(int r=0;r<rows;++r) {
                int bits=tag;
                if(tag==255) {
                    const int bit=r*3,shift=bit&7;
                    unsigned value=src[map+bit/8]>>shift;
                    if(shift>5)value|=unsigned(src[map+bit/8+1])<<(8-shift);
                    bits=1+(value&7);
                }
                require(bits>=1 && bits<=8,"invalid row precision");budget+=bits;
                require(cursor+32*bits<=int(lengths[block]),"packed row exceeds image");
                CpuTqRow row;row.bits=std::uint8_t(bits);row.state_bits=std::uint8_t(width);
                row.anchor=src[2*r]|std::uint16_t(src[2*r+1])<<8;
                row.selector=(src[2*rows+r/2]>>((r&1)*4))&15;
                row.payload.assign(src+cursor,src+cursor+32*bits);cursor+=32*bits;
                const auto decoded=decode_cpu_tq_row(row);
                std::copy(decoded.begin(),decoded.end(),expected.begin()+(block*rows+r)*256);
            }
            require(cursor==int(lengths[block]) && budget==budgets[block],"payload budget changed");
        }
        std::vector<TqBlockView> descriptors(blocks/2);
        for(int block=0;block<blocks/2;++block) {
            descriptors[block]={images.as<std::uint8_t>()+block*stride,images.as<std::uint8_t>()+(block+blocks/2)*stride,tags[block],tags[block+blocks/2]};
        }
        const float relative7=std::exp2(-2.f+4.f*7/15.f);
        require(std::abs(plan.relative-relative7)<.001f,"relative scale mismatch");
        Buffer views(descriptors.size()*sizeof(TqBlockView)),lut(32),decoded(input.size()*4);
        // Use the anchor conversion available in the CUDA runtime header.
        std::vector<unsigned short> scales(16);
        for(int i=0;i<16;++i)scales[i]=__half_as_ushort(__float2half_rn(std::exp2(-2.f+4.f*i/15.f)));
        check(cudaMemcpy(views.data,descriptors.data(),descriptors.size()*sizeof(TqBlockView),cudaMemcpyHostToDevice));
        check(cudaMemcpy(lut.data,scales.data(),32,cudaMemcpyHostToDevice));
        check(tq_decode_modes(views.as<TqBlockView>(),decoded.as<float>(),decoded.as<float>()+input.size()/2,
            lut.as<std::uint16_t>(),blocks/2,rows/4,256,nullptr));
        std::vector<float> actual(input.size());check(cudaMemcpy(actual.data(),decoded.data,actual.size()*4,cudaMemcpyDeviceToHost));
        require(actual==expected,"CUDA decoder differs from CPU row reference");
    };
    encode();verify();
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    cudaStream_t stream=nullptr;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    check(tq_encode_modes(plan,x.as<float>(),targets.as<int>(),0,1.1f,images.as<std::uint8_t>(),stride,
        sizes.as<std::uint32_t>(),flags.as<std::uint32_t>(),status.as<int>(),work.data,plan.workspace_bytes,stream));
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
    budgets={rows*8,rows*3,rows*2,rows};check(cudaMemcpy(targets.data,budgets.data(),blocks*4,cudaMemcpyHostToDevice));
    check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));verify();
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    budgets[0]=0;check(cudaMemcpy(targets.data,budgets.data(),blocks*4,cudaMemcpyHostToDevice));encode();
    int error=0;check(cudaMemcpy(&error,status.data,4,cudaMemcpyDeviceToHost));require(error!=0,"invalid budget was accepted");
}
} // namespace
int main() try {
    int devices=0;const auto available=cudaGetDeviceCount(&devices);
    if(available==cudaErrorNoDevice || available==cudaErrorInsufficientDriver || !devices){std::cout<<"SKIP: CUDA device unavailable\n";return 77;}
    check(available);
    run(TqEncodeProfile::Fast,4);run(TqEncodeProfile::Fast,12);
    TqEncodePlan quality;const int support=tq_make_encode_plan(&quality,4,8,TqEncodeProfile::Quality);
    if(!support)run(TqEncodeProfile::Quality,8);
    else if(support==int(cudaErrorNotSupported))
        std::cout<<"SKIP TQ16: full-state kernel resources unavailable\n";
    else check(support);
    std::cout<<"QSA native TQ encode/decode and graph checks PASS\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
