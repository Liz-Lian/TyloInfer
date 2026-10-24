#include "mfq_cuda_context.h"
#include "tyloinfer/kernels/cuda/mfe_ffn.h"
#include "runtime/moe_pipeline.h"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>

using mfq::cuda::MfePackedProjection;
using mfq::cuda::MoeMappedCopyDescriptor;
using mfq::cuda::SharedStorage;

static_assert(alignof(SharedStorage<MfePackedProjection,3>)==alignof(MfePackedProjection));
static_assert(alignof(SharedStorage<MoeMappedCopyDescriptor>)==alignof(MoeMappedCopyDescriptor));
static_assert(sizeof(SharedStorage<MoeMappedCopyDescriptor>)==sizeof(MoeMappedCopyDescriptor));
static_assert(std::is_trivially_default_constructible_v<SharedStorage<MoeMappedCopyDescriptor>>);

__global__ void shared_copy(const MfePackedProjection* source,MoeMappedCopyDescriptor copy,
        MfePackedProjection* output,MoeMappedCopyDescriptor* copy_output) {
    __shared__ SharedStorage<MfePackedProjection,3> projections;
    __shared__ SharedStorage<MoeMappedCopyDescriptor> descriptor;
    if(threadIdx.x<3)projections.data()[threadIdx.x]=source[threadIdx.x];
    if(!threadIdx.x)*descriptor.data()=copy;
    __syncthreads();
    if(threadIdx.x<3)output[threadIdx.x]=projections.data()[(threadIdx.x+1)%3];
    if(!threadIdx.x)*copy_output=*descriptor.data();
}

int main() {
    try {
        int devices=0;
        if(cudaGetDeviceCount(&devices)!=cudaSuccess || !devices)return 77;
        auto context=std::make_shared<mfq::cuda::Context>(0);
        mfq::cuda::Buffer source(context,3*sizeof(MfePackedProjection));
        mfq::cuda::Buffer output(context,3*sizeof(MfePackedProjection));
        mfq::cuda::Buffer copied(context,sizeof(MoeMappedCopyDescriptor));
        const auto stream=context->stream().get();
        for(int pass=0;pass<2;++pass) {
            std::array<MfePackedProjection,3> inputs{},results{};
            for(int i=0;i<3;++i) {
                inputs[i].family=pass*3+i+1;
                inputs[i].fallback=-inputs[i].family;
                inputs[i].sizes[0]=(std::int64_t{1}<<40)+inputs[i].family;
                inputs[i].decode_record_bytes=inputs[i].sizes[0]+17;
                inputs[i].nvq_bank_map=0x12345678u+inputs[i].family;
                inputs[i].nint_whole=i%2;
            }
            const MoeMappedCopyDescriptor copy{std::uint64_t(pass+1),17,31,4097},empty{};
            auto result=empty;
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(source.data(),inputs.data(),source.size(),cudaMemcpyHostToDevice,stream));
            shared_copy<<<1,32,0,stream>>>(static_cast<const MfePackedProjection*>(source.data()),copy,
                static_cast<MfePackedProjection*>(output.data()),static_cast<MoeMappedCopyDescriptor*>(copied.data()));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaGetLastError());
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(results.data(),output.data(),output.size(),cudaMemcpyDeviceToHost,stream));
            TYLOINFER_NATIVE_CUDA_CHECK(cudaMemcpyAsync(&result,copied.data(),copied.size(),cudaMemcpyDeviceToHost,stream));
            context->stream().synchronize();
            for(int i=0;i<3;++i) {
                const auto& expected=inputs[(i+1)%3];
                const auto& actual=results[i];
                if(actual.family!=expected.family || actual.fallback!=expected.fallback || actual.sizes[0]!=expected.sizes[0]
                    || actual.decode_record_bytes!=expected.decode_record_bytes || actual.nvq_bank_map!=expected.nvq_bank_map
                    || actual.nint_whole!=expected.nint_whole)
                    throw std::runtime_error("shared projection storage changed descriptor fields");
            }
            if(result.destination!=copy.destination || result.source!=copy.source || result.offset!=copy.offset || result.bytes!=copy.bytes)
                throw std::runtime_error("shared copy storage changed descriptor fields");
        }
        std::cout<<"CUDA shared descriptor storage checks passed\n";
        return 0;
    }catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n';return 1;
    }
}
