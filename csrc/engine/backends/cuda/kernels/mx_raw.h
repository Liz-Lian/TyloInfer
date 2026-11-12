#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::mx_raw {
inline constexpr int prefill_rows=64, prefill_threads=prefill_rows/16*128;
/** Describe contiguous MX buffers, geometry and optional routed metadata. */
struct Args {
    const std::uint8_t *values=nullptr,*scales=nullptr;
    const void* input=nullptr; void* output=nullptr;
    const std::int64_t* ids64=nullptr;
    const std::int32_t *ids=nullptr,*expert_local=nullptr,*ids_dst=nullptr,*expert_bounds=nullptr,*tile_bounds=nullptr,*tile_experts=nullptr;
    int rows=0,outputs=0,width=0,row_shift=0,column_shift=0,groups=1,outputs_per_group=0;
    int tokens=0,routes=0,global_experts=0,pool_experts=0,out_per_expert=0,max_tiles=0;
    bool output_float=false,routed_input=false,compact=false,tensor_core=false;
};
/** Decode MXFP8 weights to F16. */
void dequant8(const Args&,cudaStream_t);
/** Decode selected MXFP8 embedding rows. */
void embed8(const Args&,cudaStream_t);
/** Apply the original aligned or unaligned MXFP8 small-M projection. */
void small8(const Args&,cudaStream_t);
/** Apply the original MXFP8 prefill/groupwise GEMM. */
void gemm8(const Args&,cudaStream_t);
/** Decode MXFP4 weights to F16. */
void dequant4(const Args&,cudaStream_t);
/** Decode selected MXFP4 embedding rows. */
void embed4(const Args&,cudaStream_t);
/** Apply the original MXFP4 dense projection. */
void gemm4(const Args&,cudaStream_t);
/** Apply the original MXFP4 routed projection in place. */
void routed4(const Args&,cudaStream_t);
}
