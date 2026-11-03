#pragma once
#include <cuda_runtime_api.h>
#include <array>
#include <cstddef>
#include <cstdint>

namespace tyloinfer::qsa {

// Payload precision remains row-dynamic 1..8 bits in either profile.
enum class TqEncodeProfile : int {Fast=14,Quality=16};

struct TqEncodePlan {
    int blocks=0,rows_per_block=0,rows=0,state_bits=16;
    float book_rms=1,relative=1;
    std::array<int,8> waves{};
    std::size_t values=0,symbols=0,bits=0,row_images=0,row_sse=0,ids=0,counts=0,trace=0;
    std::size_t workspace_bytes=0;
};

// Plan creation queries actual kernel resources. A zero workspace budget
// requests full device occupancy; a positive budget limits temporary storage.
// No memory is allocated by these functions. Create plans before graph capture.
int tq_make_encode_plan(TqEncodePlan* plan,int blocks,int rows_per_block,
    TqEncodeProfile profile,std::size_t workspace_budget=0);

// Input: immutable [blocks,rows_per_block,256] H4/RHT modes. Budgets count
// payload bits per value summed over one block's rows, independently of scale
// metadata. Per-block budgets are optional; otherwise use default_budget.
// Output images use the existing compact layout, with flags suitable for
// TqBlockView. Graph replay needs no allocations, host reads, or synchronization.
// device_status is reset and receives asynchronous input/output-bound errors;
// check it before publishing the images. Sizes/flags of invalid images are zero.
int tq_encode_modes(const TqEncodePlan& plan,const float* input,
    const std::int32_t* budgets,int default_budget,float initial_scale,
    std::uint8_t* images,std::size_t image_stride,std::uint32_t* sizes,
    std::uint32_t* flags,int* device_status,void* workspace,
    std::size_t workspace_bytes,cudaStream_t stream);

} // namespace tyloinfer::qsa
