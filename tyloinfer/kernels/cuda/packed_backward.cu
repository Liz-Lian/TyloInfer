#include "packed_backward_launch.h"
#include "packed_backward_raw.cuh"

namespace mfq_jit_backward {
namespace raw = ::tyloi::cuda::backward_raw;
template<> void launch_cast<__half, __nv_bfloat16>(const __nv_bfloat16* x, __half* y, std::int64_t n, cudaStream_t s) { raw::launch_cast<__half, __nv_bfloat16>(x,y,n,s); }
template<> void launch_cast<__nv_bfloat16, __half>(const __half* x, __nv_bfloat16* y, std::int64_t n, cudaStream_t s) { raw::launch_cast<__nv_bfloat16, __half>(x,y,n,s); }
template<> void launch_cast<float, __half>(const __half* x, float* y, std::int64_t n, cudaStream_t s) { raw::launch_cast<float, __half>(x,y,n,s); }
void launch_split_float_reduce_to_half(const float* x,__half* y,int rows,int width,int splits,cudaStream_t s) { raw::launch_split_float_reduce_to_half(x,y,rows,width,splits,s); }
void launch_split_half_reduce_to_half(const __half* x,__half* y,int rows,int width,int splits,cudaStream_t s) { raw::launch_split_half_reduce_to_half(x,y,rows,width,splits,s); }
cublasStatus_t raw_half_gemm_nn(cublasHandle_t h,const __half* x,const __half* w,__half* y,int rows,int outputs,int width) { return raw::half_gemm_nn(h,x,w,y,rows,outputs,width); }
cublasStatus_t raw_float_gemm_nn(cublasHandle_t h,const float* x,const float* w,float* y,int rows,int outputs,int width) { return raw::float_gemm_nn(h,x,w,y,rows,outputs,width); }
}
