#include "packed_backward.h"
#include <ATen/cuda/CUDAContext.h>

namespace mfq_jit_backward {
void launch_half_gemm_nn(const __half* gradient,const __half* weight,__half* output,
        int rows,int outputs,int width,cudaStream_t stream) {
    const auto handle=at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS,"packed backward cublasSetStream failed");
    TORCH_CHECK(raw_half_gemm_nn(handle,gradient,weight,output,rows,outputs,width)==CUBLAS_STATUS_SUCCESS,"packed backward GEMM failed");
}
void launch_float_gemm_nn(const float* gradient,const float* weight,float* output,
        int rows,int outputs,int width,cudaStream_t stream) {
    const auto handle=at::cuda::getCurrentCUDABlasHandle();
    TORCH_CHECK(cublasSetStream(handle,stream)==CUBLAS_STATUS_SUCCESS,"packed backward cublasSetStream failed");
    TORCH_CHECK(raw_float_gemm_nn(handle,gradient,weight,output,rows,outputs,width)==CUBLAS_STATUS_SUCCESS,"packed backward FP32 GEMM failed");
}
void launch_dense_half_weight(const torch::Tensor& gradient,const torch::Tensor& weight,torch::Tensor& output,
        int rows,int outputs,int width,cudaStream_t stream) {
    const auto dtype=gradient.scalar_type();
    if(dtype==at::kHalf) {
        launch_half_gemm_nn(reinterpret_cast<const __half*>(gradient.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),reinterpret_cast<__half*>(output.data_ptr<at::Half>()),rows,outputs,width,stream);
        return;
    }
    if(dtype==at::kBFloat16) {
        auto half_gradient=torch::empty({rows,outputs},weight.options());
        auto half_result=torch::empty({rows,width},weight.options());
        launch_cast<__half,__nv_bfloat16>(reinterpret_cast<const __nv_bfloat16*>(gradient.data_ptr<at::BFloat16>()),
            reinterpret_cast<__half*>(half_gradient.data_ptr<at::Half>()),static_cast<int64_t>(rows)*outputs,stream);
        launch_half_gemm_nn(reinterpret_cast<const __half*>(half_gradient.data_ptr<at::Half>()),
            reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),reinterpret_cast<__half*>(half_result.data_ptr<at::Half>()),rows,outputs,width,stream);
        launch_cast<__nv_bfloat16,__half>(reinterpret_cast<const __half*>(half_result.data_ptr<at::Half>()),
            reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),static_cast<int64_t>(rows)*width,stream);
        return;
    }
    auto float_weight=torch::empty({outputs,width},weight.options().dtype(at::kFloat));
    launch_cast<float,__half>(reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),float_weight.data_ptr<float>(),static_cast<int64_t>(outputs)*width,stream);
    launch_float_gemm_nn(gradient.data_ptr<float>(),float_weight.data_ptr<float>(),output.data_ptr<float>(),rows,outputs,width,stream);
}
}
