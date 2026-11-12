#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::gdn_raw {
/** Describe recurrence buffers and the original dispatch policy. */
struct Recurrence {
    const float *q,*k,*v,*g,*beta,*state;
    float *output,*next,*decay;
    int batch,query_heads,value_heads,tokens,width,columns;
    bool kda,transposed,tiled,warp,column,short_columns,prepared_decay,pipelined,staged;
};
/** Describe fused decode inputs, output and optional state storage. */
struct Decode {
    const void *qkv,*gate,*alpha,*beta;
    const float *conv_state,*state,*weight,*dt_bias,*a_log,*norm;
    void* output;
    float *conv_out,*state_out,*workspace,*checkpoint_conv,*checkpoint_state;
    const int64_t* state_addresses;
    int batch,key_heads,value_heads,width,taps,tiles,layout,mode;
    float conv_eps; double norm_eps;
    bool silu,output_half,gate_half,transposed,inplace_conv,fixed_geometry,warp_reductions,probe;
    int64_t alpha_stride0,alpha_stride2,beta_stride0,beta_stride2,qkv_stride0,gate_stride0;
};
/** Launch recurrence with optional caller-owned prepared-decay workspace. */
void recurrence(const Recurrence&,cudaStream_t);
/** Launch fused decode and optional independent-state commit. */
void decode(const Decode&,cudaStream_t);
}
