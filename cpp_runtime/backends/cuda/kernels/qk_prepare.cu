#include "mfq_cuda_attention_ops.h"
#include "kernel_dispatch_trace.h"
#include "reduce.cuh"
#include <climits>
#include <cmath>

namespace {
struct HeadLayout { int64_t batch,head; };

// Match FP16/FP32->FP32 RMSNorm and the FP32 table RoPE arithmetic.
// Q may be a view of interleaved query/gate projection rows.
template<class Projection>
__global__ void qk_norm_rope_cache_kernel(
    const Projection* query,const Projection* key,const Projection* value,
    const float* query_weight,const float* key_weight,
    const int64_t* positions,const int64_t* write_positions,
    const float* cosine,const float* sine,float* query_out,
    __half* key_cache,__half* value_cache,
    HeadLayout q_layout,HeadLayout k_layout,HeadLayout v_layout,
    int query_heads,int key_heads,int width,int pairs,int table_length,
    int64_t capacity,float eps,float weight_offset) {
    const int combined=int(blockIdx.x),heads=query_heads+key_heads;
    const int batch=combined/heads,index=combined%heads;
    const bool is_key=index>=query_heads;
    const int head=is_key?index-query_heads:index;
    const auto layout=is_key?k_layout:q_layout;
    const auto* source=(is_key?key:query)+int64_t(batch)*layout.batch+int64_t(head)*layout.head;
    const auto* weight=is_key?key_weight:query_weight;
    float sum=0.f;
    for(int column=threadIdx.x;column<width;column+=256) {
        const float x=float(source[column]);sum+=x*x;
    }
    sum=block_sum<8>(sum);
    const float inverse=rsqrtf(sum/float(width)+eps);
    const auto normalized=[&](int column) {
        return __fmul_rn(__fmul_rn(float(source[column]),inverse),weight[column]+weight_offset);
    };
    const int64_t position=min(max(int64_t(0),positions[0]),int64_t(table_length-1));
    const int64_t slot=write_positions[0];
    const bool valid=slot>=0 && slot<capacity;
    const int64_t cache_base=((int64_t(batch)*key_heads+head)*capacity+slot)*width;
    const auto store=[&](int column,float x) {
        if(is_key) {if(valid)key_cache[cache_base+column]=__float2half_rn(x);}
        else query_out[(int64_t(batch)*query_heads+head)*width+column]=x;
    };
    for(int column=threadIdx.x;column<width;column+=256) {
        if(column<pairs) {
            const float x=normalized(column),y=normalized(column+pairs);
            const float c=cosine[position*pairs+column],s=sine[position*pairs+column];
            store(column,__fmaf_rn(x,c,-__fmul_rn(y,s)));
            store(column+pairs,__fmaf_rn(x,s,__fmul_rn(y,c)));
        }else if(column>=2*pairs)store(column,normalized(column));
        if(is_key && valid) {
            const auto at=int64_t(batch)*v_layout.batch+int64_t(head)*v_layout.head+column;
            value_cache[cache_base+column]=__float2half_rn(float(value[at]));
        }
    }
}
}

mfq_tensor_backend::Tensor qk_norm_rope_cache_cuda(
    mfq_tensor_backend::Tensor query,mfq_tensor_backend::Tensor key,mfq_tensor_backend::Tensor value,
    mfq_tensor_backend::Tensor query_weight,mfq_tensor_backend::Tensor key_weight,
    mfq_tensor_backend::Tensor positions,mfq_tensor_backend::Tensor write_positions,
    mfq_tensor_backend::Tensor cosine,mfq_tensor_backend::Tensor sine,
    mfq_tensor_backend::Tensor key_cache,mfq_tensor_backend::Tensor value_cache,
    int64_t rotary_dim,double eps,double weight_offset) {
    namespace tb=mfq_tensor_backend;
    const auto device=query.device();
    const auto input=[&](const tb::Tensor& x) {
        return x.is_cuda() && x.device()==device && (x.scalar_type()==tb::kFloat16 || x.scalar_type()==tb::kFloat32) &&
            x.dim()==4 && x.size(0)>0 && x.size(1)>0 && x.size(2)==1 && x.stride(3)==1;
    };
    TYLOINFER_RUNTIME_CHECK(input(query) && input(key) && input(value) &&
        query.size(0)==key.size(0) && query.size(3)==key.size(3) && key.sizes()==value.sizes() &&
        query.scalar_type()==key.scalar_type() && key.scalar_type()==value.scalar_type(),
        "QK preparation requires single-token CUDA FP16/FP32 head views");
    const int64_t width=query.size(3),rows=query.size(0)*(query.size(1)+key.size(1));
    TYLOINFER_RUNTIME_CHECK(width>0 && width<=INT_MAX && query.size(1)<=INT_MAX &&
        key.size(1)<=INT_MAX && query.size(1)+key.size(1)<=INT_MAX && rows<=INT_MAX &&
        rotary_dim>0 && rotary_dim<=width && rotary_dim%2==0 &&
        std::isfinite(eps) && eps>=0 && std::isfinite(weight_offset),
        "QK preparation geometry/epsilon differs");
    const auto f32=[&](const tb::Tensor& x){return x.is_cuda() && x.device()==device &&
        x.scalar_type()==tb::kFloat32 && x.is_contiguous();};
    TYLOINFER_RUNTIME_CHECK(f32(query_weight) && f32(key_weight) &&
        query_weight.dim()==1 && key_weight.dim()==1 &&
        query_weight.numel()==width && key_weight.numel()==width &&
        f32(cosine) && f32(sine) && cosine.dim()==2 && cosine.size(0)>0 &&
        cosine.size(0)<=INT_MAX && cosine.size(1)==rotary_dim/2 && sine.sizes()==cosine.sizes(),
        "QK preparation norm/rotary storage differs");
    const auto pos=[&](const tb::Tensor& x){return x.is_cuda() && x.device()==device &&
        x.scalar_type()==tb::kInt64 && x.is_contiguous() && x.dim()==1 && x.numel()==1;};
    TYLOINFER_RUNTIME_CHECK(pos(positions) && pos(write_positions),"QK preparation positions differ");
    TYLOINFER_RUNTIME_CHECK(key_cache.is_cuda() && key_cache.device()==device &&
        key_cache.scalar_type()==tb::kFloat16 && key_cache.dim()==4 && key_cache.is_contiguous() &&
        value_cache.is_cuda() && value_cache.device()==device && value_cache.is_contiguous() &&
        value_cache.scalar_type()==tb::kFloat16 && value_cache.sizes()==key_cache.sizes() &&
        key_cache.size(0)==key.size(0) && key_cache.size(1)==key.size(1) &&
        key_cache.size(2)>0 && key_cache.size(3)==width,"QK preparation cache storage differs");
    MfqCudaGuard guard(device);
    auto output=tb::empty(query.sizes(),query.options().dtype(tb::kFloat32));
    const auto layout=[](const tb::Tensor& x){return HeadLayout{x.stride(0),x.stride(1)};};
    const auto stream=mfq_current_cuda_stream();
    const auto launch=[&]<class Projection>() {
        const auto kernel=qk_norm_rope_cache_kernel<Projection>;
        kernel<<<unsigned(rows),256,0,stream>>>(
            reinterpret_cast<const Projection*>(query.data_ptr()),reinterpret_cast<const Projection*>(key.data_ptr()),
            reinterpret_cast<const Projection*>(value.data_ptr()),query_weight.data_ptr<float>(),key_weight.data_ptr<float>(),
            positions.data_ptr<int64_t>(),write_positions.data_ptr<int64_t>(),cosine.data_ptr<float>(),sine.data_ptr<float>(),
            output.data_ptr<float>(),reinterpret_cast<__half*>(key_cache.data_ptr()),reinterpret_cast<__half*>(value_cache.data_ptr()),
            layout(query),layout(key),layout(value),int(query.size(1)),int(key.size(1)),int(width),int(rotary_dim/2),
            int(cosine.size(0)),key_cache.size(2),float(eps),float(weight_offset));
        TYLOINFER_CUDA_KERNEL_LAUNCH_CHECK();
        if(mfq::cuda::dispatch_trace::enabled())mfq::cuda::dispatch_trace::record(stream,"qk_norm_rope_cache",
            reinterpret_cast<const void*>(kernel),"native heads -> FP32 Q + FP16 KV");
    };
    if(query.scalar_type()==tb::kFloat16)launch.template operator()<__half>();
    else launch.template operator()<float>();
    return output;
}
