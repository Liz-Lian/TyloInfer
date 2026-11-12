#pragma once
#include "numeric_type.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace tyloi::cuda::rope_raw {
/** Select the historical wide-position conversion arithmetic. */
enum class PositionPolicy { native_double, torch_int32 };
/** Tensor strides in elements for normalized rotary projection. */
struct Layout { int64_t batch, token, head, column; };
/** Pointer and geometry description for one normalized projection. */
struct Projection {
    const void* input; const float* weight; const void* positions;
    const float* frequencies; const int32_t* selected_axes; void* output;
    int heads,tokens,width,pairs,axes,batches,maximum,threads;
    bool wide_positions; float eps; Layout layout; void* key_cache;
    const void* cache_positions; bool wide_cache_positions; int64_t capacity;
    const void* projected_value; bool float_value; Layout value_layout;
    void* value_cache; int64_t rows; PositionPolicy position_policy;
};
/** Up to three normalized projections sharing one CUDA launch. */
struct Group { Projection entries[3]; int count; };
/** Pool and normalize the index cache in place. */
void pool(const void*,const float*,const void*,bool,const int64_t*,const float*,const int32_t*,void*,
    int,int,int,int,int,int,int,float,int64_t,int64_t,NumericType,PositionPolicy,cudaStream_t);
/** Launch one normalized rotary projection. */
void normalized(const Projection&,NumericType,cudaStream_t);
/** Launch a same-dtype normalized rotary group. */
void normalized_group(const Group&,int64_t,int,NumericType,cudaStream_t);
/** Apply cached rotary frequencies to a contiguous projection. */
void cached(const void*,const int32_t*,const float*,const int32_t*,void*,int64_t,
    int,int,int,int,int,int,int,NumericType,cudaStream_t);
/** Apply F32 rotary embedding with on-the-fly frequencies. */
void rope(const float*,const float*,float*,int,int,int,int,int,int,int,int,int,float,cudaStream_t);
/** Apply F32 rotary embedding using a frequency table. */
void table(const float*,const int64_t*,const float*,const float*,float*,int,int,int,int,int,int,int,int,int,int,cudaStream_t);
/** Apply BF16 rotary embedding using a frequency table. */
void table_bf16(const void*,const int64_t*,const float*,const float*,void*,int,int,int,int,int,cudaStream_t);
/** Rotate MiniCPM queries and write BF16 key/value cache in place. */
void minicpm(const void*,const void*,const void*,const int64_t*,const int64_t*,const float*,const float*,
    void*,void*,void*,int,int,int,int,int,cudaStream_t);
}
