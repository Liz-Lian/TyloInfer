#include "selected_raw.h"

#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>

namespace tyloi::cuda::selected_raw {
namespace {

__global__ void prepare_sparse_indices_kernel(
    const int32_t * __restrict__ source,
    int32_t * __restrict__ indices,
    half * __restrict__ mask,
    int64_t rows,
    int source_count,
    int padded_count,
    int cache_length) {
    const int64_t total = rows * padded_count;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int slot = static_cast<int>(linear % padded_count);
        const int64_t row = linear / padded_count;
        const int index = slot < source_count
            ? source[row * source_count + slot]
            : -1;
        const bool valid = index >= 0 && index < cache_length;
        indices[linear] = valid ? index : 0;
        mask[linear] = valid ? __float2half(0.0f) : __float2half(-CUDART_INF_F);
    }
}

// Each CTA retains the Metal kernel's eight warp partials and online-softmax
// recurrence. The generic wide-D path stores accumulators in its output only;
// it imposes no extra head-width limit. Published D=256/512 stay in registers.
__device__ float apply_attention_gate(float value, float gate, Sigmoid policy) {
    float sigmoid;
    if (policy == Sigmoid::native_double)
        sigmoid = float(__ddiv_rn(1.0, __dadd_rn(1.0, ::exp(-double(gate)))));
    else if (policy == Sigmoid::native_float)
        sigmoid = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-gate)));
    else
        sigmoid = 1.f / (1.f + expf(-gate));
    return __fmul_rn(value, sigmoid);
}

__device__ void store_attention(float value, float* output, int64_t offset,
    int64_t batch, int64_t token, int64_t head, int64_t column,
    const Epilogue& epilogue) {
    if (!epilogue.gate) { output[offset] = value; return; }
    const auto at = batch * epilogue.layout.batch + token * epilogue.layout.token +
        head * epilogue.layout.head + column * epilogue.layout.column;
    const float gate = epilogue.float_gate ? static_cast<const float*>(epilogue.gate)[at]
        : __half2float(static_cast<const half*>(epilogue.gate)[at]);
    value = apply_attention_gate(value, gate, epilogue.sigmoid);
    if (epilogue.half_output) static_cast<half*>(epilogue.output)[offset] = __float2half_rn(value);
    else static_cast<float*>(epilogue.output)[offset] = value;
}

template<int Slices, bool use_hot = false>
__global__ void sparse_attention_kernel(
    const void* query, const half* key, const half* value, const int32_t* indices,
    float* output, int64_t heads, int64_t kv_heads, int64_t tokens,
    int64_t width, int64_t cache_length, int64_t topk, float scale,
    bool half_query, const void* causal_positions, bool wide_positions,
    Epilogue epilogue, sparse_mma_raw::Hot hot) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int64_t head = blockIdx.x % heads;
    const int64_t row = blockIdx.x / heads;
    const int64_t token = row % tokens;
    const int64_t batch = row / tokens;
    const int64_t kv_head = head / (heads / kv_heads);
    const int64_t qb = ((batch * heads + head) * tokens + token) * width;
    const int64_t ob = ((batch * tokens + token) * heads + head) * width;
    __shared__ float partials[8];
    __shared__ float online[4];
    float accum[Slices > 0 ? Slices : 1] = {};
    if constexpr (Slices == 0) {
        for (int64_t d = tid; d < width; d += 256) output[ob + d] = 0.f;
    }
    if (tid == 0) {
        online[0] = -CUDART_INF_F;
        online[1] = online[2] = online[3] = 0.f;
    }
    __syncthreads();
    for (int64_t selected = 0; selected < topk; ++selected) {
        int64_t index;
        if (causal_positions) {
            const int64_t end = wide_positions ? static_cast<const int64_t*>(causal_positions)[token]
                : static_cast<const int32_t*>(causal_positions)[token];
            index = selected <= end ? selected : -1;
        } else index = indices[row * topk + selected];
        const bool valid = index >= 0 && index < cache_length;
        const int64_t cb = ((batch * kv_heads + kv_head) * cache_length +
            (valid ? index : 0)) * width;
        const half* source_key=key+cb;
        const half* source_value=value+cb;
        if constexpr(use_hot) {
            if(valid) {const int slot=hot.mapping[index];if(slot>=0) {
                const auto at=((batch*kv_heads+kv_head)*hot.cells+slot)*width;
                source_key=static_cast<const half*>(hot.keys)+at;
                source_value=static_cast<const half*>(hot.values)+at;
            }else if(hot.cold_mapping) {
                const auto at=((batch*kv_heads+kv_head)*hot.cold_cells+hot.cold_mapping[index])*width;
                source_key=key+at;source_value=value+at;
            }}
        }
        float dot = 0.f;
        if (valid) {
            for (int64_t d = tid; d < width; d += 256) {
                const float q = half_query ? __half2float(static_cast<const half*>(query)[qb + d])
                    : static_cast<const float*>(query)[qb + d];
                dot += q * __half2float(source_key[d]);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1)
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        if (lane == 0) partials[warp] = dot;
        __syncthreads();
        if (tid == 0) {
            float score = 0.f;
            for (int group = 0; group < 8; ++group) score += partials[group];
            score = valid ? score * scale : -CUDART_INF_F;
            const float old_max = online[0];
            const float next_max = fmaxf(old_max, score);
            const float old_scale = isfinite(old_max) ? expf(old_max - next_max) : 0.f;
            const float next_scale = valid ? expf(score - next_max) : 0.f;
            online[0] = next_max;
            online[1] = online[1] * old_scale + next_scale;
            online[2] = old_scale;
            online[3] = next_scale;
        }
        __syncthreads();
        if constexpr (Slices > 0) {
            #pragma unroll
            for (int slice = 0; slice < Slices; ++slice) {
                const int64_t d = tid + slice * 256;
                const float v = valid && d < width ? __half2float(source_value[d]) : 0.f;
                accum[slice] = accum[slice] * online[2] + v * online[3];
            }
        } else {
            for (int64_t d = tid; d < width; d += 256) {
                const float v = valid ? __half2float(source_value[d]) : 0.f;
                output[ob + d] = output[ob + d] * online[2] + v * online[3];
            }
        }
        __syncthreads();
    }
    const float inverse = online[1] > 0.f ? 1.f / online[1] : 0.f;
    if constexpr (Slices > 0) {
        #pragma unroll
        for (int slice = 0; slice < Slices; ++slice) {
            const int64_t d = tid + slice * 256;
            if (d < width) store_attention(accum[slice] * inverse, output, ob + d,
                batch, token, head, d, epilogue);
        }
    } else {
        for (int64_t d = tid; d < width; d += 256)
            store_attention(output[ob + d] * inverse, output, ob + d,
                batch, token, head, d, epilogue);
    }
}

__global__ void attention_gated_kernel(const void* attended, bool half_input,
    Layout layout, Epilogue epilogue, int64_t count,
    int64_t heads, int64_t tokens, int64_t width) {
    for (int64_t linear = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count; linear += int64_t(gridDim.x) * blockDim.x) {
        const int64_t column = linear % width, row = linear / width;
        const int64_t head = row % heads, token = (row / heads) % tokens,
            batch = row / (heads * tokens);
        const auto at = batch * layout.batch + token * layout.token + head * layout.head + column * layout.column;
        const float value = half_input ? __half2float(static_cast<const half*>(attended)[at])
            : static_cast<const float*>(attended)[at];
        store_attention(value, nullptr, linear, batch, token, head, column, epilogue);
    }
}

template<class BlockIndex, class Position>
__global__ void block_indices_to_tokens_kernel(const BlockIndex* blocks,
    const Position* positions, int32_t* output, int64_t rows, int tokens,
    int block_count, int pool, int budget, int columns) {
    for (int64_t linear = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
         linear < rows*columns; linear += int64_t(gridDim.x)*blockDim.x) {
        const int slot = int(linear%columns);
        const int64_t row = linear/columns;
        const int64_t position = positions[row%tokens];
        int64_t selected = -1;
        if (position+1 <= budget) {
            if (slot <= position) selected = slot;
        } else if (int64_t(slot) < int64_t(block_count)*pool) {
            const int64_t block = blocks[row*block_count+slot/pool];
            if (block*pool+pool-1 <= position) selected = block*pool+slot%pool;
        } else if (slot >= budget) {
            const int offset = slot-budget;
            const int64_t tail = (position+1)%pool;
            if (offset < tail) selected = position+1-tail+offset;
        }
        output[linear] = int32_t(selected);
    }
}

__global__ void prepare_causal_attention_kernel(const void* q, bool half_query,
    Layout layout, float* query, int64_t query_count, int64_t heads,
    int64_t tokens, int64_t width, const void* positions, bool wide_positions,
    int32_t* indices, half* mask, int64_t index_count, int64_t columns,
    int64_t padded_columns, int64_t capacity) {
    const int64_t first = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t step = int64_t(gridDim.x) * blockDim.x;
    for (int64_t linear = first; linear < query_count; linear += step) {
        const int64_t column = linear % width, row = linear / width;
        const int64_t token = row % tokens, head = (row / tokens) % heads,
            batch = row / (tokens * heads);
        const auto at = batch * layout.batch + token * layout.token +
            head * layout.head + column * layout.column;
        query[linear] = half_query ? __half2float(static_cast<const half*>(q)[at])
            : static_cast<const float*>(q)[at];
    }
    for (int64_t linear = first; linear < index_count; linear += step) {
        const int64_t index = linear % padded_columns,
            token = (linear / padded_columns) % tokens;
        const int64_t end = wide_positions ? static_cast<const int64_t*>(positions)[token]
            : static_cast<const int32_t*>(positions)[token];
        const bool valid = index < columns && index <= end && index < capacity;
        indices[linear] = valid ? int32_t(index) : 0;
        mask[linear] = __float2half_rn(valid ? 0.f : -CUDART_INF_F);
    }
}

__global__ void mma_reduce_gate_serial_kernel(const float* partial, const float2* metadata,
    const void* gate, void* output, Layout gate_layout, int tokens, int heads,
    int kv_heads, int width, int total_blocks, int parts, int query_columns, int head_columns,
    bool float_gate, bool half_output, Sigmoid sigmoid) {
    const int head=blockIdx.x%heads, row=blockIdx.x/heads;
    const int token=row%tokens, batch=row/tokens, lane=threadIdx.x;
    const int grouped_heads=heads/kv_heads;
    const int head_tiles=(grouped_heads+head_columns-1)/head_columns;
    const int token_tiles=(tokens+query_columns-1)/query_columns;
    const int tile=(((batch*kv_heads+head/grouped_heads)*head_tiles+
        (head%grouped_heads)/head_columns)*token_tiles+token/query_columns);
    const int tile_column=(token%query_columns)*head_columns+(head%grouped_heads)%head_columns;
    const int columns=query_columns*head_columns, last=(tile+1)*parts-1;
    const size_t output_base=size_t(row)*heads*width+head*width;
    const float* contributions=reinterpret_cast<const float*>(metadata)+size_t(total_blocks)*4*columns;
    extern __shared__ float2 factors[];
    __shared__ float normalizer;
    if(lane==0) {
        float denominator=1.f;
        if(parts>1) {
            auto statistics=metadata[size_t(last)*columns+tile_column];
            float peak=statistics.x;
            denominator=statistics.y;
            for(int step=1;step<parts;++step) {
                statistics=metadata[size_t(total_blocks+last-step)*columns+tile_column];
                const float maximum=fmaxf(peak,statistics.x);
                const float previous=peak-maximum, incoming=statistics.x-maximum;
                const float lhs=previous>=-20.f?expf(previous):0.f;
                const float rhs=incoming>=-20.f?expf(incoming):0.f;
                factors[step-1]=make_float2(lhs,rhs);
                denominator=denominator*lhs+statistics.y*rhs;
                peak=maximum;
            }
        }
        normalizer=denominator;
    }
    __syncthreads();
    for(int column=lane;column<width;column+=blockDim.x) {
        float result=partial[output_base+column];
        for(int step=1;step<parts;++step) {
            const float incoming=contributions[(size_t(last-step)*columns+tile_column)*width+column];
            const auto weights=factors[step-1];
            result=result*weights.x+incoming*weights.y;
        }
        if(parts>1)result=normalizer!=0.f?result/normalizer:0.f;
        const auto gate_index=int64_t(batch)*gate_layout.batch+int64_t(token)*gate_layout.token+
            int64_t(head)*gate_layout.head+int64_t(column)*gate_layout.column;
        const float g=float_gate?static_cast<const float*>(gate)[gate_index]
            :__half2float(static_cast<const half*>(gate)[gate_index]);
        result=apply_attention_gate(result,g,sigmoid);
        if(half_output)static_cast<half*>(output)[output_base+column]=__float2half_rn(result);
        else static_cast<float*>(output)[output_base+column]=result;
    }
}

__global__ void mma_reduce_gate_kernel(const float* partial,const float2* metadata,
    const void* gate,void* output,Layout gate_layout,int tokens,int heads,
    int kv_heads,int width,int total_blocks,int parts,int query_columns,int head_columns,
    bool float_gate,bool half_output,Sigmoid sigmoid) {
    const int head=blockIdx.x%heads,row=blockIdx.x/heads;
    const int token=row%tokens,batch=row/tokens;
    const int grouped_heads=heads/kv_heads;
    const int head_tiles=(grouped_heads+head_columns-1)/head_columns;
    const int token_tiles=(tokens+query_columns-1)/query_columns;
    const int tile=(((batch*kv_heads+head/grouped_heads)*head_tiles+
        (head%grouped_heads)/head_columns)*token_tiles+token/query_columns);
    const int tile_column=(token%query_columns)*head_columns+(head%grouped_heads)%head_columns;
    const int columns=query_columns*head_columns,last=(tile+1)*parts-1;
    const size_t output_base=size_t(row)*heads*width+head*width;
    const float* contributions=reinterpret_cast<const float*>(metadata)+size_t(total_blocks)*4*columns;
    // Like the llama.cpp stream-K fixup, each output lane combines statistics
    // and values together. Avoid a serial statistics pass and CTA barrier.
    for(int column=threadIdx.x;column<width;column+=blockDim.x) {
        float result=partial[output_base+column],denominator=1.f,peak=0.f;
        if(parts>1) {
            auto statistics=metadata[size_t(last)*columns+tile_column];
            peak=statistics.x;denominator=statistics.y;
            for(int step=1;step<parts;++step) {
                const float incoming=contributions[(size_t(last-step)*columns+tile_column)*width+column];
                statistics=metadata[size_t(total_blocks+last-step)*columns+tile_column];
                const float maximum=fmaxf(peak,statistics.x);
                const float previous=peak-maximum,next=statistics.x-maximum;
                const float lhs=previous>=-20.f?expf(previous):0.f;
                const float rhs=next>=-20.f?expf(next):0.f;
                result=result*lhs+incoming*rhs;
                denominator=denominator*lhs+statistics.y*rhs;
                peak=maximum;
            }
            result=denominator!=0.f?result/denominator:0.f;
        }
        const auto at=int64_t(batch)*gate_layout.batch+int64_t(token)*gate_layout.token+
            int64_t(head)*gate_layout.head+int64_t(column)*gate_layout.column;
        const float g=float_gate?static_cast<const float*>(gate)[at]:__half2float(static_cast<const half*>(gate)[at]);
        result=apply_attention_gate(result,g,sigmoid);
        if(half_output)static_cast<half*>(output)[output_base+column]=__float2half_rn(result);
        else static_cast<float*>(output)[output_base+column]=result;
    }
}


} // namespace

namespace {
void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("selected attention launch: ") + cudaGetErrorString(error));
}

int grid_for(int64_t count) {
    return static_cast<int>(std::min<int64_t>(65535, count / 256 + (count % 256 != 0)));
}
} // namespace

void prepare_indices(const int32_t* source, int32_t* indices, void* mask,
    int64_t rows, int source_count, int padded_count, int cache_length, cudaStream_t stream) {
    if (!rows || !padded_count) return;
    const int64_t total = rows * padded_count;
    prepare_sparse_indices_kernel<<<grid_for(total), 256, 0, stream>>>(source, indices,
        static_cast<half*>(mask), rows, source_count, padded_count, cache_length);
    check_launch();
}

void sparse(const void* query, const void* key, const void* value, const int32_t* indices,
    float* scratch_or_output, int64_t batch, int64_t heads, int64_t kv_heads,
    int64_t tokens, int64_t width, int64_t cache_length, int64_t selected,
    float scale, bool half_query, const void* positions, bool positions_i64,
    Epilogue epilogue, sparse_mma_raw::Hot hot, cudaStream_t stream) {
    if (!batch || !tokens) return;
    const int64_t blocks = batch * tokens * heads;
    if (blocks > INT_MAX) throw std::invalid_argument("sparse attention exceeds CUDA grid size");
    #define SELECTED_LAUNCH(S,H) sparse_attention_kernel<S,H><<<unsigned(blocks), 256, 0, stream>>>( \
        query, static_cast<const half*>(key), static_cast<const half*>(value), indices, \
        scratch_or_output, heads, kv_heads, tokens, width, cache_length, selected, scale, \
        half_query, positions, positions_i64, epilogue, hot)
    if (width <= 256) { if (hot.mapping) { SELECTED_LAUNCH(1,true); } else { SELECTED_LAUNCH(1,false); } }
    else if (width <= 512) { if (hot.mapping) { SELECTED_LAUNCH(2,true); } else { SELECTED_LAUNCH(2,false); } }
    else if (width <= 1024) { if (hot.mapping) { SELECTED_LAUNCH(4,true); } else { SELECTED_LAUNCH(4,false); } }
    else { if (hot.mapping) { SELECTED_LAUNCH(0,true); } else { SELECTED_LAUNCH(0,false); } }
    #undef SELECTED_LAUNCH
    check_launch();
}

void gated(const void* attended, bool half_input, Layout layout, Epilogue epilogue,
    int64_t count, int64_t heads, int64_t tokens, int64_t width, cudaStream_t stream) {
    if (!count) return;
    attention_gated_kernel<<<grid_for(count), 256, 0, stream>>>(attended, half_input,
        layout, epilogue, count, heads, tokens, width);
    check_launch();
}

void block_indices(const void* blocks, bool blocks_i64, const void* positions,
    bool positions_i64, int32_t* output, int64_t rows, int tokens,
    int block_count, int pool, int budget, int columns, cudaStream_t stream) {
    const int64_t count = rows * columns;
    if (!count) return;
    const int grid = grid_for(count);
    #define SELECTED_BLOCK(B,P) block_indices_to_tokens_kernel<B,P><<<grid,256,0,stream>>>( \
        static_cast<const B*>(blocks), static_cast<const P*>(positions), output, rows, tokens, \
        block_count, pool, budget, columns)
    if (blocks_i64) {
        if (positions_i64) { SELECTED_BLOCK(int64_t,int64_t); } else { SELECTED_BLOCK(int64_t,int32_t); }
    } else {
        if (positions_i64) { SELECTED_BLOCK(int32_t,int64_t); } else { SELECTED_BLOCK(int32_t,int32_t); }
    }
    #undef SELECTED_BLOCK
    check_launch();
}

void prepare_causal(const void* query, bool half_query, Layout layout, float* compact_query,
    int64_t query_count, int64_t heads, int64_t tokens, int64_t width,
    const void* positions, bool positions_i64, int32_t* indices, void* mask,
    int64_t index_count, int64_t columns, int64_t padded_columns,
    int64_t capacity, cudaStream_t stream) {
    const auto count = std::max(query_count, index_count);
    if (!count) return;
    prepare_causal_attention_kernel<<<grid_for(count),256,0,stream>>>(query, half_query,
        layout, compact_query, query_count, heads, tokens, width, positions, positions_i64,
        indices, static_cast<half*>(mask), index_count, columns, padded_columns, capacity);
    check_launch();
}

void reduce_gated(const float* partial, const void* metadata, const void* gate,
    void* output, Layout gate_layout, int64_t rows, int tokens, int heads, int kv_heads,
    int width, int total_blocks, int parts, int query_columns, int head_columns,
    bool float_gate, bool half_output, Sigmoid sigmoid, bool parallel, cudaStream_t stream) {
    if (!rows) return;
    if (rows > INT_MAX) throw std::invalid_argument("MMA gate reduction exceeds CUDA geometry");
    const size_t shared = parallel ? 0 : size_t(parts - 1) * sizeof(float2);
    const auto kernel = parallel ? mma_reduce_gate_kernel : mma_reduce_gate_serial_kernel;
    kernel<<<unsigned(rows),256,shared,stream>>>(partial, static_cast<const float2*>(metadata),
        gate, output, gate_layout, tokens, heads, kv_heads, width, total_blocks, parts,
        query_columns, head_columns, float_gate, half_output, sigmoid);
    check_launch();
}

} // namespace tyloi::cuda::selected_raw
