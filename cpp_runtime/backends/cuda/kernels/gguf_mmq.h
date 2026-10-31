#pragma once
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace mfq::cuda::gguf::mmq {
// Strata's contiguous expert-group MMQ interface. Original GGUF blocks are
// gathered unchanged; activations use llama.cpp's Q8_1 tensor-core layout.
bool fits(int type,std::int64_t rows,int device);
// The upstream tile loader reads a full J-wide ID tile before masking its
// inactive columns. J is bounded by its scheduler at 512 on every backend.
constexpr int id_guard=512;
std::size_t activation_bytes(std::int64_t tokens,std::int64_t columns);
void quantize(const float* input,const std::int32_t* ids,void* output,int type,
    std::int64_t columns,std::int64_t rows,cudaStream_t stream);
void swiglu_quantize(const float* gate_up,void* output,int type,
    std::int64_t columns,std::int64_t rows,float limit,cudaStream_t stream);
// Quantize a physical token once, then write identical Q8 rows to its routed
// compact positions. inverse[physical_token * routes + route] is a permutation.
void quantize_scatter(const float* input,const std::int32_t* inverse,void* output,int type,
    std::int64_t columns,std::int64_t tokens,int routes,cudaStream_t stream);
struct Product {
    const void* weights;
    int type;
    std::int64_t rows,columns;
    std::size_t expert_bytes;
    int experts;
    const void* activation;
    const std::int32_t* bounds;
    const std::int32_t* ids; // tokens entries followed by at least id_guard readable entries
    std::int64_t tokens,max_tokens;
    float* output;
    std::int64_t output_stride;
    // Optional scattered matrices, passed as kernel arguments. Entries without
    // a zero over-read requirement continue to use guarded group storage.
    const void* const* weight_addresses=nullptr;
    // A product may view part of a whole layer's compact activation matrix.
    // Zero retains the standalone product's existing row stride.
    std::int64_t activation_rows=0,activation_offset=0;
};
bool direct_fits(int type,std::int64_t rows,std::int64_t columns,int device);
class Context {
public:
    Context();
    ~Context();
    Context(const Context&)=delete;
    Context& operator=(const Context&)=delete;
    void run(const Product&,cudaStream_t);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
constexpr int group_capacity=16;
struct Copy {
    const void* source[group_capacity]{};
    void* destination=nullptr;
    std::uint64_t bytes=0,stride=0,offset=0,guard_bytes=0;
};
void gather(const Copy (&copies)[3],int experts,cudaStream_t stream);
void swiglu(const float* gate_up,float* hidden,std::int64_t tokens,std::int64_t width,float limit,cudaStream_t stream);
void scatter(const float* source,void* pairs,const std::int32_t* ids,std::int64_t tokens,std::int64_t width,cudaStream_t stream);
}
