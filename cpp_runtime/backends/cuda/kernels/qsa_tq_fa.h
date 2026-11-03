#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstddef>
#include "qsa_tq.h"

namespace tyloinfer::qsa {
struct TqPrefillAttention {
    const std::uint16_t *keys=nullptr,*values=nullptr;
    const void* query=nullptr;const float *key_signs=nullptr,*value_signs=nullptr;
    const SelectedBlock* selected=nullptr;const std::int64_t* positions=nullptr;
    void* output=nullptr;float *rotated=nullptr,*temporary=nullptr,*logmass=nullptr,*norm=nullptr,*hits=nullptr,*mass=nullptr;
    std::size_t temporary_bytes=0;
    int queries=0,heads=0,kv_heads=0,width=0,kv_rows=0,count=0;
    std::int64_t query_token_stride=0,query_head_stride=0;
    const void* gate=nullptr;std::int64_t gate_token_stride=0,gate_head_stride=0;
    bool query_half=false,gate_half=false,output_half=false;
    cudaStream_t stream=nullptr;
};
// Decoded K/V are [four-token block, KV head, token, padded channel].
// A byte stores the four visibility bits of one selected microblock.
// Reuses the retired partial-output arena. Returns cudaErrorNotSupported
// before launching anything if the device image or workspace is unsuitable.
inline std::size_t tq_prefill_attention_scratch(int queries,int heads,int kv_rows) {
    const auto q=std::size_t(queries),r=std::size_t(kv_rows),tiles=(q+7)/8;
    return q*heads*256*sizeof(float)+((q*(r/4)+3)&~std::size_t(3))+q*(r/4)*sizeof(int)+(tiles*r+tiles)*sizeof(int);
}
int tq_sparse_attention_prefill(const TqPrefillAttention& call);
// Owns executable metadata; each invocation updates its workspace pointers.
// Scratch tensors remain owned by the caller and are not retained here.
class TqPrefillGraph {
public:
    TqPrefillGraph()=default;
    ~TqPrefillGraph();
    TqPrefillGraph(const TqPrefillGraph&)=delete;
    TqPrefillGraph& operator=(const TqPrefillGraph&)=delete;
    int run(const TqPrefillAttention& call);
private:
    void reset() noexcept;
    cudaGraphExec_t executable_=nullptr;
    int device_=-1;
    std::size_t nodes_=0;
};
} // namespace tyloinfer::qsa
