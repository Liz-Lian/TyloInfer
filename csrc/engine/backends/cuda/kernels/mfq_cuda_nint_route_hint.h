#pragma once
#include "backend.h"
#include <vector>

// A serial group of projections can quantize its common input for all GS
// values in one launch. This state is capture-local, never a replay cache.
struct NintPreparedGroupInput {
    int gs=0;
    ::tyloi::Tensor codes,scale,sum,interleaved;
};
struct NintInputQuantizationScope {
    ::tyloi::Tensor input;
    std::vector<NintPreparedGroupInput> groups;
    NintInputQuantizationScope* previous=nullptr;
    NintInputQuantizationScope(::tyloi::Tensor input,const std::vector<int>& group_sizes);
    ~NintInputQuantizationScope();
    NintInputQuantizationScope(const NintInputQuantizationScope&)=delete;
    NintInputQuantizationScope& operator=(const NintInputQuantizationScope&)=delete;
};

// Internal view of a validated, fixed-capacity expert pool. The pointer and
// geometry values share the lifetime of the captured pool/workspace tensors.
struct NintSingleRouteWeight {
    int64_t pointers[9]{};
    int32_t geometry[3]{};
    int32_t local_experts=0;
    int64_t row_metadata=0;
};

bool nint_late_scale_enabled(int64_t input_width,int bits,int device);

::tyloi::Tensor nint_matmul_fixed_bits_ws_cuda(
    ::tyloi::Tensor bitstream,::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,::tyloi::Tensor subgroup_scale,
    ::tyloi::Tensor subgroup_minimum,::tyloi::Tensor neuron_scale,
    ::tyloi::Tensor neuron_minimum,::tyloi::Tensor input,
    int64_t group_size,::tyloi::Tensor quantized_input,
    ::tyloi::Tensor input_scale,int uniform_bits);

::tyloi::Tensor nint_matmul_input_mul_fixed_bits_ws_cuda(
    ::tyloi::Tensor bitstream,::tyloi::Tensor row_q_bits,
    ::tyloi::Tensor row_q_bit_offsets,::tyloi::Tensor subgroup_scale,
    ::tyloi::Tensor subgroup_minimum,::tyloi::Tensor neuron_scale,
    ::tyloi::Tensor neuron_minimum,::tyloi::Tensor input,
    ::tyloi::Tensor gate,int64_t activation_mode,int64_t group_size,
    ::tyloi::Tensor quantized_input,::tyloi::Tensor input_scale,int uniform_bits);

namespace tyloi::cuda {
bool try_nint_decode(
    NintSingleRouteWeight weight,::tyloi::Tensor row_metadata,::tyloi::Tensor expert_local,
    ::tyloi::Tensor ids,::tyloi::Tensor output,
    bool routed_input,int bits_hint,bool metadata_ready=false);
}

// Hints select code generation only. Device row metadata is checked on every
// replay, so mixed widths and mutable expert slots retain the general path.
void nint_moe_grouped_matmul_hinted_cuda(
    ::tyloi::Tensor pointers,::tyloi::Tensor params,
    ::tyloi::Tensor expert_pool,::tyloi::Tensor expert_local,
    ::tyloi::Tensor ids,::tyloi::Tensor output,
    int64_t input_width,bool routed_input,int group_hint,int bits_hint);
