#pragma once
#include "backend.h"
#include "nint_raw.h"
#include <vector>

// A serial group of projections can quantize its common input for all GS
// values in one launch. This state is capture-local, never a replay cache.
struct NintPreparedGroupInput {
    int gs=0;
    ::tyloi::DenseTensor codes,scale,sum,interleaved;
};
struct NintInputQuantizationScope {
    ::tyloi::DenseTensor input;
    std::vector<NintPreparedGroupInput> groups;
    NintInputQuantizationScope* previous=nullptr;
    NintInputQuantizationScope(::tyloi::DenseTensor input,const std::vector<int>& group_sizes);
    ~NintInputQuantizationScope();
    NintInputQuantizationScope(const NintInputQuantizationScope&)=delete;
    NintInputQuantizationScope& operator=(const NintInputQuantizationScope&)=delete;
};

bool nint_late_scale_enabled(int64_t input_width,int bits,int device);

::tyloi::DenseTensor nint_matmul_fixed_bits_ws_cuda(
    ::tyloi::DenseTensor bitstream,::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,::tyloi::DenseTensor subgroup_scale,
    ::tyloi::DenseTensor subgroup_minimum,::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor neuron_minimum,::tyloi::DenseTensor input,
    int64_t group_size,::tyloi::DenseTensor quantized_input,
    ::tyloi::DenseTensor input_scale,int uniform_bits);

::tyloi::DenseTensor nint_matmul_input_mul_fixed_bits_ws_cuda(
    ::tyloi::DenseTensor bitstream,::tyloi::DenseTensor row_q_bits,
    ::tyloi::DenseTensor row_q_bit_offsets,::tyloi::DenseTensor subgroup_scale,
    ::tyloi::DenseTensor subgroup_minimum,::tyloi::DenseTensor neuron_scale,
    ::tyloi::DenseTensor neuron_minimum,::tyloi::DenseTensor input,
    ::tyloi::DenseTensor gate,int64_t activation_mode,int64_t group_size,
    ::tyloi::DenseTensor quantized_input,::tyloi::DenseTensor input_scale,int uniform_bits);

namespace tyloi::cuda {
bool try_nint_decode(
    NintSingleRouteWeight weight,::tyloi::DenseTensor row_metadata,::tyloi::DenseTensor expert_local,
    ::tyloi::DenseTensor ids,::tyloi::DenseTensor output,
    bool routed_input,int bits_hint,bool metadata_ready=false);
}

// Hints select code generation only. Device row metadata is checked on every
// replay, so mixed widths and mutable expert slots retain the general path.
void nint_moe_grouped_matmul_hinted_cuda(
    ::tyloi::DenseTensor pointers,::tyloi::DenseTensor params,
    ::tyloi::DenseTensor expert_pool,::tyloi::DenseTensor expert_local,
    ::tyloi::DenseTensor ids,::tyloi::DenseTensor output,
    int64_t input_width,bool routed_input,int group_hint,int bits_hint);
