#pragma once
#include "backend.h"

struct Nvq1DecodeView {
    int64_t pointers[7]{}; // indices, delta, state, anchor, book, records, integer book
    int64_t sizes[4]{};
    int32_t groups=0,vectors=0,format=0,local_experts=0;
};
namespace tyloi::cuda {
bool try_nvq_decode(Nvq1DecodeView weight,
    ::tyloi::DenseTensor records,::tyloi::DenseTensor integer_book,
    ::tyloi::DenseTensor input,::tyloi::DenseTensor ids,
    ::tyloi::DenseTensor expert_local,::tyloi::DenseTensor qx,
    ::tyloi::DenseTensor xscale,::tyloi::DenseTensor output,
    int experts,int output_rows,bool input_quantized);
}
