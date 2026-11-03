#pragma once
#include "backend.h"

struct Nvq1DecodeView {
    int64_t pointers[7]{}; // indices, delta, state, anchor, book, records, integer book
    int64_t sizes[4]{};
    int32_t groups=0,vectors=0,format=0,local_experts=0;
};
namespace mfq::cuda {
bool try_nvq_decode(Nvq1DecodeView weight,
    mfq_tensor_backend::Tensor records,mfq_tensor_backend::Tensor integer_book,
    mfq_tensor_backend::Tensor input,mfq_tensor_backend::Tensor ids,
    mfq_tensor_backend::Tensor expert_local,mfq_tensor_backend::Tensor qx,
    mfq_tensor_backend::Tensor xscale,mfq_tensor_backend::Tensor output,
    int experts,int output_rows,bool input_quantized);
}
