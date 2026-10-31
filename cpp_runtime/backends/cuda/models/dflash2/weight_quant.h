#pragma once
#include "mfq_tensor_backend.h"
#include "tyloinfer/gguf_quant.h"
#include "ggml-quants.h"
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace mfq::cuda::dflash2 {
// Pack standard Q8_0 directly from checkpoint rows. Avoid a full F32 copy and
// the generic element-by-element CPU tensor conversion during model loading.
inline mfq_tensor_backend::Tensor pack_q8_weight_cpu(const mfq_tensor_backend::Tensor& source) {
    namespace tb=mfq_tensor_backend;
    if(source.is_cuda() || source.dim()!=2 || source.size(0)<1 || source.size(1)<1 || source.size(1)%32 ||
        (source.scalar_type()!=tb::kFloat32 && source.scalar_type()!=tb::kFloat16 && source.scalar_type()!=tb::kBFloat16))
        throw std::invalid_argument("DFlash2 Q8 weight requires CPU F32/F16/BF16 matrix rows divisible by 32");
    const auto input=source.contiguous();const auto rows=input.size(0),columns=input.size(1);
    const auto stride=mfq::gguf::row_bytes(GGML_TYPE_Q8_0,columns);
    auto packed=tb::empty({std::int64_t(rows*stride)},tb::TensorOptions().device(tb::kCPU).dtype(tb::kUInt8));
    mfq_parallel_for(0,rows,std::max<std::int64_t>(1,32768/columns),[&](std::int64_t first,std::int64_t last) {
        auto* output=reinterpret_cast<block_q8_0*>(packed.data_ptr<std::uint8_t>())+first*(columns/32);
        if(input.scalar_type()==tb::kFloat32) {
            quantize_row_q8_0_ref(input.data_ptr<float>()+first*columns,output,(last-first)*columns);
            return;
        }
        thread_local std::vector<float> scratch;scratch.resize(std::size_t(columns));
        for(auto row=first;row<last;++row) {
            if(input.scalar_type()==tb::kBFloat16)
                ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t*>(input.data_ptr())+row*columns,scratch.data(),columns);
            else ggml_fp16_to_fp32_row(static_cast<const ggml_fp16_t*>(input.data_ptr())+row*columns,scratch.data(),columns);
            quantize_row_q8_0_ref(scratch.data(),output,columns);output+=columns/32;
        }
    });
    return packed;
}
}
