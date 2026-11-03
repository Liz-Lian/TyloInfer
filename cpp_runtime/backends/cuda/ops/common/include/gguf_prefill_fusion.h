#pragma once
#include "mfq_tensor_backend.h"
#include <cuda_runtime_api.h>
struct GgufWeight;
#include <memory>
#include <vector>
class GgufPrefillInputScope {
    struct Impl;std::unique_ptr<Impl> impl_;
    GgufPrefillInputScope();
public:
    explicit GgufPrefillInputScope(const mfq_tensor_backend::Tensor& input);
    ~GgufPrefillInputScope();
    static bool lookup(const mfq_tensor_backend::Tensor& input,
        mfq_tensor_backend::Tensor& half,mfq_tensor_backend::Tensor& exponents);
    static std::unique_ptr<GgufPrefillInputScope> prepare_rms(
        const mfq_tensor_backend::Tensor& input,const mfq_tensor_backend::Tensor& weight,
        double eps,double offset,mfq_tensor_backend::Tensor& normalized);
};
mfq_tensor_backend::Tensor gguf_rms_gated_prefill(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& attended,const mfq_tensor_backend::Tensor& gate,
    const mfq_tensor_backend::Tensor& norm,double eps,bool silu);

struct GgufPrefillProduct {
    mfq_tensor_backend::Tensor values, row_exponents, column_exponents;
    bool defined() const { return values.defined(); }
};
// A following operator may restore values at the point of consumption.
// Empty means the ordinary streamed/reference projection should be used.
GgufPrefillProduct gguf_prefill_product(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& input);

GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& attended,const mfq_tensor_backend::Tensor& gate,
    const mfq_tensor_backend::Tensor& norm,double eps,bool silu);
GgufPrefillProduct gguf_rms_gated_product(const GgufWeight& weight,
    const mfq_tensor_backend::Tensor& attended,const GgufPrefillProduct& gate,
    const mfq_tensor_backend::Tensor& norm,double eps,bool silu);
// Materialize a scaled [tokens, columns] product for an ordinary FP32 consumer.
mfq_tensor_backend::Tensor gguf_prefill_materialize(const GgufPrefillProduct& product);
std::vector<mfq_tensor_backend::Tensor> gguf_prefill_acc_rms(
    const GgufPrefillProduct& product,const mfq_tensor_backend::Tensor& residual,
    const mfq_tensor_backend::Tensor& norm,double eps,double offset);
