#pragma once

#include "native/cuda/profile.h"
#include "backend.h"
#include "formats/mx_blob.h"

#include <cstdint>
#include <vector>

struct Mxfp8Weight {
    mfq_tensor_backend::Tensor values;
    mfq_tensor_backend::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};

struct Mxfp4Weight {
    mfq_tensor_backend::Tensor values;
    mfq_tensor_backend::Tensor scales;
    int64_t out = 0;
    int64_t neuron_len = 0;
};

mfq_tensor_backend::Tensor mxfp8_groupwise_matmul(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    mfq_tensor_backend::Tensor input,
    std::int64_t groups);
mfq_tensor_backend::Tensor mxfp8_groupwise_matmul_f32(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    mfq_tensor_backend::Tensor input,
    std::int64_t groups);
mfq_tensor_backend::Tensor mxfp8_matmul(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp8_matmul_f32(
    CudaProfiler& profiler,
    const Mxfp8Weight& weight,
    mfq_tensor_backend::Tensor input);
mfq_tensor_backend::Tensor mxfp4_matmul(
    const Mxfp4Weight& weight, mfq_tensor_backend::Tensor input);

struct Mxfp4Linear {
    Mxfp4Weight weight;

    mfq_tensor_backend::Tensor forward(mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = mxfp4_matmul(
            weight, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

struct Mxfp8Linear {
    Mxfp8Weight weight;

    mfq_tensor_backend::Tensor forward(
            CudaProfiler& profiler,
            mfq_tensor_backend::Tensor x) const {
        auto shape = x.sizes().vec();
        auto y = mxfp8_matmul(
            profiler, weight, x.reshape({-1, x.size(-1)}));
        shape.back() = y.size(-1);
        return y.reshape(shape);
    }
};

mfq_tensor_backend::Tensor dequant_mxfp4_cpu(const mfq::mxfp4::Data& source);
mfq_tensor_backend::Tensor mxfp8_cpu_reference(const Mxfp8Weight& weight);
