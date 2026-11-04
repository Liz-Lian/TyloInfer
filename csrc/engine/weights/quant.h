#pragma once

#include "engine/weights/encoding/format_features.h"
#include "engine/weights/encoding/mx_blob.h"
#include "engine/weights/mx.h"
#include "engine/weights/tensor_parallel.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "engine/weights/encoding/quant/nint_blob.h"
#include "engine/weights/encoding/quant/vq_blob.h"
#include "engine/weights/nint.h"
#include "engine/weights/vq.h"
#include "engine/weights/sq.h"
#endif

#include <cstdint>
#include <string_view>
#include <vector>

tyloi::mxfp4::Data select_mxfp4_cpu_rows(
    const tyloi::mxfp4::Data& source, const std::vector<int64_t>& rows);
tyloi::mxfp4::Data slice_mxfp4_cpu(
    const tyloi::mxfp4::Data& source, TensorParallelAxis axis,
    int64_t begin, int64_t end);
tyloi::mxfp8::Data slice_mxfp8_cpu(
    const tyloi::mxfp8::Data& source, TensorParallelAxis axis,
    int64_t begin, int64_t end);
tyloi::Mxfp8Tensor to_device_mxfp8(
    const tyloi::mxfp8::Data& source, bool cuda, int device = -1);
tyloi::Mxfp4Tensor to_device_mxfp4(
    const tyloi::mxfp4::Data& source, bool cuda, int device = -1);
::tyloi::DenseTensor dequant_mxfp4_cpu(const tyloi::mxfp4::Data& source);

#if TYLOINFER_ENABLE_MFQ_COMPAT
bool prepare_nvq_dense_groups(tyloi::NvqTensor& weight);
void prepare_nvq_integer_codebook(tyloi::NvqTensor& weight);
void prepare_nvq_decode_records(tyloi::NvqTensor& weight);
tyloi::nvq::Data slice_nvq_cpu(
    const tyloi::nvq::Data& source, TensorParallelAxis axis,
    int64_t begin, int64_t end);
tyloi::nvq::Data select_nvq_cpu_rows(
    const tyloi::nvq::Data& source, const std::vector<int64_t>& rows);
tyloi::NvqTensor to_device_nvq(
    const tyloi::nvq::Data& source, bool cuda,
    bool nvq2 = false, bool extended = false, int device = -1);
tyloi::nepq::Data select_nepq_cpu_rows(
    const tyloi::nepq::Data& source,
    const std::vector<int64_t>& rows,
    int output_per_expert,
    int selected_experts = -1);
NepqWeight to_device_nepq(const tyloi::nepq::Data& source, bool cuda);
tyloi::nint::Data slice_nint_cpu_output(
    const tyloi::nint::Data& source, int64_t begin, int64_t end);
tyloi::nint::Data slice_nint_cpu_input_groups(
    const tyloi::nint::Data& source, int64_t begin, int64_t end);
tyloi::nint::Data select_nint_cpu_rows(
    const tyloi::nint::Data& source, const std::vector<int64_t>& rows);
tyloi::nint8::Data slice_nint8_zero_cpu_output(
    const tyloi::nint8::Data& source, int64_t begin, int64_t end);
tyloi::nint8::Data slice_nint8_zero_cpu_input_groups(
    const tyloi::nint8::Data& source, int64_t begin, int64_t end);
tyloi::nint8::Data select_nint8_zero_cpu_rows(
    const tyloi::nint8::Data& source, const std::vector<int64_t>& rows);
tyloi::NintTensor to_device_nint(
    const tyloi::nint::Data& source, bool cuda, int device = -1);
tyloi::NintTensor to_device_mfe_nint(
    const tyloi::nint::Data& source,
    int local_experts,
    int out_per_expert,
    bool cuda);
tyloi::NintTensor to_device_nint8_zero(
    const tyloi::nint8::Data& source, bool cuda, int device = -1);
::tyloi::DenseTensor dequant_nint8_zero_cpu(
    const tyloi::nint8::Data& source);
tyloi::Mxfp4SqTensor to_device_mxfp4_sq(
    const std::vector<uint8_t>& payload, bool cuda, int device = -1);
tyloi::Fp8SqTensor to_device_fp8_sq(
    std::string_view dtype, const std::vector<uint8_t>& payload,
    bool cuda, int device = -1);
#endif
