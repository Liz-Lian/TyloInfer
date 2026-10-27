#pragma once

#include "formats/format_features.h"
#include "ops/common/include/mx.h"
#include "ops/common/include/parallel.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "formats/quant/vq_blob.h"
#include "ops/mfq/include/fp8_sq.h"
#include "ops/mfq/include/mxfp4_sq.h"
#include "ops/mfq/include/nint.h"
#include "ops/mfq/include/vq.h"
#endif

#include <cstdint>
#include <string_view>
#include <vector>

mfq::mxfp4::Data select_mxfp4_cpu_rows(
    const mfq::mxfp4::Data& source, const std::vector<int64_t>& rows);
mfq::mxfp4::Data slice_mxfp4_cpu(
    const mfq::mxfp4::Data& source, TensorParallelAxis axis,
    int64_t begin, int64_t end);
mfq::mxfp8::Data slice_mxfp8_cpu(
    const mfq::mxfp8::Data& source, TensorParallelAxis axis,
    int64_t begin, int64_t end);
Mxfp8Weight to_device_mxfp8(
    const mfq::mxfp8::Data& source, bool cuda, int device = -1);
Mxfp4Weight to_device_mxfp4(
    const mfq::mxfp4::Data& source, bool cuda, int device = -1);

#if TYLOINFER_ENABLE_MFQ_COMPAT
bool prepare_nvq_dense_groups(NvqWeight& weight);
void prepare_nvq_integer_codebook(NvqWeight& weight);
void prepare_nvq_decode_records(NvqWeight& weight);
mfq::nvq::Data slice_nvq_cpu(
    const mfq::nvq::Data& source, TensorParallelAxis axis,
    int64_t begin, int64_t end);
mfq::nvq::Data select_nvq_cpu_rows(
    const mfq::nvq::Data& source, const std::vector<int64_t>& rows);
NvqWeight to_device_nvq(
    const mfq::nvq::Data& source, bool cuda,
    bool nvq2 = false, bool extended = false, int device = -1);
mfq::nepq::Data select_nepq_cpu_rows(
    const mfq::nepq::Data& source,
    const std::vector<int64_t>& rows,
    int output_per_expert,
    int selected_experts = -1);
NepqWeight to_device_nepq(const mfq::nepq::Data& source, bool cuda);
mfq::nint::Data slice_nint_cpu_output(
    const mfq::nint::Data& source, int64_t begin, int64_t end);
mfq::nint::Data slice_nint_cpu_input_groups(
    const mfq::nint::Data& source, int64_t begin, int64_t end);
mfq::nint::Data select_nint_cpu_rows(
    const mfq::nint::Data& source, const std::vector<int64_t>& rows);
mfq::nint8::Data slice_nint8_zero_cpu_output(
    const mfq::nint8::Data& source, int64_t begin, int64_t end);
mfq::nint8::Data slice_nint8_zero_cpu_input_groups(
    const mfq::nint8::Data& source, int64_t begin, int64_t end);
mfq::nint8::Data select_nint8_zero_cpu_rows(
    const mfq::nint8::Data& source, const std::vector<int64_t>& rows);
NintWeight to_device_nint(
    const mfq::nint::Data& source, bool cuda, int device = -1);
NintWeight to_device_mfe_nint(
    const mfq::nint::Data& source,
    int local_experts,
    int out_per_expert,
    bool cuda);
NintWeight to_device_nint8_zero(
    const mfq::nint8::Data& source, bool cuda, int device = -1);
Mxfp4SqWeight to_device_mxfp4_sq(
    const std::vector<uint8_t>& payload, bool cuda, int device = -1);
Fp8SqWeight to_device_fp8_sq(
    std::string_view dtype, const std::vector<uint8_t>& payload,
    bool cuda, int device = -1);
#endif
