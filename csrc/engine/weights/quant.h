#pragma once

#include "formats/format_features.h"
#include "ops/common/include/mx.h"

#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "ops/mfq/include/fp8_sq.h"
#include "ops/mfq/include/mxfp4_sq.h"
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
Mxfp4SqWeight to_device_mxfp4_sq(
    const std::vector<uint8_t>& payload, bool cuda, int device = -1);
Fp8SqWeight to_device_fp8_sq(
    std::string_view dtype, const std::vector<uint8_t>& payload,
    bool cuda, int device = -1);
#endif
