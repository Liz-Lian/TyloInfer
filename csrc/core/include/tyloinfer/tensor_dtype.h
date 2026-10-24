#pragma once

#include <string_view>
#if TYLOINFER_ENABLE_MFQ_COMPAT
#include "mfq_format_compat.h"
#endif

namespace mfq {

inline bool supported_tensor_dtype(std::string_view dtype) noexcept {
#if TYLOINFER_ENABLE_MFQ_COMPAT
    dtype = canonical_format_dtype(dtype);
    if (dtype == "NINT" || dtype == "NINT8-0" || dtype == "NVQ" ||
        dtype == "MFE" || dtype == "MFED" || dtype == "NPQ" ||
        dtype == "NEPQ" || dtype == "MXFP4-SQ" ||
        dtype == "MXFP8-SQ" || dtype == "FP8-128SQ") return true;
#endif
    return dtype == "TQ" || dtype == "BF16" || dtype == "F16" || dtype == "F32"
        || dtype == "F8_E4M3" || dtype == "I32" || dtype == "I64"
        || dtype == "MXFP4" || dtype == "MXFP8" || dtype == "BLOB";
}

} // namespace mfq
