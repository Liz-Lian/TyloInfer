#include "engine/weights/encoding/format_features.h"
#include "engine/weights/encoding/tensor_dtype.h"

#include <cassert>

int main() {
    for (const auto dtype : {"TQ", "BF16", "F16", "F32", "MXFP4", "MXFP8", "F8_E4M3"})
        assert(tyloi::supported_tensor_dtype(dtype));
    for (const auto dtype : {"NINT", "NINTv2", "NINT5", "NINT8-0", "NVQ3J-L", "MFE", "MXFP4-SQ", "FP8-128SQ"})
        assert(tyloi::supported_tensor_dtype(dtype) == tyloi::mfq_compat_enabled);
    assert(!tyloi::supported_tensor_dtype("not-a-format"));
    bool called = false;
    try {
        tyloi::with_mfq_compat([&] { called = true; });
        assert(tyloi::mfq_compat_enabled);
    } catch (const std::runtime_error&) {
        assert(!tyloi::mfq_compat_enabled);
    }
    assert(called == tyloi::mfq_compat_enabled);
}
