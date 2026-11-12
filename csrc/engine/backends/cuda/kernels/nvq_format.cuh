#pragma once
#include "packed_nvq.cuh"
#include <stdexcept>
#include <type_traits>

namespace tyloi::cuda::nvq_format {
using namespace packed_nvq;
template <typename Launch>
void launch_by_format(int format, Launch && launch) {
    switch (format) {
        case kNvq1L: launch(std::integral_constant<int, kNvq1L>{}); break;
        case kNvq2: launch(std::integral_constant<int, kNvq2>{}); break;
        case kNvq3: launch(std::integral_constant<int, kNvq3>{}); break;
        case kNvq2Exec: launch(std::integral_constant<int, kNvq2Exec>{}); break;
        case kNvq2Jsc: launch(std::integral_constant<int, kNvq2Jsc>{}); break;
        case kNvq2JscExec: launch(std::integral_constant<int, kNvq2JscExec>{}); break;
        case kNpq0L: launch(std::integral_constant<int, kNpq0L>{}); break;
        case kNvq1S: launch(std::integral_constant<int, kNvq1S>{}); break;
        case kNpq0S: launch(std::integral_constant<int, kNpq0S>{}); break;
        case kNvq3Jsc: launch(std::integral_constant<int, kNvq3Jsc>{}); break;
        case kNvq3Jsc2: launch(std::integral_constant<int, kNvq3Jsc2>{}); break;
        case kNvq3Jsc512:
            launch(std::integral_constant<int, kNvq3Jsc512>{});
            break;
        case kNvq2JscL:
            launch(std::integral_constant<int, kNvq2JscL>{});
            break;
        case kNvq2JscXL:
            launch(std::integral_constant<int, kNvq2JscXL>{});
            break;
        case kNvq3JscL:
            launch(std::integral_constant<int, kNvq3JscL>{});
            break;
        case kNvq2JscXLGroupExec:
            launch(std::integral_constant<int, kNvq2JscXLGroupExec>{});
            break;
        case kNvq3JscLGroupExec:
            launch(std::integral_constant<int, kNvq3JscLGroupExec>{});
            break;
        default: throw std::runtime_error("unsupported NVQ format");
    }
}

template <typename Launch>
void launch_nepq_by_format(int format, Launch && launch) {
    switch (format) {
        case kNvq1L: launch(std::integral_constant<int, kNvq1L>{}); break;
        case kNpq0L: launch(std::integral_constant<int, kNpq0L>{}); break;
        case kNvq1S: launch(std::integral_constant<int, kNvq1S>{}); break;
        case kNpq0S: launch(std::integral_constant<int, kNpq0S>{}); break;
        default: throw std::runtime_error("unsupported NEPQ base format");
    }
}


} // namespace tyloi::cuda::nvq_format
