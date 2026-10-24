#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>

#ifndef TYLOINFER_ENABLE_MFQ_COMPAT
#define TYLOINFER_ENABLE_MFQ_COMPAT 0
#endif

namespace mfq {
inline constexpr bool mfq_compat_enabled = TYLOINFER_ENABLE_MFQ_COMPAT != 0;

template<class Function>
std::invoke_result_t<Function> with_mfq_compat(Function&& function) {
    if constexpr (mfq_compat_enabled) return std::forward<Function>(function)();
    else throw std::runtime_error("Legacy MFQ inference requires a compatibility build");
}

}
