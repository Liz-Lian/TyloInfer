set(TYLOINFER_NATIVE_GGUF_ROOT "${PROJECT_SOURCE_DIR}/third_party/splash/runtime")
set(TYLOINFER_NATIVE_PREFILL_TOKEN_BUDGET 4096)
configure_file(native_limits.h.in native_limits.h @ONLY)

function(tyloinfer_native_gguf_expand path output visited)
    if(path IN_LIST visited)
        set(${output} "" PARENT_SCOPE)
        set(${output}_VISITED "${visited}" PARENT_SCOPE)
        return()
    endif()
    list(APPEND visited "${path}")
    file(READ "${TYLOINFER_NATIVE_GGUF_ROOT}/${path}" source)
    if(path STREQUAL "metal/abi/ExecutionGeometry.h")
        string(REGEX REPLACE "#define SPLASH_PREFILL_TOKEN_BUDGET [0-9]+u"
            "#define SPLASH_PREFILL_TOKEN_BUDGET ${TYLOINFER_NATIVE_PREFILL_TOKEN_BUDGET}u" source "${source}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${TYLOINFER_NATIVE_GGUF_ROOT}/${path}")
    string(REGEX MATCHALL "#[ \t]*include[ \t]+\"[^\"]+\"" includes "${source}")
    foreach(include IN LISTS includes)
        string(REGEX REPLACE "^[^\"]*\"([^\"]+)\".*$" "\\1" dependency "${include}")
        tyloinfer_native_gguf_expand("${dependency}" expanded "${visited}")
        set(visited "${expanded_VISITED}")
        string(REPLACE "${include}" "${expanded}" source "${source}")
    endforeach()
    string(REPLACE "#pragma once" "" source "${source}")
    set(${output} "${source}" PARENT_SCOPE)
    set(${output}_VISITED "${visited}" PARENT_SCOPE)
endfunction()

tyloinfer_native_gguf_expand("metal/kernels/shared/gguf_linear.metal" TYLOINFER_NATIVE_GGUF_LINEAR_SOURCE "")
tyloinfer_native_gguf_expand("metal/kernels/decode/linear_gguf_sgmatrix.metal" TYLOINFER_NATIVE_GGUF_REGISTER_SOURCE "")
tyloinfer_native_gguf_expand("metal/kernels/shared/gguf_float.metal" TYLOINFER_NATIVE_GGUF_FLOAT_SOURCE "")
tyloinfer_native_gguf_expand("metal/kernels/shared/gguf_repack.metal" TYLOINFER_NATIVE_GGUF_REPACK_SOURCE "")
tyloinfer_native_gguf_expand("metal/kernels/shared/moe.metal" TYLOINFER_NATIVE_GGUF_MOE_SOURCE "")
tyloinfer_native_gguf_expand("metal/kernels/shared/moe_gguf.metal" TYLOINFER_NATIVE_GGUF_MOE_PROJECTION_SOURCE "")
tyloinfer_native_gguf_expand("metal/kernels/shared/hyper_connection.metal" TYLOINFER_NATIVE_GGUF_HYPER_SOURCE "")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/affine_vector.metal" TYLOINFER_NATIVE_AFFINE_VECTOR_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/affine_vector.metal")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/gguf_vector.metal" TYLOINFER_NATIVE_GGUF_VECTOR_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/gguf_vector.metal")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/single_moe.metal" TYLOINFER_NATIVE_GGUF_SINGLE_MOE_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/single_moe.metal")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/hyper_bfloat.metal" TYLOINFER_NATIVE_HYPER_BFLOAT_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/hyper_bfloat.metal")
configure_file(gguf_sources.h.in gguf_sources.h @ONLY)

add_library(tyloinfer-metal-gguf-plans OBJECT
    "${TYLOINFER_NATIVE_GGUF_ROOT}/ops/Linear.cpp"
    "${TYLOINFER_NATIVE_GGUF_ROOT}/ops/LinearGguf.cpp"
    "${TYLOINFER_NATIVE_GGUF_ROOT}/ops/MoE.cpp"
    "${TYLOINFER_NATIVE_GGUF_ROOT}/ops/HyperConnection.cpp")
target_include_directories(tyloinfer-metal-gguf-plans PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/metal"
    "${TYLOINFER_NATIVE_GGUF_ROOT}")
target_compile_features(tyloinfer-metal-gguf-plans PRIVATE cxx_std_20)
target_compile_options(tyloinfer-metal-gguf-plans PRIVATE
    -include "${CMAKE_CURRENT_BINARY_DIR}/native_limits.h")

if(BUILD_TESTING)
    add_executable(tyloinfer-native-gguf-plan-test
        "${TYLOINFER_METAL_ROOT}/tests/native_gguf_plan_test.cpp"
        $<TARGET_OBJECTS:tyloinfer-metal-gguf-plans>)
    target_include_directories(tyloinfer-native-gguf-plan-test PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/metal"
        "${TYLOINFER_NATIVE_GGUF_ROOT}")
    target_compile_features(tyloinfer-native-gguf-plan-test PRIVATE cxx_std_20)
    target_compile_options(tyloinfer-native-gguf-plan-test PRIVATE
        -include "${CMAKE_CURRENT_BINARY_DIR}/native_limits.h")
    add_test(NAME tyloinfer-native-gguf-plan-test COMMAND tyloinfer-native-gguf-plan-test)
endif()
