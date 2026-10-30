find_package(Threads REQUIRED)

set(TYLOINFER_METAL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(TYLOINFER_METAL_INCLUDE_DIRS
    ${TYLOINFER_METAL_ROOT}/storage
    ${TYLOINFER_METAL_ROOT}/storage/gguf
    ${TYLOINFER_METAL_ROOT}/storage/mlx
    ${TYLOINFER_METAL_ROOT}/storage/compat
    ${TYLOINFER_METAL_ROOT}/runtime
    ${TYLOINFER_METAL_ROOT}/ops/common
    ${TYLOINFER_METAL_ROOT}/ops/tq
    ${TYLOINFER_METAL_ROOT}/ops/gguf
    ${TYLOINFER_METAL_ROOT}/ops/mlx
    ${TYLOINFER_METAL_ROOT}/ops/mfq
    ${TYLOINFER_METAL_ROOT}/models/deepseek_v4
    ${TYLOINFER_METAL_ROOT}/models/deepseek_v41
    ${TYLOINFER_METAL_ROOT}/models/flash_next
    ${TYLOINFER_METAL_ROOT}/models/qwen35
    ${TYLOINFER_METAL_ROOT}/models/minicpmo45
    ${TYLOINFER_METAL_ROOT}/models/nemotron_h
)

set(TYLOINFER_MLX_ROOT "" CACHE PATH
    "MLX package root containing include/, lib/libmlx.dylib, and lib/mlx.metallib")
if(NOT TYLOINFER_MLX_ROOT)
    file(GLOB _TYLOINFER_LOCAL_MLX_ROOTS
        "${TYLOINFER_REPOSITORY_ROOT}/.venv/lib/python*/site-packages/mlx")
    list(LENGTH _TYLOINFER_LOCAL_MLX_ROOTS _TYLOINFER_LOCAL_MLX_ROOT_COUNT)
    if(_TYLOINFER_LOCAL_MLX_ROOT_COUNT GREATER 0)
        list(GET _TYLOINFER_LOCAL_MLX_ROOTS 0 TYLOINFER_MLX_ROOT)
    endif()
endif()

set(TYLOINFER_MLX_METALLIB_DEFAULT "${TYLOINFER_MLX_ROOT}/lib/mlx.metallib" CACHE STRING
    "Default mlx.metallib path embedded in the native Metal runtime; leave empty for relocatable packages")

foreach(_TYLOINFER_MLX_REQUIRED
        "${TYLOINFER_MLX_ROOT}/include/mlx/mlx.h"
        "${TYLOINFER_MLX_ROOT}/lib/libmlx.dylib"
        "${TYLOINFER_MLX_ROOT}/lib/libjaccl.dylib"
        "${TYLOINFER_MLX_ROOT}/lib/mlx.metallib")
    if(NOT EXISTS "${_TYLOINFER_MLX_REQUIRED}")
        message(FATAL_ERROR
            "MLX native runtime file not found: ${_TYLOINFER_MLX_REQUIRED}. "
            "Set -DTYLOINFER_MLX_ROOT=/path/to/site-packages/mlx")
    endif()
endforeach()

add_library(mfq-mlx SHARED IMPORTED GLOBAL)
set_target_properties(mfq-mlx PROPERTIES
    IMPORTED_LOCATION "${TYLOINFER_MLX_ROOT}/lib/libmlx.dylib"
    INTERFACE_INCLUDE_DIRECTORIES
        "${TYLOINFER_MLX_ROOT}/include;${TYLOINFER_MLX_ROOT}/include/metal_cpp"
    INTERFACE_COMPILE_DEFINITIONS
        "_METAL_;ACCELERATE_NEW_LAPACK"
)
if(EXISTS "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/metal_3_1")
    set_property(TARGET mfq-mlx APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES
        "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/metal_3_1")
endif()

add_library(mfq-jaccl SHARED IMPORTED GLOBAL)
set_target_properties(mfq-jaccl PROPERTIES
    IMPORTED_LOCATION "${TYLOINFER_MLX_ROOT}/lib/libjaccl.dylib"
)

find_library(TYLOINFER_METAL_FRAMEWORK Metal REQUIRED)
find_library(TYLOINFER_FOUNDATION_FRAMEWORK Foundation REQUIRED)
find_library(TYLOINFER_QUARTZCORE_FRAMEWORK QuartzCore REQUIRED)
find_library(TYLOINFER_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
find_library(TYLOINFER_AVFOUNDATION_FRAMEWORK AVFoundation REQUIRED)
find_library(TYLOINFER_COREMEDIA_FRAMEWORK CoreMedia REQUIRED)
find_library(TYLOINFER_COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
find_library(TYLOINFER_IOSURFACE_FRAMEWORK IOSurface REQUIRED)
find_library(TYLOINFER_IOKIT_FRAMEWORK IOKit REQUIRED)

add_library(mfq-ane STATIC platform/mfq_ane.mm runtime/mtp_lora.cpp)
target_include_directories(mfq-ane PUBLIC ${TYLOINFER_METAL_ROOT}/platform)
target_include_directories(mfq-ane PUBLIC ${TYLOINFER_METAL_ROOT}/runtime)
target_compile_features(mfq-ane PUBLIC cxx_std_20)
target_compile_options(mfq-ane PRIVATE -fobjc-arc)
target_link_libraries(mfq-ane PRIVATE ${TYLOINFER_FOUNDATION_FRAMEWORK} ${TYLOINFER_IOSURFACE_FRAMEWORK})

add_executable(mfq-ane-test tests/mfq_ane_test.cpp)
target_link_libraries(mfq-ane-test PRIVATE mfq-ane)
target_compile_features(mfq-ane-test PRIVATE cxx_std_20)

add_library(mfq-avfoundation-video SHARED
    platform/mfq_avfoundation_video.mm
)
target_include_directories(mfq-avfoundation-video PUBLIC
    ${TYLOINFER_METAL_ROOT}/platform
)
target_compile_features(mfq-avfoundation-video PRIVATE cxx_std_20)
target_compile_options(mfq-avfoundation-video PRIVATE -fobjc-arc)
target_link_libraries(mfq-avfoundation-video PRIVATE
    ${TYLOINFER_FOUNDATION_FRAMEWORK}
    ${TYLOINFER_AVFOUNDATION_FRAMEWORK}
    ${TYLOINFER_COREMEDIA_FRAMEWORK}
    ${TYLOINFER_COREVIDEO_FRAMEWORK}
)
set_target_properties(mfq-avfoundation-video PROPERTIES
    OUTPUT_NAME "mfq_avfoundation_video"
)

# MLX's runtime Metal compiler does not expose filesystem include paths.  Fold
# the Steel MMA implementation into the MFE prefill source at configure time
# so the native primitive can use the same tiled MMA machinery as MLX/oMLX
# without requiring a full Xcode Metal command-line toolchain at build time.
set(_TYLOINFER_STEEL_STUB_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/metal-include-stubs")
file(MAKE_DIRECTORY "${_TYLOINFER_STEEL_STUB_DIR}")
foreach(_TYLOINFER_METAL_SYSTEM_HEADER
        metal_stdlib metal_math metal_simdgroup metal_simdgroup_matrix)
    file(TOUCH
        "${_TYLOINFER_STEEL_STUB_DIR}/${_TYLOINFER_METAL_SYSTEM_HEADER}")
endforeach()
file(MAKE_DIRECTORY
    "${_TYLOINFER_STEEL_STUB_DIR}/MetalPerformancePrimitives")
file(TOUCH
    "${_TYLOINFER_STEEL_STUB_DIR}/MetalPerformancePrimitives/MetalPerformancePrimitives.h")
set(_TYLOINFER_STEEL_ENTRY
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_steel_mma_entry.h")
set(TYLOINFER_MLX_UTILS_SOURCE "#define MLX_METAL_JIT\n")
foreach(_TYLOINFER_UTILS_HEADER bf16.h bf16_math.h complex.h defines.h logging.h utils.h)
    set(_TYLOINFER_UTILS_PATH "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/${_TYLOINFER_UTILS_HEADER}")
    file(READ "${_TYLOINFER_UTILS_PATH}" _TYLOINFER_UTILS_CONTENT)
    string(REGEX REPLACE "#[ \t]*include[ \t]+\"[^\"]*\"" "" _TYLOINFER_UTILS_CONTENT "${_TYLOINFER_UTILS_CONTENT}")
    string(REPLACE "#pragma once" "" _TYLOINFER_UTILS_CONTENT "${_TYLOINFER_UTILS_CONTENT}")
    string(APPEND TYLOINFER_MLX_UTILS_SOURCE "\n${_TYLOINFER_UTILS_CONTENT}\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_TYLOINFER_UTILS_PATH}")
endforeach()
configure_file("${TYLOINFER_METAL_ROOT}/kernels/common/mfq_mlx_utils_embedded.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_mlx_utils_embedded.h" @ONLY)
file(WRITE "${_TYLOINFER_STEEL_ENTRY}"
    "#include \"mlx/backend/metal/kernels/utils.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/gemm/gemm.h\"\n"
    "#include \"mlx/backend/metal/kernels/fp_quantized.h\"\n"
    "#include \"mlx/backend/metal/kernels/quantized_utils.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=310
        -DMLX_METAL_JIT
        "-I${TYLOINFER_MLX_ROOT}/include"
        "-I${_TYLOINFER_STEEL_STUB_DIR}"
        "${_TYLOINFER_STEEL_ENTRY}"
    RESULT_VARIABLE _TYLOINFER_STEEL_EXPAND_RESULT
    OUTPUT_VARIABLE TYLOINFER_STEEL_MMA_SOURCE
    ERROR_VARIABLE _TYLOINFER_STEEL_EXPAND_ERROR)
if(NOT _TYLOINFER_STEEL_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to embed MLX Steel MMA headers: "
        "${_TYLOINFER_STEEL_EXPAND_ERROR}")
endif()
set(_TYLOINFER_STEEL_NAX_ENTRY
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_steel_nax_entry.h")
file(WRITE "${_TYLOINFER_STEEL_NAX_ENTRY}"
    "#include \"mlx/backend/metal/kernels/complex.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/gemm/mma.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/gemm/nax.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=320
        "-I${TYLOINFER_MLX_ROOT}/include"
        "-I${_TYLOINFER_STEEL_STUB_DIR}"
        "${_TYLOINFER_STEEL_NAX_ENTRY}"
    RESULT_VARIABLE _TYLOINFER_STEEL_NAX_EXPAND_RESULT
    OUTPUT_VARIABLE TYLOINFER_STEEL_NAX_SOURCE
    ERROR_VARIABLE _TYLOINFER_STEEL_NAX_EXPAND_ERROR)
if(NOT _TYLOINFER_STEEL_NAX_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to embed MLX Steel NAX headers: "
        "${_TYLOINFER_STEEL_NAX_EXPAND_ERROR}")
endif()
set(_TYLOINFER_STEEL_ATTN_ENTRY
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_steel_attn_entry.h")
file(WRITE "${_TYLOINFER_STEEL_ATTN_ENTRY}"
    "#include \"mlx/backend/metal/kernels/utils.h\"\n"
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=310
        -DMLX_METAL_JIT
        "-I${TYLOINFER_MLX_ROOT}/include"
        "-I${_TYLOINFER_STEEL_STUB_DIR}"
        "${_TYLOINFER_STEEL_ATTN_ENTRY}"
    RESULT_VARIABLE _TYLOINFER_STEEL_ATTN_EXPAND_RESULT
    OUTPUT_VARIABLE TYLOINFER_STEEL_ATTN_SOURCE
    ERROR_VARIABLE _TYLOINFER_STEEL_ATTN_EXPAND_ERROR)
if(NOT _TYLOINFER_STEEL_ATTN_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Failed to embed MLX Steel attention headers: "
        "${_TYLOINFER_STEEL_ATTN_EXPAND_ERROR}")
endif()
if(TYLOINFER_ENABLE_MFQ_COMPAT)
set(_TYLOINFER_MFE_PREFILL_METAL
    "${TYLOINFER_METAL_ROOT}/kernels/mfq/mfq_mfe_prefill.metal")
file(READ "${_TYLOINFER_MFE_PREFILL_METAL}"
    TYLOINFER_MFE_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/gemm/gemm.h\""
    ""
    TYLOINFER_MFE_PREFILL_SOURCE
    "${TYLOINFER_MFE_PREFILL_SOURCE}")
set(_TYLOINFER_NINT_PREFILL_METAL
    "${TYLOINFER_METAL_ROOT}/kernels/mfq/mfq_nint_prefill.metal")
file(READ "${_TYLOINFER_NINT_PREFILL_METAL}"
    TYLOINFER_NINT_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/gemm/gemm.h\""
    ""
    TYLOINFER_NINT_PREFILL_SOURCE
    "${TYLOINFER_NINT_PREFILL_SOURCE}")
endif()
set(_TYLOINFER_DSA_SPARSE_PREFILL_METAL
    "${TYLOINFER_METAL_ROOT}/kernels/common/mfq_dsa_sparse_prefill.metal")
file(READ "${_TYLOINFER_DSA_SPARSE_PREFILL_METAL}"
    TYLOINFER_DSA_SPARSE_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\""
    ""
    TYLOINFER_DSA_SPARSE_PREFILL_SOURCE
    "${TYLOINFER_DSA_SPARSE_PREFILL_SOURCE}")
set(_TYLOINFER_SPARSE_BLOCK_GQA_METAL
    "${TYLOINFER_METAL_ROOT}/kernels/common/mfq_sparse_block_gqa.metal")
file(READ "${_TYLOINFER_SPARSE_BLOCK_GQA_METAL}"
    TYLOINFER_SPARSE_BLOCK_GQA_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\""
    ""
    TYLOINFER_SPARSE_BLOCK_GQA_SOURCE
    "${TYLOINFER_SPARSE_BLOCK_GQA_SOURCE}")
set(_TYLOINFER_CSA2_PREFILL_METAL
    "${TYLOINFER_METAL_ROOT}/kernels/common/mfq_csa2_prefill.metal")
file(READ "${_TYLOINFER_CSA2_PREFILL_METAL}" TYLOINFER_CSA2_PREFILL_SOURCE)
string(REPLACE
    "#include \"mlx/backend/metal/kernels/steel/attn/attn.h\"" ""
    TYLOINFER_CSA2_PREFILL_SOURCE "${TYLOINFER_CSA2_PREFILL_SOURCE}")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_TYLOINFER_CSA2_PREFILL_METAL}"
    "${_TYLOINFER_MFE_PREFILL_METAL}"
    "${_TYLOINFER_NINT_PREFILL_METAL}"
    "${_TYLOINFER_DSA_SPARSE_PREFILL_METAL}"
    "${_TYLOINFER_SPARSE_BLOCK_GQA_METAL}"
    "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/fp_quantized.h"
    "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/quantized_utils.h"
    "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/utils.h"
    "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/attn/attn.h"
    "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/gemm/gemm.h"
    "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/gemm/mma.h"
    "${TYLOINFER_MLX_ROOT}/include/mlx/backend/metal/kernels/steel/gemm/nax.h")
configure_file(
    "${TYLOINFER_METAL_ROOT}/kernels/common/mfq_mfe_prefill_embedded.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/mfq_mfe_prefill_embedded.h"
    @ONLY)

file(READ "${TYLOINFER_REPOSITORY_ROOT}/tyloinfer/kernels/metal/tq_decode.h" TYLOINFER_TQ_DECODE_SOURCE)
file(READ "${TYLOINFER_REPOSITORY_ROOT}/tyloinfer/kernels/metal/tq_matmul.h" TYLOINFER_TQ_MATMUL_SOURCE)
set(_TYLOINFER_NAX_FRAGMENT_ENTRY "${CMAKE_CURRENT_BINARY_DIR}/tyloinfer_nax_fragment_entry.h")
file(WRITE "${_TYLOINFER_NAX_FRAGMENT_ENTRY}"
    "#include \"mlx/backend/metal/kernels/steel/gemm/nax.h\"\n")
execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -E -P -x c++
        -D__METAL_VERSION__=320
        "-I${TYLOINFER_MLX_ROOT}/include"
        "-I${_TYLOINFER_STEEL_STUB_DIR}"
        "${_TYLOINFER_NAX_FRAGMENT_ENTRY}"
    RESULT_VARIABLE _TYLOINFER_NAX_FRAGMENT_EXPAND_RESULT
    OUTPUT_VARIABLE TYLOINFER_NAX_FRAGMENT_SOURCE
    ERROR_VARIABLE _TYLOINFER_NAX_FRAGMENT_EXPAND_ERROR)
if(NOT _TYLOINFER_NAX_FRAGMENT_EXPAND_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to embed NAX fragment headers: ${_TYLOINFER_NAX_FRAGMENT_EXPAND_ERROR}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${TYLOINFER_REPOSITORY_ROOT}/tyloinfer/kernels/metal/tq_decode.h"
    "${TYLOINFER_REPOSITORY_ROOT}/tyloinfer/kernels/metal/tq_matmul.h")
configure_file(kernels/tq/mfq_tq_embedded.h.in mfq_tq_embedded.h @ONLY)
file(READ "${PROJECT_SOURCE_DIR}/components/ggml/src/ggml-common.h" TYLOINFER_GGUF_COMMON_SOURCE)
file(READ "${PROJECT_SOURCE_DIR}/third_party/agention-gyro/codebook_f16.inc" TYLOINFER_GYRO_CODEBOOK_SOURCE)
file(READ "${TYLOINFER_METAL_ROOT}/kernels/gguf/gguf_decode.h" TYLOINFER_GGUF_DECODE_SOURCE)
file(READ "${TYLOINFER_METAL_ROOT}/kernels/gguf/gguf_matmul.h" TYLOINFER_GGUF_MATMUL_SOURCE)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/components/ggml/src/ggml-common.h"
    "${PROJECT_SOURCE_DIR}/third_party/agention-gyro/codebook_f16.inc"
    "${TYLOINFER_METAL_ROOT}/kernels/gguf/gguf_decode.h"
    "${TYLOINFER_METAL_ROOT}/kernels/gguf/gguf_matmul.h")
configure_file(kernels/gguf/gguf_embedded.h.in gguf_embedded.h @ONLY)
configure_file(kernels/mlx/mlx_affine_nax.h.in mlx_affine_nax.h @ONLY)

add_library(mfq-metal-runtime STATIC
    storage/hf_safetensors_store.cpp
    storage/mfq_container.cpp
    storage/mfe_expert_store.cpp
    storage/mlx_quantized_row_store.cpp
    runtime/mlx_tq_expert_cache.cpp
    runtime/mlx_streamed_weight_pool.cpp
    runtime/mlx_expert_segment_cache.cpp
    storage/mlx_tq_rows.cpp
    storage/compat/mlx_legacy_tensor_compat.cpp
    runtime/mlx_axial_patch_vision.cpp
    runtime/mlx_grid_vision.cpp
    runtime/mlx_mtp.cpp
    runtime/mlx_mtp_lora.cpp
    runtime/mlx_ane_matmul.cpp
    runtime/mtp_lora_sessions.cpp
    runtime/mlx_multimodal.cpp
    runtime/mlx_pre_ln_vision.cpp
    runtime/mlx_sampling.cpp
    runtime/mlx_kernel_prepare.cpp
    runtime/mlx_cached_kernel.cpp
    runtime/mlx_kernel_cache.cpp
    runtime/mlx_builtin_kernels.cpp
    runtime/mlx_inference_warmup.cpp
    runtime/mlx_tensor.cpp
    runtime/mlx_transformer.cpp
    runtime/mlx_kv_quantization.cpp
    runtime/mlx_qsa_kv_offload.cpp
    models/dflash2/mlx_dflash2.cpp
    ops/common/mlx_linear_attention.cpp
    ops/common/mlx_moe_ops.cpp
    ops/common/mlx_expert_route_tiles.cpp
    ops/common/mlx_mx.cpp
    ops/tq/mlx_tq.cpp
    ops/common/mlx_sparse_attention.cpp
    ops/common/mlx_sparse_prefill_tiles.cpp
    ops/common/mlx_csa2.cpp
    ops/common/mlx_dsa_cache_ops.cpp
    models/deepseek_v4/deepseek_v4_model.cpp
    models/deepseek_v4/mlx_deepseek_v4_attention.cpp
    models/deepseek_v4/mlx_deepseek_v4_causal_lm.cpp
    models/deepseek_v4/mlx_deepseek_v4_dspark.cpp
    models/deepseek_v4/mlx_deepseek_v4_hc.cpp
    models/deepseek_v4/mlx_deepseek_v4_moe.cpp
    ops/common/mlx_dsa.cpp
    models/deepseek_v4/mlx_deepseek_v4_vision.cpp
    models/deepseek_v41/mlx_deepseek_v41_attention.cpp
    models/deepseek_v41/mlx_deepseek_v41_causal_lm.cpp
    models/deepseek_v41/mlx_deepseek_v41_dspark.cpp
    models/deepseek_v41/mlx_deepseek_v41_engram.cpp
    models/deepseek_v41/mlx_deepseek_v41_mhc.cpp
    models/deepseek_v41/mlx_deepseek_v41_moe.cpp
    models/deepseek_v41/mlx_deepseek_v41_vision.cpp
    models/flash_next/mlx_qwen4_causal_lm.cpp
    models/flash_next/qwen4_ops.cpp
    models/minicpmo45/mlx_minicpmo45.cpp
    models/qwen35/mlx_qwen35_causal_lm.cpp
    models/qwen35/mlx_qwen35_full_attention.cpp
    models/qwen35/mlx_qwen35_linear_attention.cpp
    models/qwen35/qwen35_model.cpp
    models/nemotron_h/mlx_nemotron_h_attention.cpp
    models/nemotron_h/mlx_nemotron_h_causal_lm.cpp
    models/nemotron_h/mlx_nemotron_h_linear_attention.cpp
    models/nemotron_h/mlx_nemotron_h_moe.cpp
    models/nemotron_h/mlx_nemotron_h_ops.cpp
    models/nemotron_h/mlx_nemotron_h_vision.cpp
    models/nemotron_h/nemotron_h_model.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    target_sources(mfq-metal-runtime PRIVATE
        ops/mfq/mlx_moe.cpp
        ops/mfq/mlx_fp8_sq.cpp
        ops/mfq/mlx_mxfp4_sq.cpp
        ops/mfq/mlx_nint.cpp
        ops/mfq/mlx_nint_rows.cpp
        ops/mfq/mlx_nint8_zero.cpp
        ops/mfq/mlx_vq.cpp
        runtime/mlx_ssd_expert_arena.cpp
        runtime/mlx_ssd_expert_cache.cpp
        ops/mfq/mlx_grouped_linear_mfq.cpp
    )
else()
    target_sources(mfq-metal-runtime PRIVATE ops/common/mlx_grouped_linear.cpp)
endif()
add_library(mfq::metal-runtime ALIAS mfq-metal-runtime)
target_sources(mfq-metal-runtime PRIVATE platform/metal_device_info.cpp)
if(BUILD_TESTING)
add_executable(tyloinfer-metal-affine-test tests/mlx_affine_test.cpp)
add_executable(tyloinfer-metal-dflash2-test tests/mlx_dflash2_test.cpp)
target_link_libraries(tyloinfer-metal-dflash2-test PRIVATE mfq-metal-engine)
target_compile_features(tyloinfer-metal-dflash2-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-dflash2-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-dflash2-test COMMAND tyloinfer-metal-dflash2-test)
target_link_libraries(tyloinfer-metal-affine-test PRIVATE mfq-metal-runtime)
target_compile_features(tyloinfer-metal-affine-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-affine-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-affine-test COMMAND tyloinfer-metal-affine-test)
add_executable(tyloinfer-metal-gguf-test tests/mlx_gguf_test.cpp)
target_link_libraries(tyloinfer-metal-gguf-test PRIVATE mfq-metal-runtime mfq-ggml)
target_include_directories(tyloinfer-metal-gguf-test PRIVATE "${PROJECT_SOURCE_DIR}/components/ggml/src")
target_compile_features(tyloinfer-metal-gguf-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-gguf-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-gguf-test COMMAND tyloinfer-metal-gguf-test)
add_test(NAME tyloinfer-metal-gguf-heterogeneous-test COMMAND tyloinfer-metal-gguf-test heterogeneous)
add_test(NAME tyloinfer-metal-gguf-gyro-test COMMAND tyloinfer-metal-gguf-test gyro)
add_test(NAME tyloinfer-metal-gguf-gyro-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-test> gyro)
add_test(NAME tyloinfer-metal-gguf-heterogeneous-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-test> heterogeneous)
add_test(NAME tyloinfer-metal-gguf-native-stack-test COMMAND tyloinfer-metal-gguf-test native-stack)
add_test(NAME tyloinfer-metal-gguf-native-hyper-test COMMAND tyloinfer-metal-gguf-test native-hyper)
add_test(NAME tyloinfer-metal-gguf-native-stack-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-test> native-stack)
add_test(NAME tyloinfer-metal-gguf-native-hyper-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-test> native-hyper)
add_test(NAME tyloinfer-metal-gguf-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-test>)
add_executable(tyloinfer-metal-gguf-cache-test tests/mlx_gguf_expert_cache_test.cpp)
add_executable(tyloinfer-metal-gguf-deepseek41-test tests/mlx_gguf_deepseek41_test.cpp)
target_link_libraries(tyloinfer-metal-gguf-deepseek41-test PRIVATE mfq-metal-runtime mfq-ggml)
target_include_directories(tyloinfer-metal-gguf-deepseek41-test PRIVATE "${PROJECT_SOURCE_DIR}/components/ggml/src")
target_compile_features(tyloinfer-metal-gguf-deepseek41-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-gguf-deepseek41-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-gguf-deepseek41-test COMMAND tyloinfer-metal-gguf-deepseek41-test)
add_test(NAME tyloinfer-metal-gguf-deepseek41-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-deepseek41-test>)
add_executable(tyloinfer-metal-gguf-gyro-rotation-test tests/mlx_gguf_gyro_test.cpp)
target_link_libraries(tyloinfer-metal-gguf-gyro-rotation-test PRIVATE mfq-metal-runtime mfq-ggml)
target_compile_features(tyloinfer-metal-gguf-gyro-rotation-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-gguf-gyro-rotation-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-gguf-gyro-rotation-test COMMAND tyloinfer-metal-gguf-gyro-rotation-test)
add_test(NAME tyloinfer-metal-gguf-gyro-rotation-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-gyro-rotation-test>)
target_link_libraries(tyloinfer-metal-gguf-cache-test PRIVATE mfq-metal-runtime mfq-ggml)
target_include_directories(tyloinfer-metal-gguf-cache-test PRIVATE "${PROJECT_SOURCE_DIR}/components/ggml/src")
target_compile_features(tyloinfer-metal-gguf-cache-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-gguf-cache-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-gguf-cache-test COMMAND tyloinfer-metal-gguf-cache-test)
add_test(NAME tyloinfer-metal-gguf-cache-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-gguf-cache-test>)
add_executable(tyloinfer-metal-affine-cache-test tests/mlx_affine_expert_cache_test.cpp)
target_link_libraries(tyloinfer-metal-affine-cache-test PRIVATE mfq-metal-runtime)
target_compile_features(tyloinfer-metal-affine-cache-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-affine-cache-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-affine-cache-test COMMAND tyloinfer-metal-affine-cache-test)
add_test(NAME tyloinfer-metal-affine-cache-no-nax-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:tyloinfer-metal-affine-cache-test>)
add_executable(tyloinfer-metal-compiled-prepare-test tests/mlx_compiled_prepare_test.cpp)
target_link_libraries(tyloinfer-metal-compiled-prepare-test PRIVATE mfq-metal-runtime)
target_compile_features(tyloinfer-metal-compiled-prepare-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-compiled-prepare-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-compiled-prepare-test COMMAND tyloinfer-metal-compiled-prepare-test)
add_executable(tyloinfer-metal-cached-kernel-test tests/mlx_cached_kernel_test.cpp)
target_link_libraries(tyloinfer-metal-cached-kernel-test PRIVATE mfq-metal-runtime)
target_compile_features(tyloinfer-metal-cached-kernel-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-cached-kernel-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-cached-kernel-test COMMAND tyloinfer-metal-cached-kernel-test)
endif()
target_include_directories(mfq-metal-runtime PUBLIC
    ${TYLOINFER_METAL_INCLUDE_DIRS}
)
target_include_directories(mfq-metal-runtime PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(mfq-metal-runtime PUBLIC
    mfq-core
    mfq-models
    mfq-engine
    mfq-mlx
    mfq-jaccl
    mfq-ane
    Threads::Threads
    ${TYLOINFER_METAL_FRAMEWORK}
    ${TYLOINFER_FOUNDATION_FRAMEWORK}
    ${TYLOINFER_QUARTZCORE_FRAMEWORK}
    ${TYLOINFER_ACCELERATE_FRAMEWORK}
    ${TYLOINFER_IOKIT_FRAMEWORK}
)
target_link_libraries(mfq-metal-runtime PRIVATE mfq-json)
target_compile_features(mfq-metal-runtime PUBLIC cxx_std_20)
target_compile_definitions(mfq-metal-runtime PUBLIC
    TYLOINFER_NATIVE_MLX_RUNTIME=1
    TYLOINFER_MLX_METALLIB_DEFAULT="${TYLOINFER_MLX_METALLIB_DEFAULT}"
)

function(tyloinfer_add_metal_format target)
    add_library(${target} OBJECT ${ARGN})
    target_include_directories(${target} PRIVATE
        ${TYLOINFER_METAL_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${target} PRIVATE mfq-core mfq-models mfq-engine mfq-mlx mfq-json)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE TYLOINFER_NATIVE_MLX_RUNTIME=1)
    target_sources(mfq-metal-runtime PRIVATE $<TARGET_OBJECTS:${target}>)
endfunction()

add_subdirectory(ops/gguf/native)
target_sources(mfq-metal-runtime PRIVATE $<TARGET_OBJECTS:tyloinfer-metal-gguf-plans>)

tyloinfer_add_metal_format(tyloinfer-metal-gguf-ops
    ops/gguf/native/mlx_native_gguf.cpp
    ops/gguf/mlx_gguf.cpp
    ops/gguf/mlx_gguf_mixed.cpp
    ops/gguf/mlx_gguf_rotation.cpp
    ops/gguf/mlx_gguf_prefill.cpp
    ops/gguf/mlx_gguf_routes.cpp
    ops/gguf/mlx_gguf_staged.cpp
    ops/gguf/mlx_gguf_split.cpp
    ops/gguf/mlx_gguf_mhc.cpp
    storage/gguf/mlx_gguf_rows.cpp
    storage/gguf/mlx_gguf_expert_cache.cpp)
target_include_directories(tyloinfer-metal-gguf-ops PRIVATE
    "${TYLOINFER_METAL_ROOT}/ops/gguf/native"
    "${TYLOINFER_METAL_ROOT}/ops/gguf/native/metal"
    "${CMAKE_CURRENT_BINARY_DIR}/ops/gguf/native"
    "${PROJECT_SOURCE_DIR}/third_party/splash/runtime")
target_compile_options(tyloinfer-metal-gguf-ops PRIVATE
    -include "${CMAKE_CURRENT_BINARY_DIR}/ops/gguf/native/native_limits.h")
tyloinfer_add_metal_format(tyloinfer-metal-mlx-ops
    ops/mlx/mlx_affine.cpp
    ops/mlx/mlx_affine_staged.cpp
    ops/mlx/mlx_affine_mhc.cpp
    ops/mlx/mlx_affine_indirect.cpp
    storage/mlx/mlx_affine_rows.cpp
    storage/mlx/mlx_affine_expert_cache.cpp)
target_link_libraries(tyloinfer-metal-gguf-ops PRIVATE mfq-ggml)

add_executable(mfq-metal-tq-test tests/mlx_tq_test.cpp)
target_link_libraries(mfq-metal-tq-test PRIVATE mfq-metal-runtime)
target_include_directories(mfq-metal-tq-test PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_compile_features(mfq-metal-tq-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-tq-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_executable(tyloinfer-metal-tq-benchmark tests/mlx_tq_benchmark.cpp)
target_link_libraries(tyloinfer-metal-tq-benchmark PRIVATE mfq-metal-runtime)
target_compile_features(tyloinfer-metal-tq-benchmark PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-tq-benchmark PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
if(BUILD_TESTING)
    add_test(NAME tyloinfer-metal-tq-nemotron-test COMMAND mfq-metal-tq-test --nemotron)
    add_test(NAME tyloinfer-metal-tq-container-test COMMAND mfq-metal-tq-test unused --expert-containers)
    add_test(NAME tyloinfer-metal-tq-streamed-expert-test COMMAND mfq-metal-tq-test --streamed-experts)
    add_test(NAME tyloinfer-metal-tq-streamed-expert-no-nax-test
        COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:mfq-metal-tq-test> --streamed-experts)
    add_test(NAME tyloinfer-metal-tq-indirect-residency-test COMMAND mfq-metal-tq-test --indirect-residency)
    add_test(NAME tyloinfer-metal-tq-prefill-reduction-test COMMAND mfq-metal-tq-test --prefill-reduction)
    add_test(NAME tyloinfer-metal-tq-projection-cache-test COMMAND mfq-metal-tq-test --projection-cache)
    add_test(NAME tyloinfer-metal-tq-projection-cache-no-nax-test
        COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:mfq-metal-tq-test> --projection-cache)
    add_test(NAME tyloinfer-metal-tq-expansion-vector-test COMMAND mfq-metal-tq-test --expansion-vectors)
    add_test(NAME tyloinfer-metal-tq-column-input-test COMMAND mfq-metal-tq-test --column-inputs)
    add_test(NAME tyloinfer-metal-tq-gated-residual-test COMMAND mfq-metal-tq-test --gated-residual)
    add_test(NAME tyloinfer-metal-tq-gated-residual-no-nax-test
        COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1 $<TARGET_FILE:mfq-metal-tq-test> --gated-residual)
    add_test(NAME tyloinfer-metal-tq-compact-reuse-test COMMAND mfq-metal-tq-test --compact-reuse)
endif()

if(TARGET mfq-paged-prefix-cache)
    add_executable(mfq-metal-tq-model-benchmark tests/mlx_tq_model_benchmark.cpp)
    target_link_libraries(mfq-metal-tq-model-benchmark PRIVATE mfq-metal-runtime mfq-paged-prefix-cache)
    target_compile_features(mfq-metal-tq-model-benchmark PRIVATE cxx_std_20)
    set_target_properties(mfq-metal-tq-model-benchmark PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
endif()

add_library(mfq-metal-engine STATIC
    runtime/mlx_engine.cpp
    runtime/mlx_generation_job.cpp
    runtime/mlx_server_components.cpp
    runtime/mlx_dflash2_serving.cpp
)
target_link_libraries(mfq-metal-engine PUBLIC mfq-metal-runtime mfq-engine)
target_compile_features(mfq-metal-engine PUBLIC cxx_std_20)

add_executable(mfq-metal-engine-test tests/mlx_engine_test.cpp)
target_link_libraries(mfq-metal-engine-test PRIVATE mfq-metal-engine)
target_compile_features(mfq-metal-engine-test PRIVATE cxx_std_20)
add_test(NAME mfq-metal-engine-test COMMAND mfq-metal-engine-test)

add_executable(mfq-qsa-kv-offload-test tests/qsa_kv_offload_test.cpp runtime/mlx_paged_session_codec.cpp)
target_link_libraries(mfq-qsa-kv-offload-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-qsa-kv-offload-test PRIVATE cxx_std_20)
add_test(NAME mfq-qsa-kv-offload-test COMMAND mfq-qsa-kv-offload-test)

add_executable(mfq-decode-metal
    apps/mfq_decode_mlx.cpp
)
add_dependencies(mfq-decode-metal mfq-avfoundation-video)
target_link_libraries(mfq-decode-metal PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-decode-metal PRIVATE cxx_std_20)
target_link_options(mfq-decode-metal PRIVATE -Wl,-dead_strip)

if(TARGET mfq-paged-prefix-cache)
    target_sources(mfq-decode-metal PRIVATE
        runtime/mlx_paged_session_codec.cpp
    )

    add_executable(mfq-metal-paged-session-codec-test
        runtime/mlx_paged_session_codec.cpp
        tests/mlx_paged_session_codec_test.cpp
    )
    target_link_libraries(mfq-metal-paged-session-codec-test PRIVATE
        mfq-metal-runtime
        mfq-paged-prefix-cache
    )
    target_compile_features(
        mfq-metal-paged-session-codec-test PRIVATE cxx_std_20)
    add_test(
        NAME mfq-metal-paged-session-codec-test
        COMMAND mfq-metal-paged-session-codec-test
    )
endif()

set_target_properties(mfq-decode-metal PROPERTIES
    BUILD_RPATH "@executable_path/lib"
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "@executable_path/lib"
)

# Both native executables live in this binary directory and use the same
# relocatable MLX runtime bundle. A single dependency target prevents parallel
# links from copying the same runtime files at once.
set(_TYLOINFER_METAL_RUNTIME_BUNDLE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/lib")
option(TYLOINFER_BUILD_PRECOMPILED_METAL "Build bundled MFQ Metal libraries" OFF)
add_executable(mfq-metal-source-export
    apps/mfq_metal_source_export.cpp runtime/mlx_builtin_kernels.cpp)
target_include_directories(mfq-metal-source-export PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/ops/gguf/native"
    "${TYLOINFER_METAL_ROOT}/runtime" "${TYLOINFER_METAL_ROOT}/ops/common" "${TYLOINFER_METAL_ROOT}/ops/tq" "${TYLOINFER_METAL_ROOT}/ops/mfq" "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_features(mfq-metal-source-export PRIVATE cxx_std_20)
if(TYLOINFER_BUILD_PRECOMPILED_METAL)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND xcrun --find metal OUTPUT_VARIABLE _TYLOINFER_METAL_COMPILER
        RESULT_VARIABLE _TYLOINFER_METAL_COMPILER_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND xcrun --find metallib OUTPUT_VARIABLE _TYLOINFER_METALLIB_COMPILER
        RESULT_VARIABLE _TYLOINFER_METALLIB_COMPILER_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND xcrun --sdk macosx --show-sdk-path OUTPUT_VARIABLE _TYLOINFER_METAL_SDK
        RESULT_VARIABLE _TYLOINFER_METAL_SDK_RESULT OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _TYLOINFER_METAL_COMPILER_RESULT EQUAL 0 OR NOT _TYLOINFER_METALLIB_COMPILER_RESULT EQUAL 0
            OR NOT _TYLOINFER_METAL_SDK_RESULT EQUAL 0)
        message(FATAL_ERROR "Precompiled Metal libraries require the Xcode Metal Toolchain (metal and metallib)")
    endif()
    set(_TYLOINFER_AOT_ROOT "${_TYLOINFER_METAL_RUNTIME_BUNDLE_DIR}/mfq-metal")
    add_custom_command(OUTPUT "${_TYLOINFER_AOT_ROOT}/manifest.json"
        COMMAND $<TARGET_FILE:mfq-metal-source-export> "${CMAKE_CURRENT_BINARY_DIR}/metal-sources"
        COMMAND "${Python3_EXECUTABLE}" "${TYLOINFER_REPOSITORY_ROOT}/packaging/compile_metal_libraries.py"
            "${CMAKE_CURRENT_BINARY_DIR}/metal-sources" "${_TYLOINFER_AOT_ROOT}"
            --metal "${_TYLOINFER_METAL_COMPILER}" --metallib "${_TYLOINFER_METALLIB_COMPILER}" --sdk "${_TYLOINFER_METAL_SDK}"
        DEPENDS mfq-metal-source-export "${TYLOINFER_REPOSITORY_ROOT}/packaging/compile_metal_libraries.py"
        VERBATIM)
    add_custom_target(mfq-metal-precompiled DEPENDS "${_TYLOINFER_AOT_ROOT}/manifest.json")
    add_dependencies(mfq-decode-metal mfq-metal-precompiled)
    if(TARGET mfq-perplexity)
        add_dependencies(mfq-perplexity mfq-metal-precompiled)
    endif()
endif()
set(_TYLOINFER_METAL_RUNTIME_BUNDLE_STAMP
    "${_TYLOINFER_METAL_RUNTIME_BUNDLE_DIR}/.mfq-runtime-bundle.stamp")
add_custom_command(
    OUTPUT "${_TYLOINFER_METAL_RUNTIME_BUNDLE_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${_TYLOINFER_METAL_RUNTIME_BUNDLE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TYLOINFER_MLX_ROOT}/lib/libmlx.dylib"
        "${_TYLOINFER_METAL_RUNTIME_BUNDLE_DIR}/libmlx.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TYLOINFER_MLX_ROOT}/lib/libjaccl.dylib"
        "${_TYLOINFER_METAL_RUNTIME_BUNDLE_DIR}/libjaccl.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TYLOINFER_MLX_ROOT}/lib/mlx.metallib"
        "${_TYLOINFER_METAL_RUNTIME_BUNDLE_DIR}/mlx.metallib"
    DEPENDS
        "${TYLOINFER_MLX_ROOT}/lib/libmlx.dylib"
        "${TYLOINFER_MLX_ROOT}/lib/libjaccl.dylib"
        "${TYLOINFER_MLX_ROOT}/lib/mlx.metallib"
    COMMAND_EXPAND_LISTS
    COMMENT "Building the shared native Metal runtime bundle"
)

if(TARGET mfq-runtime-communication)
    target_link_libraries(mfq-decode-metal PRIVATE mfq-metal-engine mfq-runtime-communication)
    target_compile_definitions(
        mfq-decode-metal PRIVATE TYLOINFER_METAL_RUNTIME_COMMUNICATION=1)

    add_executable(mfq-perplexity
        apps/mfq_perplexity_mlx.cpp
    )
    target_link_libraries(mfq-perplexity PRIVATE
        mfq-metal-runtime
        mfq-runtime-communication
    )
    target_compile_features(mfq-perplexity PRIVATE cxx_std_20)
    set_target_properties(mfq-perplexity PROPERTIES
        BUILD_RPATH "@executable_path/lib"
        BUILD_WITH_INSTALL_RPATH ON
        INSTALL_RPATH "@executable_path/lib"
    )
endif()

add_custom_command(
    OUTPUT "${_TYLOINFER_METAL_RUNTIME_BUNDLE_STAMP}"
    APPEND
    COMMAND ${CMAKE_COMMAND} -E touch
        "${_TYLOINFER_METAL_RUNTIME_BUNDLE_STAMP}"
)
add_custom_target(mfq-metal-runtime-bundle
    DEPENDS "${_TYLOINFER_METAL_RUNTIME_BUNDLE_STAMP}")
add_dependencies(mfq-decode-metal mfq-metal-runtime-bundle)
if(TARGET mfq-perplexity)
    add_dependencies(mfq-perplexity mfq-metal-runtime-bundle)
    if(TARGET mfq-metal-precompiled)
        add_dependencies(mfq-perplexity mfq-metal-precompiled)
    endif()
endif()

add_executable(tyloinfer-metal-native-projection-test tests/mlx_mx_test.cpp)
target_link_libraries(tyloinfer-metal-native-projection-test PRIVATE mfq-metal-runtime)
target_compile_features(tyloinfer-metal-native-projection-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-native-projection-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-native-projection-test COMMAND tyloinfer-metal-native-projection-test)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-container-test
    tests/mfq_container_test.cpp
)
target_link_libraries(mfq-metal-container-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-container-test PRIVATE cxx_std_20)

enable_testing()
add_test(NAME mfq-metal-container-test COMMAND mfq-metal-container-test)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-nint-test
    tests/mlx_nint_test.cpp
)
target_link_libraries(mfq-metal-nint-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-nint-test COMMAND mfq-metal-nint-test)
add_test(NAME mfq-metal-nint-portable-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1
        $<TARGET_FILE:mfq-metal-nint-test>)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-kernel-prepare-test tests/mlx_kernel_prepare_test.cpp)
target_link_libraries(mfq-metal-kernel-prepare-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-kernel-prepare-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-kernel-prepare-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME mfq-metal-kernel-prepare-test COMMAND mfq-metal-kernel-prepare-test)
add_test(NAME mfq-metal-kernel-prepare-portable-test
    COMMAND ${CMAKE_COMMAND} -E env TYLOINFER_METAL_DISABLE_NAX=1
        $<TARGET_FILE:mfq-metal-kernel-prepare-test>)

endif()

add_executable(mfq-metal-inference-warmup-test tests/mlx_inference_warmup_test.cpp)
target_link_libraries(mfq-metal-inference-warmup-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-inference-warmup-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-inference-warmup-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME mfq-metal-inference-warmup-test COMMAND mfq-metal-inference-warmup-test)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-nint-rows-test tests/mlx_nint_rows_test.cpp)
target_link_libraries(mfq-metal-nint-rows-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-nint-rows-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-rows-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME mfq-metal-nint-rows-test COMMAND mfq-metal-nint-rows-test)

endif()

add_executable(mfq-metal-memory-residency-test tests/mlx_memory_residency_test.cpp)
target_link_libraries(mfq-metal-memory-residency-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-memory-residency-test PRIVATE cxx_std_20)
target_compile_options(mfq-metal-memory-residency-test PRIVATE -UNDEBUG)
set_target_properties(mfq-metal-memory-residency-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME mfq-metal-memory-residency-test COMMAND mfq-metal-memory-residency-test)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-mx-test
    tests/mlx_mx_test.cpp
)
target_link_libraries(mfq-metal-mx-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mx-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mx-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-mx-test COMMAND mfq-metal-mx-test)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-fp8-sq-test
    tests/mlx_fp8_sq_test.cpp
)
target_link_libraries(mfq-metal-fp8-sq-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-fp8-sq-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-fp8-sq-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-fp8-sq-test COMMAND mfq-metal-fp8-sq-test)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-mxfp4-sq2-test
    tests/mlx_mxfp4_sq2_test.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq2-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq2-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq2-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-mxfp4-sq2-test
    COMMAND mfq-metal-mxfp4-sq2-test
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-mxfp4-sq2-benchmark
    benchmarks/mlx_mxfp4_sq2_benchmark.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq2-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq2-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq2-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-mxfp4-sq3-test
    tests/mlx_mxfp4_sq3_test.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq3-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq3-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq3-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-mxfp4-sq3-test
    COMMAND mfq-metal-mxfp4-sq3-test
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-mxfp4-sq3-benchmark
    benchmarks/mlx_mxfp4_sq3_benchmark.cpp
)
target_link_libraries(mfq-metal-mxfp4-sq3-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mxfp4-sq3-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mxfp4-sq3-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

add_executable(mfq-metal-mx-benchmark
    benchmarks/mlx_mx_benchmark.cpp
)
target_link_libraries(mfq-metal-mx-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mx-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mx-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-nint-benchmark
    benchmarks/mlx_nint_benchmark.cpp
)
target_link_libraries(mfq-metal-nint-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

add_executable(mfq-metal-dense-smallm-benchmark
    benchmarks/mlx_dense_smallm_benchmark.cpp
)
target_link_libraries(mfq-metal-dense-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_include_directories(mfq-metal-dense-smallm-benchmark PRIVATE
    ${TYLOINFER_REPOSITORY_ROOT}/bench
)
target_compile_features(
    mfq-metal-dense-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-dense-smallm-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-mfe-smallm-benchmark
    benchmarks/mlx_mfe_smallm_benchmark.cpp
)
target_link_libraries(mfq-metal-mfe-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-mfe-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mfe-smallm-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-vq-smallm-benchmark
    benchmarks/mlx_vq_smallm_benchmark.cpp
)
target_link_libraries(mfq-metal-vq-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-vq-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-vq-smallm-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-mfe-prefill-benchmark
    benchmarks/mlx_mfe_prefill_benchmark.cpp
)
target_link_libraries(mfq-metal-mfe-prefill-benchmark PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-mfe-prefill-benchmark PRIVATE cxx_std_20)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-vq-benchmark
    benchmarks/mlx_vq_benchmark.cpp
)
target_link_libraries(mfq-metal-vq-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-vq-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-vq-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-grouped-linear-benchmark
    benchmarks/mlx_grouped_linear_benchmark.cpp
)
target_link_libraries(mfq-metal-grouped-linear-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-grouped-linear-benchmark PRIVATE cxx_std_20)
set_target_properties(mfq-metal-grouped-linear-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-qwen35-projection-profile
    tools/qwen35_projection_profile.cpp
)
target_link_libraries(
    mfq-metal-qwen35-projection-profile PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-projection-profile PRIVATE cxx_std_20
)
set_target_properties(
    mfq-metal-qwen35-projection-profile PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

add_executable(mfq-metal-qwen35-decode-overhead-profile
    tools/qwen35_decode_overhead_profile.cpp
)
target_link_libraries(
    mfq-metal-qwen35-decode-overhead-profile PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-decode-overhead-profile PRIVATE cxx_std_20
)
set_target_properties(
    mfq-metal-qwen35-decode-overhead-profile PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-nint8-zero-test
    tests/mlx_nint8_zero_test.cpp
)
target_link_libraries(mfq-metal-nint8-zero-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-nint8-zero-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-nint8-zero-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-nint8-zero-test
    COMMAND mfq-metal-nint8-zero-test
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-grouped-linear-test
    tests/mlx_grouped_linear_test.cpp
)
target_link_libraries(mfq-metal-grouped-linear-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-grouped-linear-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-grouped-linear-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-grouped-linear-test
    COMMAND mfq-metal-grouped-linear-test
)

endif()

add_executable(mfq-metal-deepseek-v4-hc-test
    tests/mlx_deepseek_v4_hc_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-hc-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-hc-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-hc-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-hc-test
    COMMAND mfq-metal-deepseek-v4-hc-test
)

add_executable(mfq-metal-deepseek-v41-fusions-test
    tests/mlx_deepseek_v41_fusions_test.cpp
)
target_link_libraries(mfq-metal-deepseek-v41-fusions-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-deepseek-v41-fusions-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-deepseek-v41-fusions-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-deepseek-v41-fusions-test COMMAND mfq-metal-deepseek-v41-fusions-test)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-deepseek-v4-model-test
    tests/deepseek_v4_model_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-model-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-model-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-model-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-model-test
    COMMAND mfq-metal-deepseek-v4-model-test
)

endif()

add_executable(mfq-metal-hf-safetensors-store-test
    storage/hf_safetensors_store.cpp
    tests/hf_safetensors_store_test.cpp
)

target_include_directories(mfq-metal-hf-safetensors-store-test PRIVATE
    ${TYLOINFER_METAL_ROOT}/storage
)
target_link_libraries(mfq-metal-hf-safetensors-store-test PRIVATE mfq-model-source)
target_compile_features(
    mfq-metal-hf-safetensors-store-test PRIVATE cxx_std_20)
add_test(
    NAME mfq-metal-hf-safetensors-store-test
    COMMAND mfq-metal-hf-safetensors-store-test
)

add_executable(mfq-metal-quantized-row-store-test
    tests/mlx_quantized_row_store_test.cpp
)
target_link_libraries(mfq-metal-quantized-row-store-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-quantized-row-store-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-quantized-row-store-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-quantized-row-store-test
    COMMAND mfq-metal-quantized-row-store-test
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-ssd-expert-benchmark
    benchmarks/ssd_expert_benchmark.cpp
)
target_link_libraries(
    mfq-metal-ssd-expert-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-expert-benchmark PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-expert-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-deepseek-v4-moe-smallm-benchmark
    benchmarks/mlx_deepseek_v4_moe_smallm_benchmark.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-moe-smallm-benchmark PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-moe-smallm-benchmark PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-moe-smallm-benchmark PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-ssd-expert-arena-test
    tests/mlx_ssd_expert_arena_test.cpp
)
target_link_libraries(
    mfq-metal-ssd-expert-arena-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-expert-arena-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-expert-arena-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-ssd-expert-arena-test
    COMMAND mfq-metal-ssd-expert-arena-test
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-ssd-expert-real-smoke
    tools/ssd_expert_real_smoke.cpp
)
target_link_libraries(
    mfq-metal-ssd-expert-real-smoke PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-expert-real-smoke PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-expert-real-smoke PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

endif()

add_executable(mfq-metal-ssd-hf-model-smoke
    tools/ssd_hf_model_smoke.cpp
)
target_link_libraries(
    mfq-metal-ssd-hf-model-smoke PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-hf-model-smoke PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-hf-model-smoke PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

add_executable(mfq-metal-ssd-hf-prefill-smoke
    tools/ssd_hf_prefill_smoke.cpp
)
target_link_libraries(
    mfq-metal-ssd-hf-prefill-smoke PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-ssd-hf-prefill-smoke PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-ssd-hf-prefill-smoke PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)

add_executable(mfq-metal-deepseek-v4-attention-test
    tests/mlx_deepseek_v4_attention_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-attention-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-attention-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-attention-test
    COMMAND mfq-metal-deepseek-v4-attention-test
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-deepseek-v4-causal-lm-test
    tests/mlx_deepseek_v4_causal_lm_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-causal-lm-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-causal-lm-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-causal-lm-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-causal-lm-test
    COMMAND mfq-metal-deepseek-v4-causal-lm-test
)

endif()

add_executable(mfq-metal-deepseek-v4-sparse-test
    tests/mlx_deepseek_v4_sparse_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-sparse-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-sparse-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-sparse-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-sparse-test
    COMMAND mfq-metal-deepseek-v4-sparse-test
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-deepseek-v4-moe-test
    tests/mlx_deepseek_v4_moe_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-moe-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-moe-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-moe-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-moe-test
    COMMAND mfq-metal-deepseek-v4-moe-test
)

endif()

add_executable(mfq-metal-mtp-test tests/mlx_mtp_test.cpp)
target_link_libraries(mfq-metal-mtp-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-mtp-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-mtp-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-mtp-test COMMAND mfq-metal-mtp-test)

add_executable(mfq-mtp-lora-sessions-test tests/mtp_lora_sessions_test.cpp)
target_link_libraries(mfq-mtp-lora-sessions-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-mtp-lora-sessions-test PRIVATE cxx_std_20)
add_test(NAME mfq-mtp-lora-sessions-test COMMAND mfq-mtp-lora-sessions-test)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-deepseek-v4-dspark-test
    tests/mlx_deepseek_v4_dspark_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-dspark-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-dspark-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-dspark-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-dspark-test
    COMMAND mfq-metal-deepseek-v4-dspark-test
)

endif()

add_executable(mfq-metal-deepseek-v4-vision-test
    tests/mlx_deepseek_v4_vision_test.cpp
)
target_link_libraries(
    mfq-metal-deepseek-v4-vision-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-deepseek-v4-vision-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-deepseek-v4-vision-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-deepseek-v4-vision-test
    COMMAND mfq-metal-deepseek-v4-vision-test
)

add_executable(mfq-metal-deepseek-v41-vision-test
    tests/mlx_deepseek_v41_vision_test.cpp
)
target_link_libraries(mfq-metal-deepseek-v41-vision-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-deepseek-v41-vision-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-deepseek-v41-vision-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-deepseek-v41-vision-test COMMAND mfq-metal-deepseek-v41-vision-test)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_test(NAME mfq-metal-deepseek-v41-vision-routing-test COMMAND mfq-metal-moe-test --test-v41-vision-routing)
    add_test(NAME mfq-metal-deepseek-v41-prefix-test COMMAND mfq-metal-moe-test --test-v41-prefix)
endif()

add_executable(mfq-metal-moe-ops-test
    tests/mlx_moe_ops_test.cpp
)
target_link_libraries(mfq-metal-moe-ops-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-moe-ops-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-moe-ops-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-moe-ops-test
    COMMAND mfq-metal-moe-ops-test
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-moe-test
    runtime/mlx_paged_session_codec.cpp
    tests/mlx_moe_test.cpp
)
target_link_libraries(mfq-metal-moe-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-moe-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-moe-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-moe-test
    COMMAND mfq-metal-moe-test
)
add_test(NAME mfq-metal-moe-compat-test COMMAND mfq-metal-moe-test --test-compat)
add_test(NAME mfq-metal-moe-portable-test COMMAND mfq-metal-moe-test)
set_tests_properties(mfq-metal-moe-portable-test PROPERTIES ENVIRONMENT "TYLOINFER_METAL_DISABLE_NAX=1")
add_test(NAME mfq-metal-moe-prepare-test COMMAND mfq-metal-moe-test --test-prepare)
add_test(NAME mfq-metal-model-prepare-test COMMAND mfq-metal-moe-test --test-model-prepare)
option(TYLOINFER_TEST_ANE "Run private ANE hardware tests" OFF)
if(TYLOINFER_TEST_ANE)
    add_test(NAME mfq-ane-test COMMAND mfq-ane-test)
    add_test(NAME mfq-metal-ane-autograd-test COMMAND mfq-metal-moe-test --test-ane-autograd)
    add_test(NAME mfq-metal-mtp-ttt-test COMMAND mfq-metal-moe-test --test-mtp-ttt)
endif()

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-vq-test
    tests/mlx_vq_test.cpp
)
target_link_libraries(mfq-metal-vq-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-vq-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-vq-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-vq-test
    COMMAND mfq-metal-vq-test
)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-tensor-test
    tests/mlx_tensor_test.cpp
)
target_link_libraries(mfq-metal-tensor-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-tensor-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-tensor-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-tensor-test COMMAND mfq-metal-tensor-test)

endif()

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-dynamic-geometry-test
    tests/mlx_dynamic_geometry_test.cpp
)
target_link_libraries(mfq-metal-dynamic-geometry-test PRIVATE mfq-metal-runtime)
target_compile_features(mfq-metal-dynamic-geometry-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-dynamic-geometry-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-dynamic-geometry-test COMMAND mfq-metal-dynamic-geometry-test)

endif()

add_executable(mfq-metal-transformer-test
    tests/mlx_transformer_test.cpp
)
target_link_libraries(mfq-metal-transformer-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-transformer-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-transformer-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-transformer-test
    COMMAND mfq-metal-transformer-test
)

add_executable(mfq-metal-multimodal-test
    tests/mlx_multimodal_test.cpp
)
target_link_libraries(mfq-metal-multimodal-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-multimodal-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-multimodal-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-multimodal-test
    COMMAND mfq-metal-multimodal-test
)

add_executable(mfq-metal-linear-attention-test
    tests/mlx_linear_attention_test.cpp
)
target_link_libraries(mfq-metal-linear-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-linear-attention-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-linear-attention-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-linear-attention-test
    COMMAND mfq-metal-linear-attention-test
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_executable(mfq-metal-qwen35-config-test
    tests/qwen35_config_test.cpp
)
target_link_libraries(mfq-metal-qwen35-config-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-config-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen35-config-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-config-test
    COMMAND mfq-metal-qwen35-config-test
)

endif()

add_executable(mfq-metal-qwen4-config-test
    tests/qwen4_config_test.cpp
)

target_link_libraries(mfq-metal-qwen4-config-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen4-config-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen4-config-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen4-config-test
    COMMAND mfq-metal-qwen4-config-test
)

add_executable(mfq-metal-qwen4-batching-test tests/mlx_qwen4_batching_test.cpp)
target_link_libraries(mfq-metal-qwen4-batching-test PRIVATE mfq-metal-runtime mfq-engine)
target_compile_features(mfq-metal-qwen4-batching-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen4-batching-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")

add_executable(mfq-metal-sparse-attention-test
    tests/mlx_sparse_attention_test.cpp
)
target_link_libraries(mfq-metal-sparse-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-sparse-attention-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-sparse-attention-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-sparse-attention-test
    COMMAND mfq-metal-sparse-attention-test
)

add_executable(tyloinfer-metal-dense-hc-test tests/mlx_dense_hc_test.cpp)
target_link_libraries(tyloinfer-metal-dense-hc-test PRIVATE mfq-metal-runtime)
target_compile_features(tyloinfer-metal-dense-hc-test PRIVATE cxx_std_20)
set_target_properties(tyloinfer-metal-dense-hc-test PROPERTIES BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib")
add_test(NAME tyloinfer-metal-dense-hc-test COMMAND tyloinfer-metal-dense-hc-test)

add_executable(mfq-metal-qwen-gated-hc-test
    tests/qwen_gated_hc_test.cpp
)
target_link_libraries(mfq-metal-qwen-gated-hc-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen-gated-hc-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-qwen-gated-hc-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen-gated-hc-test
    COMMAND mfq-metal-qwen-gated-hc-test
)
add_test(NAME mfq-metal-qwen-gated-hc-no-nax-test COMMAND mfq-metal-qwen-gated-hc-test)
set_tests_properties(mfq-metal-qwen-gated-hc-no-nax-test PROPERTIES
    ENVIRONMENT "TYLOINFER_METAL_DISABLE_NAX=1;MFQ_METAL_DISABLE_NAX=1")

add_executable(mfq-metal-grid-vision-test
    tests/mlx_grid_vision_test.cpp
)
target_link_libraries(
    mfq-metal-grid-vision-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-grid-vision-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-grid-vision-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-grid-vision-test
    COMMAND mfq-metal-grid-vision-test
)

add_executable(mfq-metal-qwen35-full-attention-test
    tests/mlx_qwen35_full_attention_test.cpp
)
target_link_libraries(
    mfq-metal-qwen35-full-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-full-attention-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-qwen35-full-attention-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-full-attention-test
    COMMAND mfq-metal-qwen35-full-attention-test
)

add_executable(mfq-metal-qwen35-linear-attention-test
    tests/mlx_qwen35_linear_attention_test.cpp
)
target_link_libraries(
    mfq-metal-qwen35-linear-attention-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-linear-attention-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-qwen35-linear-attention-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-linear-attention-test
    COMMAND mfq-metal-qwen35-linear-attention-test
)

add_executable(mfq-metal-sampling-test
    tests/mlx_sampling_test.cpp
)
target_link_libraries(mfq-metal-sampling-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(mfq-metal-sampling-test PRIVATE cxx_std_20)
set_target_properties(mfq-metal-sampling-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-sampling-test
    COMMAND mfq-metal-sampling-test
)

add_executable(mfq-metal-qwen35-causal-lm-test
    tests/mlx_qwen35_causal_lm_test.cpp
)
target_link_libraries(
    mfq-metal-qwen35-causal-lm-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-causal-lm-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-qwen35-causal-lm-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-qwen35-causal-lm-test
    COMMAND mfq-metal-qwen35-causal-lm-test
)

add_executable(mfq-metal-nemotron-test
    tests/mlx_nemotron_h_test.cpp
)
target_link_libraries(
    mfq-metal-nemotron-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-nemotron-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-nemotron-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(NAME mfq-metal-nemotron-conv-test
    COMMAND mfq-metal-nemotron-test --check-cached-conv)
add_executable(mfq-metal-minicpmo45-test
    tests/mlx_minicpmo45_test.cpp
)
target_link_libraries(
    mfq-metal-minicpmo45-test PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-minicpmo45-test PRIVATE cxx_std_20)
set_target_properties(
    mfq-metal-minicpmo45-test PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
add_test(
    NAME mfq-metal-minicpmo45-test
    COMMAND mfq-metal-minicpmo45-test
)

add_executable(mfq-metal-qwen35-grouped-parity
    tools/qwen35_grouped_parity.cpp
)
target_link_libraries(
    mfq-metal-qwen35-grouped-parity PRIVATE
    mfq-metal-runtime
)
target_compile_features(
    mfq-metal-qwen35-grouped-parity PRIVATE cxx_std_20
)
set_target_properties(
    mfq-metal-qwen35-grouped-parity PROPERTIES
    BUILD_RPATH "${TYLOINFER_MLX_ROOT}/lib"
)
