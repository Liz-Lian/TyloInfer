# Native CUDA backend. Compute, execution and loading targets do not require
# the optional service transport; only the runtime service below does.

set(TYLOINFER_CUDA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
# Single source of the shared CUDA kernels; the Python extension builds the same files.
set(TYLOINFER_CUDA_KERNEL_ROOT "${TYLOINFER_CUDA_ROOT}/kernels")
# TQ kernels stay in the Python package because they are installed wheel resources.
set(TYLOINFER_CUDA_TQ_ROOT "${TYLOINFER_REPOSITORY_ROOT}/tyloinfer/kernels/cuda")

set(TYLOINFER_CUDA_ARCHITECTURES "" CACHE STRING
    "Optional CUDA architecture override; otherwise use CMAKE_CUDA_ARCHITECTURES, CUDAARCHS or native")
if(NOT "${TYLOINFER_CUDA_ARCHITECTURES}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}")
elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(_mfq_cuda_architectures native)
    if(NOT "$ENV{CUDAARCHS}" STREQUAL "")
        set(_mfq_cuda_architectures "$ENV{CUDAARCHS}")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "${_mfq_cuda_architectures}" CACHE STRING
        "CUDA architectures compiled into MFQ CUDA targets")
    unset(_mfq_cuda_architectures)
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Keep the override cache empty unless explicitly set, so later changes to the
# standard CMake option are honored on reconfigure.
set(TYLOINFER_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_library(mfq-cuda-storage STATIC
    ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/expert_store.cpp
)
add_library(mfq::cuda-storage ALIAS mfq-cuda-storage)
target_include_directories(mfq-cuda-storage PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-cuda-storage PUBLIC Threads::Threads)
target_compile_features(mfq-cuda-storage PUBLIC cxx_std_20)
set_target_properties(mfq-cuda-storage PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-cuda-storage PRIVATE /utf-8 /EHsc)
endif()

else()
    add_library(mfq-cuda-storage INTERFACE)
    add_library(mfq::cuda-storage ALIAS mfq-cuda-storage)
endif()

add_library(mfq-cuda-core STATIC
    ${TYLOINFER_CUDA_ROOT}/native/env.cpp
    ${TYLOINFER_CUDA_ROOT}/native/context.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/tensor.cpp
    ${TYLOINFER_CUDA_ROOT}/native/tensor/tensor.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/blas.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/creation.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/indexing.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/ops.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/reduction.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/signal.cu
    ${TYLOINFER_CUDA_ROOT}/native/tensor/sort.cu
)
add_library(mfq::cuda-core ALIAS mfq-cuda-core)
# The native context/tensor units only need their own headers plus the CUDA SDK;
# the high-level kernel op declarations in include/ stay out of the interface.
target_include_directories(mfq-cuda-core PUBLIC
    ${TYLOINFER_CUDA_ROOT}/native
    ${TYLOINFER_CUDA_ROOT}/native/tensor
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_include_directories(mfq-cuda-core PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(mfq-cuda-core PUBLIC CUDA::cudart CUDA::cublas mfq-host-parallel)
if(WIN32)
    target_link_libraries(mfq-cuda-core PUBLIC dxgi)
endif()
target_compile_features(mfq-cuda-core PUBLIC cxx_std_20)
target_compile_options(mfq-cuda-core PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)
set_target_properties(mfq-cuda-core PROPERTIES
    CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

set(TYLOINFER_CUDA_KERNEL_SOURCES
    ${TYLOINFER_CUDA_KERNEL_ROOT}/acc.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/activation.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/attention.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/attention_mma.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/deepseek_v4_attention.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/deepseek_v4_hc.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/deepseek_v41.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/embedding.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/selected_attention.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/qwen4_exp.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/ple_fused.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/glm5_next.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/gated_delta_net.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/gdn_decode_window.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/glm_dsa.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/kv_cache.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/paged_kv.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/moe.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mx_matmul.cu
    # nepq.cu is only the generic Hadamard transform pair; the default model path calls it.
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nepq.cu
    ${TYLOINFER_CUDA_TQ_ROOT}/tq_ops.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/norm.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/rope.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/sampling.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/selective_state_scan.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/ssm_conv.cu
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_KERNEL_SOURCES
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_compact.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_parity_read.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_cached_reduce.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_shared.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_token_groups.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_prepare.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_launch.cpp
    ${TYLOINFER_CUDA_KERNEL_ROOT}/fp8_sq.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mxfp4_sq.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nepq_residual.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nint_matmul.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nint_short_decode.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nint_q8_decode.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_matmul.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_routed_reuse.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/nvq1_decode.cu
    )
endif()

add_library(mfq-cuda-native-kernels STATIC
    ${TYLOINFER_CUDA_KERNEL_SOURCES}
    ${TYLOINFER_CUDA_ROOT}/kernels/float_projection.cu
    ${TYLOINFER_CUDA_ROOT}/kernels/kv_quantization.cu
    ${TYLOINFER_CUDA_ROOT}/kernels/kv_hot_cache.cu
    ${TYLOINFER_CUDA_ROOT}/kernels/shared_gate_fused.cu
    ${TYLOINFER_CUDA_ROOT}/kernels/moe/pipeline.cu
    ${TYLOINFER_CUDA_ROOT}/apps/ggml_cuda_compat.cu
)
add_library(mfq::cuda-native-kernels ALIAS mfq-cuda-native-kernels)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    target_sources(mfq-cuda-native-kernels PRIVATE
        ${TYLOINFER_CUDA_ROOT}/kernels/gr_prefill_blas.cu
        ${TYLOINFER_CUDA_ROOT}/kernels/gated_residual_fused.cu
    )
endif()

# Keep the shared packed math in incremental builds even when nvcc's depfile
# omits headers from a heavily inlined translation unit on Windows.
set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nint.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_compact.h;${TYLOINFER_CUDA_KERNEL_ROOT}/glu.cuh")
set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_compact.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nvq_compact_dot.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_jsc_bank_map.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_dense_window32.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nint.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_compact.h")
set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_parity_read.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_gu_prequantized.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_gu_group_quant.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_gu_transfer_routes.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nvq_compact_dot.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nvq_parity_read.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_jsc_bank_map.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_dense_window32.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nint.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h")
set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_token_groups.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_token_groups.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nint_tokens.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nint.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_e8_dense_group.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_d4_dense_group.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_dense_window32.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_jsc_bank_map.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nint_route_group.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nint_whole.cuh")
set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/nint_matmul.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nint.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/nint_route_group.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/glu.cuh")
set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/nvq_matmul.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/packed_nvq.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/glu.cuh")
set_property(SOURCE
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_compact.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_parity_read.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_shared.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_token_groups.cu
    ${TYLOINFER_CUDA_ROOT}/kernels/moe/pipeline.cu
    APPEND PROPERTY OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/shared_storage.cuh")
set_source_files_properties(${TYLOINFER_CUDA_ROOT}/kernels/gated_residual_fused.cu PROPERTIES
    COMPILE_OPTIONS "--maxrregcount=80")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set_property(SOURCE ${TYLOINFER_CUDA_ROOT}/kernels/gated_residual_fused.cu APPEND
        PROPERTY COMPILE_OPTIONS "--split-compile=0")
endif()
target_compile_definitions(mfq-cuda-native-kernels PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1
)
target_include_directories(mfq-cuda-native-kernels PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${TYLOINFER_CUDA_ROOT}
    ${TYLOINFER_REPOSITORY_ROOT}
    ${TYLOINFER_CUDA_ROOT}/include
    ${TYLOINFER_CUDA_ROOT}/native
    ${TYLOINFER_CUDA_ROOT}/native/tensor
    ${TYLOINFER_CUDA_KERNEL_ROOT}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(mfq-cuda-native-kernels
    PUBLIC mfq-cuda-core CUDA::cuda_driver
    PRIVATE mfq-formats mfq-ggml-internal-headers mfq-ggml
)
target_compile_options(mfq-cuda-native-kernels PRIVATE
    "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
)
set_target_properties(mfq-cuda-native-kernels PROPERTIES
    CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    foreach(mfq_cuda_target mfq-cuda-core mfq-cuda-native-kernels)
        target_compile_options(${mfq_cuda_target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        )
    endforeach()
endif()

if(BUILD_TESTING)
    function(mfq_add_cuda_test target source)
        add_executable(${target} ${source})
        target_link_libraries(${target} PRIVATE mfq-cuda-core ${ARGN})
        target_include_directories(${target} PRIVATE ${TYLOINFER_CUDA_ROOT}/include)
        set_target_properties(${target} PROPERTIES
            CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
            CUDA_RUNTIME_LIBRARY Shared
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>"
            "$<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CXX_COMPILER_ID:MSVC>,/UNDEBUG,-UNDEBUG>>")
        if(MSVC)
            target_compile_options(${target} PRIVATE
                "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
                "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
        endif()
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cuda-mfe-expert-store-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_mfe_expert_store_test.cpp)
    target_link_libraries(mfq-cuda-mfe-expert-store-test PRIVATE
        mfq-cuda-storage)
    target_compile_features(mfq-cuda-mfe-expert-store-test PRIVATE cxx_std_20)
    add_test(NAME mfq-cuda-mfe-expert-store-test
        COMMAND mfq-cuda-mfe-expert-store-test)
    endif()

    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-mxfp4-sq-test ${TYLOINFER_CUDA_ROOT}/tests/mfq_mxfp4_sq_test.cu)
    target_compile_definitions(mfq-mxfp4-sq-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-mxfp4-sq-test PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
    target_link_libraries(mfq-mxfp4-sq-test PRIVATE
        mfq-core mfq-cuda-core mfq-cuda-native-kernels)
    set_target_properties(mfq-mxfp4-sq-test PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    endif()

    add_executable(mfq-paged-kv-attention-bench
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_paged_kv_attention_bench.cu)
    target_compile_definitions(mfq-paged-kv-attention-bench PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-paged-kv-attention-bench PRIVATE
        ${TYLOINFER_CUDA_ROOT}/include)
    target_link_libraries(mfq-paged-kv-attention-bench PRIVATE
        mfq-cuda-core mfq-cuda-native-kernels)
    set_target_properties(mfq-paged-kv-attention-bench PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cuda-nint-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_nint_bench.cpp)
    target_compile_definitions(mfq-cuda-nint-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-cuda-nint-bench PRIVATE ${TYLOINFER_CUDA_ROOT}/include)
    target_link_libraries(mfq-cuda-nint-bench PRIVATE mfq-cuda-core mfq-cuda-native-kernels)
    target_compile_features(mfq-cuda-nint-bench PRIVATE cxx_std_20)
    endif()

    mfq_add_cuda_test(mfq-cuda-context-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_cuda_context_test.cu)
    add_executable(mfq-cuda-env-test ${TYLOINFER_CUDA_ROOT}/tests/native/env_test.cpp)
    target_include_directories(mfq-cuda-env-test PRIVATE ${TYLOINFER_CUDA_ROOT})
    target_link_libraries(mfq-cuda-env-test PRIVATE mfq-cuda-core)
    target_compile_features(mfq-cuda-env-test PRIVATE cxx_std_20)
    add_test(NAME mfq-cuda-env-test COMMAND mfq-cuda-env-test)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    mfq_add_cuda_test(mfq-cuda-shared-storage-test
        ${TYLOINFER_CUDA_ROOT}/tests/cuda_shared_storage_test.cu)
    target_include_directories(mfq-cuda-shared-storage-test PRIVATE
        ${TYLOINFER_REPOSITORY_ROOT} ${TYLOINFER_CUDA_ROOT} ${TYLOINFER_CUDA_ROOT}/native/tensor)
    endif()
    mfq_add_cuda_test(mfq-cuda-activation-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_cuda_activation_test.cu
        mfq-cuda-native-kernels)
    mfq_add_cuda_test(mfq-recurrent-mixer-ops-test
        ${TYLOINFER_CUDA_ROOT}/tests/recurrent_mixer_ops_test.cpp
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-recurrent-mixer-ops-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    if(MSVC)
        target_compile_options(mfq-recurrent-mixer-ops-test PRIVATE /EHsc)
    mfq_add_cuda_test(mfq-nint-q8-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_nint_q8_test.cu
        mfq-cuda-native-kernels)
    mfq_add_cuda_test(mfq-nint-rows-cuda-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_nint_rows_test.cu
        mfq-cuda-native-kernels mfq-core)
    target_compile_definitions(mfq-nint-rows-cuda-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_compile_definitions(mfq-nint-q8-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-packed-prefill-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_packed_prefill_test.cu
        mfq-cuda-native-kernels mfq-core)
    target_compile_definitions(mfq-packed-prefill-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-packed-prefill-test PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
    endif()
    mfq_add_cuda_test(mfq-paged-kv-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_paged_kv_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-paged-kv-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-deepseek-v41-hc-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_deepseek_v41_hc_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-deepseek-v41-hc-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-deepseek-v41-hc-test PRIVATE
        ${TYLOINFER_REPOSITORY_ROOT})
    mfq_add_cuda_test(mfq-native-tensor-cuda-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_native_tensor_cuda_test.cu)
    add_test(NAME mfq-contiguous-convert-test
        COMMAND mfq-native-tensor-cuda-test --contiguous-convert-check)
    set_tests_properties(mfq-contiguous-convert-test PROPERTIES SKIP_RETURN_CODE 77)
    mfq_add_cuda_test(mfq-cuda-storage-offset-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_cuda_storage_offset_test.cu
        mfq-cuda-native-kernels)
    target_compile_definitions(mfq-cuda-storage-offset-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    mfq_add_cuda_test(mfq-flash-next-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_flash_next_test.cu
        mfq-cuda-native-kernels mfq-json)
    target_compile_definitions(mfq-flash-next-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    mfq_add_cuda_test(mfq-nint-row-pipeline-test tests/nint_row_pipeline_test.cpp mfq-model-source mfq-cuda-native-kernels)
    target_compile_definitions(mfq-nint-row-pipeline-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    endif()
    target_include_directories(mfq-flash-next-test PRIVATE
        ${TYLOINFER_CUDA_ROOT}
        ${PROJECT_SOURCE_DIR}
        ${TYLOINFER_REPOSITORY_ROOT}
        ${TYLOINFER_CUDA_ROOT}/models
        ${TYLOINFER_CUDA_ROOT}/ops/common/include
        ${TYLOINFER_CUDA_ROOT}/ops/tq/include
        ${TYLOINFER_CUDA_ROOT}/ops/mfq/include
        ${TYLOINFER_CUDA_ROOT}/../../core)
endif()

set(TYLOINFER_NCCL_ROOT "" CACHE PATH
    "Optional NCCL installation root for tensor-parallel inference")
if(UNIX)
    find_library(TYLOINFER_NCCL_LIBRARY
        NAMES nccl libnccl.so.2
        HINTS
            "${TYLOINFER_NCCL_ROOT}/lib"
            "${TYLOINFER_NCCL_ROOT}/lib64"
            "$ENV{NCCL_ROOT}/lib"
            "$ENV{NCCL_ROOT}/lib64")
    find_path(TYLOINFER_NCCL_INCLUDE_DIR
        NAMES nccl.h
        HINTS
            "${TYLOINFER_NCCL_ROOT}/include"
            "$ENV{NCCL_ROOT}/include")
endif()

find_package(OpenMP QUIET)

function(mfq_configure_cuda_backend_target target)
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${TYLOINFER_REPOSITORY_ROOT}
        ${TYLOINFER_CUDA_ROOT}
        ${TYLOINFER_CUDA_ROOT}/include
        ${TYLOINFER_CUDA_ROOT}/native
        ${TYLOINFER_CUDA_ROOT}/native/tensor
        ${TYLOINFER_CUDA_ROOT}/ops/common/include
        ${TYLOINFER_CUDA_ROOT}/ops/tq/include
        ${TYLOINFER_CUDA_ROOT}/ops/mfq/include
        ${TYLOINFER_CUDA_KERNEL_ROOT}
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${target} PRIVATE
        mfq-formats
        mfq-cpu-ops
        mfq-json
        mfq-ggml-internal-headers
    )
    target_compile_features(${target} PRIVATE cxx_std_20)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
    if(TYLOINFER_NCCL_LIBRARY AND TYLOINFER_NCCL_INCLUDE_DIR)
        target_include_directories(${target} PRIVATE "${TYLOINFER_NCCL_INCLUDE_DIR}")
        target_link_libraries(${target} PRIVATE "${TYLOINFER_NCCL_LIBRARY}")
        target_compile_definitions(${target} PRIVATE TYLOINFER_HAVE_NCCL=1)
    endif()
endfunction()

# Shared Engine/model templates are instantiated only by backend runtime/loading
# units. Kernel/quant ops have no Engine, model-family, scheduler or transport link.
function(mfq_configure_cuda_runtime_target target)
    mfq_configure_cuda_backend_target(${target})
    target_include_directories(${target} PRIVATE
        ${TYLOINFER_CUDA_ROOT}/models
        ${TYLOINFER_CUDA_ROOT}/commands
        ${TYLOINFER_CUDA_ROOT}/engine
        ${PROJECT_SOURCE_DIR}/engine/models
    )
    target_link_libraries(${target} PRIVATE
        mfq-model-source
        mfq-quant-readers
        mfq-engine
        mfq-prefix-blocks
        mfq-qsa-kv-store
        mfq-cpu-execution
    )
endfunction()

set(TYLOINFER_CUDA_MODEL_SOURCES
    ${TYLOINFER_CUDA_ROOT}/models/common/causal_model_ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/common/full_block.cpp
    ${TYLOINFER_CUDA_ROOT}/models/common/axial_patch_vision_ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/common/grid_vision.cpp
    ${TYLOINFER_CUDA_ROOT}/models/deepseek_v4/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/deepseek_v41/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/deepseek_v41/session.cpp
    ${TYLOINFER_CUDA_ROOT}/models/deepseek_v41/engram.cpp
    ${TYLOINFER_CUDA_ROOT}/models/deepseek_v41/prefill.cpp
    ${TYLOINFER_CUDA_ROOT}/models/deepseek_v41/dspark.cpp
    ${TYLOINFER_CUDA_ROOT}/models/deepseek_v41/vision.cpp
    ${TYLOINFER_CUDA_ROOT}/models/qwen4_exp/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/state/kv_offload.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/state/kv_memory.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/state/kv_vmm.cpp
    ${TYLOINFER_CUDA_ROOT}/models/glm5_next/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/gemma4/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/glm_dsa/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/minicpmo45/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/nemotron_h/ops.cpp
    ${TYLOINFER_CUDA_ROOT}/models/qwen35/ops.cpp
)


set(TYLOINFER_CUDA_RUNTIME_OP_SOURCES
    ${TYLOINFER_CUDA_ROOT}/ops/ffn/mlp.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/ffn/residual.cpp
    ${TYLOINFER_CUDA_ROOT}/native/profile.cpp
    ${TYLOINFER_CUDA_ROOT}/native/collectives.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/attention/attention.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/attention/full.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/attention/kv.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/attention/linear.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/norm/rms.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/rope.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/format.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/moe_routes.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/gated_residual.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/tq/tq.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/tq/moe.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/mx.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/quant_linear.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/projection.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/common/mmq.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_RUNTIME_OP_SOURCES
    ${TYLOINFER_CUDA_ROOT}/ops/mfq/fp8_sq.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/mfq/mxfp4_sq.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/mfq/moe.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/mfq/mfe.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/mfq/nint.cpp
    ${TYLOINFER_CUDA_ROOT}/ops/mfq/vq.cpp
    )
endif()

add_library(mfq-cuda-ops OBJECT ${TYLOINFER_CUDA_RUNTIME_OP_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-ops)
target_link_libraries(mfq-cuda-ops PRIVATE mfq-quant-readers)
target_compile_definitions(mfq-cuda-ops PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TYLOINFER_CUDA_EXECUTION_SOURCES
    ${TYLOINFER_CUDA_ROOT}/engine/executor/execution.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/attention.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/ffn.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/experts.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/tq.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/tq_moe.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/decode_window.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/graph_registry.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_EXECUTION_SOURCES
        ${TYLOINFER_CUDA_ROOT}/engine/executor/moe_ffn_pipeline.cpp
        ${TYLOINFER_CUDA_ROOT}/engine/executor/moe_batch.cpp)
endif()
add_library(mfq-cuda-execution OBJECT ${TYLOINFER_CUDA_EXECUTION_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-execution)
target_link_libraries(mfq-cuda-execution PRIVATE mfq-cpu-execution)
target_compile_definitions(mfq-cuda-execution PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-execution PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TYLOINFER_CUDA_LOADING_SOURCES
    ${TYLOINFER_CUDA_ROOT}/engine/state/kv_cache.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/state/paged_kv.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/tq/load.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/tq/image.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/tq/cache.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/tq/routed.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/tq/rows.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/load.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/models/load.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_LOADING_SOURCES
    ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/expert_cache.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/load.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/prepare.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/quant_range_source.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/host_expert_cache.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/residency.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/weights/nint_row_pipeline.cpp
    )
else()
    list(APPEND TYLOINFER_CUDA_LOADING_SOURCES
        ${TYLOINFER_CUDA_ROOT}/engine/weights/moe/disabled.cpp)
endif()
add_library(mfq-cuda-loading OBJECT ${TYLOINFER_CUDA_LOADING_SOURCES})
mfq_configure_cuda_runtime_target(mfq-cuda-loading)
target_compile_definitions(mfq-cuda-loading PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-loading PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Concrete Engine/model bindings stay in this source group for both native and
# optional Torch builds. Extra targets do not reduce their header dependencies.
set(TYLOINFER_CUDA_INSTANTIATION_SOURCES
    ${TYLOINFER_CUDA_ROOT}/engine/executor/setup.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/models/components.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/decode_graph.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/engine.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/generation.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/state/session/text_cache.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/state/session/deepseek_v41_paged.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/mtp.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/executor/qwen35_mtp.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/batch/batch.cpp
    ${TYLOINFER_CUDA_ROOT}/engine/batch/qwen35.cpp
    ${TYLOINFER_CUDA_ROOT}/models/minicpmo45/components.cpp
    ${TYLOINFER_CUDA_ROOT}/commands/cli.cpp
    ${TYLOINFER_CUDA_ROOT}/commands/reference.cpp
    ${TYLOINFER_CUDA_ROOT}/commands/minicpmo45.cpp
    ${TYLOINFER_CUDA_MODEL_SOURCES}
)
add_library(mfq-cuda-runtime STATIC
    $<TARGET_OBJECTS:mfq-cuda-ops>
    $<TARGET_OBJECTS:mfq-cuda-execution>
    $<TARGET_OBJECTS:mfq-cuda-loading>
    ${TYLOINFER_CUDA_INSTANTIATION_SOURCES}
)
add_library(mfq::cuda-runtime ALIAS mfq-cuda-runtime)
mfq_configure_cuda_runtime_target(mfq-cuda-runtime)

if(BUILD_TESTING)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cuda-format-bench
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_format_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-format-bench)
    target_compile_definitions(mfq-cuda-format-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-format-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-e8-dense-probe
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/e8_dense_group_probe.cpp
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/e8_dense_group_probe.cu
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/d4_dense_group_probe.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-e8-dense-probe)
    target_include_directories(mfq-cuda-e8-dense-probe PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
    target_compile_definitions(mfq-cuda-e8-dense-probe PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-e8-dense-probe PRIVATE mfq-cuda-runtime)
    set_target_properties(mfq-cuda-e8-dense-probe PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}" CUDA_RUNTIME_LIBRARY Shared
        CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_executable(mfq-mfe-e8-local-bounds-test
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_e8_local_bounds_test.cpp)
    target_include_directories(mfq-mfe-e8-local-bounds-test PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
    target_link_libraries(mfq-mfe-e8-local-bounds-test PRIVATE CUDA::cudart)
    target_compile_features(mfq-mfe-e8-local-bounds-test PRIVATE cxx_std_20)
    add_test(NAME mfq-mfe-e8-local-bounds-test COMMAND mfq-mfe-e8-local-bounds-test)
    add_executable(mfq-mfe-e8-default-dispatch-test
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_e8_default_dispatch_test.cpp)
    target_include_directories(mfq-mfe-e8-default-dispatch-test PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
    target_link_libraries(mfq-mfe-e8-default-dispatch-test PRIVATE mfq-cuda-native-kernels)
    target_compile_features(mfq-mfe-e8-default-dispatch-test PRIVATE cxx_std_20)
    add_test(NAME mfq-mfe-e8-default-dispatch-test COMMAND mfq-mfe-e8-default-dispatch-test)
    add_executable(mfq-cuda-format-moe-bench
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_moe_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-format-moe-bench)
    target_compile_definitions(mfq-cuda-format-moe-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-format-moe-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-real-mfe-bench
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_mfe_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-real-mfe-bench)
    target_compile_definitions(mfq-cuda-real-mfe-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-real-mfe-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-mfe-nvq-bank-map-probe
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_nvq_bank_map_probe.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-mfe-nvq-bank-map-probe)
    target_compile_definitions(mfq-cuda-mfe-nvq-bank-map-probe PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-mfe-nvq-bank-map-probe PRIVATE mfq-cuda-runtime)
    endif()
    # Generated device snapshot keeps occupancy experiments out of the runtime.
    # Refresh with bench/cuda_format_comparison/generate_mfe_gu_register_probe.py.
    if(EXISTS "${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_gu_register_probe_body.inc")
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
        add_executable(mfq-cuda-mfe-gu-register-probe
            ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_gu_register_probe.cpp
            ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfe_gu_register_probe.cu)
        mfq_configure_cuda_runtime_target(mfq-cuda-mfe-gu-register-probe)
        target_include_directories(mfq-cuda-mfe-gu-register-probe PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
        target_compile_definitions(mfq-cuda-mfe-gu-register-probe PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
        target_link_libraries(mfq-cuda-mfe-gu-register-probe PRIVATE mfq-cuda-runtime)
        target_compile_options(mfq-cuda-mfe-gu-register-probe PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
        set_target_properties(mfq-cuda-mfe-gu-register-probe PROPERTIES
            CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}" CUDA_RUNTIME_LIBRARY Shared
            CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    endif()
    add_executable(mfq-cuda-nvq-byte-signs-test
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq_byte_signs_test.cu)
    target_include_directories(mfq-cuda-nvq-byte-signs-test PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
    target_link_libraries(mfq-cuda-nvq-byte-signs-test PRIVATE CUDA::cudart)
    set_target_properties(mfq-cuda-nvq-byte-signs-test PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}" CUDA_STANDARD 17 CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME mfq-cuda-nvq-byte-signs-test COMMAND mfq-cuda-nvq-byte-signs-test)
    add_executable(mfq-cuda-real-gr-bench
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_gr_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-real-gr-bench)
    target_compile_definitions(mfq-cuda-real-gr-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-real-gr-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-gr-batch-probe
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_gr_batch_probe.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-gr-batch-probe)
    target_compile_definitions(mfq-cuda-gr-batch-probe PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-gr-batch-probe PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-nint-dense-decode-probe
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/nint_dense_decode_probe.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-nint-dense-decode-probe)
    target_include_directories(mfq-cuda-nint-dense-decode-probe PRIVATE ${TYLOINFER_REPOSITORY_ROOT}/bench)
    target_compile_definitions(mfq-cuda-nint-dense-decode-probe PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-nint-dense-decode-probe PRIVATE mfq-cuda-runtime)
    endif()
    add_executable(mfq-cuda-shared-gate-bench
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_shared_gate_bench.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-shared-gate-bench)
    set_target_properties(mfq-cuda-shared-gate-bench PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(mfq-cuda-shared-gate-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-shared-gate-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-bf16-mmvf-bench
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/mfq_bf16_mmvf_bench.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-bf16-mmvf-bench)
    set_target_properties(mfq-cuda-bf16-mmvf-bench PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(mfq-cuda-bf16-mmvf-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-bf16-mmvf-bench PRIVATE mfq-cuda-runtime)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cuda-nvq-pair-indices-probe
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq_pair_indices_probe.cu)
    target_include_directories(mfq-cuda-nvq-pair-indices-probe PRIVATE ${TYLOINFER_REPOSITORY_ROOT})
    set_target_properties(mfq-cuda-nvq-pair-indices-probe PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
    add_executable(mfq-cuda-nvq-compressed-test
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq_compressed_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-nvq-compressed-test)
    target_compile_definitions(mfq-cuda-nvq-compressed-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-nvq-compressed-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-nvq-compressed-test COMMAND mfq-cuda-nvq-compressed-test)
    set_tests_properties(mfq-cuda-nvq-compressed-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-cuda-format-nint-test
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/nint_dense_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-format-nint-test)
    target_compile_definitions(mfq-cuda-format-nint-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-format-nint-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-format-nint-test COMMAND mfq-cuda-format-nint-test)
    set_tests_properties(mfq-cuda-format-nint-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-cuda-routed-nint-test
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/nint_routed_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-routed-nint-test)
    target_compile_definitions(mfq-cuda-routed-nint-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-routed-nint-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-routed-nint-test COMMAND mfq-cuda-routed-nint-test)
    set_tests_properties(mfq-cuda-routed-nint-test PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(mfq-cuda-routed-nvq1-test
        ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_format_comparison/nvq1_routed_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-routed-nvq1-test)
    target_compile_definitions(mfq-cuda-routed-nvq1-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-routed-nvq1-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-routed-nvq1-test COMMAND mfq-cuda-routed-nvq1-test)
    set_tests_properties(mfq-cuda-routed-nvq1-test PROPERTIES SKIP_RETURN_CODE 77)
    add_test(NAME mfq-cuda-routed-nint-occupancy-test COMMAND mfq-cuda-routed-nint-test)
    set_tests_properties(mfq-cuda-routed-nint-occupancy-test PROPERTIES
        SKIP_RETURN_CODE 77 ENVIRONMENT "TYLOINFER_NINT_ROUTE_BLOCKS=8;TYLOINFER_NINT_ROUTE_LANES=32")
    endif()

    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cuda-nvq-moe-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_nvq_moe_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-nvq-moe-bench)
    target_compile_definitions(mfq-cuda-nvq-moe-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-nvq-moe-bench PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cuda-mfe-ffn-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_mfe_ffn_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-mfe-ffn-bench)
    target_compile_definitions(mfq-cuda-mfe-ffn-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-mfe-ffn-bench PRIVATE mfq-cuda-runtime)
    endif()
    add_executable(mfq-cuda-wire-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_moe_wire_bench.cu)
    mfq_configure_cuda_runtime_target(mfq-cuda-wire-bench)
    target_compile_definitions(mfq-cuda-wire-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-wire-bench PRIVATE mfq-cuda-runtime)
    target_compile_options(mfq-cuda-wire-bench PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-O3>")
    set_target_properties(mfq-cuda-wire-bench PROPERTIES CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}" CUDA_STANDARD 20)

    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cuda-mfe-released-test ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_mfe_released_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-mfe-released-test)
    target_compile_definitions(mfq-cuda-mfe-released-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-mfe-released-test PRIVATE mfq-cuda-runtime)
    add_executable(mfq-sq-linear-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_sq_linear_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-sq-linear-test)
    target_compile_definitions(mfq-sq-linear-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-sq-linear-test PRIVATE mfq-cuda-runtime)
    add_executable(mfq-cpu-quant-linear-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_cpu_quant_linear_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cpu-quant-linear-test)
    target_compile_definitions(mfq-cpu-quant-linear-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cpu-quant-linear-test PRIVATE mfq-cuda-runtime)
    add_executable(mfq-quant-range-cache-test
        ${TYLOINFER_CUDA_ROOT}/tests/mfq_quant_range_cache_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-quant-range-cache-test)
    target_compile_definitions(mfq-quant-range-cache-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-quant-range-cache-test PRIVATE mfq-cuda-runtime)
    endif()
    add_executable(mfq-mxfp8-layout-test ${TYLOINFER_CUDA_ROOT}/tests/mxfp8_layout_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-mxfp8-layout-test)
    target_compile_definitions(mfq-mxfp8-layout-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-mxfp8-layout-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-mxfp8-layout-test COMMAND mfq-mxfp8-layout-test)
    add_executable(mfq-nope-latent-moe-test ${TYLOINFER_CUDA_ROOT}/tests/nope_latent_moe_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-nope-latent-moe-test)
    target_compile_definitions(mfq-nope-latent-moe-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-nope-latent-moe-test PRIVATE
        mfq-cuda-runtime mfq-cuda-native-kernels mfq-ggml)
    add_test(NAME mfq-nope-latent-moe-test COMMAND mfq-nope-latent-moe-test)
    set_tests_properties(mfq-nope-latent-moe-test PROPERTIES SKIP_RETURN_CODE 77)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-moe-ffn-pipeline-test ${TYLOINFER_CUDA_ROOT}/tests/moe_ffn_pipeline_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-moe-ffn-pipeline-test)
    target_compile_definitions(mfq-moe-ffn-pipeline-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-moe-ffn-pipeline-test PRIVATE mfq-cuda-runtime)
    endif()
    add_executable(mfq-cuda-dma-batch-test ${TYLOINFER_CUDA_ROOT}/tests/dma_copy_batch_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-dma-batch-test)
    target_compile_definitions(mfq-cuda-dma-batch-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-dma-batch-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-cuda-dma-batch-test COMMAND mfq-cuda-dma-batch-test)
    add_executable(mfq-cuda-warp-multi-sum-test ${TYLOINFER_CUDA_ROOT}/tests/warp_multi_sum_test.cu)
    target_include_directories(mfq-cuda-warp-multi-sum-test PRIVATE ${TYLOINFER_CUDA_KERNEL_ROOT})
    target_link_libraries(mfq-cuda-warp-multi-sum-test PRIVATE CUDA::cudart)
    set_target_properties(mfq-cuda-warp-multi-sum-test PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared)
    add_test(NAME mfq-cuda-warp-multi-sum-test COMMAND mfq-cuda-warp-multi-sum-test)
    add_executable(mfq-gdn-decode-window-test ${TYLOINFER_CUDA_ROOT}/tests/gdn_decode_window_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-gdn-decode-window-test)
    target_compile_definitions(mfq-gdn-decode-window-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-gdn-decode-window-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-gdn-decode-window-test COMMAND mfq-gdn-decode-window-test)
    # Every QSA/turboquant variant reuses the legacy window-attention fixture.
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-window-attention-test ${TYLOINFER_CUDA_ROOT}/tests/decode_window_attention_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-window-attention-test)
    target_compile_definitions(mfq-window-attention-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-window-attention-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-window-attention-test COMMAND mfq-window-attention-test)
    add_test(NAME mfq-cuda-qsa-live-scores-test COMMAND mfq-window-attention-test --qsa-live-score-check)
    add_test(NAME mfq-cuda-qsa-pool-write-test COMMAND mfq-window-attention-test --qsa-pool-write-check)
    add_test(NAME mfq-cuda-qsa-history-prime-test COMMAND mfq-window-attention-test --qsa-history-prime-check)
    add_test(NAME mfq-cuda-qsa-kv-offload-test COMMAND mfq-window-attention-test --qsa-kv-offload-check)
    foreach(bits 0 2.5 4)
        add_test(NAME mfq-cuda-qsa-stream-graph-${bits}-test COMMAND mfq-window-attention-test --qsa-stream-graph-check)
        set_tests_properties(mfq-cuda-qsa-stream-graph-${bits}-test PROPERTIES
            ENVIRONMENT "TYLOINFER_KV_TURBOQUANT_BITS=${bits};TYLOINFER_KV_STREAM_GRAPHS=1;TYLOINFER_MEMORY_ARCHITECTURE=discrete")
    endforeach()
    add_test(NAME mfq-cuda-turboquant-kv-test COMMAND mfq-window-attention-test --turboquant-kv-check)
    add_test(NAME mfq-cuda-turboquant-qsa-test COMMAND mfq-window-attention-test --turboquant-qsa-check)
    set_tests_properties(mfq-cuda-turboquant-qsa-test PROPERTIES ENVIRONMENT "TYLOINFER_KV_TURBOQUANT_BITS=3.5")
    endif()
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    mfq_configure_cuda_runtime_target(mfq-nint-row-pipeline-test)
    target_link_libraries(mfq-nint-row-pipeline-test PRIVATE mfq-cuda-runtime)
    add_executable(mfq-host-expert-cache-test
        ${TYLOINFER_CUDA_ROOT}/tests/moe_host_expert_cache_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-host-expert-cache-test)
    target_compile_definitions(mfq-host-expert-cache-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-host-expert-cache-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-host-expert-cache-test COMMAND mfq-host-expert-cache-test)
    add_executable(mfq-ssd-expert-streaming-test ${TYLOINFER_CUDA_ROOT}/tests/ssd_expert_streaming_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-ssd-expert-streaming-test)
    target_compile_definitions(mfq-ssd-expert-streaming-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-ssd-expert-streaming-test PRIVATE mfq-cuda-runtime)
    endif()
    if(MSVC)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
        target_compile_options(mfq-ssd-expert-streaming-test PRIVATE /EHsc)
        target_compile_options(mfq-host-expert-cache-test PRIVATE /EHsc)
    endif()
        # Cancellation checks must unwind CUDA stream guards before recovery.
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
        target_compile_options(mfq-nint-row-pipeline-test PRIVATE /EHsc)
    endif()
    add_executable(mfq-float-projection-test ${TYLOINFER_CUDA_ROOT}/tests/float_projection_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-float-projection-test)
    target_compile_definitions(mfq-float-projection-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-float-projection-test PRIVATE mfq-cuda-runtime)
    add_test(NAME mfq-float-projection-test COMMAND mfq-float-projection-test)
    add_test(NAME mfq-gr-prefill-blas-test COMMAND mfq-float-projection-test --prefill-blas-check)
    endif()

    add_executable(mfq-deepseek-v41-ced-test ${TYLOINFER_CUDA_ROOT}/tests/deepseek_v41_ced_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-ced-test)
    target_link_libraries(mfq-deepseek-v41-ced-test PRIVATE mfq-cuda-runtime)
    target_compile_definitions(mfq-deepseek-v41-ced-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    add_test(NAME mfq-deepseek-v41-ced-test COMMAND mfq-deepseek-v41-ced-test)
    set_tests_properties(mfq-deepseek-v41-ced-test PROPERTIES SKIP_RETURN_CODE 77)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-deepseek-v41-engram-test ${TYLOINFER_CUDA_ROOT}/tests/deepseek_v41_engram_test.cpp)
    target_link_libraries(mfq-deepseek-v41-engram-test PRIVATE mfq-cuda-runtime)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-engram-test)
    target_compile_definitions(mfq-deepseek-v41-engram-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    add_test(NAME mfq-deepseek-v41-engram-test COMMAND mfq-deepseek-v41-engram-test)
    set_tests_properties(mfq-deepseek-v41-engram-test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    add_executable(mfq-deepseek-v41-vision-test ${TYLOINFER_CUDA_ROOT}/tests/deepseek_v41_vision_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-vision-test)
    target_link_libraries(mfq-deepseek-v41-vision-test PRIVATE mfq-cuda-runtime)
    target_compile_definitions(mfq-deepseek-v41-vision-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    add_test(NAME mfq-deepseek-v41-vision-test COMMAND mfq-deepseek-v41-vision-test)
    set_tests_properties(mfq-deepseek-v41-vision-test PROPERTIES SKIP_RETURN_CODE 77)
    if(MSVC)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
        target_compile_options(mfq-deepseek-v41-engram-test PRIVATE /EHsc)
    endif()
        target_compile_options(mfq-deepseek-v41-vision-test PRIVATE /EHsc)
        target_compile_options(mfq-deepseek-v41-ced-test PRIVATE /EHsc)
    endif()
    add_executable(mfq-deepseek-v41-indexer-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_deepseek_v41_indexer_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-indexer-bench)
    target_compile_definitions(mfq-deepseek-v41-indexer-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-deepseek-v41-indexer-bench PRIVATE mfq-cuda-runtime)
    if(MSVC)
        target_compile_options(mfq-deepseek-v41-indexer-bench PRIVATE /EHsc)
    endif()
    add_executable(mfq-deepseek-v41-hc-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cuda_deepseek_v41_hc_bench.cpp)
    mfq_configure_cuda_runtime_target(mfq-deepseek-v41-hc-bench)
    target_compile_definitions(mfq-deepseek-v41-hc-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-deepseek-v41-hc-bench PRIVATE mfq-cuda-runtime)
    if(MSVC)
        target_compile_options(mfq-deepseek-v41-hc-bench PRIVATE /EHsc)
    endif()
    add_executable(mfq-cuda-deepseek-v41-axial-vision-test
        ${TYLOINFER_CUDA_ROOT}/tests/deepseek_v41_axial_vision_gpu_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-deepseek-v41-axial-vision-test)
    target_compile_definitions(mfq-cuda-deepseek-v41-axial-vision-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-deepseek-v41-axial-vision-test PRIVATE mfq-cuda-runtime)
    target_compile_features(mfq-cuda-deepseek-v41-axial-vision-test PRIVATE cxx_std_20)
    add_test(NAME mfq-cuda-deepseek-v41-axial-vision-test
        COMMAND mfq-cuda-deepseek-v41-axial-vision-test)
    set_tests_properties(mfq-cuda-deepseek-v41-axial-vision-test PROPERTIES SKIP_RETURN_CODE 77)
endif()
target_link_libraries(mfq-cuda-runtime PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cublas
    mfq-cuda-core
    mfq-cuda-native-kernels
    mfq-cuda-storage
)
target_compile_definitions(mfq-cuda-runtime PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1
    NOMINMAX
)
if(MSVC)
    foreach(target mfq-cuda-ops mfq-cuda-execution mfq-cuda-loading mfq-cuda-runtime)
        target_compile_options(${target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>"
            "$<$<COMPILE_LANGUAGE:CXX>:/EHsc>"
            "$<$<COMPILE_LANGUAGE:CXX>:/bigobj>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
        )
    endforeach()
endif()
set_target_properties(mfq-cuda-runtime PROPERTIES
    CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
    CUDA_RUNTIME_LIBRARY Shared
)

if(BUILD_TESTING)
    add_executable(mfq-cuda-weight-loader-test
        ${TYLOINFER_CUDA_ROOT}/tests/weight_loader_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-weight-loader-test)
    target_compile_definitions(mfq-cuda-weight-loader-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-weight-loader-test PRIVATE mfq-cuda-runtime)
    if(MSVC)
        target_compile_options(mfq-cuda-weight-loader-test PRIVATE /EHsc)
    endif()
    add_test(NAME mfq-cuda-weight-loader-test COMMAND mfq-cuda-weight-loader-test)
    set_tests_properties(mfq-cuda-weight-loader-test PROPERTIES SKIP_RETURN_CODE 77)
    target_link_libraries(mfq-flash-next-test PRIVATE
        mfq-cuda-runtime)
    add_executable(mfq-cuda-execution-scope-test
        ${TYLOINFER_CUDA_ROOT}/tests/cuda_execution_scope_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-execution-scope-test)
    target_compile_definitions(mfq-cuda-execution-scope-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-execution-scope-test PRIVATE
        mfq-cuda-runtime)
    add_test(NAME mfq-cuda-execution-scope-test
        COMMAND mfq-cuda-execution-scope-test)
    if(MSVC)
        target_compile_options(mfq-cuda-execution-scope-test PRIVATE /EHsc)
    endif()
    # Runtime-state fixtures below exercise legacy MFQ operators.
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cuda-runtime-state-test
        ${TYLOINFER_CUDA_ROOT}/tests/cuda_runtime_state_test.cpp)
    mfq_configure_cuda_runtime_target(mfq-cuda-runtime-state-test)
    target_compile_definitions(mfq-cuda-runtime-state-test PRIVATE
        TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
    target_link_libraries(mfq-cuda-runtime-state-test PRIVATE mfq-cuda-runtime)
    if(MSVC)
        # Cancellation checks must unwind CUDA stream guards before recovery.
        target_compile_options(mfq-cuda-runtime-state-test PRIVATE /EHsc)
    endif()
    add_test(NAME mfq-cuda-runtime-state-test COMMAND mfq-cuda-runtime-state-test)
    add_test(NAME tyloinfer-cuda-tq-parallel-test COMMAND mfq-cuda-runtime-state-test --tq-tp-check)
    add_test(NAME mfq-cuda-mfe-prepare-test COMMAND mfq-cuda-runtime-state-test --mfe-prepare-check)
    add_test(NAME mfq-cuda-mfe-transfer-test COMMAND mfq-cuda-runtime-state-test --mfe-transfer-check)
    add_test(NAME mfq-cuda-mapped-copy-test COMMAND mfq-cuda-runtime-state-test --mapped-copy-check)
    add_test(NAME mfq-cuda-sampling-keys-test COMMAND mfq-cuda-runtime-state-test --sampling-check)
    add_test(NAME mfq-cuda-transfer-copy-test COMMAND mfq-cuda-runtime-state-test --transfer-copy-check)
    set_tests_properties(mfq-cuda-runtime-state-test PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(tyloinfer-cuda-tq-parallel-test PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(mfq-cuda-mfe-prepare-test mfq-cuda-mfe-transfer-test mfq-cuda-mapped-copy-test PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(mfq-cuda-sampling-keys-test PROPERTIES SKIP_RETURN_CODE 77)
    set_tests_properties(mfq-cuda-transfer-copy-test PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endif()

add_library(mfq-cuda-eval STATIC ${TYLOINFER_CUDA_ROOT}/eval/kl.cpp)
mfq_configure_cuda_runtime_target(mfq-cuda-eval)
target_include_directories(mfq-cuda-eval PRIVATE ${TYLOINFER_CUDA_ROOT}/storage)
target_compile_definitions(mfq-cuda-eval PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-cuda-eval PUBLIC mfq-cuda-runtime)

add_library(mfq-cuda-diagnostics STATIC
    ${TYLOINFER_CUDA_ROOT}/diagnostics/attention_checks.cpp
    ${TYLOINFER_CUDA_ROOT}/diagnostics/backend_checks.cpp
    ${TYLOINFER_CUDA_ROOT}/diagnostics/linear_checks.cpp
    ${TYLOINFER_CUDA_ROOT}/diagnostics/moe_checks.cpp
    ${TYLOINFER_CUDA_ROOT}/diagnostics/runtime_checks.cpp
    ${TYLOINFER_CUDA_ROOT}/diagnostics/session_checks.cpp
)
mfq_configure_cuda_runtime_target(mfq-cuda-diagnostics)
target_include_directories(mfq-cuda-diagnostics PRIVATE ${TYLOINFER_CUDA_ROOT}/storage)
target_compile_definitions(mfq-cuda-diagnostics PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-cuda-diagnostics PUBLIC mfq-cuda-runtime)
if(MSVC)
    foreach(target mfq-cuda-diagnostics mfq-cuda-eval)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc /bigobj)
    endforeach()
endif()

if(TARGET mfq-runtime-communication)
    add_executable(mfq-runtime
        ${TYLOINFER_CUDA_ROOT}/apps/runtime_main.cpp
        ${TYLOINFER_CUDA_ROOT}/commands/runtime.cpp
    )
    target_link_libraries(mfq-runtime PRIVATE
        mfq-cuda-runtime mfq-runtime-communication)
    target_include_directories(mfq-runtime PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${TYLOINFER_CUDA_ROOT}
        ${TYLOINFER_CUDA_ROOT}/include
        ${TYLOINFER_CUDA_ROOT}/engine
    )
    target_compile_features(mfq-runtime PRIVATE cxx_std_20)
    set_target_properties(mfq-runtime PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    )
endif()

add_executable(mfq-diagnostics
    ${TYLOINFER_CUDA_ROOT}/apps/diagnostics_main.cpp
    ${TYLOINFER_CUDA_ROOT}/commands/diagnostics.cpp
)
mfq_configure_cuda_runtime_target(mfq-diagnostics)
target_include_directories(mfq-diagnostics PRIVATE ${TYLOINFER_CUDA_ROOT}/storage)
target_compile_definitions(mfq-diagnostics PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-diagnostics PRIVATE mfq-cuda-diagnostics)
target_compile_features(mfq-diagnostics PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(mfq-diagnostics PRIVATE /utf-8 /EHsc /bigobj)
endif()
set_target_properties(mfq-diagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-eval
    ${TYLOINFER_CUDA_ROOT}/apps/eval_main.cpp
    ${TYLOINFER_CUDA_ROOT}/commands/eval.cpp
)
mfq_configure_cuda_runtime_target(mfq-eval)
target_include_directories(mfq-eval PRIVATE ${TYLOINFER_CUDA_ROOT}/storage)
target_compile_definitions(mfq-eval PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-eval PRIVATE
    mfq-cuda-eval mfq-cuda-runtime mfq-cuda-native-kernels mfq-ggml)
target_compile_features(mfq-eval PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(mfq-eval PRIVATE /utf-8 /EHsc /bigobj)
endif()
set_target_properties(mfq-eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(BUILD_TESTING)
    add_test(
        NAME mfq-deepseek-v41-runtime-check
        COMMAND mfq-diagnostics --check-deepseek-v41)
    add_test(NAME mfq-diagnostics-help COMMAND mfq-diagnostics --help)
    add_test(NAME mfq-eval-help COMMAND mfq-eval --help)
    add_test(NAME mfq-eval-kl-help COMMAND mfq-eval kl --help)
    add_test(NAME mfq-runtime-cli-options COMMAND mfq-runtime --transport stdio --continuous-batching 2 --help)
    add_test(NAME mfq-diagnostics-cli-options COMMAND mfq-diagnostics --ids 1,2 --gen 3 --help)
    add_test(NAME mfq-eval-cli-options COMMAND mfq-eval kl --kl-base x --kl-evaluator optimized --help)
    add_test(NAME mfq-runtime-rejects-check
        COMMAND mfq-runtime --check-mfq-container)
    add_test(NAME mfq-runtime-rejects-kl
        COMMAND mfq-runtime --kl-base reference.bin)
    add_test(NAME mfq-runtime-rejects-bad-number
        COMMAND mfq-runtime --port 3x --help)
    add_test(NAME mfq-runtime-rejects-missing-value
        COMMAND mfq-runtime --model --help)
    add_test(NAME mfq-runtime-rejects-token-input
        COMMAND mfq-runtime --ids 1,2 --help)
    add_test(NAME mfq-diagnostics-rejects-bad-ids
        COMMAND mfq-diagnostics --ids 1,x --help)
    add_test(NAME mfq-diagnostics-rejects-kl
        COMMAND mfq-diagnostics --kl-base reference.bin)
    add_test(NAME mfq-diagnostics-rejects-missing-value
        COMMAND mfq-diagnostics --check-linear --check-backend-argmax)
    add_test(NAME mfq-eval-rejects-check
        COMMAND mfq-eval kl --check-mfq-container)
    add_test(NAME mfq-eval-rejects-bad-number
        COMMAND mfq-eval kl --kl-chunks 1x --help)
    set_tests_properties(
        mfq-runtime-rejects-check mfq-runtime-rejects-kl
        mfq-runtime-rejects-bad-number mfq-runtime-rejects-missing-value
        mfq-runtime-rejects-token-input mfq-diagnostics-rejects-bad-ids
        mfq-diagnostics-rejects-kl mfq-diagnostics-rejects-missing-value
        mfq-eval-rejects-check mfq-eval-rejects-bad-number
        PROPERTIES WILL_FAIL TRUE)
endif()

option(TYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME
    "Build the optional LibTorch CUDA runtime used only for A/B validation"
    OFF)
if(TYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME)
    if(NOT TARGET mfq-runtime-communication)
        message(FATAL_ERROR
            "TYLOINFER_BUILD_TORCH_REFERENCE_RUNTIME requires TYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON")
    endif()
    find_package(Python REQUIRED COMPONENTS Interpreter Development)
    if(NOT TARGET CUDA::nvToolsExt)
        find_library(NVTOOLSEXT_LIBRARY
            NAMES nvToolsExt nvToolsExt64_1
            PATHS "${CUDAToolkit_LIBRARY_DIR}" "$ENV{CUDA_PATH}/lib/x64"
            NO_DEFAULT_PATH)
        if(NVTOOLSEXT_LIBRARY)
            add_library(CUDA::nvToolsExt UNKNOWN IMPORTED)
            set_target_properties(CUDA::nvToolsExt PROPERTIES
                IMPORTED_LOCATION "${NVTOOLSEXT_LIBRARY}")
        else()
            add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
        endif()
    endif()
    find_package(Torch REQUIRED)
    add_executable(mfq-runtime-torch
        ${TYLOINFER_CUDA_ROOT}/apps/runtime_main.cpp
        ${TYLOINFER_CUDA_ROOT}/commands/runtime.cpp
        ${TYLOINFER_CUDA_RUNTIME_OP_SOURCES}
        ${TYLOINFER_CUDA_EXECUTION_SOURCES}
        ${TYLOINFER_CUDA_LOADING_SOURCES}
        ${TYLOINFER_CUDA_INSTANTIATION_SOURCES}
        ${TYLOINFER_CUDA_ROOT}/apps/ggml_cuda_compat.cu
        ${TYLOINFER_CUDA_KERNEL_SOURCES}
    )
    mfq_configure_cuda_runtime_target(mfq-runtime-torch)
    target_include_directories(mfq-runtime-torch PRIVATE ${Python_INCLUDE_DIRS})
    target_link_libraries(mfq-runtime-torch PRIVATE
        mfq-runtime-communication
        ${TORCH_LIBRARIES}
        Python::Python
        CUDA::cuda_driver
        CUDA::cudart
        CUDA::cublas
        mfq-cuda-storage
    )
    target_compile_definitions(mfq-runtime-torch PRIVATE NOMINMAX)
    target_compile_options(mfq-runtime-torch PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>"
    )
    set_target_properties(mfq-runtime-torch PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

if(NOT WIN32 AND TARGET mfq-runtime)
    set(TYLOINFER_SYSTEM_RUNTIME_DIR
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "directory containing the host libstdc++ runtime")
    set_property(TARGET mfq-runtime PROPERTY
        BUILD_RPATH "${TYLOINFER_SYSTEM_RUNTIME_DIR}")
endif()

set_source_files_properties(
    ${TYLOINFER_CUDA_KERNEL_ROOT}/attention_mma.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/deepseek_v4_attention.cu
    PROPERTIES COMPILE_OPTIONS "--use_fast_math;--extended-lambda"
)
set_source_files_properties(
    ${TYLOINFER_CUDA_KERNEL_ROOT}/deepseek_v4_hc.cu
    PROPERTIES COMPILE_OPTIONS "--extended-lambda"
)

set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_shared.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nint_tokens.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_shared.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_shared_short.cuh")

set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_cached_reduce.cu PROPERTIES
    OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_device.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_cached_reduce.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nvq_compact_dot.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_nvq_parity_read.cuh;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn.h;${TYLOINFER_CUDA_KERNEL_ROOT}/mfe_ffn_launch.h")
