# Native CUDA backend. Compute, execution and loading targets do not require
# the optional service transport; only the runtime service below does.

set(TYLOINFER_CUDA_ROOT "${PROJECT_SOURCE_DIR}/engine/backends/cuda")
# Single source of the shared CUDA kernels; the Python extension builds the same files.
set(TYLOINFER_CUDA_KERNEL_ROOT "${PROJECT_SOURCE_DIR}/engine/backends/cuda/kernels")

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

add_library(mfq-cuda-core STATIC
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/env.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/device.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/stream.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/memory.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/graph.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/context.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/tensor.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/tensor.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/blas.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/creation.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/indexing.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/ops.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/reduction.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/signal.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor/sort.cu
)
add_library(mfq::cuda-core ALIAS mfq-cuda-core)
# The native context/tensor units only need their own headers plus the CUDA SDK;
# the kernel declarations stay out of the interface.
target_include_directories(mfq-cuda-core PUBLIC
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_include_directories(mfq-cuda-core PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(mfq-cuda-core PUBLIC CUDA::cudart CUDA::cublas mfq-host-parallel)
target_link_libraries(mfq-cuda-core PUBLIC mfq-tensor)
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
    ${TYLOINFER_CUDA_KERNEL_ROOT}/tq_ops.cu
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
    ${TYLOINFER_CUDA_KERNEL_ROOT}/float_projection.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/kv_quantization.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/kv_hot_cache.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/shared_gate_fused.cu
    ${TYLOINFER_CUDA_KERNEL_ROOT}/moe/pipeline.cu
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/ggml.cu
)
add_library(mfq::cuda-native-kernels ALIAS mfq-cuda-native-kernels)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    target_sources(mfq-cuda-native-kernels PRIVATE
        ${TYLOINFER_CUDA_KERNEL_ROOT}/gr_prefill_blas.cu
        ${TYLOINFER_CUDA_KERNEL_ROOT}/gated_residual_fused.cu
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
    ${TYLOINFER_CUDA_KERNEL_ROOT}/moe/pipeline.cu
    APPEND PROPERTY OBJECT_DEPENDS "${TYLOINFER_CUDA_KERNEL_ROOT}/shared_storage.cuh")
set_source_files_properties(${TYLOINFER_CUDA_KERNEL_ROOT}/gated_residual_fused.cu PROPERTIES
    COMPILE_OPTIONS "--maxrregcount=80")
if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set_property(SOURCE ${TYLOINFER_CUDA_KERNEL_ROOT}/gated_residual_fused.cu APPEND
        PROPERTY COMPILE_OPTIONS "--split-compile=0")
endif()
target_compile_definitions(mfq-cuda-native-kernels PRIVATE
    TYLOINFER_NATIVE_CUDA=1
)
target_include_directories(mfq-cuda-native-kernels PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${TYLOINFER_REPOSITORY_ROOT}
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor
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
        ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native
        ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/tensor
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

set(TYLOINFER_CUDA_OP_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/tensor.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/ffn.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/mlp.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/dense.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/residual.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/profile.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/vmm.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/route_stats.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/collectives.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/attention/attention.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/attention/kv.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/attention/linear.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/norm/rms.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/rope/rope.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/routes.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/gated.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/tq.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/tq.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/mx.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/quant_linear.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/projection.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/expert_projection.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/group.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/mmq.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_OP_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/fp8_sq.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/mxfp4_sq.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/moe.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/ffn/mfe.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/nint.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/ops/projection/vq.cpp
    )
endif()

add_library(mfq-cuda-ops OBJECT ${TYLOINFER_CUDA_OP_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-ops)
target_link_libraries(mfq-cuda-ops PRIVATE mfq-quant-readers)
target_compile_definitions(mfq-cuda-ops PRIVATE
    TYLOINFER_NATIVE_CUDA=1 NOMINMAX)
set_target_properties(mfq-cuda-ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TYLOINFER_CUDA_EXECUTION_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/execution.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/options.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/config.cpp
    ${PROJECT_SOURCE_DIR}/engine/budget.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/executor.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/decode_graph.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/batch_graph.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/tq.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/tq_moe.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/decode_window.cpp
    ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/graph_registry.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_EXECUTION_SOURCES
        ${PROJECT_SOURCE_DIR}/engine/backends/cuda/executor/moe_batch.cpp)
endif()
add_library(mfq-cuda-execution OBJECT ${TYLOINFER_CUDA_EXECUTION_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-execution)
target_link_libraries(mfq-cuda-execution PRIVATE mfq-cpu-execution mfq-engine)
target_compile_definitions(mfq-cuda-execution PRIVATE
    TYLOINFER_NATIVE_CUDA=1 NOMINMAX)
set_target_properties(mfq-cuda-execution PROPERTIES POSITION_INDEPENDENT_CODE ON)

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

include(${PROJECT_SOURCE_DIR}/engine/cuda.cmake)
