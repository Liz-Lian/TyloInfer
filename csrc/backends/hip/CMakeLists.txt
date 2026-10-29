# Native HIP backend (AMD ROCm).
#
# Mirrors backends/cuda: the shared kernels live in the Python package
# (TYLOINFER_HIP_KERNEL_ROOT), tensor/storage/engine/model code in this tree.
# Kernel sources were produced by hipify and are fixed up incrementally;
# tensor-core tile paths that rely on NVIDIA-only MMA intrinsics are
# excluded from the source lists until they gain MFMA or direct-path
# implementations.

if(NOT TARGET mfq-runtime-communication)
    message(FATAL_ERROR
        "The HIP runtime requires TYLOINFER_BUILD_RUNTIME_COMMUNICATION=ON")
endif()

enable_language(HIP)

set(TYLOINFER_HIP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(TYLOINFER_HIP_KERNEL_ROOT "${TYLOINFER_REPOSITORY_ROOT}/tyloinfer/kernels/hip")

set(TYLOINFER_HIP_ARCHITECTURES "gfx942" CACHE STRING
    "HIP architectures compiled into MFQ HIP targets")

find_package(hip REQUIRED CONFIG
    HINTS "${ROCM_PATH}" "$ENV{ROCM_PATH}" /opt/rocm)
find_package(hipblas REQUIRED CONFIG
    HINTS "${ROCM_PATH}" "$ENV{ROCM_PATH}" /opt/rocm)
find_package(Threads REQUIRED)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_library(mfq-hip-storage STATIC
    ${TYLOINFER_HIP_ROOT}/storage/mfe_expert_store.cpp
)
add_library(mfq::hip-storage ALIAS mfq-hip-storage)
target_include_directories(mfq-hip-storage PUBLIC
    ${TYLOINFER_HIP_ROOT}/storage
)
target_link_libraries(mfq-hip-storage PUBLIC Threads::Threads)
target_compile_definitions(mfq-hip-storage PUBLIC __HIP_PLATFORM_AMD__=1)
target_compile_features(mfq-hip-storage PUBLIC cxx_std_20)
set_target_properties(mfq-hip-storage PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

else()
    add_library(mfq-hip-storage INTERFACE)
    add_library(mfq::hip-storage ALIAS mfq-hip-storage)
endif()

add_library(mfq-hip-core STATIC
    ${TYLOINFER_HIP_ROOT}/src/mfq_cuda_context.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor.cpp
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_f32_matmul_lt.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor_blas.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor_creation.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor_indexing.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor_ops.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor_reduction.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor_signal.hip
    ${TYLOINFER_HIP_ROOT}/src/mfq_native_tensor_sort.hip
)
add_library(mfq::hip-core ALIAS mfq-hip-core)
target_include_directories(mfq-hip-core PUBLIC
    ${TYLOINFER_HIP_ROOT}/include
    ${TYLOINFER_HIP_KERNEL_ROOT}
)
target_link_libraries(mfq-hip-core PUBLIC hip::host hipblas mfq-host-parallel)
find_package(hipblaslt QUIET CONFIG HINTS "${ROCM_PATH}" "$ENV{ROCM_PATH}" /opt/rocm)
if(TARGET roc::hipblaslt)
    target_link_libraries(mfq-hip-core PRIVATE roc::hipblaslt)
    target_compile_definitions(mfq-hip-core PRIVATE TYLOINFER_HAVE_HIPBLASLT=1)
    # ROCm wheels copy development libraries without their kernel data. Load
    # the runtime copy so hipBLASLt can find its adjacent Tensile kernels.
    get_filename_component(TYLOINFER_ROCM_WHEEL_BASE "${hipblaslt_DIR}/../../../.." ABSOLUTE)
    set(TYLOINFER_ROCM_WHEEL_LIB "${TYLOINFER_ROCM_WHEEL_BASE}/_rocm_sdk_libraries/lib")
    if(EXISTS "${TYLOINFER_ROCM_WHEEL_LIB}/hipblaslt/library")
        # The wheel's LD_LIBRARY_PATH also contains the incomplete devel copy.
        target_link_options(mfq-hip-core INTERFACE
            "LINKER:--disable-new-dtags,-rpath,${TYLOINFER_ROCM_WHEEL_LIB}")
    endif()
endif()
target_compile_features(mfq-hip-core PUBLIC cxx_std_20)
set_target_properties(mfq-hip-core PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES "${TYLOINFER_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

set(TYLOINFER_HIP_KERNEL_SOURCES
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_gr.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_gr_dense.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_shared.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_moe.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_projection.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_prefill.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_q8.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/tq_input_gate.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/acc.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/activation.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/attention.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/attention_mma.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/embedding.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/selected_attention.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/qwen4_exp.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/glm5_next.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/gated_delta_net.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/gdn_stateful_decode.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/gdn_decode_window.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/kv_cache.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/paged_kv.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/moe.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/moe_route.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/mx_matmul.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/mxfp8_grouped.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/norm.hip
    # The signed Hadamard transform is also used by the sparse indexer.
    ${TYLOINFER_HIP_KERNEL_ROOT}/nepq.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/rope.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/sampling.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/ssm_conv.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/selective_state_scan.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/deepseek_v41.hip
    ${TYLOINFER_HIP_ROOT}/kernels/kv_quantization.hip
    ${TYLOINFER_HIP_ROOT}/kernels/kv_hot_cache.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/deepseek_v4_hc.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/deepseek_v4_attention.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/deepseek_v4_fp4.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/deepseek_v41_index.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/deepseek_v41_block_sort.hip
    ${TYLOINFER_HIP_KERNEL_ROOT}/glm_dsa.hip
)

if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_HIP_KERNEL_SOURCES
        ${TYLOINFER_HIP_KERNEL_ROOT}/mfe_ffn.hip
        ${TYLOINFER_HIP_KERNEL_ROOT}/mfe_ffn_launch.cpp
        ${TYLOINFER_HIP_KERNEL_ROOT}/fp8_sq.hip
        ${TYLOINFER_HIP_KERNEL_ROOT}/mxfp4_sq.hip
        ${TYLOINFER_HIP_KERNEL_ROOT}/nepq_residual.hip
        ${TYLOINFER_HIP_KERNEL_ROOT}/nint_matmul.hip
        ${TYLOINFER_HIP_KERNEL_ROOT}/nvq_matmul.hip
    )
endif()

add_library(mfq-hip-native-kernels STATIC
    ${TYLOINFER_HIP_KERNEL_SOURCES}
    ${TYLOINFER_HIP_ROOT}/runtime/moe_pipeline.hip
    ${TYLOINFER_HIP_ROOT}/apps/ggml_cuda_compat.hip
)
add_library(mfq::hip-native-kernels ALIAS mfq-hip-native-kernels)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    target_sources(mfq-hip-native-kernels PRIVATE
        ${TYLOINFER_HIP_ROOT}/kernels/float_projection.hip
        ${TYLOINFER_HIP_ROOT}/kernels/gated_residual_fused.hip
    )
endif()
target_compile_definitions(mfq-hip-native-kernels PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1
)
target_include_directories(mfq-hip-native-kernels PRIVATE
    ${TYLOINFER_REPOSITORY_ROOT}
    ${PROJECT_SOURCE_DIR}
    ${TYLOINFER_HIP_ROOT}/include
    ${TYLOINFER_HIP_KERNEL_ROOT}
    ${TYLOINFER_REPOSITORY_ROOT}/csrc/components/ggml/src/ggml-hip/vendors
)
target_link_libraries(mfq-hip-native-kernels
    PUBLIC mfq-hip-core
    PRIVATE mfq-core mfq-ggml-internal-headers
)
# nvq_matmul's heavily templated tile kernels take ~25 minutes at -O3 for
# gfx942 with negligible runtime difference on the memory-bound decode path;
# compile it at -O2 (the later flag wins over the config -O3).
set_source_files_properties(
    ${TYLOINFER_HIP_KERNEL_ROOT}/nvq_matmul.hip
    PROPERTIES COMPILE_OPTIONS "-O2"
)
set_target_properties(mfq-hip-native-kernels PROPERTIES
    HIP_STANDARD 20
    HIP_STANDARD_REQUIRED ON
    HIP_ARCHITECTURES "${TYLOINFER_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

find_package(OpenMP QUIET)

function(mfq_configure_hip_backend_target target)
    target_include_directories(${target} PRIVATE
        ${TYLOINFER_REPOSITORY_ROOT}
        ${PROJECT_SOURCE_DIR}
        ${TYLOINFER_HIP_ROOT}
        ${TYLOINFER_HIP_ROOT}/include
        ${TYLOINFER_HIP_ROOT}/ops/common/include
        ${TYLOINFER_HIP_ROOT}/ops/tq/include
        ${TYLOINFER_HIP_ROOT}/ops/mfq/include
        ${TYLOINFER_HIP_ROOT}/storage
        ${TYLOINFER_HIP_KERNEL_ROOT}
    )
    target_link_libraries(${target} PRIVATE
        mfq-core
        mfq-cpu-ops
        mfq-json
        mfq-ggml-internal-headers
    )
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(${target} PRIVATE __HIP_PLATFORM_AMD__=1)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

function(mfq_configure_hip_runtime_target target)
    mfq_configure_hip_backend_target(${target})
    target_include_directories(${target} PRIVATE
        ${TYLOINFER_HIP_ROOT}/models
        ${TYLOINFER_HIP_ROOT}/commands
        ${TYLOINFER_HIP_ROOT}/engine
    )
    target_link_libraries(${target} PRIVATE mfq-models mfq-engine)
endfunction()

set(TYLOINFER_HIP_MODEL_SOURCES
    ${TYLOINFER_HIP_ROOT}/models/deepseek_v4/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/nemotron_h/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/deepseek_v41/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/deepseek_v41/engram.cpp
    ${TYLOINFER_HIP_ROOT}/models/deepseek_v41/prefill.cpp
    ${TYLOINFER_HIP_ROOT}/models/deepseek_v41/vision.cpp
    ${TYLOINFER_HIP_ROOT}/models/deepseek_v41/session.cpp
    ${TYLOINFER_HIP_ROOT}/models/deepseek_v41/dspark.cpp
    ${TYLOINFER_HIP_ROOT}/models/qwen4_exp/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/glm5_next/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/gemma4/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/glm_dsa/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/minicpmo45/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/minicpmo45/components.cpp
    ${TYLOINFER_HIP_ROOT}/models/qwen35/ops.cpp
    ${TYLOINFER_HIP_ROOT}/models/qwen35/batch_state.cpp
)

set(TYLOINFER_HIP_TRANSFORMER_SOURCES
    ${TYLOINFER_HIP_ROOT}/core/causal_model.cpp
    ${TYLOINFER_HIP_ROOT}/core/decode_graph.cpp
    ${TYLOINFER_HIP_ROOT}/core/ffn.cpp
    ${TYLOINFER_HIP_ROOT}/core/full_block.cpp
    ${TYLOINFER_HIP_ROOT}/core/axial_patch_vision.cpp
    ${TYLOINFER_HIP_ROOT}/core/kv_cache.cpp
    ${TYLOINFER_HIP_ROOT}/core/rope.cpp
)

set(TYLOINFER_HIP_RUNTIME_OP_SOURCES
    ${TYLOINFER_HIP_ROOT}/ops/common/cuda_execution.cpp
    ${TYLOINFER_HIP_ROOT}/ops/common/format.cpp
    ${TYLOINFER_HIP_ROOT}/ops/common/moe_routes.cpp
    ${TYLOINFER_HIP_ROOT}/ops/common/gated_residual.cpp
    ${TYLOINFER_HIP_ROOT}/ops/common/mx.cpp
    ${TYLOINFER_HIP_ROOT}/ops/common/quant_linear.cpp
    ${TYLOINFER_HIP_ROOT}/ops/tq/tq.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_HIP_RUNTIME_OP_SOURCES
    ${TYLOINFER_HIP_ROOT}/ops/mfq/fp8_sq.cpp
    ${TYLOINFER_HIP_ROOT}/ops/mfq/mxfp4_sq.cpp
    ${TYLOINFER_HIP_ROOT}/ops/mfq/moe.cpp
    ${TYLOINFER_HIP_ROOT}/ops/mfq/mfe_ffn.cpp
    ${TYLOINFER_HIP_ROOT}/ops/mfq/nint.cpp
    ${TYLOINFER_HIP_ROOT}/ops/mfq/vq.cpp
    )
endif()

add_library(mfq-hip-ops OBJECT ${TYLOINFER_HIP_RUNTIME_OP_SOURCES})
mfq_configure_hip_backend_target(mfq-hip-ops)
target_compile_definitions(mfq-hip-ops PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-hip-ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TYLOINFER_HIP_LOADING_SOURCES
    ${TYLOINFER_HIP_ROOT}/storage/load_options.cpp
    ${TYLOINFER_HIP_ROOT}/storage/model_loader.cpp
    ${TYLOINFER_HIP_ROOT}/runtime/graph_registry.cpp
    ${TYLOINFER_HIP_ROOT}/runtime/mfq_cache_api.cpp
    ${TYLOINFER_HIP_ROOT}/runtime/decode_window.cpp
    ${TYLOINFER_HIP_ROOT}/storage/weight_loader.cpp
    ${TYLOINFER_HIP_ROOT}/storage/transformer_loader.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_HIP_LOADING_SOURCES
    ${TYLOINFER_HIP_ROOT}/storage/moe_expert_cache.cpp
    ${TYLOINFER_HIP_ROOT}/storage/moe_ffn_pipeline.cpp
    ${TYLOINFER_HIP_ROOT}/storage/moe_quant_range_source.cpp
    ${TYLOINFER_HIP_ROOT}/runtime/moe_host_expert_cache.cpp
    ${TYLOINFER_HIP_ROOT}/runtime/moe_residency.cpp
    ${TYLOINFER_HIP_ROOT}/runtime/nint_row_pipeline.cpp
    )
endif()
add_library(mfq-hip-loading OBJECT ${TYLOINFER_HIP_LOADING_SOURCES})
mfq_configure_hip_runtime_target(mfq-hip-loading)
target_compile_definitions(mfq-hip-loading PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-hip-loading PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(TYLOINFER_HIP_INSTANTIATION_SOURCES
    ${TYLOINFER_HIP_ROOT}/engine/cuda_engine.cpp
    ${TYLOINFER_HIP_ROOT}/engine/generation.cpp
    ${TYLOINFER_HIP_ROOT}/storage/text_session_cache.cpp
    ${TYLOINFER_HIP_ROOT}/storage/qwen4_paged_session.cpp
    ${TYLOINFER_HIP_ROOT}/storage/deepseek_v41_paged_session.cpp
    ${TYLOINFER_HIP_ROOT}/engine/mtp.cpp
    ${TYLOINFER_HIP_ROOT}/engine/cuda_batching.cpp
    ${TYLOINFER_HIP_ROOT}/commands/cli.cpp
    ${TYLOINFER_HIP_ROOT}/commands/minicpmo45.cpp
    ${TYLOINFER_HIP_MODEL_SOURCES}
    ${TYLOINFER_HIP_TRANSFORMER_SOURCES}
)
add_library(mfq-hip-runtime STATIC
    $<TARGET_OBJECTS:mfq-hip-ops>
    $<TARGET_OBJECTS:mfq-hip-loading>
    ${TYLOINFER_HIP_INSTANTIATION_SOURCES}
)
add_library(mfq::hip-runtime ALIAS mfq-hip-runtime)
mfq_configure_hip_runtime_target(mfq-hip-runtime)
target_link_libraries(mfq-hip-runtime PRIVATE
    hip::host
    hipblas
    mfq-hip-core
    mfq-hip-native-kernels
    mfq-hip-storage
)
target_compile_definitions(mfq-hip-runtime PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1
    NOMINMAX
)
set_target_properties(mfq-hip-runtime PROPERTIES
    HIP_ARCHITECTURES "${TYLOINFER_HIP_ARCHITECTURES}"
    POSITION_INDEPENDENT_CODE ON
)

add_library(mfq-hip-eval STATIC ${TYLOINFER_HIP_ROOT}/eval/kl.cpp)
mfq_configure_hip_runtime_target(mfq-hip-eval)
target_include_directories(mfq-hip-eval PRIVATE ${TYLOINFER_HIP_ROOT}/storage)
target_compile_definitions(mfq-hip-eval PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-hip-eval PUBLIC mfq-hip-runtime)

add_library(mfq-hip-diagnostics STATIC
    ${TYLOINFER_HIP_ROOT}/diagnostics/attention_checks.cpp
    ${TYLOINFER_HIP_ROOT}/diagnostics/backend_checks.cpp
    ${TYLOINFER_HIP_ROOT}/diagnostics/linear_checks.cpp
    ${TYLOINFER_HIP_ROOT}/diagnostics/moe_checks.cpp
    ${TYLOINFER_HIP_ROOT}/diagnostics/runtime_checks.cpp
    ${TYLOINFER_HIP_ROOT}/diagnostics/session_checks.cpp
)
mfq_configure_hip_runtime_target(mfq-hip-diagnostics)
target_include_directories(mfq-hip-diagnostics PRIVATE ${TYLOINFER_HIP_ROOT}/storage)
target_compile_definitions(mfq-hip-diagnostics PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-hip-diagnostics PUBLIC mfq-hip-runtime)

add_executable(mfq-runtime
    ${TYLOINFER_HIP_ROOT}/apps/runtime_main.cpp
    ${TYLOINFER_HIP_ROOT}/commands/runtime.cpp
)
target_link_libraries(mfq-runtime PRIVATE
    mfq-hip-runtime mfq-runtime-communication)
target_include_directories(mfq-runtime PRIVATE
    ${TYLOINFER_HIP_ROOT}
    ${TYLOINFER_HIP_ROOT}/include
    ${TYLOINFER_HIP_ROOT}/engine
)
target_compile_features(mfq-runtime PRIVATE cxx_std_20)
set_target_properties(mfq-runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-diagnostics
    ${TYLOINFER_HIP_ROOT}/apps/diagnostics_main.cpp
    ${TYLOINFER_HIP_ROOT}/commands/diagnostics.cpp
)
mfq_configure_hip_runtime_target(mfq-diagnostics)
target_include_directories(mfq-diagnostics PRIVATE ${TYLOINFER_HIP_ROOT}/storage)
target_compile_definitions(mfq-diagnostics PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-diagnostics PRIVATE mfq-hip-diagnostics)
target_compile_features(mfq-diagnostics PRIVATE cxx_std_20)
set_target_properties(mfq-diagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

add_executable(mfq-eval
    ${TYLOINFER_HIP_ROOT}/apps/eval_main.cpp
    ${TYLOINFER_HIP_ROOT}/commands/eval.cpp
)
mfq_configure_hip_runtime_target(mfq-eval)
target_include_directories(mfq-eval PRIVATE ${TYLOINFER_HIP_ROOT}/storage)
target_compile_definitions(mfq-eval PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-eval PRIVATE
    mfq-hip-eval mfq-hip-runtime mfq-hip-native-kernels mfq-ggml)
target_compile_features(mfq-eval PRIVATE cxx_std_20)
set_target_properties(mfq-eval PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

if(BUILD_TESTING)
    add_executable(mfq-hip-decode-test ${TYLOINFER_HIP_ROOT}/tests/decode_hotspots_test.cpp)
    mfq_configure_hip_backend_target(mfq-hip-decode-test)
    target_compile_definitions(mfq-hip-decode-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_compile_features(mfq-hip-decode-test PRIVATE cxx_std_20)
    target_link_libraries(mfq-hip-decode-test PRIVATE
        mfq-hip-runtime mfq-hip-native-kernels mfq-hip-core mfq-core)
    add_test(NAME mfq-hip-decode-test COMMAND mfq-hip-decode-test)
    set_tests_properties(mfq-hip-decode-test PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(mfq-hip-tq-test ${TYLOINFER_HIP_ROOT}/tests/tq_projection_test.cpp)
    mfq_configure_hip_backend_target(mfq-hip-tq-test)
    target_compile_definitions(mfq-hip-tq-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_compile_features(mfq-hip-tq-test PRIVATE cxx_std_20)
    target_link_libraries(mfq-hip-tq-test PRIVATE
        mfq-hip-runtime mfq-hip-native-kernels mfq-hip-core mfq-core)
    add_test(NAME mfq-hip-tq-test COMMAND mfq-hip-tq-test
        ${TYLOINFER_REPOSITORY_ROOT}/tests/fixtures/hip-tq-native-subgroup.bin
        ${TYLOINFER_REPOSITORY_ROOT}/tests/fixtures/hip-tq-native-short-groups.bin
        ${TYLOINFER_REPOSITORY_ROOT}/tests/fixtures/hip-tq-native-columns.bin)
    set_tests_properties(mfq-hip-tq-test PROPERTIES SKIP_RETURN_CODE 77)

    # The CUDA backend's test sources compile directly as HIP translation
    # units: hipcc accepts <<<>>> launches, and tests/cuda_compat maps the
    # cuda_* runtime API names onto hip.
    set(TYLOINFER_HIP_TEST_INCLUDES
        ${TYLOINFER_REPOSITORY_ROOT}/csrc/core
        ${TYLOINFER_HIP_ROOT}/tests/cuda_compat
        ${TYLOINFER_REPOSITORY_ROOT}/csrc/components/ggml/src/ggml-hip/vendors
        ${TYLOINFER_REPOSITORY_ROOT}
        ${PROJECT_SOURCE_DIR}
        ${TYLOINFER_HIP_ROOT}
        ${TYLOINFER_HIP_ROOT}/include
        ${TYLOINFER_HIP_ROOT}/ops/common/include
        ${TYLOINFER_HIP_ROOT}/ops/tq/include
        ${TYLOINFER_HIP_ROOT}/ops/mfq/include
        ${TYLOINFER_HIP_ROOT}/storage
        ${TYLOINFER_HIP_KERNEL_ROOT}
    )
    function(mfq_add_hip_test target source)
        add_executable(${target} ${source})
        set_source_files_properties(${source} PROPERTIES LANGUAGE HIP)
        target_include_directories(${target} PRIVATE ${TYLOINFER_HIP_TEST_INCLUDES})
        target_compile_definitions(${target} PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
        target_compile_features(${target} PRIVATE cxx_std_20)
        set_target_properties(${target} PROPERTIES
            HIP_STANDARD 20
            HIP_STANDARD_REQUIRED ON
            HIP_ARCHITECTURES "${TYLOINFER_HIP_ARCHITECTURES}"
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/hip-tests"
        )
        add_test(NAME ${target} COMMAND ${target})
        set_tests_properties(${target} PROPERTIES SKIP_RETURN_CODE 77)
    endfunction()

    mfq_add_hip_test(mfq-hip-context-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_cuda_context_test.cu)
    target_link_libraries(mfq-hip-context-test PRIVATE mfq-hip-core)

    mfq_add_hip_test(mfq-hip-wave-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_hip_wave_test.hip)
    target_link_libraries(mfq-hip-wave-test PRIVATE hip::host)

    mfq_add_hip_test(mfq-hip-activation-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_cuda_activation_test.cu)
    target_link_libraries(mfq-hip-activation-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels)

    mfq_add_hip_test(mfq-hip-attention-mma-test
        ${TYLOINFER_HIP_ROOT}/tests/attention_mma_test.cu)
    target_link_libraries(mfq-hip-attention-mma-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-core mfq-ggml)

    mfq_add_hip_test(mfq-hip-attention-decode-test
        ${TYLOINFER_HIP_ROOT}/tests/attention_decode_test.cu)
    target_link_libraries(mfq-hip-attention-decode-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels)

    mfq_add_hip_test(mfq-hip-native-tensor-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_native_tensor_cuda_test.cu)
    target_link_libraries(mfq-hip-native-tensor-test PRIVATE mfq-hip-core)

    if(TYLOINFER_ENABLE_MFQ_COMPAT)
    mfq_add_hip_test(mfq-hip-nint-q8-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_nint_q8_test.cu)
    target_link_libraries(mfq-hip-nint-q8-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-hip-storage
        mfq-hip-runtime mfq-runtime-communication mfq-model-graph)
    add_executable(mfq-hip-nint-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/hip_nint_bench.cpp)
    target_include_directories(mfq-hip-nint-bench PRIVATE ${TYLOINFER_HIP_TEST_INCLUDES})
    target_compile_definitions(mfq-hip-nint-bench PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_link_libraries(mfq-hip-nint-bench PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-hip-storage
        mfq-hip-runtime mfq-runtime-communication mfq-model-graph)
    target_compile_features(mfq-hip-nint-bench PRIVATE cxx_std_20)
    mfq_add_hip_test(mfq-hip-nint-rows-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_nint_rows_test.cu)
    target_link_libraries(mfq-hip-nint-rows-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-core)
    mfq_add_hip_test(mfq-hip-packed-prefill-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_packed_prefill_test.cu)
    target_link_libraries(mfq-hip-packed-prefill-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels mfq-core)
    endif()

    mfq_add_hip_test(mfq-hip-paged-kv-test
        ${TYLOINFER_HIP_ROOT}/tests/mfq_paged_kv_test.cu)
    target_link_libraries(mfq-hip-paged-kv-test PRIVATE
        mfq-hip-core mfq-hip-native-kernels)
endif()

set(TYLOINFER_SYSTEM_RUNTIME_DIR
    "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
    "directory containing the host libstdc++ runtime")
set_property(TARGET mfq-runtime PROPERTY
    BUILD_RPATH "${TYLOINFER_SYSTEM_RUNTIME_DIR}")

if(BUILD_TESTING)
    # This bridge contains host code only. Match the runtime's C++ compiler
    # for the model/template ABI; GPU work is supplied by the linked kernels.
    set_source_files_properties(${TYLOINFER_HIP_ROOT}/tests/qwen38_flash_next_test.hip
        PROPERTIES LANGUAGE CXX)
    add_executable(mfq-hip-flash-next-test ${TYLOINFER_HIP_ROOT}/tests/qwen38_flash_next_test.hip
        ${TYLOINFER_HIP_ROOT}/tests/quant_kv_check.cpp
        ${TYLOINFER_HIP_ROOT}/tests/hot_kv_check.cpp
        ${TYLOINFER_HIP_ROOT}/tests/qwen4_graph_reuse_check.cpp
        ${TYLOINFER_HIP_ROOT}/tests/qwen4_session_check.cpp
        ${TYLOINFER_HIP_ROOT}/tests/qwen4_predictor_graph_check.cpp)
    target_compile_definitions(mfq-hip-flash-next-test PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1)
    target_include_directories(mfq-hip-flash-next-test PRIVATE ${TYLOINFER_HIP_TEST_INCLUDES}
        ${PROJECT_SOURCE_DIR} ${TYLOINFER_HIP_ROOT}/models ${TYLOINFER_HIP_ROOT}/runtime)
    target_compile_features(mfq-hip-flash-next-test PRIVATE cxx_std_20)
    set_target_properties(mfq-hip-flash-next-test PROPERTIES HIP_STANDARD 20
        HIP_ARCHITECTURES "${TYLOINFER_HIP_ARCHITECTURES}" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/hip-tests")
    target_link_libraries(mfq-hip-flash-next-test PRIVATE mfq-hip-runtime mfq-hip-native-kernels
        mfq-hip-core mfq-core mfq-json mfq-model-graph mfq-runtime-communication)
    add_test(NAME mfq-hip-flash-next COMMAND mfq-hip-flash-next-test)
    set_tests_properties(mfq-hip-flash-next PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 60)
endif()

if(BUILD_TESTING)
    mfq_add_hip_test(mfq-hip-recurrent-mixer-test ${TYLOINFER_HIP_ROOT}/tests/recurrent_mixer_ops_test.hip)
    target_link_libraries(mfq-hip-recurrent-mixer-test PRIVATE mfq-hip-core mfq-hip-native-kernels)
endif()
