if(TYLOINFER_ENABLE_MFQ_COMPAT)
add_library(mfq-cuda-storage STATIC
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/expert_store.cpp
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

# Remaining runtime bindings share Engine and weight dependencies. Kernel/quant
# ops have no Engine, model-family, scheduler or transport link.
function(mfq_configure_cuda_runtime_target target)
    mfq_configure_cuda_backend_target(${target})
    target_link_libraries(${target} PRIVATE
        mfq-model-source
        mfq-quant-readers
        mfq-engine
        mfq-prefix-blocks
        mfq-qsa-kv-store
        mfq-cpu-execution
    )
endfunction()

set(TYLOINFER_CUDA_LOADING_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/state/kv_cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/state/recurrent.cpp
    ${PROJECT_SOURCE_DIR}/engine/state/paged_kv.cpp
    ${PROJECT_SOURCE_DIR}/engine/state/kv_offload.cpp
    ${PROJECT_SOURCE_DIR}/engine/state/kv_memory.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/image.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/routed.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/rows.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/loader.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/routed_cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/workspace.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/group.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/format.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/quant.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/memory.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/experts.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/load.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    list(APPEND TYLOINFER_CUDA_LOADING_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/expert_cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/prepare.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/quant_range_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/host_expert_cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/moe/residency.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/nint_row_pipeline.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/nint.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/vq.cpp
    )
else()
    list(APPEND TYLOINFER_CUDA_LOADING_SOURCES
        ${PROJECT_SOURCE_DIR}/engine/weights/moe/disabled.cpp)
endif()
add_library(mfq-cuda-loading OBJECT ${TYLOINFER_CUDA_LOADING_SOURCES})
mfq_configure_cuda_runtime_target(mfq-cuda-loading)
target_compile_definitions(mfq-cuda-loading PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-loading PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(${PROJECT_SOURCE_DIR}/engine/models ${CMAKE_CURRENT_BINARY_DIR}/models)

# Remaining legacy bindings are shared by native and optional Torch builds.
# Concrete Engine selection is compiled from engine/load.cpp.
set(TYLOINFER_CUDA_INSTANTIATION_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/generate.cpp
    ${PROJECT_SOURCE_DIR}/engine/mtp.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/components.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/session/cache.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/session/paged.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/session_codec.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen4_exp/session_codec.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/minicpmo45/components.cpp
    ${PROJECT_SOURCE_DIR}/commands/cli.cpp
    ${PROJECT_SOURCE_DIR}/commands/reference.cpp
    ${PROJECT_SOURCE_DIR}/commands/minicpmo45.cpp
    ${TYLOINFER_CUDA_MODEL_SOURCES}
)
add_library(mfq-cuda-runtime STATIC
    $<TARGET_OBJECTS:mfq-cuda-ops>
    $<TARGET_OBJECTS:mfq-cuda-execution>
    $<TARGET_OBJECTS:mfq-cuda-loading>
    $<TARGET_OBJECTS:mfq-cuda-models>
    ${TYLOINFER_CUDA_INSTANTIATION_SOURCES}
)
add_library(mfq::cuda-runtime ALIAS mfq-cuda-runtime)
mfq_configure_cuda_runtime_target(mfq-cuda-runtime)
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
    foreach(target mfq-cuda-ops mfq-cuda-execution mfq-cuda-loading mfq-cuda-models mfq-cuda-runtime)
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
    POSITION_INDEPENDENT_CODE ON
)

include(${PROJECT_SOURCE_DIR}/diagnostics/cuda.cmake)
include(${PROJECT_SOURCE_DIR}/apps/cuda.cmake)
