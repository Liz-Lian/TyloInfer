if(TARGET mfq-runtime-communication)
    add_executable(mfq-runtime
        ${PROJECT_SOURCE_DIR}/apps/runtime_main.cpp
        ${PROJECT_SOURCE_DIR}/commands/runtime.cpp
    )
    target_link_libraries(mfq-runtime PRIVATE
        mfq-cuda-runtime mfq-runtime-communication)
    target_include_directories(mfq-runtime PRIVATE
        ${PROJECT_SOURCE_DIR}
    )
    target_compile_features(mfq-runtime PRIVATE cxx_std_20)
    set_target_properties(mfq-runtime PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
    )
endif()

add_executable(mfq-diagnostics
    ${PROJECT_SOURCE_DIR}/apps/diagnostics_main.cpp
    ${PROJECT_SOURCE_DIR}/commands/diagnostics.cpp
)
mfq_configure_cuda_runtime_target(mfq-diagnostics)
target_compile_definitions(mfq-diagnostics PRIVATE TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
target_link_libraries(mfq-diagnostics PRIVATE mfq-cuda-diagnostics)
target_compile_features(mfq-diagnostics PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(mfq-diagnostics PRIVATE /utf-8 /EHsc /bigobj)
endif()
set_target_properties(mfq-diagnostics PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
)

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
        ${PROJECT_SOURCE_DIR}/apps/runtime_main.cpp
        ${PROJECT_SOURCE_DIR}/commands/runtime.cpp
        ${TYLOINFER_CUDA_RUNTIME_OP_SOURCES}
        ${TYLOINFER_CUDA_EXECUTION_SOURCES}
        ${TYLOINFER_CUDA_LOADING_SOURCES}
        ${TYLOINFER_CUDA_INSTANTIATION_SOURCES}
        ${TYLOINFER_MODELS_CUDA_SOURCES}
        ${PROJECT_SOURCE_DIR}/engine/backends/cuda/native/ggml.cu
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
