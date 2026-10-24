cmake_minimum_required(VERSION 3.26)
project(mfq_cpp_runtime LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TYLOINFER_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

option(TYLOINFER_ENABLE_MFQ_COMPAT "Include legacy MFQ inference formats" OFF)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    add_compile_definitions(TYLOINFER_ENABLE_MFQ_COMPAT=1)
else()
    add_compile_definitions(TYLOINFER_ENABLE_MFQ_COMPAT=0)
endif()

option(
    TYLOINFER_BUILD_METAL_RUNTIME
    "Build the native MLX/Metal runtime on Apple Silicon"
    ${APPLE}
)

include(CheckLanguage)
check_language(CUDA)
if(NOT APPLE AND CMAKE_CUDA_COMPILER AND NOT TYLOINFER_BUILD_HIP_RUNTIME)
    set(_TYLOINFER_BUILD_CUDA_RUNTIME_DEFAULT ON)
else()
    set(_TYLOINFER_BUILD_CUDA_RUNTIME_DEFAULT OFF)
endif()
option(
    TYLOINFER_BUILD_CUDA_RUNTIME
    "Build the native CUDA runtime (requires the CUDA toolkit)"
    ${_TYLOINFER_BUILD_CUDA_RUNTIME_DEFAULT}
)
unset(_TYLOINFER_BUILD_CUDA_RUNTIME_DEFAULT)

check_language(HIP)
if(NOT APPLE AND CMAKE_HIP_COMPILER AND NOT TYLOINFER_BUILD_CUDA_RUNTIME)
    set(_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT ON)
else()
    set(_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT OFF)
endif()
option(
    TYLOINFER_BUILD_HIP_RUNTIME
    "Build the native HIP runtime on AMD ROCm hosts"
    ${_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT}
)
unset(_TYLOINFER_BUILD_HIP_RUNTIME_DEFAULT)

if(TYLOINFER_BUILD_CUDA_RUNTIME AND TYLOINFER_BUILD_HIP_RUNTIME)
    message(FATAL_ERROR
        "Select either TYLOINFER_BUILD_CUDA_RUNTIME or TYLOINFER_BUILD_HIP_RUNTIME; use separate build directories for both backends")
endif()

if(APPLE)
    set(_TYLOINFER_BUILD_RUNTIME_COMMUNICATION_DEFAULT OFF)
else()
    set(_TYLOINFER_BUILD_RUNTIME_COMMUNICATION_DEFAULT ON)
endif()
option(
    TYLOINFER_BUILD_RUNTIME_COMMUNICATION
    "Build private HTTP/stdio communication modules for native runtimes"
    ${_TYLOINFER_BUILD_RUNTIME_COMMUNICATION_DEFAULT}
)
unset(_TYLOINFER_BUILD_RUNTIME_COMMUNICATION_DEFAULT)

if(TYLOINFER_BUILD_METAL_RUNTIME AND NOT APPLE)
    message(FATAL_ERROR "TYLOINFER_BUILD_METAL_RUNTIME requires macOS")
endif()

# Common code is split by responsibility. JSON is defined first because the
# canonical model graph and several public policy headers consume it.
add_subdirectory(components/json)
add_subdirectory(core)
add_subdirectory(backends/cpu)
add_subdirectory(models)

if(TYLOINFER_BUILD_RUNTIME_COMMUNICATION OR TYLOINFER_BUILD_METAL_RUNTIME)
    add_subdirectory(components/ggml)
    add_subdirectory(components/tokenizer)
endif()

if(TYLOINFER_BUILD_RUNTIME_COMMUNICATION)
    add_subdirectory(components/http)
endif()

add_subdirectory(engine)
add_subdirectory(scheduler)

if(TYLOINFER_BUILD_RUNTIME_COMMUNICATION)
    add_subdirectory(transport)
endif()

if(TYLOINFER_BUILD_METAL_RUNTIME)
    enable_language(OBJCXX)
    # Keep the established build output path and executable locations.
    add_subdirectory(backends/metal metal)
endif()

if(TYLOINFER_BUILD_CUDA_RUNTIME)
    add_subdirectory(backends/cuda cuda)
endif()

if(TYLOINFER_BUILD_HIP_RUNTIME)
    add_subdirectory(backends/hip hip)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
