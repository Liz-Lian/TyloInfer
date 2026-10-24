# Thread primitives do not pull in expert scheduling or model loading.
find_package(Threads REQUIRED)
add_library(mfq-host-parallel STATIC
    native/host_parallel.cpp
    native/host_parallel.h
    native/host_copy.cpp
    native/host_copy.h
    native/host_memory.cpp
    native/host_memory.h
    native/topology.cpp
    native/topology.h
    native/cpu_affinity_win.h
)
target_include_directories(mfq-host-parallel PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_compile_features(mfq-host-parallel PUBLIC cxx_std_17)
target_link_libraries(mfq-host-parallel PUBLIC Threads::Threads)
set_target_properties(mfq-host-parallel PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(mfq::host-parallel ALIAS mfq-host-parallel)

if(MSVC)
    target_compile_options(mfq-host-parallel PRIVATE /EHsc)
endif()

add_library(mfq-cpu-execution STATIC
    execution/cpu_expert_pool.cpp
    execution/cpu_expert_pool.h
    execution/cpu_measurement.cpp
    execution/cpu_measurement.h
)
target_include_directories(mfq-cpu-execution PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_compile_features(mfq-cpu-execution PUBLIC cxx_std_17)
target_link_libraries(mfq-cpu-execution PUBLIC mfq-host-parallel)
set_target_properties(mfq-cpu-execution PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Compressed host math; shared by CUDA offload and other host consumers.
add_library(mfq-cpu-ops STATIC
    ops/common/host_memory_copy.cpp
    ops/common/host_memory_copy.h
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    target_sources(mfq-cpu-ops PRIVATE
        ops/mfq/packed_nint.cpp ops/mfq/quant_dot.cpp ops/mfq/nvq_group.cpp)
endif()
target_include_directories(mfq-cpu-ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/common
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/mfq)
target_include_directories(mfq-cpu-ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_link_libraries(mfq-cpu-ops PUBLIC mfq-host-parallel)
target_compile_features(mfq-cpu-ops PUBLIC cxx_std_17)
set_target_properties(mfq-cpu-ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64|i.86)$")
    target_sources(mfq-cpu-ops PRIVATE ops/common/host_memory_copy_avx2.cpp)
    if(TYLOINFER_ENABLE_MFQ_COMPAT)
        target_sources(mfq-cpu-ops PRIVATE ops/mfq/packed_nint_avx2.cpp ops/mfq/quant_dot_avx2.cpp ops/mfq/nvq_group_avx2.cpp)
    endif()
    target_compile_definitions(mfq-cpu-ops PRIVATE TYLOINFER_CPU_PACKED_AVX2=1)
    if(MSVC)
        set_source_files_properties(ops/mfq/packed_nint_avx2.cpp ops/mfq/quant_dot_avx2.cpp ops/mfq/nvq_group_avx2.cpp ops/common/host_memory_copy_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2;/fp:precise")
    else()
        set_source_files_properties(ops/mfq/packed_nint_avx2.cpp ops/mfq/quant_dot_avx2.cpp ops/mfq/nvq_group_avx2.cpp ops/common/host_memory_copy_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
    endif()
endif()
add_library(mfq::cpu-ops ALIAS mfq-cpu-ops)

add_library(mfq-qsa-kv-store STATIC
    storage/qsa_blocks.cpp
    storage/qsa_blocks.h
)
target_include_directories(mfq-qsa-kv-store PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/storage
)
target_link_libraries(mfq-qsa-kv-store PRIVATE Threads::Threads)
target_compile_features(mfq-qsa-kv-store PUBLIC cxx_std_17)
set_target_properties(mfq-qsa-kv-store PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mfq-prefix-blocks STATIC
    storage/prefix_blocks.cpp
    storage/prefix_blocks.h
)
target_include_directories(mfq-prefix-blocks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/storage
)
target_link_libraries(mfq-prefix-blocks PRIVATE Threads::Threads)
target_compile_features(mfq-prefix-blocks PUBLIC cxx_std_17)
set_target_properties(mfq-prefix-blocks PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(mfq-cpu-execution PRIVATE /EHsc)
    target_compile_options(mfq-qsa-kv-store PRIVATE /utf-8 /EHsc)
    target_compile_options(mfq-prefix-blocks PRIVATE /utf-8 /EHsc)
endif()

if(BUILD_TESTING AND TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cpu-nvq-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cpu_nvq_bench.cpp)
    target_link_libraries(mfq-cpu-nvq-bench PRIVATE mfq-cpu-ops)
endif()
