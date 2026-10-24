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

if(BUILD_TESTING AND TYLOINFER_ENABLE_MFQ_COMPAT)
    add_executable(mfq-cpu-nvq-bench ${TYLOINFER_REPOSITORY_ROOT}/bench/cpu_nvq_bench.cpp)
    target_link_libraries(mfq-cpu-nvq-bench PRIVATE mfq-cpu-ops)
endif()
