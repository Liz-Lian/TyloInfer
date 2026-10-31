# Compressed host math; shared by CUDA offload and other host consumers.
add_library(mfq-cpu-ops STATIC
    ops/common/host_memory_copy.cpp
    ops/common/host_memory_copy.h
    ops/common/gguf_cpu.cpp
)
if(TYLOINFER_ENABLE_MFQ_COMPAT)
    target_sources(mfq-cpu-ops PRIVATE
        ops/mfq/packed_nint.cpp ops/mfq/quant_dot.cpp ops/mfq/nvq_group.cpp)
endif()
target_include_directories(mfq-cpu-ops PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/common
    ${CMAKE_CURRENT_SOURCE_DIR}/ops/mfq)
target_link_libraries(mfq-cpu-ops PUBLIC mfq-host-parallel)
target_link_libraries(mfq-cpu-ops PRIVATE mfq-ggml mfq-ggml-internal-headers)
target_compile_features(mfq-cpu-ops PUBLIC cxx_std_17)
set_target_properties(mfq-cpu-ops PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64|i.86)$")
    # The GGUF row arithmetic actually uses AVX2, FMA and half conversion.
    # Only these translation units get the ISA flags; baseline dispatch is safe.
    target_sources(mfq-cpu-ops PRIVATE ops/common/gguf_cpu_avx2.cpp ops/common/gguf_cpu_q2_avx2.cpp ops/common/gguf_cpu_prepare_avx2.cpp)
    target_compile_definitions(mfq-cpu-ops PRIVATE TYLOINFER_GGUF_CPU_AVX2=1)
    if(MSVC)
        set_source_files_properties(ops/common/gguf_cpu_avx2.cpp ops/common/gguf_cpu_q2_avx2.cpp ops/common/gguf_cpu_prepare_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "/arch:AVX2;/fp:precise")
    else()
        set_source_files_properties(ops/common/gguf_cpu_avx2.cpp ops/common/gguf_cpu_q2_avx2.cpp ops/common/gguf_cpu_prepare_avx2.cpp
            PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c;-ffp-contract=off")
    endif()
    # Compile wide rows independently; baseline feature detection stays free
    # of AVX-512. The check compiles only and never runs on the build host.
    include(CheckCXXSourceCompiles)
    set(_gguf_saved_required_flags "${CMAKE_REQUIRED_FLAGS}")
    # Probe both 256-bit VNNI encodings without executing on the build host.
    # Only their row functions use the extensions; baseline code stays AVX2.
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "${_gguf_saved_required_flags} /arch:AVX2")
    else()
        set(CMAKE_REQUIRED_FLAGS "${_gguf_saved_required_flags} -mavx2 -mavxvnni")
    endif()
    check_cxx_source_compiles([=[
        #include <immintrin.h>
        __m256i probe(__m256i a,__m256i b,__m256i c) {
            return _mm256_dpwssd_avx_epi32(_mm256_dpbusd_avx_epi32(a,b,c),b,c);
        }
        int main(){return 0;}
    ]=] TYLOINFER_COMPILER_GGUF_VNNI_VEX)
    if(TYLOINFER_COMPILER_GGUF_VNNI_VEX)
        target_compile_definitions(mfq-cpu-ops PRIVATE STRATA_AVXVNNI=1)
    endif()
    if(NOT MSVC)
        set(CMAKE_REQUIRED_FLAGS "${_gguf_saved_required_flags} -mavx2 -mavx512vl -mavx512vnni")
    endif()
    check_cxx_source_compiles([=[
        #include <immintrin.h>
        __m256i probe(__m256i a,__m256i b,__m256i c) {
            return _mm256_dpwssd_epi32(_mm256_dpbusd_epi32(a,b,c),b,c);
        }
        int main(){return 0;}
    ]=] TYLOINFER_COMPILER_GGUF_VNNI_EVEX)
    if(TYLOINFER_COMPILER_GGUF_VNNI_EVEX)
        target_compile_definitions(mfq-cpu-ops PRIVATE TYLOINFER_GGUF_VNNI_EVEX=1)
    endif()
    set(CMAKE_REQUIRED_FLAGS "${_gguf_saved_required_flags}")
    if(MSVC)
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} /arch:AVX512")
    else()
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma -mf16c")
    endif()
    check_cxx_source_compiles([=[
        #include <immintrin.h>
        __m256i probe(__m512i a){return _mm512_extracti64x4_epi64(_mm512_mask_sub_epi8(a,3,a,a),1);}
        int main(){return 0;}
    ]=] TYLOINFER_COMPILER_GGUF_AVX512)
    set(CMAKE_REQUIRED_FLAGS "${_gguf_saved_required_flags}")
    if(TYLOINFER_COMPILER_GGUF_AVX512)
        target_sources(mfq-cpu-ops PRIVATE ops/common/gguf_cpu_avx512.cpp)
        target_compile_definitions(mfq-cpu-ops PRIVATE TYLOINFER_GGUF_CPU_AVX512=1)
        if(MSVC)
            set_source_files_properties(ops/common/gguf_cpu_avx512.cpp
                PROPERTIES COMPILE_OPTIONS "/arch:AVX512;/fp:precise")
        else()
            set_source_files_properties(ops/common/gguf_cpu_avx512.cpp
                PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma;-mf16c;-ffp-contract=off")
        endif()
    endif()
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
