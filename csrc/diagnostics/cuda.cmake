add_library(mfq-cuda-diagnostics STATIC
    ${PROJECT_SOURCE_DIR}/diagnostics/attention_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/backend_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/linear_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/operator_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/experts.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/ffn.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/gemma_moe.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/trace.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/engines.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/source.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/text.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/vision.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/mtp_bench.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/mtp.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/prefill.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/qwen_mtp.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/batch.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/tokens.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/compare.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/policy.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/qwen_decode.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/qwen_variants.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/session_checks.cpp
)
mfq_configure_cuda_engine_target(mfq-cuda-diagnostics)
target_compile_definitions(mfq-cuda-diagnostics PRIVATE NOMINMAX)
target_link_libraries(mfq-cuda-diagnostics PUBLIC mfq-cuda-engine)

if(MSVC)
    target_compile_options(mfq-cuda-diagnostics PRIVATE /utf-8 /EHsc /bigobj)
endif()
