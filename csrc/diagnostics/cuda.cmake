add_library(mfq-cuda-diagnostics STATIC
    ${PROJECT_SOURCE_DIR}/diagnostics/attention_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/backend_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/linear_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/moe_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/runtime_checks.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/session_checks.cpp
)
mfq_configure_cuda_engine_target(mfq-cuda-diagnostics)
target_compile_definitions(mfq-cuda-diagnostics PRIVATE TYLOINFER_NATIVE_CUDA=1 NOMINMAX)
target_link_libraries(mfq-cuda-diagnostics PUBLIC mfq-cuda-engine)
