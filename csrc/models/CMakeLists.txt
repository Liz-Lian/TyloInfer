set(TYLOINFER_CUDA_QWEN_SOURCES
    ${PROJECT_SOURCE_DIR}/models/qwen35/config.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/model.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/forward.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/predictor.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/state.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/batch.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/mtp_state.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/mtp.cpp
    ${PROJECT_SOURCE_DIR}/models/mtp.cpp
)
set(TYLOINFER_CUDA_QWEN_SOURCES "${TYLOINFER_CUDA_QWEN_SOURCES}" PARENT_SCOPE)

add_library(mfq-cuda-models OBJECT ${TYLOINFER_CUDA_QWEN_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-models)
target_compile_definitions(mfq-cuda-models PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-models PROPERTIES POSITION_INDEPENDENT_CODE ON)
