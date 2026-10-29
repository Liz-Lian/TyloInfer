set(TYLOINFER_MODELS_CUDA_SOURCES
    ${PROJECT_SOURCE_DIR}/models/minicpmo45/model.cpp
    ${PROJECT_SOURCE_DIR}/models/minicpmo45/state.cpp
    ${PROJECT_SOURCE_DIR}/models/minicpmo45/forward.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/config.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/load.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/model.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/forward.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/generation.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/predictor.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/state.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/batch.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/batch_ops.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/mtp_state.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/mtp.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen35/components.cpp
    ${PROJECT_SOURCE_DIR}/models/engine.cpp
    ${PROJECT_SOURCE_DIR}/models/qwen4_exp/batch.cpp
)
set(TYLOINFER_MODELS_CUDA_SOURCES "${TYLOINFER_MODELS_CUDA_SOURCES}" PARENT_SCOPE)

add_library(mfq-cuda-models OBJECT ${TYLOINFER_MODELS_CUDA_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-models)
target_link_libraries(mfq-cuda-models PRIVATE mfq-engine)
target_compile_definitions(mfq-cuda-models PRIVATE
    TYLOINFER_NATIVE_CUDA_RUNTIME=1 NOMINMAX)
set_target_properties(mfq-cuda-models PROPERTIES POSITION_INDEPENDENT_CODE ON)
