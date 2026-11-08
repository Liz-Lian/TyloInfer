set(TYLOINFER_CUDA_MODEL_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/axial_patch_encoder.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/vision/grid_encoder.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v4/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/components.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/state.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/engram.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/prefill.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/dspark.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/deepseek_v41/vision.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/attention.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen4_exp/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen4_exp/layer.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen4_exp/components.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen4_exp/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/glm5_next/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/glm5_next/components.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/gemma4/forward.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/gemma4/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/gemma4/model.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/gemma4/state.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/glm_dsa/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/minicpmo45/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/nemotron_h/ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/nemotron_h/components.cpp
)
set(TYLOINFER_CUDA_MODEL_SOURCES "${TYLOINFER_CUDA_MODEL_SOURCES}" PARENT_SCOPE)

set(TYLOINFER_MODELS_CUDA_SOURCES
    ${PROJECT_SOURCE_DIR}/engine/models/minicpmo45/model.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/minicpmo45/state.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/minicpmo45/forward.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/config.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/model.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/forward.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/predictor.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/state.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/batch.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/batch_ops.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/mtp_state.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/mtp.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen35/components.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/load.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/qwen4_exp/batch.cpp
)
set(TYLOINFER_MODELS_CUDA_SOURCES "${TYLOINFER_MODELS_CUDA_SOURCES}" PARENT_SCOPE)

add_library(mfq-cuda-models OBJECT ${TYLOINFER_MODELS_CUDA_SOURCES})
mfq_configure_cuda_backend_target(mfq-cuda-models)
target_link_libraries(mfq-cuda-models PRIVATE mfq-engine)
target_compile_definitions(mfq-cuda-models PRIVATE
    TYLOINFER_NATIVE_CUDA=1 NOMINMAX)
set_target_properties(mfq-cuda-models PROPERTIES POSITION_INDEPENDENT_CODE ON)
