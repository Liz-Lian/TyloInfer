# DenseTensor lives outside mfq-engine so native-backend targets can link the
# tensor implementation without the whole engine.
add_library(mfq-tensor STATIC tensor.cpp)
add_library(mfq::tensor ALIAS mfq-tensor)
target_include_directories(mfq-tensor PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_features(mfq-tensor PUBLIC cxx_std_20)
set_target_properties(mfq-tensor PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mfq-engine STATIC
    config.cpp
    generation/mtp/metrics.cpp
    models/prepare.cpp
    models/deepseek_v4/input.cpp
    models/minicpmo45/input.cpp
    models/deepseek_v41/axial_patch_vision.cpp
    models/common/vision/grid_vision.cpp
    models/deepseek_v4/config.cpp
    models/deepseek_v41/config.cpp
    models/gemma4/config.cpp
    models/glm5_next/config.cpp
    models/glm_dsa/config.cpp
    models/minicpmo45/config.cpp
    models/nemotron_h/config.cpp
    models/nemotron_h/vision.cpp
    models/qwen4_exp/config.cpp
    requests/constraint.cpp
    requests/emitter.cpp
    requests/output.cpp
    requests/policy.cpp
    requests/processor.cpp
    requests/tokenizer.cpp
    state/session/paged.cpp
    state/session/prefix_cache.cpp
    state/qsa_sequence.cpp
)
add_library(mfq::engine ALIAS mfq-engine)
target_include_directories(mfq-engine
    PUBLIC
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/components/tokenizer/include
        ${PROJECT_SOURCE_DIR}/components/tokenizer/chat
    PRIVATE
        ${PROJECT_SOURCE_DIR}/components/tokenizer
        ${PROJECT_SOURCE_DIR}/components/tokenizer/src
)
target_compile_features(mfq-engine PUBLIC cxx_std_20)
target_link_libraries(mfq-engine PUBLIC
    mfq-tensor
    mfq-model-source
    mfq-model-graph
    mfq-formats
    mfq-json
    mfq-prefix-blocks
    mfq-qsa-kv-store
)
set_target_properties(mfq-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(mfq-engine PUBLIC mfq-tokenizer)
target_compile_definitions(mfq-engine PUBLIC TYLOINFER_ENGINE_TEXT=1)
