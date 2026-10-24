add_library(mfq-engine STATIC
    executor/config.cpp
    executor/mtp_metrics.cpp
    models/common/axial_patch_vision.cpp
    models/common/model_config.cpp
    models/common/vision/grid_vision.cpp
    models/deepseek_v4/config.cpp
    models/deepseek_v41/config.cpp
    models/gemma4/config.cpp
    models/glm5_next/config.cpp
    models/glm_dsa/config.cpp
    models/minicpmo45/config.cpp
    models/nemotron_h/config.cpp
    models/nemotron_h/vision.cpp
    models/qwen35/config.cpp
    models/qwen4_exp/config.cpp
    requests/constraint.cpp
    requests/emitter.cpp
    requests/inference.cpp
    requests/policy.cpp
    requests/processor.cpp
    requests/table.cpp
    requests/tokenizer.cpp
    state/paged.cpp
    state/prefix_cache.cpp
    state/qsa_sequence.cpp
)
add_library(mfq::engine ALIAS mfq-engine)
target_include_directories(mfq-engine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/components/tokenizer/include
        ${PROJECT_SOURCE_DIR}/components/tokenizer/chat
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/components/tokenizer
        ${PROJECT_SOURCE_DIR}/components/tokenizer/src
)
target_compile_features(mfq-engine PUBLIC cxx_std_20)
target_link_libraries(mfq-engine PUBLIC
    mfq-model-source
    mfq-model-graph
    mfq-formats
    mfq-json
    mfq-prefix-blocks
    mfq-qsa-kv-store
)
set_target_properties(mfq-engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(TARGET mfq-tokenizer)
    target_link_libraries(mfq-engine PUBLIC mfq-tokenizer)
    target_compile_definitions(mfq-engine PUBLIC TYLOINFER_ENGINE_TEXT=1)
endif()
