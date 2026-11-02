add_library(mfq-model-graph STATIC
    model_graph.cpp
    model_graph.h
    model_config.cpp
    model_config.h
    gguf.cpp
    gguf.h
    hf.cpp
    hf.h
    legacy_graph.cpp
    legacy_graph.h
    legacy_names.cpp
    legacy_names.h
    mlx_names.h)
add_library(mfq::model-graph ALIAS mfq-model-graph)
target_include_directories(mfq-model-graph PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-model-graph PRIVATE mfq-json)
target_compile_features(mfq-model-graph PUBLIC cxx_std_17)
set_target_properties(mfq-model-graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mfq-model-source STATIC
    ../source.cpp
    ../source.h
    tq_fixture_source.cpp
    tq_fixture_source.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/hf_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/gguf_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/gguf_source.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq_source.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/hf_source.h
)
add_library(mfq::model-source ALIAS mfq-model-source)
target_include_directories(mfq-model-source PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-model-source PUBLIC mfq-model-graph mfq-weight-source
    PRIVATE mfq-json)
target_compile_features(mfq-model-source PUBLIC cxx_std_17)
set_target_properties(mfq-model-source PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-model-source PRIVATE /EHsc)
endif()
