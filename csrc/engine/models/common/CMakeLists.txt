add_library(mfq-model-graph STATIC
    model_graph.cpp
    model_graph.h
    legacy_graph.cpp
    legacy_graph.h)
add_library(mfq::model-graph ALIAS mfq-model-graph)
target_include_directories(mfq-model-graph PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-model-graph PRIVATE mfq-json)
target_compile_features(mfq-model-graph PUBLIC cxx_std_17)
set_target_properties(mfq-model-graph PROPERTIES POSITION_INDEPENDENT_CODE ON)
