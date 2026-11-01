add_library(mfq-model-source STATIC
    model_source.cpp
    mfq_model_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mfq.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mapped.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/buffered.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/tensor.h
    hf_model_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mlx.cpp
    gguf_model_source.cpp
    gguf_model_source.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mlx.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mfq.h
    mlx_tensor_names.h
    hf_safetensors_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/source.cpp
    tq_fixture_source.cpp
    compat/mfq_legacy_tensor_names.cpp
    compat/mfq_legacy_tensor_names.h
    model_source.h
    read.h
    mfq_model_source.h
    hf_model_source.h
    hf_safetensors_source.h
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/source.h
    tq_fixture_source.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mapped.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/buffered.h
)
add_library(mfq::model-source ALIAS mfq-model-source)
target_include_directories(mfq-model-source PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(mfq-model-source PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(mfq-model-source PUBLIC mfq-model-graph mfq-formats mfq-io mfq-quant-readers
    PRIVATE mfq-json mfq-ggml)
target_compile_features(mfq-model-source PUBLIC cxx_std_17)
set_target_properties(mfq-model-source PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-model-source PRIVATE /EHsc)
endif()
