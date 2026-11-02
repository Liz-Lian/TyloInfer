add_library(mfq-model-source STATIC
    model_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mfq.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mapped.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/buffered.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/files.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/json.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/safetensors.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/hf.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/hf.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/tensor.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/hf_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/assets.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/mlx.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/gguf.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/gguf.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/gguf.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/gguf_source.cpp
    ${PROJECT_SOURCE_DIR}/engine/models/common/gguf_source.h
    ${PROJECT_SOURCE_DIR}/engine/weights/mlx.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/gguf.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mfq.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/json.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/safetensors.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/hf.cpp
    ${PROJECT_SOURCE_DIR}/engine/weights/source/hf.h
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/source.cpp
    tq_fixture_source.cpp
    model_source.h
    read.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq_source.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/mfq.h
    ${PROJECT_SOURCE_DIR}/engine/models/common/hf_source.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/assets.h
    ${PROJECT_SOURCE_DIR}/engine/weights/tq/source.h
    tq_fixture_source.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/mapped.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/buffered.h
    ${PROJECT_SOURCE_DIR}/engine/weights/source/files.h
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
