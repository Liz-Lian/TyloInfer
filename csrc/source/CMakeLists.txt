add_library(mfq-model-source STATIC
    model_source.cpp
    mfq_model_source.cpp
    hf_model_source.cpp
    mlx_affine_source.cpp
    gguf_model_source.cpp
    gguf_model_source.h
    mlx_affine_source.h
    mlx_tensor_names.h
    hf_safetensors_source.cpp
    tq_source.cpp
    tq_fixture_source.cpp
    compat/mfq_format_compat.h
    compat/mfq_legacy_model_graph.cpp
    compat/mfq_legacy_model_graph.h
    compat/mfq_legacy_tensor_names.cpp
    compat/mfq_legacy_tensor_names.h
    model_source.h
    read.h
    mfq_model_source.h
    hf_model_source.h
    hf_safetensors_source.h
    tq_source.h
    tq_fixture_source.h
    mapped_prefetch_scope.h
    mapped_read_profile.h
    mapped_tensor_pages.h
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
