add_library(mfq-formats INTERFACE)
target_include_directories(mfq-formats INTERFACE
    ${PROJECT_SOURCE_DIR})

add_library(mfq-quant-readers STATIC
    bytes.h
    dense_blob.cpp
    dense_blob.h
    fp8_sq_blob.cpp
    fp8_sq_blob.h
    quant/nint_rows.cpp
    quant/nint_blob.cpp
    quant/nint_blob.h
    quant/nvq_rows.cpp
    quant/vq_blob.cpp
    quant/vq_blob.h
    quant/mfe_quant_expert_store.cpp
    quant/mfe_index.cpp
    quant/mfe_index.h
    quant/mfe_blob.cpp
    quant/mfe_blob.h
    mlx_affine.cpp
    mlx_affine.h
    mx_blob.cpp
    mx_blob.h
    tq_table.cpp
    tq_table.h
)
target_include_directories(mfq-quant-readers PRIVATE
    ${PROJECT_SOURCE_DIR})
target_link_libraries(mfq-quant-readers PUBLIC mfq-formats mfq-io
    PRIVATE mfq-host-parallel mfq-json)
target_compile_features(mfq-quant-readers PUBLIC cxx_std_17 PRIVATE cxx_std_20)
set_target_properties(mfq-quant-readers PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(MSVC)
    target_compile_options(mfq-quant-readers PRIVATE /EHsc)
endif()
