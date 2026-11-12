add_library(mfq-cuda-reference STATIC ${PROJECT_SOURCE_DIR}/eval/reference.cpp)
mfq_configure_cuda_engine_target(mfq-cuda-reference)
target_compile_definitions(mfq-cuda-reference PRIVATE TYLOINFER_NATIVE_CUDA=1 NOMINMAX)
target_link_libraries(mfq-cuda-reference PUBLIC mfq-cuda-engine)

add_library(mfq-cuda-eval STATIC
    ${PROJECT_SOURCE_DIR}/eval/kl.cpp
    ${PROJECT_SOURCE_DIR}/eval/minicpmo45.cpp
    ${PROJECT_SOURCE_DIR}/eval/tensors.cpp
)
mfq_configure_cuda_engine_target(mfq-cuda-eval)
target_compile_definitions(mfq-cuda-eval PRIVATE TYLOINFER_NATIVE_CUDA=1 NOMINMAX)
target_link_libraries(mfq-cuda-eval PUBLIC mfq-cuda-engine)

if(MSVC)
    target_compile_options(mfq-cuda-reference PRIVATE /utf-8 /EHsc /bigobj)
    target_compile_options(mfq-cuda-eval PRIVATE /utf-8 /EHsc /bigobj)
endif()
