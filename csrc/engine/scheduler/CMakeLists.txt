add_library(mfq-runtime-scheduler STATIC
    scheduler.cpp
)
if(TYLOINFER_BUILD_CUDA_RUNTIME)
    target_sources(mfq-runtime-scheduler PRIVATE request.cpp)
    target_compile_definitions(mfq-runtime-scheduler PUBLIC TYLOINFER_NATIVE_CUDA_RUNTIME=1)
endif()
add_library(mfq::runtime-scheduler ALIAS mfq-runtime-scheduler)
target_include_directories(mfq-runtime-scheduler PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(mfq-runtime-scheduler PUBLIC mfq-engine)
target_compile_features(mfq-runtime-scheduler PUBLIC cxx_std_17)
set_target_properties(mfq-runtime-scheduler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
