add_library(mfq-scheduler STATIC
    scheduler.cpp
)
if(TYLOINFER_BUILD_CUDA_ENGINE)
    target_sources(mfq-scheduler PRIVATE request.cpp)
    target_compile_definitions(mfq-scheduler PUBLIC TYLOINFER_NATIVE_CUDA=1)
endif()
add_library(mfq::scheduler ALIAS mfq-scheduler)
target_include_directories(mfq-scheduler PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(mfq-scheduler PUBLIC mfq-engine)
target_compile_features(mfq-scheduler PUBLIC cxx_std_17)
set_target_properties(mfq-scheduler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
