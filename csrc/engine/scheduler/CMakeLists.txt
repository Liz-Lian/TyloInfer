add_library(mfq-runtime-scheduler STATIC
    scheduler.cpp
)
add_library(mfq::runtime-scheduler ALIAS mfq-runtime-scheduler)
target_include_directories(mfq-runtime-scheduler PUBLIC
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(mfq-runtime-scheduler PUBLIC mfq-engine)
target_compile_features(mfq-runtime-scheduler PUBLIC cxx_std_17)
set_target_properties(mfq-runtime-scheduler PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
