find_package(Threads REQUIRED)

add_library(mfq-qsa-kv-store STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/qsa_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/qsa_blocks.h
)
target_include_directories(mfq-qsa-kv-store PUBLIC
    ${PROJECT_SOURCE_DIR}/engine/state
)
target_link_libraries(mfq-qsa-kv-store PRIVATE Threads::Threads)
target_compile_features(mfq-qsa-kv-store PUBLIC cxx_std_17)
set_target_properties(mfq-qsa-kv-store PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(mfq-prefix-blocks STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/session/prefix_blocks.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/session/prefix_blocks.h
)
target_include_directories(mfq-prefix-blocks PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/session
)
target_link_libraries(mfq-prefix-blocks PRIVATE Threads::Threads)
target_compile_features(mfq-prefix-blocks PUBLIC cxx_std_17)
set_target_properties(mfq-prefix-blocks PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(mfq-qsa-kv-store PRIVATE /utf-8 /EHsc)
    target_compile_options(mfq-prefix-blocks PRIVATE /utf-8 /EHsc)
endif()
