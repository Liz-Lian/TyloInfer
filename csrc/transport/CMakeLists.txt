# Private HTTP/stdio transport for Engine applications.

find_package(Threads REQUIRED)

add_library(mfq-transport STATIC
    include/transport.h
    src/common.cpp
    src/common.h
    src/http.cpp
    src/stdio.cpp
)
add_library(mfq::transport ALIAS mfq-transport)
target_include_directories(mfq-transport PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_include_directories(mfq-transport PRIVATE
    ${PROJECT_SOURCE_DIR}
)
target_link_libraries(mfq-transport
    PUBLIC
        mfq-engine
    PRIVATE
        mfq-http
        mfq-json
        Threads::Threads
)
target_compile_definitions(mfq-transport PRIVATE NOMINMAX)
if(WIN32)
    target_link_libraries(mfq-transport PRIVATE advapi32)
endif()
target_compile_features(mfq-transport PUBLIC cxx_std_17)
set_target_properties(mfq-transport PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
    target_compile_options(mfq-transport PRIVATE /utf-8 /EHsc)
endif()
