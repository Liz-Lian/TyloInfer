# Private HTTP/stdio transport for Engine applications.

find_package(Threads REQUIRED)

function(mfq_add_transport target engine_target)
    set(source_dir ${CMAKE_CURRENT_FUNCTION_LIST_DIR})
    add_library(${target} STATIC
        ${source_dir}/include/transport.h
        ${source_dir}/src/common.cpp
        ${source_dir}/src/common.h
        ${source_dir}/src/http.cpp
        ${source_dir}/src/stdio.cpp
    )
    target_include_directories(${target} PUBLIC
        ${source_dir}/include
    )
    target_include_directories(${target} PRIVATE
        ${PROJECT_SOURCE_DIR}
    )
    target_link_libraries(${target}
        PUBLIC ${engine_target}
        PRIVATE mfq-http mfq-json Threads::Threads
    )
    target_compile_definitions(${target} PRIVATE NOMINMAX)
    if(WIN32)
        target_link_libraries(${target} PRIVATE advapi32)
    endif()
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    if(MSVC)
        target_compile_options(${target} PRIVATE /utf-8 /EHsc)
    endif()
endfunction()

mfq_add_transport(mfq-transport mfq-engine)
add_library(mfq::transport ALIAS mfq-transport)
