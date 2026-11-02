cmake_minimum_required(VERSION 3.26)

foreach(required SERVER DESTINATION DUMPBIN)
    if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
        message(FATAL_ERROR "${required} is required")
    endif()
endforeach()

# 在构建后递归解析 PE 导入，不把整个开发环境或 Python DLL 打入产品。
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM windows+pe)
set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL dumpbin)
set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND "${DUMPBIN}")
file(GLOB staged_libraries "${DESTINATION}/*.dll")
file(TO_CMAKE_PATH "$ENV{SystemRoot}/System32/" system_directory)
set(system_pattern "^")
string(LENGTH "${system_directory}" length)
math(EXPR last "${length} - 1")
foreach(index RANGE ${last})
    string(SUBSTRING "${system_directory}" ${index} 1 character)
    if(character MATCHES "^[A-Za-z]$")
        string(TOLOWER "${character}" lower)
        string(TOUPPER "${character}" upper)
        string(APPEND system_pattern "[${lower}${upper}]")
    elseif(character STREQUAL "/")
        string(APPEND system_pattern "[/\\\\]")
    elseif(character MATCHES "^[.+:]$")
        string(APPEND system_pattern "[${character}]")
    else()
        string(APPEND system_pattern "${character}")
    endif()
endforeach()
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${SERVER}"
    LIBRARIES ${staged_libraries}
    DIRECTORIES ${SEARCH_DIRECTORIES}
    RESOLVED_DEPENDENCIES_VAR resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    CONFLICTING_DEPENDENCIES_PREFIX conflicts
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-" "^nvcuda\\.dll$"
    POST_EXCLUDE_REGEXES "${system_pattern}"
)
if(unresolved)
    message(FATAL_ERROR "Incomplete native runtime: unresolved=${unresolved}")
endif()
foreach(name IN LISTS conflicts_FILENAMES)
    string(TOLOWER "${name}" lower_name)
    # 开发库目录可能自带旧 CRT；产品统一使用已暂存的 VS redistributable。
    if(NOT lower_name MATCHES "^(vcruntime|msvcp|concrt|vcomp)[0-9][a-z0-9_]*\\.dll$"
            OR NOT EXISTS "${DESTINATION}/${name}")
        message(FATAL_ERROR "Conflicting native runtime dependency ${name}: ${conflicts_${name}}")
    endif()
endforeach()
foreach(library IN LISTS resolved)
    get_filename_component(name "${library}" NAME)
    string(TOLOWER "${name}" name)
    if(name MATCHES "^(python|torch|libtorch|c10)")
        message(FATAL_ERROR "Forbidden product runtime dependency: ${library}")
    endif()
    file(COPY_FILE "${library}" "${DESTINATION}/${name}" ONLY_IF_DIFFERENT)
endforeach()
# 最终仅允许服务同目录与系统库满足导入，防止开发环境掩盖缺失的传递依赖。
file(GLOB staged_libraries "${DESTINATION}/*.dll")
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${SERVER}"
    LIBRARIES ${staged_libraries}
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    PRE_EXCLUDE_REGEXES "^api-ms-" "^ext-ms-" "^nvcuda\\.dll$"
    POST_EXCLUDE_REGEXES "${system_pattern}"
)
if(unresolved)
    message(FATAL_ERROR "Staged native runtime is incomplete: ${unresolved}")
endif()
message(STATUS "Native runtime dependency closure staged in ${DESTINATION}")
