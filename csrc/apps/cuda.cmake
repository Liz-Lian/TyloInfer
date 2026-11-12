if(TARGET mfq-transport-cuda)
    add_executable(mfq-runtime
        ${PROJECT_SOURCE_DIR}/apps/runtime_main.cpp
        ${PROJECT_SOURCE_DIR}/commands/runtime.cpp
        ${PROJECT_SOURCE_DIR}/commands/cli.cpp
        ${PROJECT_SOURCE_DIR}/commands/reference.cpp
        ${PROJECT_SOURCE_DIR}/commands/minicpmo45.cpp
    )
    target_link_libraries(mfq-runtime PRIVATE
        mfq-cuda-engine mfq-cuda-reference mfq-cuda-eval mfq-transport-cuda)
endif()

add_executable(mfq-diagnostics
    ${PROJECT_SOURCE_DIR}/apps/diagnostics_main.cpp
    ${PROJECT_SOURCE_DIR}/commands/diagnostics.cpp
    ${PROJECT_SOURCE_DIR}/commands/cli.cpp
    ${PROJECT_SOURCE_DIR}/commands/minicpmo45.cpp
)
target_link_libraries(mfq-diagnostics PRIVATE mfq-cuda-diagnostics mfq-cuda-eval)

add_executable(mfq-numerics
    ${PROJECT_SOURCE_DIR}/apps/numerics_main.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/case.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/tensor.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/basic.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/attn.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/sample.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/gdn.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/flash.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/flash_graph.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/flash_reject.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/mfq.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/sq4.cpp
    ${PROJECT_SOURCE_DIR}/diagnostics/numerics/tq.cpp
)
target_link_libraries(mfq-numerics PRIVATE mfq-cuda-engine CUDA::cudart CUDA::cublas)

add_executable(mfq-eval
    ${PROJECT_SOURCE_DIR}/apps/eval_main.cpp
    ${PROJECT_SOURCE_DIR}/commands/eval.cpp
    ${PROJECT_SOURCE_DIR}/commands/cli.cpp
)
target_link_libraries(mfq-eval PRIVATE mfq-cuda-eval)

if(TARGET mfq-cuda-torch-engine)
    add_executable(mfq-runtime-torch
        ${PROJECT_SOURCE_DIR}/apps/runtime_main.cpp
        ${PROJECT_SOURCE_DIR}/commands/runtime.cpp
        ${PROJECT_SOURCE_DIR}/commands/cli.cpp
        ${PROJECT_SOURCE_DIR}/commands/reference.cpp
        ${PROJECT_SOURCE_DIR}/commands/minicpmo45.cpp
    )
    target_link_libraries(mfq-runtime-torch PRIVATE
        mfq-cuda-torch-engine mfq-transport-torch)
    set_target_properties(mfq-runtime-torch PROPERTIES
        CUDA_ARCHITECTURES "${TYLOINFER_CUDA_ARCHITECTURES}"
        CUDA_RUNTIME_LIBRARY Shared
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

foreach(target mfq-runtime mfq-diagnostics mfq-numerics mfq-eval mfq-runtime-torch)
    if(TARGET ${target})
        mfq_configure_cuda_engine_target(${target})
        target_compile_definitions(${target} PRIVATE NOMINMAX)
        if(MSVC)
            target_compile_options(${target} PRIVATE /utf-8 /EHsc /bigobj)
        endif()
        set_target_properties(${target} PROPERTIES
            CUDA_RUNTIME_LIBRARY Shared
            RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}"
        )
    endif()
endforeach()

if(NOT WIN32 AND TARGET mfq-runtime)
    set(TYLOINFER_SYSTEM_RUNTIME_DIR
        "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" CACHE PATH
        "directory containing the host libstdc++ runtime")
    set_property(TARGET mfq-runtime PROPERTY
        BUILD_RPATH "${TYLOINFER_SYSTEM_RUNTIME_DIR}")
endif()
