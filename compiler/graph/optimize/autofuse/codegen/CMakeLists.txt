add_subdirectory(api_call)
add_subdirectory(reg_api_call)
add_subdirectory(micro_api_call)
add_subdirectory(vec_func_call)
add_subdirectory(cube_api_call)
target_sources(aihac_codegen PRIVATE
    codegen_kernel.cpp
    codegen_tiling.cpp
    codegen_tiling_data.cpp
    codegen_infershape.cpp
    codegen.cpp
)

target_include_directories(aihac_codegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CODE_ROOT_DIR}/common
    ${CODE_ROOT_DIR}/att
)