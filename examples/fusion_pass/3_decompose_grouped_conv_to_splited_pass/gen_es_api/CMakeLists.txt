message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

###########  generate eager stype api  ######################
set(CMAKE_CURRENT_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
message(STATUS "Binary directory: ${CMAKE_CURRENT_BINARY_DIR}")
set(common_compile_options
        -std=c++17
        -g
        -Wall
)
set(common_compile_definitions
        _GLIBCXX_USE_CXX11_ABI=0
)
# 1. 创建输出目录
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated_ops)

# 2. 在构建目录创建占位源文件
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/generated_ops/placeholder.cpp
        "// Placeholder file - will be replaced during build\n"
        "void placeholder_function() {}\n")

# 3. 使用 file(GLOB) 获取文件列表（但使用 CONFIGURE_DEPENDS 特性）
file(GLOB GENERATED_CPP_FILES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_BINARY_DIR}/generated_ops/*.cpp"
)
link_directories(${ASCEND_PATH}/libs)
# 4. 定义代码生成命令
add_custom_command(
        OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/generated_ops/generated_code.flag
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_CURRENT_BINARY_DIR}/generated_ops/*.cpp
        COMMAND ${CMAKE_COMMAND} -E env ASCEND_OPP_PATH=${ASCEND_PATH}/opp env LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ASCEND_PATH}/lib64 ${ASCEND_PATH}/bin/gen_esb --output_dir=${CMAKE_CURRENT_BINARY_DIR}/generated_ops
        COMMAND ${CMAKE_COMMAND} -E touch ${CMAKE_CURRENT_BINARY_DIR}/generated_ops/generated_code.flag
        COMMENT "Generating eager style graph builder code..."
        VERBATIM
)
# 5. 创建自定义目标触发代码生成
add_custom_target(generate_es_api ALL
        DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/generated_ops/generated_code.flag)

# 临时处理，后续run包发布了es api的so后，去掉这一步
###########  generate eager stype api so  ######################

add_library(esb_generated_shared SHARED
        ${CMAKE_CURRENT_BINARY_DIR}/generated_ops/placeholder.cpp
        ${GENERATED_CPP_FILES}
)
set_source_files_properties(${GENERATED_CPP_FILES}
        PROPERTIES
        GENERATED TRUE
)
add_dependencies(esb_generated_shared generate_es_api)
target_compile_options(esb_generated_shared PRIVATE ${common_compile_options})
target_compile_definitions(esb_generated_shared PRIVATE ${common_compile_definitions})

target_include_directories(esb_generated_shared PUBLIC
        ${CMAKE_CURRENT_BINARY_DIR}/generated_ops
        ${PROJECT_SOURCE_DIR}/gen_es_api/temp
        ${ASCEND_PATH}/opp/built-in/op_proto/inc
        ${INCLUDE_DIR}/graph
        ${INCLUDE_DIR}/ge
        ${INCLUDE_DIR}/register
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external
)
target_link_directories(esb_generated_shared PRIVATE
        ${ASCEND_PATH}/lib64/stub
        ${ASCEND_PATH}/lib64/
)
target_link_libraries(esb_generated_shared
        PUBLIC
        eager_style_graph_builder_base
)