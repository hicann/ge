cmake_minimum_required(VERSION 3.5.1)
project(decompose_grouped_conv_to_splited_pass)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_PATH $ENV{ASCEND_HOME_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit/latest)
endif ()

set(INCLUDE_DIR ${ASCEND_PATH}/include)
aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(common_compile_options
        -std=c++17
        -g
        -Wall
)

set(common_compile_definitions
        _GLIBCXX_USE_CXX11_ABI=0
)
message(STATUS "ASCEND_PATH: ${ASCEND_PATH}")

add_subdirectory(gen_es_api)

############ decompose_grouped_conv_to_splited_pass.so ############
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

set_target_properties(${PROJECT_NAME} PROPERTIES output_name ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME} PRIVATE
        ${common_compile_options}

)

target_compile_definitions(${PROJECT_NAME} PRIVATE
        ${common_compile_definitions}
)

target_include_directories(${PROJECT_NAME} PRIVATE
        ${ASCEND_PATH}/opp/built-in/op_proto/inc
        ${INCLUDE_DIR}/graph
        ${INCLUDE_DIR}/ge
        ${INCLUDE_DIR}/register
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external
        ${CMAKE_CURRENT_BINARY_DIR}/generated_ops
)

target_link_directories(${PROJECT_NAME} PRIVATE
        ${ASCEND_PATH}/lib64/stub
        ${ASCEND_PATH}/lib64/ #todo delete  eager_style_graph_builder_base change to stub
)
target_link_options(${PROJECT_NAME} PRIVATE
)
target_link_libraries(${PROJECT_NAME} PRIVATE
        -Wl,--no-as-needed
        graph
        register
        ge_compiler
        esb_generated_shared
        eager_style_graph_builder_base
        -Wl,--as-needed
)

add_dependencies(${PROJECT_NAME} esb_generated_shared)
