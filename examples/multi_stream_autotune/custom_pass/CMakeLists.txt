cmake_minimum_required(VERSION 3.5.1)
project(ge_ms_autotune_pass)

set(ASCEND_HOME_PATH "" CACHE PATH "CANN installation root")

if (NOT ASCEND_HOME_PATH)
    if (DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else ()
        message(FATAL_ERROR "ASCEND_HOME_PATH is required")
    endif ()
endif ()

add_library(${PROJECT_NAME} SHARED src/ge_ms_autotune_pass.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
target_compile_options(${PROJECT_NAME} PRIVATE -g -Wall)
target_compile_definitions(${PROJECT_NAME} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_HOME_PATH}/include/graph
    ${ASCEND_HOME_PATH}/include/ge
    ${ASCEND_HOME_PATH}/include/register
    ${ASCEND_HOME_PATH}/include
)
target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_HOME_PATH}/lib64/stub
    ${ASCEND_HOME_PATH}/lib64
)
target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    ge_compiler
    -Wl,--as-needed
)
