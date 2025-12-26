file(GLOB_RECURSE CODEGEN_API_CALL_SRCS CONFIGURE_DEPENDS "*.cpp")

target_sources(aihac_codegen PRIVATE
    ${CODEGEN_API_CALL_SRCS}
)

target_include_directories(aihac_codegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CODE_ROOT_DIR}/common
    ${CODE_ROOT_DIR}/att
)