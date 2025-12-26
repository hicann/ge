file(GLOB_RECURSE CODEGEN_COMMON_SRCS CONFIGURE_DEPENDS "*.cpp" "*.cc")

target_include_directories(aihac_codegen PRIVATE
        ${ASCEND_INSTALL_PATH}/pkg_inc
        ${ASCEND_INSTALL_PATH}/pkg_inc/base
)

target_sources(aihac_codegen PRIVATE
    ${CODEGEN_COMMON_SRCS}
)
