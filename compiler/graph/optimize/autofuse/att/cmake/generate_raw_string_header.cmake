# 简化版本，使用 echo 和 cat 组合
function(generate_raw_string_header TARGET_NAME OUTPUT_FILE)
    # 解析参数
    set(INPUT_FILES ${ARGN})

    # 创建自定义命令
    add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND bash -c "echo 'R\"===(' > ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} && cat ${INPUT_FILES} >> ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE} && echo ')===\"' >> ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}"
            DEPENDS ${INPUT_FILES}
            COMMENT "Generating ${OUTPUT_FILE} from ${INPUT_FILES}"
            VERBATIM
    )

    # 创建自定义目标
    add_custom_target(${TARGET_NAME}_text ALL
            DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${OUTPUT_FILE}
    )

    # 创建接口库
    add_library(${TARGET_NAME} INTERFACE)

    # 设置接口库的包含目录
    target_include_directories(${TARGET_NAME} INTERFACE
            ${CMAKE_CURRENT_BINARY_DIR}
            ..
    )

    # 添加依赖关系
    add_dependencies(${TARGET_NAME} ${TARGET_NAME}_text)
endfunction()