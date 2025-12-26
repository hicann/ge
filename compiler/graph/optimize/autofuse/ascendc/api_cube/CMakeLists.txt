set(ascendc_api_cube_src
    matmul.h
    matmul/mat_mul_v3_tiling_key_public.h
    matmul/mat_mul_tiling_key.h
    matmul/mat_mul_v3_common.h
    matmul/mat_mul_tiling_data.h
    matmul/batch_mat_mul_v3_tiling_key.h
    matmul/mat_mul_asw_block.h
    matmul/mat_mul_dasw_block.h
    matmul/mat_mul_asw_kernel.h
    matmul/mat_mul_stream_k_block.h
    matmul/mat_mul_stream_k_kernel.h
    matmul/mat_mul_v3_full_load_kernel_helper.h
    matmul/mat_mul_full_load.h
    matmul/mm_copy_cube_out.h
    matmul/mm_custom_mm_policy.h
    matmul/mat_mul_fixpipe_opti.h
    matmul/block_scheduler_aswt.h
    matmul/block_scheduler_streamk.h
    matmul/mat_mul_fixpipe_opti_basic_act.h
    matmul/batch_mat_mul_v3_matmul2mul_block_scheduler.h
    matmul/batch_mat_mul_v3_matmul2mul_act.h
    matmul/mat_mul_pingpong_basic_act.h
    matmul/mat_mul_input_k_eq_zero_clear_output.h
    matmul/mat_mul_streamk_basic_act.h
    matmul/batch_mat_mul_v3_asw_kernel_advanced.h
    matmul/batch_mat_mul_v3_asw_block_advanced.h
    matmul/batch_mat_mul_v3_dasw_block_advanced.h
    matmul/batch_mat_mul_v3_asw_al1_full_load_kernel_advanced.h
    matmul/batch_mat_mul_v3_asw_bl1_full_load_kernel_advanced.h
    matmul/batch_mat_mul_v3_iterbatch_block_advanced.h
    matmul/batch_mat_mul_v3_iterbatch_kernel_advanced.h
    matmul/batch_mat_mul_v3_iterbatch_basicapi_block_scheduler.h
    matmul/batch_mat_mul_v3_iterbatch_basicapi_act.h
    batch_matmul.h
    )

set(generated_cube_files)

foreach(header ${ascendc_api_cube_src})
    get_filename_component(name_part ${header} NAME_WLE)
    set(str_header "${name_part}_str.h")
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND cat ${header}
                | sed "'1i\\R\"===('"
                | sed "'$$a\\)===\"'"
                > ${CMAKE_CURRENT_BINARY_DIR}/${str_header}
        DEPENDS ${header}
    )
    list(APPEND generated_cube_files ${CMAKE_CURRENT_BINARY_DIR}/${str_header})
endforeach()

add_custom_target(generated_cube_files ALL DEPENDS ${generated_cube_files})

add_library(ascendc_api_cube_extend INTERFACE)
target_include_directories(ascendc_api_cube_extend INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
add_dependencies(ascendc_api_cube_extend generated_cube_files)
