include(${GE_METADEF_DIR}/graph/ascendc_ir/generator/generator.cmake)

add_library(ascir_builtin_ops SHARED
        v1/ascir_builtin_ops.cpp
        v2/ascir_builtin_ops.cpp
        ../reg_func/v1/pow.cpp
        ../reg_func/v1/pad.cpp
        ../reg_func/v1/rsqrt.cpp
        ../reg_func/v1/where.cpp
        ../reg_func/v1/broadcast.cpp
        ../reg_func/v1/clipbyvalue.cpp
        ../reg_func/v1/compare.cpp
        ../reg_func/v1/concat.cpp
        ../reg_func/v1/div.cpp
        ../reg_func/v1/gather.cpp
        ../reg_func/v1/logical_and.cpp
        ../reg_func/v1/logical_not.cpp
        ../reg_func/v1/logical_or.cpp
        ../reg_func/v1/sigmoid.cpp
        ../reg_func/v1/sign.cpp
        ../reg_func/v1/sub.cpp
        ../reg_func/v1/truediv.cpp
        ../reg_func/v1/isnan.cpp
        ../reg_func/v1/tanh.cpp
        ../reg_func/v1/zerosLike.cpp
        ../reg_func/v1/reduce.cpp
        ../reg_func/v1/cast.cpp
        ../reg_func/v1/axpy.cpp
        ../reg_func/v1/isfinite.cpp
        ../reg_func/v1/default_reg_func.cpp
        ../reg_func/v1/erf.cpp
        ../reg_func/v2/reduce_v2.cpp
        ../reg_func/v2/erf_v2.cpp
        ../reg_func/v2/gelu_v2.cpp
        ../reg_func/v2/tanh_v2.cpp
        ../reg_func/v2/gather_v2.cpp
	    ../reg_func/v1/split.cpp
        ../reg_func/v2/pow_v2.cpp
        )

target_link_options(ascir_builtin_ops PRIVATE
        -rdynamic
        -Wl,-Bsymbolic
        -Wl,--exclude-libs,All
        )

target_link_libraries(ascir_builtin_ops PRIVATE aihac_ir_register ascend_protobuf)

target_include_directories(ascir_builtin_ops PRIVATE
        ../reg_func/v1/default_reg_func.h
        ../reg_func/v2/default_reg_func_v2.h
        ${GE_METADEF_INC_DIR}
        ${GE_METADEF_INC_DIR}/graph
        ${GE_METADEF_INC_DIR}/graph/ascendc_ir
        ${GE_METADEF_INC_DIR}/external
        ${GE_METADEF_DIR}/graph/ascendc_ir/generator
        )

set(ops_header_dir ${CMAKE_BINARY_DIR}/ascir_builtin_ops)
set_target_properties(ascir_ops_header_generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ops_header_dir})
message("ops_header_dir is " ${ops_header_dir}/ascir_ops.h)
ascir_generate(ascir_builtin_ops ${ops_header_dir} ${CMAKE_CURRENT_BINARY_DIR}/libascir_builtin_ops.so ${ops_header_dir}/ascir_ops.h)

add_custom_target(ascir_builtin_ops_header ALL DEPENDS ${ops_header_dir}/ascir_ops.h)
add_library(ascir_ops_headers INTERFACE)
target_include_directories(ascir_ops_headers INTERFACE ${ops_header_dir})
add_dependencies(ascir_ops_headers ascir_builtin_ops_header)

################################################################################
target_sources(aihac_codegen PRIVATE
    v1/ascir_builtin_ops.cpp
    v2/ascir_builtin_ops.cpp
    ../reg_func/v1/pow.cpp
    ../reg_func/v1/pad.cpp
    ../reg_func/v1/rsqrt.cpp
    ../reg_func/v1/where.cpp
    ../reg_func/v1/broadcast.cpp
    ../reg_func/v1/clipbyvalue.cpp
    ../reg_func/v1/compare.cpp
    ../reg_func/v1/concat.cpp
    ../reg_func/v1/div.cpp
    ../reg_func/v1/gather.cpp
    ../reg_func/v1/logical_and.cpp
    ../reg_func/v1/logical_not.cpp
    ../reg_func/v1/logical_or.cpp
    ../reg_func/v1/sigmoid.cpp
    ../reg_func/v1/sign.cpp
    ../reg_func/v1/sub.cpp
    ../reg_func/v1/truediv.cpp
    ../reg_func/v1/isnan.cpp
    ../reg_func/v1/tanh.cpp
    ../reg_func/v1/zerosLike.cpp
    ../reg_func/v1/reduce.cpp
    ../reg_func/v1/cast.cpp
    ../reg_func/v1/axpy.cpp
    ../reg_func/v1/isfinite.cpp
    ../reg_func/v1/default_reg_func.cpp
    ../reg_func/v1/erf.cpp
    ../reg_func/v2/reduce_v2.cpp
    ../reg_func/v2/erf_v2.cpp
    ../reg_func/v2/gelu_v2.cpp
    ../reg_func/v2/tanh_v2.cpp
    ../reg_func/v1/split.cpp
    ../reg_func/v2/gather_v2.cpp
    ../reg_func/v2/pow_v2.cpp
)

target_include_directories(aihac_codegen PRIVATE
    ${GE_METADEF_INC_DIR}
    ${GE_METADEF_INC_DIR}/graph
    ${GE_METADEF_INC_DIR}/graph/ascendc_ir
    ${GE_METADEF_INC_DIR}/external
    ${GE_METADEF_DIR}/graph/ascendc_ir/generator
    ${ops_header_dir}
)
