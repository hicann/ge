add_library(test_regbase_ascendc OBJECT
    ./test_floor_div.cpp
    ./test_cast.cpp
    ./test_compare.cpp
    ./test_where.cpp
    ./test_reduce_init.cpp
    )
target_link_libraries(test_regbase_ascendc
    ascendc_api_regbase_extend
    tikicpulib_ascend910_9599
    depends
    GTest::gtest)
target_include_directories(test_regbase_ascendc PUBLIC
    ${CODE_ROOT_DIR}/ascendc
)
