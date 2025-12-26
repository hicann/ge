set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
add_subdirectory(att)
add_subdirectory(ascendc)
add_subdirectory(codegen)
add_subdirectory(py_module)
add_subdirectory(autofuse)
add_subdirectory(e2e)
add_subdirectory(common)
add_subdirectory(ascir)
find_program(PYTEST_EXECUTABLE pytest)

if(PYTEST_EXECUTABLE)
  message(STATUS "Found pytest: ${PYTEST_EXECUTABLE}")
  add_subdirectory(python)
else()
  message(STATUS "pytest not found")
endif()
add_subdirectory(optimize)

add_executable(test_main test_main.cpp)
target_link_libraries(test_main PRIVATE
    # libpem_davinci.so 本来应该有test_e2e对应款型的测试引入
    # 在修改了att编译后引入找不到的问题，当前先手动增加规避
    ${ASCEND_ROOT}/tools/simulator/Ascend910B1/lib/libpem_davinci.so
)

target_link_libraries(test_main PRIVATE
    atrace
    error_manager
    aihac_symbolizer
    test_ascir_ut
    test_ascendc
    test_codegen
    test_py_module
    test_e2e
    runtime
    metadef
    GTest::gtest
)
add_dependencies(test_main ascir_ops_headers)
gtest_discover_tests(test_main)

add_executable(test_main_regbase test_main.cpp)
target_link_libraries(test_main_regbase PRIVATE
    # libpem_davinci.so 本来应该有test_e2e对应款型的测试引入
    # 在修改了att编译后引入找不到的问题，当前先手动增加规避
    ${ASCEND_ROOT}/tools/simulator/Ascend910_9599/lib/libpem_davinci.so
)

target_link_libraries(test_main_regbase PRIVATE
    atrace
    error_manager
    aihac_symbolizer
    test_regbase_ascendc
    runtime
    metadef
    GTest::gtest
)
add_dependencies(test_main_regbase ascir_ops_headers)
gtest_discover_tests(test_main_regbase)