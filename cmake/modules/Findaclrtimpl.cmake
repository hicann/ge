# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (aclrtimpl_FOUND)
    message(STATUS "Package aclrtimpl has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS acl_rt_impl aclrtimpl_headers)
    list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
    if(TARGET "${_cmake_expected_target}")
        list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
    else()
        list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
    endif()
endforeach()
unset(_cmake_expected_target)

if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
    unset(_cmake_targets_defined)
    unset(_cmake_targets_not_defined)
    unset(_cmake_expected_targets)
    unset(CMAKE_IMPORT_FILE_VERSION)
    cmake_policy(POP)
    return()
endif()

if(NOT _cmake_targets_defined STREQUAL "")
    string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
    string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
    message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)

find_path(aclrtimpl_INCLUDE_DIR
    NAMES acl/acl_rt.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_library(acl_rt_impl_SHARED_LIBRARY
    NAMES libacl_rt_impl.so
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(aclrtimpl
    FOUND_VAR
        aclrtimpl_FOUND
    REQUIRED_VARS
        aclrtimpl_INCLUDE_DIR
        acl_rt_impl_SHARED_LIBRARY
)

if(aclrtimpl_FOUND)
    include(CMakePrintHelpers)
    message(STATUS "Variables in aclrtimpl module:")
    cmake_print_variables(aclrtimpl_INCLUDE_DIR)
    cmake_print_variables(acl_rt_impl_SHARED_LIBRARY)

    add_library(acl_rt_impl SHARED IMPORTED)
    set_target_properties(acl_rt_impl PROPERTIES
        INTERFACE_LINK_LIBRARIES "aclrtimpl_headers"
        IMPORTED_LOCATION "${acl_rt_impl_SHARED_LIBRARY}"
    )

    add_library(aclrtimpl_headers INTERFACE IMPORTED)
    set_target_properties(aclrtimpl_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${aclrtimpl_INCLUDE_DIR};${aclrtimpl_INCLUDE_DIR}/acl;${BASE_DIR}/api/acl/include_other_module;${BASE_DIR}/api/acl/include_other_module/acl"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS acl_rt_impl
        PROPERTIES INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS aclrtimpl_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()
