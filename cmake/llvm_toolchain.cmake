# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER clang CACHE PATH "llvm compiler")
set(CMAKE_CXX_COMPILER clang++ CACHE PATH "GCC CXX32 compiler")
set(CMAKE_AS llvm-as CACHE STRING "Cross AS")
set(CMAKE_LINKER ld.lld CACHE STRING "Cross linker")
set(CMAKE_CPP clang-cpp CACHE STRING "Cross CPP")
set(CMAKE_AR llvm-ar)
set(CMAKE_NM llvm-nm)
set(CMAKE_OBJCOPY llvm-objcopy CACHE STRING "Cross OBJCOPY")
set(CMAKE_READELF llvm-readelf CACHE STRING "Cross READELF")
set(CMAKE_STRIP llvm-strip CACHE STRING "Cross STRIP")
set(CMAKE_RANLIB llvm-ranlib)

set(CMAKE_SKIP_BUILD_RPATH TRUE CACHE BOOL "" FORCE)
set(CMAKE_C_CREATE_SHARED_LIBRARY "<CMAKE_LINKER> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")

set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_C_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")