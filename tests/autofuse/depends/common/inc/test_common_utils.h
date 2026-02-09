/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TESTS_AUTOFUSE_DEPENDS_COMMON_TEST_COMMON_UTILS_H_
#define TESTS_AUTOFUSE_DEPENDS_COMMON_TEST_COMMON_UTILS_H_

#include <cstdlib>

namespace autofuse {
namespace test {

// 清理测试用例生成的临时文件和stub拷贝
inline void CleanupTestArtifacts() {
  // 删除stub相关目录和文件
  system("rm -rf ./stub ./tiling ./register ./graph ./lib ./kernel_tiling");
  // 删除公共文件
  system("rm -f ./op_log.h ./autofuse_tiling_func_common.h");
  // 删除日志文件
  system("rm -f *.log");
  // 删除生成的二进制文件
  system("rm -f ./tiling_func_main ./tiling_func_main_concat ./tiling_func_main_transpose ./tiling_func_main_softmax");
  // 删除生成的tiling data和func文件
  system("rm -f ./*_tiling_data.h ./*_tiling_func.cpp ./tiling_func_main_*.cpp");
}

}  // namespace test
}  // namespace autofuse

#endif  // TESTS_AUTOFUSE_DEPENDS_COMMON_TEST_COMMON_UTILS_H_
