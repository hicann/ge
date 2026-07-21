/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include "macro_utils/dt_public_scope.h"
#include "common/debug/memory_dumper.h"
#include "common/dump/exception_dumper.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestMemoryDumper : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestMemoryDumper, OpenFile_success) {
  MemoryDumper memory_dumper;
  const std::string dump_file_path = "./test_file";
  EXPECT_NE(MemoryDumper::OpenFile(dump_file_path), -1);
}

TEST_F(UtestMemoryDumper, OpenLongFile_success) {
  MemoryDumper memory_dumper;
  std::string dump_file_path = "./test_file_";
  std::string back_end(560, 'a');
  dump_file_path += back_end;
  EXPECT_NE(MemoryDumper::OpenFile(dump_file_path), -1);
}

TEST_F(UtestMemoryDumper, DumpToFile_EmptyData) {
  const char *filename = "./empty_dump_test";
  const uint8_t data[] = {0x01, 0x02};
  EXPECT_NE(MemoryDumper::DumpToFile(filename, data, 0U), SUCCESS);
}

TEST_F(UtestMemoryDumper, DumpToFile_ValidData) {
  const char *filename = "./dump_test_file";
  const uint8_t data[] = {0x01, 0x02, 0x03, 0x04};
  EXPECT_EQ(MemoryDumper::DumpToFile(filename, data, sizeof(data)), SUCCESS);
  (void)remove(filename);
}
}  // namespace ge
