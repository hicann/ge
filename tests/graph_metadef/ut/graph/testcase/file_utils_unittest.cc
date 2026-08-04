/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include "graph_metadef/graph/utils/file_utils.h"
#include <gtest/gtest.h>
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {
class UtestFileUtils : public testing::Test {
 public:
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestFileUtils, RealPathIsNull) {
  const char *path = nullptr;
  std::string res;
  res = ge::RealPath(path);
  EXPECT_EQ(res, "");
}

TEST_F(UtestFileUtils, RealPathIsNotExist) {
  const char *path = "D:/UTTest/aabbccddaaddbcasdaj.txt";
  std::string res;
  res = ge::RealPath(path);
  EXPECT_EQ(res, "");
}

TEST_F(UtestFileUtils, CreateDirPathIsNull) {
  std::string directory_path;
  int32_t ret = ge::CreateDir(directory_path);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestFileUtils, CreateDirSuccess) {
  std::string directory_path = "D:\\123\\456";
  int32_t ret = ge::CreateDir(directory_path);
  EXPECT_EQ(ret, 0);
  int delete_ret = remove(directory_path.c_str());
  EXPECT_EQ(delete_ret, 0);
}

TEST_F(UtestFileUtils, CreateDirPathIsGreaterThanMaxPath) {
  std::string directory_path;
  for (int i = 0; i < 4000; i++) {
    directory_path.append(std::to_string(i));
  }
  int ret = 0;
  ret = ge::CreateDir(directory_path);
  EXPECT_EQ(ret, -1);
}

TEST_F(UtestFileUtils, RealPath) {
  ASSERT_EQ(ge::RealPath(nullptr), "");
}

TEST_F(UtestFileUtils, CreateDir) {
  ASSERT_EQ(ge::CreateDir("~/test"), 0);
}

TEST_F(UtestFileUtils, GetBinFileFromFileSuccess) {
  std::string so_bin = "./opsptoro.so";
  system(("touch " + so_bin).c_str());
  system(("echo '123' > " + so_bin).c_str());
  uint32_t data_len;
  std::unique_ptr<char_t[]> so_data = GetBinFromFile(so_bin, data_len);
  ASSERT_NE(so_data, nullptr);
  ASSERT_EQ(data_len, 4);
  ASSERT_EQ(so_data.get()[0], '1');
  ASSERT_EQ(so_data.get()[1], '2');
  ASSERT_EQ(so_data.get()[2], '3');

  system(("rm -f " + so_bin).c_str());
}

TEST_F(UtestFileUtils, GetBinFileFromFileSuccess_offset) {
  std::string so_bin = "./opsptoro.so";
  system(("touch " + so_bin).c_str());
  system(("echo '123' > " + so_bin).c_str());
  size_t data_len = 4;
  size_t offset = 0;
  std::unique_ptr<ge::char_t[]> so_data = GetBinFromFile(so_bin, offset, data_len);
  ASSERT_NE(so_data, nullptr);
  ASSERT_EQ(data_len, 4);
  ASSERT_EQ(so_data.get()[0], '1');
  ASSERT_EQ(so_data.get()[1], '2');
  ASSERT_EQ(so_data.get()[2], '3');

  ASSERT_EQ(GetBinFromFile(so_bin, static_cast<ge::char_t *>(so_data.get()), data_len), GRAPH_SUCCESS);
  ASSERT_NE(so_data, nullptr);
  ASSERT_EQ(data_len, 4);
  ASSERT_EQ(so_data.get()[0], '1');
  ASSERT_EQ(so_data.get()[1], '2');
  ASSERT_EQ(so_data.get()[2], '3');
  system(("rm -f " + so_bin).c_str());
}

TEST_F(UtestFileUtils, GetBinFilePathNullFail) {
  std::string so_bin = "";
  uint32_t data_len;
  std::unique_ptr<char_t[]> so_data = GetBinFromFile(so_bin, data_len);
  ASSERT_EQ(so_data, nullptr);
}

TEST_F(UtestFileUtils, GetBinFileOpenPathFail) {
  std::string so_bin = "./opsptoro.so";
  uint32_t data_len;
  ASSERT_EQ(GetBinFromFile(so_bin, data_len), nullptr);
}

TEST_F(UtestFileUtils, WriteBinToFileSuccess) {
  std::string so_bin = "./opsptoro.so";
  uint32_t data_len = 4;
  char so_data[4] = {'1', '2', '3'};
  ASSERT_EQ(WriteBinToFile(so_bin, so_data, data_len), GRAPH_SUCCESS);
  ASSERT_EQ(SaveBinToFile(so_data, data_len, so_bin), GRAPH_SUCCESS);
  system(("rm -f " + so_bin).c_str());
}

TEST_F(UtestFileUtils, WriteBinToFile_OK_FilePathNoDirName) {
  std::string file_name = "file_name_without_dir_prefix.txt";
  uint32_t data_len = 4;
  char so_data[4] = {'1', '2', '3'};
  ASSERT_EQ(WriteBinToFile(file_name, so_data, data_len), GRAPH_SUCCESS);
  ASSERT_EQ(SaveBinToFile(so_data, data_len, file_name), GRAPH_SUCCESS);
  system(("rm -f " + file_name).c_str());
}

TEST_F(UtestFileUtils, WriteBinToFilePathNullFail) {
  std::string so_bin = "";
  uint32_t data_len = 4;
  char so_data[4] = {'1', '2', '3'};
  ASSERT_EQ(WriteBinToFile(so_bin, so_data, data_len), PARAM_INVALID);
}

TEST_F(UtestFileUtils, GetSanitizedNameCase0) {
  std::string file_name = "ge_proto_a/b\\c";
  ASSERT_EQ(GetRegulatedName(file_name), "ge_proto_a_b_c");
}

TEST_F(UtestFileUtils, WriteBinToFileFdNullData) {
  ASSERT_EQ(WriteBinToFile(1, nullptr, 10), GRAPH_FAILED);
}

TEST_F(UtestFileUtils, WriteBinToFileFdZeroLen) {
  char data[4] = {'1', '2', '3'};
  ASSERT_EQ(WriteBinToFile(1, reinterpret_cast<char_t *>(data), 0), GRAPH_FAILED);
}

TEST_F(UtestFileUtils, SaveBinToFileNullData) {
  ASSERT_EQ(SaveBinToFile(nullptr, 10, "./test_file_for_ut.bin"), GRAPH_FAILED);
}

TEST_F(UtestFileUtils, SaveBinToFileZeroLen) {
  char data[4] = {'1', '2', '3'};
  ASSERT_EQ(SaveBinToFile(data, 0, "./test_file_for_ut.bin"), GRAPH_FAILED);
}

TEST_F(UtestFileUtils, GetBinDataFromFileNotOpen) {
  uint32_t data_len = 0;
  ASSERT_EQ(GetBinDataFromFile("./nonexistent_file_for_ut.bin", data_len), nullptr);
}

TEST_F(UtestFileUtils, GetBinFromFileBufferNotOpen) {
  std::string dir_path = "./test_dir_for_ut_buffernotopen";
  system(("mkdir -p " + dir_path).c_str());
  size_t data_len = 10;
  char buffer[10];
  ASSERT_EQ(GetBinFromFile(dir_path, buffer, data_len), GRAPH_FAILED);
  system(("rmdir " + dir_path).c_str());
}

TEST_F(UtestFileUtils, WriteBinToFileOpenFail) {
  std::string dir_path = "./test_dir_for_ut_openfail";
  system(("mkdir -p " + dir_path).c_str());
  std::string subdir = dir_path + "/subdir";
  system(("mkdir -p " + subdir).c_str());
  uint32_t data_len = 4;
  char data[4] = {'1', '2', '3'};
  ASSERT_EQ(WriteBinToFile(subdir, data, data_len), GRAPH_FAILED);
  system(("rmdir " + subdir).c_str());
  system(("rmdir " + dir_path).c_str());
}

TEST_F(UtestFileUtils, CreateDirFailUnderFile) {
  std::string file_path = "./test_file_for_ut_createdir";
  system(("touch " + file_path).c_str());
  std::string dir_path = file_path + "/subdir";
  int32_t ret = ge::CreateDir(dir_path);
  EXPECT_NE(ret, 0);
  system(("rm -f " + file_path).c_str());
}

TEST_F(UtestFileUtils, ScandirInvalidPath) {
  mmDirent **entry_list = nullptr;
  int32_t count = Scandir("/nonexistent_path_for_ut_12345", &entry_list, nullptr, nullptr);
  EXPECT_LT(count, 0);
}

TEST_F(UtestFileUtils, GetAscendWorkPathInvalid) {
  setenv("ASCEND_WORK_PATH", "/dev/null/invalid_path_for_ut", 1);
  std::string work_path;
  Status ret = GetAscendWorkPath(work_path);
  EXPECT_EQ(ret, FAILED);
  unsetenv("ASCEND_WORK_PATH");
}

TEST_F(UtestFileUtils, GetAscendWorkPathNotSet) {
  unsetenv("ASCEND_WORK_PATH");
  std::string work_path;
  Status ret = GetAscendWorkPath(work_path);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(work_path, "");
}

TEST_F(UtestFileUtils, CreateDirEExist) {
  std::string dir_path = "./test_dir_for_ut_eexist";
  system(("mkdir -p " + dir_path).c_str());
  int32_t ret = ge::CreateDir(dir_path);
  EXPECT_EQ(ret, 0);
  system(("rmdir " + dir_path).c_str());
}

TEST_F(UtestFileUtils, GetSanitizedNameTest) {
  std::string input = "file:/name\\with*bad?chars";
  std::string result = GetSanitizedName(input);
  EXPECT_EQ(result.find('/'), std::string::npos);
  EXPECT_EQ(result.find('\\'), std::string::npos);
  EXPECT_EQ(result.find(':'), std::string::npos);
  EXPECT_EQ(result.find('*'), std::string::npos);
  EXPECT_EQ(result.find('?'), std::string::npos);
}

TEST_F(UtestFileUtils, SplitFilePathTest) {
  std::string dir_path;
  std::string file_name;
  SplitFilePath("/a/b/c.txt", dir_path, file_name);
  EXPECT_EQ(dir_path, "/a/b");
  EXPECT_EQ(file_name, "c.txt");

  dir_path.clear();
  file_name.clear();
  SplitFilePath("filename_only", dir_path, file_name);
  EXPECT_EQ(dir_path, "");
  EXPECT_EQ(file_name, "filename_only");

  dir_path.clear();
  file_name.clear();
  SplitFilePath("", dir_path, file_name);
  EXPECT_EQ(dir_path, "");
  EXPECT_EQ(file_name, "");
}

TEST_F(UtestFileUtils, CreateDirectoryTest) {
  std::string dir_path = "./test_dir_for_ut_createdir_func";
  int32_t ret = ge::CreateDirectory(dir_path);
  EXPECT_EQ(ret, 0);
  system(("rmdir " + dir_path).c_str());
}
}  // namespace ge
