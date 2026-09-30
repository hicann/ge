/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_utils.h"
#include "framework/common/zip_archive_reader.h"
#include "framework/common/zip_archive_writer.h"
#include "ge/ge_ir_build.h"
#include <gtest/gtest.h>
#include <fstream>
#include <unordered_set>
#include "common/env_path.h"
#include "minizip/zip.h"
#include "mmpa/mmpa_api.h"

namespace ge {

class ZipArchiveUt : public ::testing::Test {
 public:
  void SetUp() override {
    const ::testing::TestInfo *test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    test_case_name = test_info->test_case_name();  // ZipArchiveUt
    test_work_dir = EnvPath().GetOrCreateCaseTmpPath(test_case_name);
  }
  void TearDown() override {
    EnvPath().RemoveRfCaseTmpPath(test_case_name);
  }
  static void CreateTestZipArchive(const std::string &archive_path,
                                   const std::vector<std::pair<std::string, std::string>> &entries,
                                   const bool compress = true) {
    zipFile zf = zipOpen64(archive_path.c_str(), 0);
    ASSERT_NE(zf, nullptr);
    int32_t compress_flag = compress ? Z_DEFAULT_COMPRESSION : Z_NO_COMPRESSION;
    int32_t method = compress ? Z_DEFLATED : Z_BINARY;

    for (const auto &entry : entries) {
      const std::string &file_name = entry.first;
      const std::string &content = entry.second;

      zip_fileinfo zi;
      memset_s(&zi, sizeof(zi), 0, sizeof(zi));

      auto ret =
          zipOpenNewFileInZip64(zf, file_name.c_str(), &zi, nullptr, 0, nullptr, 0, nullptr, method, compress_flag, 1);

      if (ret != ZIP_OK) {
        (void)zipClose(zf, nullptr);
        return;
      }

      if (!content.empty()) {
        ret = zipWriteInFileInZip(zf, content.data(), static_cast<unsigned>(content.size()));
        if (ret != ZIP_OK) {
          (void)zipCloseFileInZip(zf);
          (void)zipClose(zf, nullptr);
          return;
        }
      }

      (void)zipCloseFileInZip(zf);
    }

    (void)zipClose(zf, nullptr);
  }

  static std::vector<uint8_t> ReadFileToVector(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
      return {};
    }
    file.seekg(0, std::ios::end);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size < 0) {
      return {};
    }
    std::vector<uint8_t> buffer(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size)) {
      return {};
    }

    return buffer;
  }

  static bool WriteFileToZip(gert::ZipArchiveWriter &writer, const std::string &entry, const std::string &path,
                             bool compress = true) {
    auto data = ReadFileToVector(path);
    return writer.WriteBytes(entry, data.data(), data.size(), compress);
  }

  std::string CreateTempFile(const std::string &file_name, const size_t file_size = 100) {
    const std::string full_path = PathUtils::Join({test_work_dir, file_name});
    std::ofstream ofs(full_path, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
      return {};
    }

    constexpr size_t kBlockSize = 4096;
    const std::string block(kBlockSize, 'a');
    size_t written = 0;
    while (written < file_size) {
      const size_t to_write = std::min(kBlockSize, file_size - written);
      ofs.write(block.data(), to_write);
      written += to_write;
    }
    ofs.close();
    return full_path;
  }

  void CheckExtractedFiles(const std::string &zipfile_path,
                           const std::unordered_set<std::string> &expected_entries) const {
    const auto file_buf = ReadFileToVector(zipfile_path);
    gert::ZipArchiveReader unzip_file(file_buf.data(), file_buf.size());
    ASSERT_TRUE(unzip_file.IsGood());
    const auto file_names = unzip_file.ListFiles();
    ASSERT_EQ(expected_entries.size(), file_names.size());
    for (const auto &entry : expected_entries) {
      size_t buff_size = 0UL;
      const auto buff_data = unzip_file.ExtractToMem(PathUtils::Join({kZipFileBaseName, entry}), buff_size);
      ASSERT_NE(buff_data, nullptr);
    }
  }

 public:
  std::string test_case_name;
  std::string test_work_dir;
  const std::string kZipFileBaseName = "fake_test";
};

TEST_F(ZipArchiveUt, TestZipArchiveReader_Fail_InvalidFileOrData) {
  gert::ZipArchiveReader unzip_file_invalid_data(nullptr, 0);
  EXPECT_EQ(unzip_file_invalid_data.IsGood(), false);
}

TEST_F(ZipArchiveUt, TestZipArchiveReader_Ok_DecompressArchive) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"example/demo1.txt", "Hello from demo1!\nThis is example."},
      {"doc/doc1.txt", "Document 1 inside zip.\n-- EOF --"},
  };
  CreateTestZipArchive(archive_path, entries);
  // 测试从内存读取与解压功能
  {
    const auto zip_file_buf = ReadFileToVector(archive_path);
    gert::ZipArchiveReader archive(zip_file_buf.data(), zip_file_buf.size());
    EXPECT_EQ(archive.IsGood(), true);
    const auto file_names = archive.ListFiles();
    ASSERT_EQ(file_names.size(), 2);
    for (const auto &file_name : file_names) {
      size_t buff_size = 0UL;
      const auto buff_data = archive.ExtractToMem(file_name, buff_size);
      ASSERT_NE(buff_data, nullptr);
    }
  }
}

TEST_F(ZipArchiveUt, TestZipArchiveReader_Ok_ExtractToMem) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  std::string data_str1 = "1234test_zip_archive";
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"example/demo1.txt", data_str1},
  };
  CreateTestZipArchive(archive_path, entries);
  // 测试从内存读取与解压功能
  {
    const auto file_buf = ReadFileToVector(archive_path);
    gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
    EXPECT_EQ(archive.IsGood(), true);
    const auto file_names = archive.ListFiles();
    ASSERT_EQ(file_names.size(), 1);
    for (const auto &file_name : file_names) {
      size_t buff_size = 0UL;
      const auto buff_data = archive.ExtractToMem(file_name, buff_size);
      ASSERT_NE(buff_data, nullptr);
      EXPECT_EQ(buff_size, data_str1.size());
      ASSERT_TRUE(std::memcmp(buff_data.get(), data_str1.data(), buff_size) == 0);
    }
  }
}

TEST_F(ZipArchiveUt, TestZipArchiveReader_Ok_ExtractToMemNoCompression) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  std::string data_str1(123456, 'c');
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"example1/demo1.txt", data_str1},
      {"example2/demo2.txt", data_str1},
  };
  CreateTestZipArchive(archive_path, entries, false);
  // 测试从内存读取与解压功能
  {
    const auto file_buf = ReadFileToVector(archive_path);
    gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
    EXPECT_EQ(archive.IsGood(), true);
    const auto file_names = archive.ListFiles();
    ASSERT_EQ(file_names.size(), 2);
    for (const auto &file_name : file_names) {
      size_t buff_size = 0UL;
      const auto buff_data = archive.ExtractToMem(file_name, buff_size);
      ASSERT_NE(buff_data, nullptr);
      EXPECT_EQ(buff_size, data_str1.size());
      ASSERT_TRUE(std::memcmp(buff_data.get(), data_str1.data(), buff_size) == 0);
      ASSERT_GE(buff_data.get(), file_buf.data());
      ASSERT_LT(buff_data.get(), file_buf.data() + file_buf.size());
    }
  }
}

TEST_F(ZipArchiveUt, TestZipArchiveReader_Ok_ExtractToMemEmptyNoCompressionAfterListFiles) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"empty.bin", ""},
  };
  CreateTestZipArchive(archive_path, entries, false);

  const auto file_buf = ReadFileToVector(archive_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 1);

  size_t buff_size = 1UL;
  const auto buff_data = archive.ExtractToMem(file_names[0], buff_size);
  ASSERT_NE(buff_data, nullptr);
  EXPECT_EQ(buff_size, 0U);
}

TEST_F(ZipArchiveUt, TestZipArchiveReader_Ok_ExtractManyNoCompressionEntriesAfterListFiles) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  constexpr size_t kEntryCount = 1500U;
  std::vector<std::pair<std::string, std::string>> entries;
  entries.reserve(kEntryCount);
  for (size_t i = 0U; i < kEntryCount; ++i) {
    entries.emplace_back("kernels/kernel_" + std::to_string(i) + ".o", "kernel_bin_payload_" + std::to_string(i));
  }
  CreateTestZipArchive(archive_path, entries, false);

  const auto file_buf = ReadFileToVector(archive_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), entries.size());

  for (size_t i = file_names.size(); i > 0U; --i) {
    const size_t index = i - 1U;
    size_t buff_size = 0UL;
    const auto buff_data = archive.ExtractToMem(file_names[index], buff_size);
    ASSERT_NE(buff_data, nullptr);
    ASSERT_EQ(buff_size, entries[index].second.size());
    ASSERT_EQ(std::memcmp(buff_data.get(), entries[index].second.data(), buff_size), 0);
  }
}

TEST_F(ZipArchiveUt, TestZipArchiveReader_Ok_ExtractToMemNoCompressionWithoutListFiles) {
  const std::string archive_path = PathUtils::Join({test_work_dir, "__test.zip"});
  const std::vector<std::pair<std::string, std::string>> entries = {
      {"kernels/kernel_0.o", "kernel_bin_payload_0"},
      {"kernels/kernel_1.o", "kernel_bin_payload_1"},
  };
  CreateTestZipArchive(archive_path, entries, false);

  const auto file_buf = ReadFileToVector(archive_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());

  size_t buff_size = 0UL;
  const auto buff_data = archive.ExtractToMem(entries[1].first, buff_size);
  ASSERT_NE(buff_data, nullptr);
  ASSERT_EQ(buff_size, entries[1].second.size());
  ASSERT_EQ(std::memcmp(buff_data.get(), entries[1].second.data(), buff_size), 0);
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_InvalidArchiveName) {
  gert::ZipArchiveWriter zip_writer("");
  EXPECT_FALSE(zip_writer.IsMemFileOpened());
  gert::ZipArchiveWriter zip_writer2(".");
  EXPECT_FALSE(zip_writer.IsMemFileOpened());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_InvaidFileOrDataBuffIsNull) {
  const auto zipfile_path = PathUtils::Join({test_work_dir, "invalid_case.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  ASSERT_FALSE(zip_writer.WriteBytes("data/fake_data.bin", nullptr, 123));
  ASSERT_FALSE(zip_writer.WriteBytes("data/fake_data.bin", zipfile_path.data(), 0));
  ASSERT_FALSE(zip_writer.WriteBytes("", zipfile_path.data(), 123));
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_StateAfterFinalization) {
  const auto zipfile_path = PathUtils::Join({test_work_dir, "invalid_case.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  EXPECT_TRUE(zip_writer.IsMemFileOpened());
  EXPECT_TRUE(zip_writer.SaveModelDataToFile());
  EXPECT_FALSE(zip_writer.IsMemFileOpened());
  auto file_data = ReadFileToVector(CreateTempFile("fake_test.txt"));
  EXPECT_FALSE(zip_writer.WriteBytes("test.txt", file_data.data(), file_data.size()));
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_WriteBytesAndFileSucc) {
  const std::string zipfile_name = kZipFileBaseName + ".zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  const std::string buffer = "123-abc-TestZipArchiveWriter_Ok_WriteBytesSucc";
  const std::string arc_name = "ok/file1.txt";
  const std::string file_path = CreateTempFile("fake_test.txt");
  const std::string arc_name2 = "ok/ok/file2.txt";

  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
  EXPECT_TRUE(WriteFileToZip(zip_writer, arc_name2, file_path));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());

  // 解压并校验内容
  CheckExtractedFiles(zipfile_path, {arc_name, arc_name2});
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_SaveModelDataToBuffer) {
  const std::string zipfile_name = kZipFileBaseName + ".zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  const std::string buffer = "123-abc-TestZipArchiveWriter_Ok_SaveModelDataToBuffer";
  const std::string arc_name = "ok/file1.txt";
  const std::string file_path = CreateTempFile("fake_test.txt");
  const std::string arc_name2 = "ok/ok/file2.txt";
  gert::GertBuffer model;

  {
    gert::ZipArchiveWriter zip_writer(zipfile_path);
    ASSERT_TRUE(zip_writer.IsMemFileOpened());
    EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
    EXPECT_TRUE(WriteFileToZip(zip_writer, arc_name2, file_path));
    ASSERT_TRUE(zip_writer.SaveModelData(model, false));
    ASSERT_FALSE(zip_writer.IsMemFileOpened());
  }

  ASSERT_NE(model.data, nullptr);
  ASSERT_GT(model.length, 0U);
  gert::ZipArchiveReader archive(model.data.get(), model.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  const std::unordered_set<std::string> expect_files = {
      PathUtils::Join({kZipFileBaseName, arc_name}),
      PathUtils::Join({kZipFileBaseName, arc_name2}),
  };
  ASSERT_EQ(file_names.size(), expect_files.size());
  for (const auto &file_name : file_names) {
    EXPECT_EQ(expect_files.count(file_name), 1U);
  }

  size_t extracted_size = 0UL;
  const auto extracted = archive.ExtractToMem(PathUtils::Join({kZipFileBaseName, arc_name}), extracted_size);
  ASSERT_NE(extracted, nullptr);
  EXPECT_EQ(extracted_size, buffer.size());
  EXPECT_EQ(std::memcmp(extracted.get(), buffer.data(), extracted_size), 0);
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_SaveModelDataToFileThroughUnifiedApi) {
  const std::string zipfile_name = kZipFileBaseName + "_unified.zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  const std::string buffer = "123-abc-TestZipArchiveWriter_Ok_SaveModelDataToFileThroughUnifiedApi";
  const std::string arc_name = "ok/file1.txt";
  gert::GertBuffer model;

  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
  ASSERT_TRUE(zip_writer.SaveModelData(model, true));
  ASSERT_FALSE(zip_writer.IsMemFileOpened());
  EXPECT_EQ(model.data, nullptr);
  EXPECT_EQ(mmAccess2(zipfile_path.c_str(), M_F_OK), EN_OK);

  const auto file_buf = ReadFileToVector(zipfile_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 1U);
  EXPECT_EQ(file_names[0], PathUtils::Join({kZipFileBaseName + "_unified", arc_name}));
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_RepeatedAddSameFile) {
  const std::string zipfile_name = kZipFileBaseName + ".zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  const std::string buffer = "123-abc-TestZipArchiveWriter_Ok_WriteBytesSucc";
  const std::string arc_name = "ok/file1.txt";
  const std::string file_path = CreateTempFile("fake_test.txt");
  const std::string arc_name2 = "ok/ok/file2.txt";

  // 重复添加相同文件
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
  EXPECT_TRUE(zip_writer.WriteBytes(arc_name, buffer.data(), buffer.size()));
  EXPECT_TRUE(WriteFileToZip(zip_writer, arc_name2, file_path));
  EXPECT_TRUE(WriteFileToZip(zip_writer, arc_name2, file_path));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());

  // 解压并校验内容
  CheckExtractedFiles(zipfile_path, {arc_name, arc_name2});
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_LargeDataWriteTriggersMemGrow) {
  const std::string zipfile_name = kZipFileBaseName + "_grow.zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());

  constexpr size_t kLargeDataSize = 128UL * 1024UL;
  std::vector<uint8_t> large_data(kLargeDataSize, 0xAB);
  EXPECT_TRUE(zip_writer.WriteBytes("large/data.bin", large_data.data(), large_data.size(), false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  const auto file_buf = ReadFileToVector(zipfile_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 1U);
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_CompressedWriteTriggersMemGrow) {
  const std::string zipfile_name = kZipFileBaseName + "_compressed_grow.zip";
  const auto zipfile_path = PathUtils::Join({test_work_dir, zipfile_name});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());

  std::string compressible_data(200000, 'X');
  EXPECT_TRUE(zip_writer.WriteBytes("compressed/data.txt", compressible_data.data(), compressible_data.size(), true));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_MultipleFilesWithDifferentCompression) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "mixed_compression.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());

  const std::string uncompressed_data = "uncompressed_data_12345";
  EXPECT_TRUE(zip_writer.WriteBytes("raw/data.bin", uncompressed_data.data(), uncompressed_data.size(), false));

  const std::string compressed_data(10000, 'Z');
  EXPECT_TRUE(zip_writer.WriteBytes("compressed/data.bin", compressed_data.data(), compressed_data.size(), true));

  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  const auto file_buf = ReadFileToVector(zipfile_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 2U);
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_WriteFileAfterClose) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "closed_writer.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());

  const std::string data = "test_data";
  EXPECT_FALSE(zip_writer.WriteBytes("test.txt", data.data(), data.size()));
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_SaveModelDataToBufferWithLargeData) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "buffer_large.zip"});
  gert::GertBuffer model;

  {
    gert::ZipArchiveWriter zip_writer(zipfile_path);
    ASSERT_TRUE(zip_writer.IsMemFileOpened());

    constexpr size_t kDataSize = 256UL * 1024UL;
    std::vector<uint8_t> data(kDataSize, 0xCD);
    EXPECT_TRUE(zip_writer.WriteBytes("large.bin", data.data(), data.size(), false));
    ASSERT_TRUE(zip_writer.SaveModelData(model, false));
  }

  ASSERT_NE(model.data, nullptr);
  ASSERT_GT(model.length, 0U);

  gert::ZipArchiveReader archive(model.data.get(), model.length);
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 1U);
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Fail_ArchivePathEndsWithSlash) {
  gert::ZipArchiveWriter zip_writer(test_work_dir + "/");
  EXPECT_FALSE(zip_writer.IsMemFileOpened());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_ArchivePathNoExtension) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "no_ext_archive"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const std::string buffer = "test_data_no_ext";
  EXPECT_TRUE(zip_writer.WriteBytes("data.txt", buffer.data(), buffer.size()));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_ArchivePathNoSlash) {
  const std::string zipfile_name = "no_slash.zip";
  gert::ZipArchiveWriter zip_writer(zipfile_name);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const std::string buffer = "test_data_no_slash";
  EXPECT_TRUE(zip_writer.WriteBytes("data.txt", buffer.data(), buffer.size()));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());
  (void)std::remove(zipfile_name.c_str());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_ArchivePathDotOnly) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, ".zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const std::string buffer = "test_data_dot";
  EXPECT_TRUE(zip_writer.WriteBytes("data.txt", buffer.data(), buffer.size()));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());
  ASSERT_FALSE(zip_writer.IsMemFileOpened());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_WriteBytesNoCompressionLargeData) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "large_nocompress.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());

  constexpr size_t kDataSize = 200UL * 1024UL;
  std::vector<uint8_t> data(kDataSize, 0x42);
  EXPECT_TRUE(zip_writer.WriteBytes("large_data.bin", data.data(), data.size(), false));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  const auto file_buf = ReadFileToVector(zipfile_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 1U);
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_MultipleEntriesWithGrowth) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "multi_growth.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());

  for (int i = 0; i < 10; ++i) {
    std::vector<uint8_t> data(32UL * 1024UL, static_cast<uint8_t>(i));
    EXPECT_TRUE(zip_writer.WriteBytes("entry_" + std::to_string(i) + ".bin", data.data(), data.size(), false));
  }
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  const auto file_buf = ReadFileToVector(zipfile_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 10U);
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_WriteEndOfFileTwice) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "double_close.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());
  const std::string buffer = "test_double_close";
  EXPECT_TRUE(zip_writer.WriteBytes("data.txt", buffer.data(), buffer.size()));
  EXPECT_TRUE(zip_writer.WriteEndOfFile());
  EXPECT_TRUE(zip_writer.WriteEndOfFile());
}

TEST_F(ZipArchiveUt, TestZipArchiveWriter_Ok_SaveModelDataToFileWithCompressedData) {
  const std::string zipfile_path = PathUtils::Join({test_work_dir, "compressed_save.zip"});
  gert::ZipArchiveWriter zip_writer(zipfile_path);
  ASSERT_TRUE(zip_writer.IsMemFileOpened());

  std::string data(50000, 'A');
  EXPECT_TRUE(zip_writer.WriteBytes("compressed.bin", data.data(), data.size(), true));
  ASSERT_TRUE(zip_writer.SaveModelDataToFile());

  const auto file_buf = ReadFileToVector(zipfile_path);
  gert::ZipArchiveReader archive(file_buf.data(), file_buf.size());
  ASSERT_TRUE(archive.IsGood());
  const auto file_names = archive.ListFiles();
  ASSERT_EQ(file_names.size(), 1U);
}

}  // namespace ge
