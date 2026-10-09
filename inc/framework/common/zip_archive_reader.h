/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_ZIP_ARCHIVE_READER_H_
#define INC_FRAMEWORK_COMMON_ZIP_ARCHIVE_READER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "minizip/unzip.h"
#include "framework/om2/model_data/gert_model_data.h"

namespace gert {
struct MemoryFileReadonly {
  const uint8_t *buffer;  // Read-only buffer managed by caller.
  uint64_t length;        // Actual read-only buffer content length.
  uint64_t position;      // Current position.
};

class ZipArchiveReader {
 public:
  /**
   * Constructs a ZipArchiveReader object using a ZIP archive that already
   * resides in memory.
   * @param data Pointer to the beginning of the ZIP data in memory.
   * @param length Size of the ZIP data in bytes.
   */
  ZipArchiveReader(const uint8_t *data, const size_t length);
  ~ZipArchiveReader();
  bool IsGood() const {
    return (zip_handle_ != nullptr) && entry_cache_ready_;
  }
  /**
   * Lists all regular files in the ZIP archive, excluding directories.
   * @return Vector containing relative paths of all files in the archive.
   */
  std::vector<std::string> ListFiles() const;
  ge::ReadonlyByteBuffer ExtractToMem(const std::string &entry_name, size_t &buff_size) const;
  /**
   * Checks if an entry exists in the ZIP archive.
   * @param entry_name Filename (relative path) within the ZIP archive.
   * @return true if the entry exists, false otherwise.
   */
  bool HasEntry(const std::string &entry_name) const;
  /**
   * Finds the full entry name by path relative to the archive root directory.
   * @param relative_path Path without the archive root prefix (e.g. "data/model_0/model_meta.json").
   * @return Full entry name, empty string if not found.
   */
  std::string FindEntry(const std::string &relative_path) const;
  bool HasEntryByRelativePath(const std::string &relative_path) const;
  /**
   * Lists all entries whose path relative to the archive root starts with the prefix.
   * @param relative_prefix Prefix relative to the archive root (e.g. "data/model_0/runtime/").
   * @return Vector of full entry names.
   */
  std::vector<std::string> ListFilesByRelativePrefix(const std::string &relative_prefix) const;

 private:
  struct CachedZipEntry {
    unz64_file_pos file_pos;
    uint64_t uncompressed_size = 0U;
    uint64_t raw_data_offset = 0U;
    // 仅未压缩 entry 的 raw_data_offset 可直接用于零拷贝读取。
    bool raw_data_ready = false;
  };

  bool BuildEntryCache();
  bool CacheCurrentEntry(const std::string &entry_name, const unz_file_info64 &file_info);
  bool GoToEntry(const std::string &entry_name) const;
  bool GetCachedRawData(const std::string &entry_name, size_t &buff_size, ge::ReadonlyByteBuffer &raw_data) const;
  bool GetRawDataOffset(const size_t pos_in_central_dir, const size_t buff_size, uint64_t &raw_data_offset) const;

 private:
  MemoryFileReadonly mem_file_{};
  unzFile zip_handle_ = nullptr;
  std::unordered_map<std::string, CachedZipEntry> entry_cache_;
  std::unordered_map<std::string, std::string> relative_entry_cache_;
  std::vector<std::string> entry_names_;
  bool entry_cache_ready_ = false;
};
}  // namespace gert

#endif  // INC_FRAMEWORK_COMMON_ZIP_ARCHIVE_READER_H_
