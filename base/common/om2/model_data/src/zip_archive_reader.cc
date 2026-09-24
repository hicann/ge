/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/zip_archive_reader.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "common/checker.h"
#include "common/debug/log.h"
#include "common/scope_guard.h"

#include "zip_archive_mem_file.h"

namespace gert {
namespace {
constexpr uint32_t kMaxFileNameLength = 4096U;
constexpr uint64_t kMaxCompressedEntriesUncompressedSize = 10ULL * 1024ULL * 1024ULL * 1024ULL;

voidpf ZCALLBACK MemOpenFileFuncReadonly(voidpf opaque, const void *filename, int mode) {
  (void)filename;
  (void)mode;
  const auto mem_file_readonly = static_cast<MemoryFileReadonly *>(opaque);
  if (mem_file_readonly == nullptr) {
    GELOGE(ge::FAILED, "[MEMZIP] Opaque pointer is null. Cannot initialize memory file.");
    return nullptr;
  }
  mem_file_readonly->position = 0;
  return mem_file_readonly;
}

uLong ZCALLBACK MemReadFileFuncReadonly(voidpf opaque, voidpf stream, void *buf, uLong size) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  (void)opaque;
  return MemReadFileImpl(mem_file, buf, size);
}

uLong ZCALLBACK MemWriteFileFuncReadonly(voidpf opaque, voidpf stream, const void *buf, uLong size) {
  (void)opaque;
  (void)stream;
  (void)buf;
  (void)size;
  return 0;
}

ZPOS64_T ZCALLBACK MemTell64FileFuncReadonly(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  (void)opaque;
  if (mem_file == nullptr) {
    return static_cast<ZPOS64_T>(-1);
  }
  return mem_file->position;
}

long ZCALLBACK MemSeek64FileFuncReadonly(voidpf opaque, voidpf stream, ZPOS64_T offset, int origin) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  (void)opaque;
  return MemSeek64FileImpl(mem_file, offset, origin);
}

int ZCALLBACK MemCloseFileFuncReadonly(voidpf opaque, voidpf stream) {
  (void)opaque;
  (void)stream;
  return kMemZipOk;
}

int ZCALLBACK MemErrorFileFuncReadonly(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFileReadonly *>(stream);
  (void)opaque;

  if (mem_file == nullptr) {
    return kMemZipError;
  }

  return kMemZipOk;
}

void FillMemFileFuncReadonly(zlib_filefunc64_def *file_func_def, MemoryFileReadonly *mem_file) {
  file_func_def->zopen64_file = MemOpenFileFuncReadonly;
  file_func_def->zread_file = MemReadFileFuncReadonly;
  file_func_def->zwrite_file = MemWriteFileFuncReadonly;
  file_func_def->ztell64_file = MemTell64FileFuncReadonly;
  file_func_def->zseek64_file = MemSeek64FileFuncReadonly;
  file_func_def->zclose_file = MemCloseFileFuncReadonly;
  file_func_def->zerror_file = MemErrorFileFuncReadonly;
  file_func_def->opaque = mem_file;
}

constexpr uint32_t kCdHeaderFixedSize = 46U;
constexpr uint32_t kCdCompressedSizeOffset = 20U;
constexpr uint32_t kCdUncompressedSizeOffset = 24U;
constexpr uint32_t kCdFileNameLenOffset = 28U;
constexpr uint32_t kCdExtraLenOffset = 30U;
constexpr uint32_t kCdLocalFileHeaderOffset = 42U;
constexpr uint32_t kCdHeaderMagicNum = 0x02014b50;
constexpr uint16_t kCdExtraMagicNum = 0x0001;
constexpr uint16_t kLfHeaderFixedSize = 30U;
constexpr uint32_t kLfNameLenOffset = 26U;
constexpr uint32_t kLfExtraLenOffset = 28U;
constexpr uint32_t kLfHeaderMagicNum = 0x04034b50;
constexpr size_t kBitsPerByte = 8;
constexpr size_t kBytesPerUint16 = sizeof(uint16_t);
constexpr size_t kBytesPerUint32 = sizeof(uint32_t);
constexpr size_t kBytesPerUint64 = sizeof(uint64_t);
constexpr size_t kBitsPerUint16 = kBitsPerByte * kBytesPerUint16;
constexpr size_t kBitsPerUint32 = kBitsPerByte * kBytesPerUint32;

struct ZipEntryInfo {
  uint64_t compressed_size;
  uint64_t uncompressed_size;
  uint64_t local_file_header_offset;
};

uint16_t ReadLE16(const uint8_t *p) {
  return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << kBitsPerByte);
}

uint32_t ReadLE32(const uint8_t *p) {
  return static_cast<uint32_t>(ReadLE16(p)) | (static_cast<uint32_t>(ReadLE16(p + kBytesPerUint16)) << kBitsPerUint16);
}

uint64_t ReadLE64(const uint8_t *p) {
  return static_cast<uint64_t>(ReadLE32(p)) | (static_cast<uint64_t>(ReadLE32(p + kBytesPerUint32)) << kBitsPerUint32);
}

bool ParseCentralDirEntry(const MemoryFileReadonly &buffer, const uint64_t pos_in_central_dir,
                          ZipEntryInfo &entry_info) {
  GE_ASSERT_TRUE(pos_in_central_dir + kCdHeaderFixedSize <= buffer.length, "Invalid central directory position");
  const uint8_t *entry_buff = buffer.buffer + pos_in_central_dir;

  const uint32_t magic_num = ReadLE32(entry_buff);
  GE_ASSERT_TRUE(magic_num == kCdHeaderMagicNum, "Invalid central directory file header, magic = %u", magic_num);

  entry_info.compressed_size = ReadLE32(entry_buff + kCdCompressedSizeOffset);
  entry_info.uncompressed_size = ReadLE32(entry_buff + kCdUncompressedSizeOffset);

  const uint16_t name_len = ReadLE16(entry_buff + kCdFileNameLenOffset);
  const uint16_t extra_len = ReadLE16(entry_buff + kCdExtraLenOffset);
  entry_info.local_file_header_offset = ReadLE32(entry_buff + kCdLocalFileHeaderOffset);

  if (entry_info.compressed_size == MAXU32 || entry_info.uncompressed_size == MAXU32 ||
      entry_info.local_file_header_offset == MAXU32) {
    GE_ASSERT_TRUE(pos_in_central_dir + kCdHeaderFixedSize + name_len + extra_len <= buffer.length,
                   "Invalid extra field position");

    const uint8_t *extra_buf = entry_buff + kCdHeaderFixedSize + name_len;
    size_t offset = 0;
    while (offset + kBytesPerUint32 <= extra_len) {
      const uint16_t header_id = ReadLE16(extra_buf + offset);
      const uint16_t data_size = ReadLE16(extra_buf + offset + kBytesPerUint16);
      offset += kBytesPerUint32;

      if (header_id == kCdExtraMagicNum) {
        if ((entry_info.uncompressed_size == MAXU32) && (offset + kBytesPerUint64 <= extra_len)) {
          entry_info.uncompressed_size = ReadLE64(extra_buf + offset);
          offset += kBytesPerUint64;
        }
        if ((entry_info.compressed_size == MAXU32) && (offset + kBytesPerUint64 <= extra_len)) {
          entry_info.compressed_size = ReadLE64(extra_buf + offset);
          offset += kBytesPerUint64;
        }
        if ((entry_info.local_file_header_offset == MAXU32) && (offset + kBytesPerUint64 <= extra_len)) {
          entry_info.local_file_header_offset = ReadLE64(extra_buf + offset);
          offset += kBytesPerUint64;
        }
        break;
      } else {
        offset += data_size;
      }
    }
  }
  return true;
}

bool LocateFileDataOffset(const MemoryFileReadonly &buffer, const uint64_t local_file_header_offset,
                          uint64_t &raw_data_offset) {
  GE_ASSERT_TRUE(local_file_header_offset + kLfHeaderFixedSize <= buffer.length, "Invalid local file header position");
  const uint8_t *header_buff = buffer.buffer + local_file_header_offset;

  const uint32_t header_magic_num = ReadLE32(header_buff);
  GE_ASSERT_TRUE(header_magic_num == kLfHeaderMagicNum, "Invalid local file header, magic = %u", header_magic_num);

  const int64_t name_len = ReadLE16(header_buff + kLfNameLenOffset);
  const int64_t extra_len = ReadLE16(header_buff + kLfExtraLenOffset);
  raw_data_offset = local_file_header_offset + kLfHeaderFixedSize + name_len + extra_len;
  GE_ASSERT_TRUE(raw_data_offset <= buffer.length, "Invalid file data offset");

  return true;
}
}  // namespace

ZipArchiveReader::ZipArchiveReader(const uint8_t *data, const size_t length) : mem_file_{data, length, 0} {
  if (mem_file_.buffer == nullptr || mem_file_.length == 0) {
    GELOGE(ge::FAILED, "Invalid zip archive data, data is [%p] and size is [%zu]", mem_file_.buffer, mem_file_.length);
    return;
  }

  zlib_filefunc64_def file_funcs;
  FillMemFileFuncReadonly(&file_funcs, &mem_file_);
  zip_handle_ = unzOpen2_64(nullptr, &file_funcs);
  if (zip_handle_ == nullptr) {
    GELOGE(ge::FAILED, "Failed to open ZIP file from memory");
    return;
  }

  if (!BuildEntryCache()) {
    entry_cache_.clear();
    entry_names_.clear();
    entry_cache_ready_ = false;
  }
}

ZipArchiveReader::~ZipArchiveReader() {
  if (zip_handle_ != nullptr) {
    unzClose(zip_handle_);
  }
}

std::vector<std::string> ZipArchiveReader::ListFiles() const {
  GE_ASSERT_TRUE(IsGood(), "Invalid status of archive");
  return entry_names_;
}

bool ZipArchiveReader::BuildEntryCache() {
  GE_ASSERT_NOTNULL(zip_handle_, "Invalid status of archive");

  entry_cache_.clear();
  relative_entry_cache_.clear();
  entry_names_.clear();
  entry_cache_ready_ = false;
  uint64_t total_uncompressed_size = 0U;
  auto uz_ret = unzGoToFirstFile(zip_handle_);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to go to the first file in the archive, ret = %d", uz_ret);

  while (uz_ret == UNZ_OK) {
    std::vector<char> name_buff(kMaxFileNameLength, '\0');
    unz_file_info64 file_info{};
    uz_ret =
        unzGetCurrentFileInfo64(zip_handle_, &file_info, name_buff.data(), name_buff.size(), nullptr, 0, nullptr, 0);
    GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to get the current file information, ret = %d", uz_ret);
    GE_ASSERT_TRUE(file_info.uncompressed_size <= std::numeric_limits<size_t>::max(),
                   "Uncompressed size of entry exceeds size_t range, size = %llu",
                   static_cast<unsigned long long>(file_info.uncompressed_size));
    const std::string file_name(name_buff.data());
    if (!file_name.empty() && file_name.back() != '/') {
      if (file_info.compression_method != Z_NO_COMPRESSION) {
        if (file_info.uncompressed_size > kMaxCompressedEntriesUncompressedSize - total_uncompressed_size) {
          const std::string value = std::to_string(file_info.uncompressed_size);
          const std::string reason = "Total uncompressed size of compressed entries exceeds the 10 GiB limit.";
          (void)REPORT_PREDEFINED_ERR_MSG(
              "E10001", std::vector<const char *>({"parameter", "value", "reason"}),
              std::vector<const char *>({"uncompressed_size", value.c_str(), reason.c_str()}));
          GELOGE(ge::FAILED,
                 "Total uncompressed size of compressed entries exceeds limit, current = %llu, limit = %llu",
                 static_cast<unsigned long long>(file_info.uncompressed_size),
                 static_cast<unsigned long long>(kMaxCompressedEntriesUncompressedSize));
          return false;
        }
        total_uncompressed_size += file_info.uncompressed_size;
      }
      GE_ASSERT_TRUE(CacheCurrentEntry(file_name, file_info));
      entry_names_.emplace_back(file_name);
    }
    uz_ret = unzGoToNextFile(zip_handle_);
  }

  GE_ASSERT_TRUE(uz_ret == UNZ_END_OF_LIST_OF_FILE, "unzGoToNextFile failed, ret=%d", uz_ret);
  entry_cache_ready_ = true;
  return true;
}

bool ZipArchiveReader::CacheCurrentEntry(const std::string &entry_name, const unz_file_info64 &file_info) {
  unz64_file_pos file_pos{};
  auto uz_ret = unzGetFilePos64(zip_handle_, &file_pos);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to get file position for [%s], ret = %d", entry_name.c_str(), uz_ret);

  CachedZipEntry cached_entry;
  cached_entry.file_pos = file_pos;
  cached_entry.uncompressed_size = file_info.uncompressed_size;
  if (file_info.compression_method == Z_NO_COMPRESSION) {
    GE_ASSERT_TRUE(
        GetRawDataOffset(file_pos.pos_in_zip_directory, file_info.uncompressed_size, cached_entry.raw_data_offset));
    cached_entry.raw_data_ready = true;
  }
  const auto emplace_ret = entry_cache_.emplace(entry_name, cached_entry);
  GE_ASSERT_TRUE(emplace_ret.second, "Duplicate zip entry [%s]", entry_name.c_str());
  const auto pos = entry_name.find('/');
  const auto relative = (pos == std::string::npos) ? entry_name : entry_name.substr(pos + 1U);
  relative_entry_cache_[relative] = entry_name;
  return true;
}

bool ZipArchiveReader::GoToEntry(const std::string &entry_name) const {
  GE_ASSERT_TRUE(IsGood(), "Invalid status of archive");
  const auto iter = entry_cache_.find(entry_name);
  if (iter != entry_cache_.end()) {
    auto file_pos = iter->second.file_pos;
    const auto uz_ret = unzGoToFilePos64(zip_handle_, &file_pos);
    GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to go to file [%s] by cached position, ret = %d", entry_name.c_str(),
                   uz_ret);
    return true;
  }

  GELOGE(ge::FAILED, "Failed to locate file [%s] in zip entry cache", entry_name.c_str());
  return false;
}

bool ZipArchiveReader::GetCachedRawData(const std::string &entry_name, size_t &buff_size,
                                        ge::ReadonlyByteBuffer &raw_data) const {
  raw_data = ge::ReadonlyByteBuffer(nullptr, ge::ConditionalDeleter{false});
  GE_ASSERT_TRUE(IsGood(), "Invalid status of archive");
  const auto iter = entry_cache_.find(entry_name);
  if (iter == entry_cache_.end()) {
    GELOGE(ge::FAILED, "Failed to locate file [%s] in zip entry cache", entry_name.c_str());
    return false;
  }
  if (!iter->second.raw_data_ready) {
    return true;
  }

  const auto &cached_entry = iter->second;
  GE_ASSERT_TRUE(cached_entry.raw_data_offset <= mem_file_.length);
  GE_ASSERT_TRUE(cached_entry.uncompressed_size <= mem_file_.length - cached_entry.raw_data_offset);
  buff_size = cached_entry.uncompressed_size;
  raw_data = ge::ReadonlyByteBuffer(mem_file_.buffer + cached_entry.raw_data_offset, ge::ConditionalDeleter{false});
  return true;
}

bool ZipArchiveReader::GetRawDataOffset(const size_t pos_in_central_dir, const size_t buff_size,
                                        uint64_t &raw_data_offset) const {
  ZipEntryInfo entry_info{};
  GE_ASSERT_TRUE(ParseCentralDirEntry(mem_file_, pos_in_central_dir, entry_info));
  GE_ASSERT_TRUE(entry_info.compressed_size == entry_info.uncompressed_size,
                 "uncompressed_size and compressed_size must be equal when loading raw data");
  GE_ASSERT_TRUE(buff_size == entry_info.uncompressed_size, "buff_size is %zu, but uncompressed_size is %zu", buff_size,
                 entry_info.uncompressed_size);
  GE_ASSERT_TRUE(LocateFileDataOffset(mem_file_, entry_info.local_file_header_offset, raw_data_offset));
  GE_ASSERT_TRUE(raw_data_offset <= mem_file_.length);
  GE_ASSERT_TRUE(entry_info.uncompressed_size <= mem_file_.length - raw_data_offset);
  return true;
}

bool ZipArchiveReader::HasEntry(const std::string &entry_name) const {
  if (!IsGood()) {
    return false;
  }
  return entry_cache_.find(entry_name) != entry_cache_.end();
}

std::string ZipArchiveReader::FindEntry(const std::string &relative_path) const {
  if (!IsGood()) {
    return "";
  }
  const auto it = relative_entry_cache_.find(relative_path);
  return (it != relative_entry_cache_.end()) ? it->second : "";
}

bool ZipArchiveReader::HasEntryByRelativePath(const std::string &relative_path) const {
  return !FindEntry(relative_path).empty();
}

std::vector<std::string> ZipArchiveReader::ListFilesByRelativePrefix(const std::string &relative_prefix) const {
  std::vector<std::string> result;
  if (!IsGood()) {
    return result;
  }
  for (const auto &entry : entry_names_) {
    const auto pos = entry.find('/');
    const auto relative = (pos == std::string::npos) ? entry : entry.substr(pos + 1U);
    if (relative.find(relative_prefix) == 0U) {
      result.push_back(entry);
    }
  }
  return result;
}

ge::ReadonlyByteBuffer ZipArchiveReader::ExtractToMem(const std::string &entry_name, size_t &buff_size) const {
  GE_ASSERT_TRUE(IsGood(), "Invalid status of archive");

  ge::ReadonlyByteBuffer raw_data(nullptr, ge::ConditionalDeleter{false});
  GE_ASSERT_TRUE(GetCachedRawData(entry_name, buff_size, raw_data));
  if (raw_data != nullptr) {
    return raw_data;
  }

  GE_ASSERT_TRUE(GoToEntry(entry_name), "Failed to locate file [%s]", entry_name.c_str());

  auto uz_ret = unzOpenCurrentFile(zip_handle_);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to open file [%s], ret = %d", entry_name.c_str(), uz_ret);
  GE_MAKE_GUARD(zipfile_guard, [this]() { (void)unzCloseCurrentFile(zip_handle_); });

  unz_file_info64 file_info{};
  uz_ret = unzGetCurrentFileInfo64(zip_handle_, &file_info, nullptr, 0, nullptr, 0, nullptr, 0);
  GE_ASSERT_TRUE(uz_ret == UNZ_OK, "Failed to get the current file information, ret = %d", uz_ret);
  buff_size = file_info.uncompressed_size;

  auto mutable_buffer = std::make_unique<uint8_t[]>(buff_size);
  GE_ASSERT_NOTNULL(mutable_buffer, "Failed to allocate buffer, size = %zu", buff_size);
  size_t total_read = 0;
  int32_t bytes_read = 0;
  do {
    const uint32_t remaining = static_cast<uint32_t>(
        std::min<size_t>(buff_size - total_read, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
    bytes_read = unzReadCurrentFile(zip_handle_, mutable_buffer.get() + total_read, remaining);

    GE_ASSERT_TRUE(bytes_read >= 0, "Failed to read file [%s], ret = %d", entry_name.c_str(), bytes_read);
    total_read += static_cast<size_t>(bytes_read);
  } while (bytes_read > 0);

  GE_ASSERT_TRUE(total_read == buff_size, "Failed to extract file [%s], expected = %zu bytes, actual = %zu bytes",
                 entry_name.c_str(), buff_size, total_read);
  GELOGI("Successfully extract file [%s], total_read = %d bytes", entry_name.c_str(), total_read);

  return ge::ReadonlyByteBuffer(mutable_buffer.release(), ge::ConditionalDeleter{true});
}

}  // namespace gert
