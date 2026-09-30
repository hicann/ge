/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/zip_archive_writer.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <vector>

#include "common/checker.h"
#include "common/debug/log.h"
#include "common/math/ge_math_util.h"
#include "common/scope_guard.h"
#include "mmpa/mmpa_api.h"
#include "securec.h"

#include "zip_archive_mem_file.h"

namespace gert {
namespace {
// mmpa 版建目录（语义与 graph_metadef file_utils 的 CreateDir 一致：逐级检查+创建）。
int32_t CheckAndMkdir(const char *tmp_dir_path, const mmMode_t mode) {
  if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
    const int32_t ret = mmMkdir(tmp_dir_path, mode);
    if (ret != 0) {
      if (errno == EEXIST) {
        return 0;
      }
      GELOGW("[MEMZIP] Create directory %s failed, reason:%s", tmp_dir_path, strerror(errno));
      return ret;
    }
  }
  return 0;
}

int32_t CreateDir(const std::string &directory_path) {
  if (directory_path.empty() || directory_path.length() >= static_cast<size_t>(MMPA_MAX_PATH)) {
    GELOGW("[MEMZIP] Directory path invalid: %s", directory_path.c_str());
    return -1;
  }
  constexpr uint32_t mkdir_mode =
      static_cast<uint32_t>(M_IRUSR) | static_cast<uint32_t>(M_IWUSR) | static_cast<uint32_t>(M_IXUSR);
  const auto mode = static_cast<mmMode_t>(mkdir_mode);
  std::string current_path;
  current_path.reserve(directory_path.length());
  for (const char c : directory_path) {
    current_path += c;
    if ((c == '/') || (c == '\\')) {
      const int32_t ret = CheckAndMkdir(current_path.c_str(), mode);
      if (ret != 0) {
        return ret;
      }
    }
  }
  return CheckAndMkdir(directory_path.c_str(), mode);
}

ge::Status SaveBinToFile(const char *const data, size_t length, const std::string &file_path) {
  if (data == nullptr || length == 0UL) {
    GELOGE(ge::FAILED, "[MEMZIP] SaveBinToFile param invalid: data is nullptr or length is zero.");
    return ge::FAILED;
  }
  const size_t split_pos = file_path.find_last_of('/');
  const std::string dir_path = (split_pos == std::string::npos) ? "" : file_path.substr(0U, split_pos);
  if (!dir_path.empty() && (CreateDir(dir_path) != 0)) {
    GELOGE(ge::FAILED, "[MEMZIP] Create directory failed, path: %s", file_path.c_str());
    return ge::FAILED;
  }
  std::ofstream ofs(file_path, std::ios::binary | std::ios::trunc);
  if (!ofs.is_open()) {
    GELOGE(ge::FAILED, "[MEMZIP] Open file failed, path: %s", file_path.c_str());
    return ge::FAILED;
  }
  ofs.write(data, static_cast<std::streamsize>(length));
  ofs.close();
  if (ofs.fail()) {
    GELOGE(ge::FAILED, "[MEMZIP] Write data to file failed, path: %s", file_path.c_str());
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

constexpr uint64_t kMemInitialCapacity = 64UL * 1024UL;
constexpr uint32_t kMemGrowFactor = 2U;
constexpr uint32_t kMaxWriteSize = std::numeric_limits<uint32_t>::max();
constexpr int64_t kBufSize = 16384L;        // same as UNZ_BUFSIZE
constexpr size_t kBufVectorSize = 16384UL;  // same as UNZ_BUFSIZE

int MemGrow(MemoryFile *mem_file, const uint64_t new_size) {
  if (new_size <= mem_file->capacity) {
    return kMemZipOk;
  }
  uint64_t new_capacity = mem_file->capacity;
  if (new_capacity == 0) {
    new_capacity = kMemInitialCapacity;
  }
  while (new_capacity < new_size) {
    if (ge::CheckUint64MulOverflow(new_capacity, kMemGrowFactor) != ge::SUCCESS) {
      GELOGE(ge::FAILED,
             "[MEMZIP] Memory file capacity overflow: current_capacity[%lu bytes], grow_factor[%u], target_size[%zu "
             "bytes]",
             new_capacity, kMemGrowFactor, new_size);
      return kMemZipError;
    }
    new_capacity *= kMemGrowFactor;
  }
  const auto new_buffer = static_cast<uint8_t *>(std::malloc(new_capacity));
  if (new_buffer == nullptr) {
    GELOGE(ge::FAILED, "[MEMZIP] Failed to allocate memory: target_capacity[%lu bytes], error_msg[%s]", new_capacity,
           strerror(errno));
    return kMemZipError;
  }
  if ((mem_file->buffer != nullptr) && (mem_file->length > 0)) {
    const auto ret = GeMemcpy(new_buffer, new_capacity, mem_file->buffer, mem_file->length);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED,
             "[MEMZIP] Failed to copy memory: dest_ptr[%p], dest_max[%lu], src_ptr[%p], src_size[%lu], ret=%d",
             new_buffer, new_capacity, mem_file->buffer, mem_file->capacity, ret);
      std::free(new_buffer);
      return kMemZipError;
    }
  }
  std::free(mem_file->buffer);
  mem_file->buffer = new_buffer;
  mem_file->capacity = new_capacity;
  return kMemZipOk;
}

voidpf ZCALLBACK MemOpenFileFuncWithBuffer(voidpf opaque, const void *filename, int mode) {
  (void)filename;
  const auto mem_file = static_cast<MemoryFile *>(opaque);
  if (mem_file == nullptr) {
    return nullptr;
  }
  mem_file->buffer = nullptr;
  mem_file->length = 0UL;
  mem_file->capacity = 0UL;
  mem_file->position = 0UL;
  mem_file->error = kMemZipOk;
  mem_file->grow_mode = 1;
  mem_file->release_from_outside = 1;
  const auto mode_value = static_cast<uint32_t>(mode);
  if ((mode_value & static_cast<uint32_t>(ZLIB_FILEFUNC_MODE_CREATE)) != 0U) {
    if (MemGrow(mem_file, kMemInitialCapacity) != kMemZipOk) {
      GELOGE(ge::FAILED, "[MEMZIP] Failed to allocate initial capacity[%zu bytes]", kMemInitialCapacity);
      return nullptr;
    }
  }
  return mem_file;
}

uLong ZCALLBACK MemReadFileFunc(voidpf opaque, voidpf stream, void *buf, uLong size) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;
  if (mem_file == nullptr || mem_file->error != kMemZipOk) {
    return 0;
  }
  return MemReadFileImpl(mem_file, buf, size);
}

uLong ZCALLBACK MemWriteFileFunc(voidpf opaque, voidpf stream, const void *buf, uLong size) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;
  if (mem_file == nullptr || mem_file->error != kMemZipOk) {
    GELOGE(ge::FAILED, "[MEMZIP] Opaque pointer is null or file is invalid");
    return 0;
  }
  if (size == 0) {
    return 0;
  }
  if (mem_file->position > std::numeric_limits<uint64_t>::max() - size) {
    GELOGE(ge::FAILED, "[MEMZIP] Position overflow: position=%zu, size=%zu", mem_file->position, size);
    mem_file->error = kMemZipError;
    return 0;
  }
  const uint64_t new_size = mem_file->position + size;
  if (new_size > mem_file->capacity) {
    if (MemGrow(mem_file, new_size) != kMemZipOk) {
      GELOGE(ge::FAILED, "[MEMZIP] Failed to expand memory capacity[%zu bytes]", new_size);
      mem_file->error = kMemZipError;
      return 0;
    }
  }
  const auto ret = memcpy_s(&mem_file->buffer[mem_file->position], size, buf, size);
  if (ret != EOK) {
    GELOGE(ge::FAILED,
           "[MEMZIP] Failed to copy, ret=%d: dest_base_ptr[%p], dest_position[%zu], dest_max[%zu], src_ptr[%p], "
           "src_size[%zu]",
           ret, mem_file->buffer, mem_file->position, mem_file->capacity - mem_file->position, buf, size);
    return 0;
  }
  mem_file->position += size;
  if (mem_file->position > mem_file->length) {
    mem_file->length = mem_file->position;
  }
  return size;
}

ZPOS64_T ZCALLBACK MemTell64FileFunc(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;
  if (mem_file == nullptr) {
    return static_cast<ZPOS64_T>(-1);
  }
  return mem_file->position;
}

long ZCALLBACK MemSeek64FileFunc(voidpf opaque, voidpf stream, ZPOS64_T offset, int origin) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;
  if (mem_file == nullptr) {
    GELOGE(ge::FAILED, "[MEMZIP] Get invalid memory file");
    return kMemZipError;
  }
  const auto ret = MemSeek64FileImpl(mem_file, offset, origin);
  if (ret != kMemZipOk) {
    GELOGE(ge::FAILED, "[MEMZIP] Failed to seek, current_position=%zu, offset=%zu, origin=%d, file_length=%zu",
           mem_file->position, static_cast<uint64_t>(offset), origin, mem_file->length);
  }
  return ret;
}

int ZCALLBACK MemCloseFileFunc(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;
  if (mem_file == nullptr) {
    return kMemZipError;
  }
  if ((mem_file->buffer != nullptr) && (mem_file->release_from_outside == 0)) {
    std::free(mem_file->buffer);
    *mem_file = {};
  }
  return kMemZipOk;
}

int ZCALLBACK MemErrorFileFunc(voidpf opaque, voidpf stream) {
  const auto mem_file = static_cast<MemoryFile *>(stream);
  (void)opaque;
  if (mem_file == nullptr) {
    return kMemZipError;
  }
  return mem_file->error;
}

void FillMemFileFuncWithBuffer(zlib_filefunc64_def *file_func_def, MemoryFile *mem_file) {
  file_func_def->zopen64_file = &MemOpenFileFuncWithBuffer;
  file_func_def->zread_file = &MemReadFileFunc;
  file_func_def->zwrite_file = &MemWriteFileFunc;
  file_func_def->ztell64_file = &MemTell64FileFunc;
  file_func_def->zseek64_file = &MemSeek64FileFunc;
  file_func_def->zclose_file = &MemCloseFileFunc;
  file_func_def->zerror_file = &MemErrorFileFunc;
  file_func_def->opaque = mem_file;
}

std::string GetBaseName(const std::string &path) {
  if (path.empty()) {
    return "";
  }
  const auto pos_slash = path.find_last_of('/');
  std::string file_name = (pos_slash == std::string::npos) ? path : path.substr(pos_slash + 1U);
  if (file_name.empty()) {
    return "";
  }
  const auto pos_dot = file_name.find_last_of('.');
  if ((pos_dot == std::string::npos) || (pos_dot == 0)) {
    return file_name;
  }
  return file_name.substr(0, pos_dot);
}
}  // namespace

ZipArchiveWriter::ZipArchiveWriter(const std::string &archive_path)
    : archive_path_(archive_path), base_name_(GetBaseName(archive_path)) {
  if (!InitArchive()) {
    GELOGE(ge::FAILED, "Failed to initialize archive [%s]", archive_path.c_str());
  }
}

ZipArchiveWriter::~ZipArchiveWriter() {
  (void)WriteEndOfFile();
  if (mem_file_.buffer != nullptr) {
    std::free(mem_file_.buffer);
    mem_file_ = {};
  }
}

bool ZipArchiveWriter::WriteFile(const std::string &entry_name, const std::string &src_file_path, const bool compress) {
  GE_ASSERT_TRUE(IsMemFileOpened(), "Invalid status of archive [%s]", archive_path_.c_str());
  GE_ASSERT_TRUE(!entry_name.empty(), "Entry name cannot be empty");
  const std::string arc_name_with_prefix = base_name_ + "/" + entry_name;
  if (files_written_.count(arc_name_with_prefix) != 0) {
    GELOGW("Duplicate file entry '%s' detected and ignored, src_file_name [%s]", arc_name_with_prefix.c_str(),
           src_file_path.c_str());
    return true;
  }

  std::ifstream fin(src_file_path, std::ios::binary);
  GE_ASSERT_TRUE(fin.is_open(), "Failed to open file [%s]", src_file_path.c_str());
  (void)fin.seekg(0, std::ios::end);
  const auto file_size = fin.tellg();
  (void)fin.seekg(0, std::ios::beg);

  zip_fileinfo file_info{};
  const int compression_method = compress ? Z_DEFLATED : Z_BINARY;
  const int compression_level = compress ? Z_DEFAULT_COMPRESSION : Z_NO_COMPRESSION;
  auto ret = zipOpenNewFileInZip64(zip_handle_, arc_name_with_prefix.c_str(), &file_info, nullptr, 0, nullptr, 0,
                                   nullptr, compression_method, compression_level, 1);
  GE_ASSERT_TRUE(ret == ZIP_OK, "Failed to open zip entry [%s], ret = %d", arc_name_with_prefix.c_str(), ret);
  GE_MAKE_GUARD(close_file_in_zip, [this]() { (void)zipCloseFileInZip(zip_handle_); });

  std::vector<char> buffer(kBufVectorSize);
  auto remaining = static_cast<int64_t>(file_size);
  while (remaining > 0) {
    const int64_t chunk = remaining > kBufSize ? kBufSize : remaining;
    (void)fin.read(buffer.data(), chunk);
    const auto read_bytes = fin.gcount();
    GE_ASSERT_TRUE(read_bytes == chunk, "Failed to read from file [%s], expected = %zu bytes, bytes_read = %zu bytes",
                   src_file_path.c_str(), chunk, read_bytes);

    ret = zipWriteInFileInZip(zip_handle_, buffer.data(), static_cast<uint32_t>(chunk));
    GE_ASSERT_TRUE(ret == ZIP_OK, "zipWriteInFileInZip failed for [%s], ret = %d, bytes_left = %zu bytes",
                   arc_name_with_prefix.c_str(), ret, remaining);
    remaining -= chunk;
  }
  GELOGI("Successfully written [%s] to archive, file_size = %zu bytes", src_file_path.c_str(), file_size);
  (void)files_written_.insert(arc_name_with_prefix);
  return true;
}

bool ZipArchiveWriter::WriteBytes(const std::string &entry_name, const void *data, const size_t data_size,
                                  const bool compress) {
  GE_ASSERT_TRUE(IsMemFileOpened(), "Invalid status of archive [%s]", archive_path_.c_str());
  GE_ASSERT_TRUE(!entry_name.empty(), "Entry name cannot be empty");
  GE_ASSERT_NOTNULL(data, "Pointer data is null, arc_name is [%s]", entry_name.c_str());
  GE_ASSERT_TRUE(data_size > 0, "Data size must be greater than zero, arc_name is [%s]", entry_name.c_str());
  const std::string arc_name_with_prefix = base_name_ + "/" + entry_name;
  if (files_written_.count(arc_name_with_prefix) != 0) {
    GELOGW("Duplicate file entry '%s' detected and ignored", arc_name_with_prefix.c_str());
    return true;
  }

  zip_fileinfo file_info{};
  const int compression_method = compress ? Z_DEFLATED : Z_BINARY;
  const int compression_level = compress ? Z_DEFAULT_COMPRESSION : Z_NO_COMPRESSION;
  auto ret = zipOpenNewFileInZip64(zip_handle_, arc_name_with_prefix.c_str(), &file_info, nullptr, 0, nullptr, 0,
                                   nullptr, compression_method, compression_level, 1);
  GE_ASSERT_TRUE(ret == ZIP_OK, "Failed to open file [%s] in zip, ret = %d", arc_name_with_prefix.c_str(), ret);
  GE_MAKE_GUARD(close_file_in_zip, [this]() { (void)zipCloseFileInZip(zip_handle_); });

  auto pdata = static_cast<const uint8_t *>(data);
  size_t remaining = data_size;
  while (remaining > 0) {
    const uint32_t chunk = remaining > kMaxWriteSize ? kMaxWriteSize : static_cast<uint32_t>(remaining);

    ret = zipWriteInFileInZip(zip_handle_, pdata, chunk);
    GE_ASSERT_TRUE(ret == ZIP_OK, "zipWriteInFileInZip failed for [%s], ret = %d, bytes_left = %zu bytes",
                   arc_name_with_prefix.c_str(), ret, remaining);

    pdata = &pdata[chunk];
    remaining -= chunk;
  }

  GELOGI("Successfully written [%s] to archive, data_size = %zu bytes", entry_name.c_str(), data_size);
  (void)files_written_.insert(arc_name_with_prefix);
  return true;
}

bool ZipArchiveWriter::WriteEndOfFile() {
  if (IsMemFileOpened()) {
    GE_MAKE_GUARD(zip_close_guard, [this]() { zip_handle_ = nullptr; });
    return zipClose(zip_handle_, nullptr) == ZIP_OK;
  }
  return true;
}

bool ZipArchiveWriter::SaveModelData(gert::GertBuffer &model, bool save_to_file) {
  return save_to_file ? SaveModelDataToFile() : SaveModelDataToBuffer(model);
}

bool ZipArchiveWriter::SaveModelDataToFile() {
  GE_ASSERT_TRUE(WriteEndOfFile());
  GE_ASSERT_SUCCESS(SaveBinToFile(reinterpret_cast<const char *>(mem_file_.buffer), mem_file_.length, archive_path_));
  return true;
}

bool ZipArchiveWriter::SaveModelDataToBuffer(gert::GertBuffer &model) {
  GE_ASSERT_TRUE(WriteEndOfFile());
  GE_ASSERT_NOTNULL(mem_file_.buffer);
  GE_ASSERT_TRUE(mem_file_.length > 0U);
  model.length = mem_file_.length;
  model.data = std::shared_ptr<uint8_t>(mem_file_.buffer, [](uint8_t *ptr) { std::free(ptr); });
  mem_file_ = {};
  return true;
}

bool ZipArchiveWriter::InitArchive() {
  GE_ASSERT_TRUE(!base_name_.empty(), "The base_name_ of is empty");
  zlib_filefunc64_def file_funcs;
  FillMemFileFuncWithBuffer(&file_funcs, &mem_file_);
  zip_handle_ = zipOpen2_64(archive_path_.c_str(), APPEND_STATUS_CREATE, nullptr, &file_funcs);
  GE_ASSERT_NOTNULL(zip_handle_, "Failed to open archive [%s]", archive_path_.c_str());
  return true;
}

}  // namespace gert
