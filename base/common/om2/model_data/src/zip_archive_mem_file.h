/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 本头文件仅供 zip_archive_reader.cc / zip_archive_writer.cc 内部使用，不对外暴露。
// 承载两边共用的内存文件回调辅助代码；引入者需先 include 自己的公共头
// （zip_archive_reader.h 或 zip_archive_writer.h），以提供 MemoryFileReadonly / MemoryFile 完整类型。
#ifndef BASE_COMMON_OM2_MODEL_DATA_SRC_ZIP_ARCHIVE_MEM_FILE_H_
#define BASE_COMMON_OM2_MODEL_DATA_SRC_ZIP_ARCHIVE_MEM_FILE_H_

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "minizip/ioapi.h"
#include "securec.h"

#include "common/debug/log.h"

namespace gert {
namespace {
constexpr int32_t kMemZipOk = 0;
constexpr int32_t kMemZipError = -1;

// 本地实现替代 graph_metadef 的 math_util/file_utils，避免外部头与跨库符号依赖。
inline ge::Status GeMemcpy(uint8_t *dst_ptr, size_t dst_size, const uint8_t *src_ptr, const size_t src_size) {
  if ((dst_ptr == nullptr) || (src_ptr == nullptr) || (dst_size < src_size)) {
    GELOGE(ge::FAILED, "[MEMZIP] GeMemcpy param invalid: dst_size=%zu, src_size=%zu", dst_size, src_size);
    return ge::PARAM_INVALID;
  }
  size_t offset = 0U;
  size_t remain_size = src_size;
  do {
    const size_t copy_size = (remain_size > SECUREC_MEM_MAX_LEN) ? SECUREC_MEM_MAX_LEN : remain_size;
    const errno_t err = memcpy_s((dst_ptr + offset), copy_size, (src_ptr + offset), copy_size);
    if (err != EOK) {
      GELOGE(ge::FAILED, "[MEMZIP] memcpy_s failed: dst_size=%zu, src_size=%zu, offset=%zu, err=%d", dst_size, src_size,
             offset, err);
      return ge::PARAM_INVALID;
    }
    offset += copy_size;
    remain_size -= copy_size;
  } while (remain_size > 0U);
  return ge::SUCCESS;
}

template <typename T>
uLong MemReadFileImpl(T *mem_file, void *buf, const uLong size) {
  if (mem_file == nullptr) {
    return 0;
  }
  uLong bytes_to_read = size;
  if (mem_file->position + bytes_to_read > mem_file->length) {
    bytes_to_read = mem_file->length - mem_file->position;
  }
  if (bytes_to_read > 0) {
    const auto ret = GeMemcpy(static_cast<uint8_t *>(buf), size, &mem_file->buffer[mem_file->position], bytes_to_read);
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED,
             "[MEMZIP] Failed to copy, ret=%d: dest_ptr[%p], dest_max[%zu], src_base_ptr[%p], src_position[%zu], "
             "src_size[%zu]",
             ret, buf, size, mem_file->buffer, mem_file->position, bytes_to_read);
      return 0;
    }
    mem_file->position += bytes_to_read;
  }
  return bytes_to_read;
}

template <typename T>
long MemSeek64FileImpl(T *mem_file, const ZPOS64_T offset, const int origin) {
  if (mem_file == nullptr) {
    return kMemZipError;
  }
  uint64_t new_position;
  switch (origin) {
    case ZLIB_FILEFUNC_SEEK_CUR:
      new_position = mem_file->position + offset;
      break;
    case ZLIB_FILEFUNC_SEEK_END:
      new_position = mem_file->length + offset;
      break;
    case ZLIB_FILEFUNC_SEEK_SET:
      new_position = offset;
      break;
    default:
      return kMemZipError;
  }
  if (new_position > mem_file->length) {
    return kMemZipError;
  }
  mem_file->position = new_position;
  return kMemZipOk;
}
}  // namespace
}  // namespace gert

#endif  // BASE_COMMON_OM2_MODEL_DATA_SRC_ZIP_ARCHIVE_MEM_FILE_H_
