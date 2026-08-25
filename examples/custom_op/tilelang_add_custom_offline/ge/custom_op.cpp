/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include "graph/custom_op.h"
#include "acl/acl_rt.h"

using namespace ge;

namespace {
constexpr const char *kKernelSourceFile = "add_custom_kernel.py";
constexpr const char *kCallFuncName = "call";
constexpr uint32_t kSerializeMagic = 0x4F504B4EU;  // "OPKN" in ASCII, custom format identifier
constexpr uint32_t kSerializeVersion = 1U;
constexpr size_t kMaxSoSize = 100U * 1024U * 1024U;  // 100 MB
constexpr size_t kMaxKeyLen = 256U;
constexpr size_t kMaxEntryCount = 64U;

using CallFunc = void (*)(void *x_ptr, void *y_ptr, void *z_ptr, void *stream);

struct KernelEntry {
  std::vector<uint8_t> so_data;
  void *handle = nullptr;
  CallFunc call_func = nullptr;
};

using KernelEntryMap = std::map<std::string, KernelEntry>;

std::string GetCurrentLibraryDir() {
  Dl_info info{};
  if ((dladdr(reinterpret_cast<void *>(&GetCurrentLibraryDir), &info) == 0) || (info.dli_fname == nullptr)) {
    return {};
  }
  const std::string library_path = info.dli_fname;
  const auto pos = library_path.find_last_of('/');
  if (pos == std::string::npos) {
    return ".";
  }
  if (pos == 0U) {
    return "/";
  }
  return library_path.substr(0U, pos);
}

std::string GetKernelSourcePath() {
  const auto library_dir = GetCurrentLibraryDir();
  if (library_dir.empty()) {
    return {};
  }
  if (library_dir == "/") {
    return library_dir + kKernelSourceFile;
  }
  return library_dir + "/" + kKernelSourceFile;
}

std::string BuildBinaryKey(int64_t shape_size) {
  return std::to_string(shape_size);
}

bool ReadFileToVector(const std::string &path, std::vector<uint8_t> &data) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    return false;
  }
  std::ostringstream buffer;
  buffer << file.rdbuf();
  const auto &str = buffer.str();
  data.assign(str.begin(), str.end());
  return true;
}

void WriteU32LE(std::vector<uint8_t> &buf, uint32_t val) {
  buf.push_back(static_cast<uint8_t>(val & 0xFFU));
  buf.push_back(static_cast<uint8_t>((val >> 8U) & 0xFFU));
  buf.push_back(static_cast<uint8_t>((val >> 16U) & 0xFFU));
  buf.push_back(static_cast<uint8_t>((val >> 24U) & 0xFFU));
}

bool ReadU32LE(const std::vector<uint8_t> &buf, size_t &offset, uint32_t &val) {
  if (offset + sizeof(uint32_t) > buf.size()) {
    return false;
  }
  val = static_cast<uint32_t>(buf[offset]) | (static_cast<uint32_t>(buf[offset + 1U]) << 8U) |
        (static_cast<uint32_t>(buf[offset + 2U]) << 16U) | (static_cast<uint32_t>(buf[offset + 3U]) << 24U);
  offset += sizeof(uint32_t);
  return true;
}

bool ReadBytesSafe(const std::vector<uint8_t> &buf, size_t &offset, size_t len, std::vector<uint8_t> &out) {
  if (len > buf.size() || offset > buf.size() - len) {
    return false;
  }
  out.assign(buf.data() + offset, buf.data() + offset + len);
  offset += len;
  return true;
}

bool WriteVectorToMemfd(const std::vector<uint8_t> &data, std::string &memfd_path) {
  int fd = memfd_create("tilelang_kernel", MFD_CLOEXEC);
  if (fd < 0) {
    return false;
  }
  ssize_t written = 0;
  const size_t total = data.size();
  while (static_cast<size_t>(written) < total) {
    ssize_t ret = write(fd, data.data() + written, total - static_cast<size_t>(written));
    if (ret < 0) {
      (void)close(fd);
      return false;
    }
    written += ret;
  }
  memfd_path = "/proc/self/fd/" + std::to_string(fd);
  return true;
}

graphStatus LoadSoFromData(const std::vector<uint8_t> &so_data, void *&handle, CallFunc &call_func) {
  std::string memfd_path;
  if (!WriteVectorToMemfd(so_data, memfd_path)) {
    std::cerr << "Failed to create memfd for kernel .so" << std::endl;
    return GRAPH_FAILED;
  }

  handle = dlopen(memfd_path.c_str(), RTLD_NOW);
  if (handle == nullptr) {
    std::cerr << "dlopen failed: " << dlerror() << std::endl;
    return GRAPH_FAILED;
  }

  dlerror();
  call_func = reinterpret_cast<CallFunc>(dlsym(handle, kCallFuncName));
  const char *error = dlerror();
  if (error != nullptr) {
    std::cerr << "dlsym '" << kCallFuncName << "' failed: " << error << std::endl;
    (void)dlclose(handle);
    handle = nullptr;
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

int ExecuteSubprocess(const char *const argv[], std::string &output) {
  std::ostringstream cmd_stream;
  for (size_t i = 0U; argv[i] != nullptr; ++i) {
    if (i > 0U) {
      cmd_stream << " ";
    }
    cmd_stream << argv[i];
  }
  const std::string cmd = cmd_stream.str() + " 2>&1";

  FILE *pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return -1;
  }
  char buffer[256];
  size_t n = 0U;
  while ((n = fread(buffer, 1U, sizeof(buffer), pipe)) > 0U) {
    output.append(buffer, n);
  }
  int status = pclose(pipe);
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  return -1;
}
}  // namespace

class AddCustomOffline : public CompilableOp, public PortableOp, public EagerExecuteOp, public ShapeInferOp {
 public:
  ~AddCustomOffline() {
    for (auto &entry : kernel_entries_) {
      if (entry.second.handle != nullptr) {
        (void)dlclose(entry.second.handle);
      }
    }
  }

  graphStatus Compile(gert::OpCompileContext *ctx) override {
    if (ctx == nullptr) {
      std::cerr << "Compile context is null" << std::endl;
      return GRAPH_FAILED;
    }

    const gert::Tensor *input_x = ctx->GetInputTensor(0);
    if (input_x == nullptr) {
      std::cerr << "Compile: GetInputTensor(0) failed" << std::endl;
      return GRAPH_FAILED;
    }

    const int64_t n = input_x->GetShapeSize();
    const std::string key = BuildBinaryKey(n);

    std::lock_guard<std::mutex> guard(mutex_);
    if (kernel_entries_.find(key) != kernel_entries_.end()) {
      std::cout << "TileLang kernel already compiled for key=" << key << std::endl;
      return GRAPH_SUCCESS;
    }

    const std::string py_path = GetKernelSourcePath();
    if (py_path.empty()) {
      std::cerr << "Failed to locate TileLang kernel source: " << kKernelSourceFile << std::endl;
      return GRAPH_FAILED;
    }

    char so_tmpl[] = "/tmp/tilelang_offline_XXXXXX.so";
    int tmp_fd = mkstemps(so_tmpl, 3);
    if (tmp_fd < 0) {
      std::cerr << "Failed to create temp file for kernel .so" << std::endl;
      return GRAPH_FAILED;
    }
    std::string so_path(so_tmpl);

    std::ostringstream n_str;
    n_str << n;
    const std::string n_arg = n_str.str();
    const char *const argv[] = {"python3", py_path.c_str(), n_arg.c_str(), so_path.c_str(), nullptr};
    std::cout << "Compiling TileLang kernel (same-machine NPU required)" << std::endl;

    std::string output;
    const int status = ExecuteSubprocess(argv, output);
    if (status != 0) {
      std::cerr << "TileLang compilation failed (exit=" << status << "):" << std::endl;
      std::cerr << output << std::endl;
      (void)close(tmp_fd);
      (void)unlink(so_path.c_str());
      return GRAPH_FAILED;
    }
    std::cout << output;
    (void)close(tmp_fd);

    std::vector<uint8_t> so_data;
    if (!ReadFileToVector(so_path, so_data)) {
      std::cerr << "Failed to read compiled .so: " << so_path << std::endl;
      (void)unlink(so_path.c_str());
      return GRAPH_FAILED;
    }
    (void)unlink(so_path.c_str());

    void *handle = nullptr;
    CallFunc call_func = nullptr;
    if (LoadSoFromData(so_data, handle, call_func) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    kernel_entries_[key] = {std::move(so_data), handle, call_func};
    std::cout << "TileLang kernel compiled and loaded, key=" << key
              << ", so_size=" << kernel_entries_[key].so_data.size() << std::endl;
    return GRAPH_SUCCESS;
  }

  graphStatus Serialize(std::vector<uint8_t> &buffer) override {
    std::lock_guard<std::mutex> guard(mutex_);
    WriteU32LE(buffer, kSerializeMagic);
    WriteU32LE(buffer, kSerializeVersion);
    WriteU32LE(buffer, static_cast<uint32_t>(kernel_entries_.size()));

    for (const auto &entry : kernel_entries_) {
      const auto &key = entry.first;
      const auto &so_data = entry.second.so_data;
      WriteU32LE(buffer, static_cast<uint32_t>(key.size()));
      buffer.insert(buffer.end(), reinterpret_cast<const uint8_t *>(key.data()),
                    reinterpret_cast<const uint8_t *>(key.data()) + key.size());
      WriteU32LE(buffer, static_cast<uint32_t>(so_data.size()));
      buffer.insert(buffer.end(), so_data.data(), so_data.data() + so_data.size());
    }

    std::cout << "Serialized " << kernel_entries_.size() << " kernel(s), total buffer size=" << buffer.size()
              << std::endl;
    return GRAPH_SUCCESS;
  }

  graphStatus Deserialize(const std::vector<uint8_t> &buffer) override {
    std::lock_guard<std::mutex> guard(mutex_);
    size_t offset = 0U;
    uint32_t magic = 0U;
    uint32_t version = 0U;
    uint32_t count = 0U;

    if (!ReadU32LE(buffer, offset, magic) || magic != kSerializeMagic) {
      std::cerr << "Deserialize: invalid magic" << std::endl;
      return GRAPH_FAILED;
    }
    if (!ReadU32LE(buffer, offset, version) || version != kSerializeVersion) {
      std::cerr << "Deserialize: unsupported version=" << version << std::endl;
      return GRAPH_FAILED;
    }
    if (!ReadU32LE(buffer, offset, count)) {
      std::cerr << "Deserialize: failed to read count" << std::endl;
      return GRAPH_FAILED;
    }
    if (count == 0U) {
      std::cerr << "Deserialize: empty entry count" << std::endl;
      return GRAPH_FAILED;
    }
    if (count > kMaxEntryCount) {
      std::cerr << "Deserialize: entry count " << count << " exceeds limit " << kMaxEntryCount << std::endl;
      return GRAPH_FAILED;
    }

    KernelEntryMap temp_entries;
    std::set<std::string> seen_keys;
    for (uint32_t i = 0U; i < count; ++i) {
      if (DeserializeOneEntry(buffer, offset, i, seen_keys, temp_entries) != GRAPH_SUCCESS) {
        ReleaseHandles(temp_entries);
        return GRAPH_FAILED;
      }
    }

    if (offset != buffer.size()) {
      std::cerr << "Deserialize: trailing data after " << count << " entries, offset=" << offset
                << ", buffer_size=" << buffer.size() << std::endl;
      ReleaseHandles(temp_entries);
      return GRAPH_FAILED;
    }

    kernel_entries_ = std::move(temp_entries);
    std::cout << "Deserialized " << kernel_entries_.size() << " kernel(s)" << std::endl;
    return GRAPH_SUCCESS;
  }

 private:
  void ReleaseHandles(KernelEntryMap &entries) {
    for (auto &e : entries) {
      if (e.second.handle != nullptr) {
        (void)dlclose(e.second.handle);
      }
    }
  }

  graphStatus DeserializeOneEntry(const std::vector<uint8_t> &buffer, size_t &offset, uint32_t index,
                                  std::set<std::string> &seen_keys, KernelEntryMap &temp_entries) {
    uint32_t key_len = 0U;
    if (!ReadU32LE(buffer, offset, key_len)) {
      std::cerr << "Deserialize: failed to read key_len at entry " << index << std::endl;
      return GRAPH_FAILED;
    }
    if (key_len == 0U || key_len > kMaxKeyLen) {
      std::cerr << "Deserialize: invalid key_len " << key_len << " at entry " << index << std::endl;
      return GRAPH_FAILED;
    }
    std::vector<uint8_t> key_bytes;
    if (!ReadBytesSafe(buffer, offset, key_len, key_bytes)) {
      std::cerr << "Deserialize: failed to read key at entry " << index << std::endl;
      return GRAPH_FAILED;
    }
    std::string key(key_bytes.begin(), key_bytes.end());
    if (seen_keys.count(key) > 0) {
      std::cerr << "Deserialize: duplicate key '" << key << "' at entry " << index << std::endl;
      return GRAPH_FAILED;
    }

    uint32_t so_size = 0U;
    if (!ReadU32LE(buffer, offset, so_size)) {
      std::cerr << "Deserialize: failed to read so_size at entry " << index << std::endl;
      return GRAPH_FAILED;
    }
    if (so_size == 0U || so_size > kMaxSoSize) {
      std::cerr << "Deserialize: invalid so_size " << so_size << " at entry " << index << std::endl;
      return GRAPH_FAILED;
    }
    std::vector<uint8_t> so_data;
    if (!ReadBytesSafe(buffer, offset, so_size, so_data)) {
      std::cerr << "Deserialize: failed to read so_data at entry " << index << std::endl;
      return GRAPH_FAILED;
    }

    void *handle = nullptr;
    CallFunc call_func = nullptr;
    if (LoadSoFromData(so_data, handle, call_func) != GRAPH_SUCCESS) {
      return GRAPH_FAILED;
    }

    seen_keys.insert(key);
    temp_entries[key] = {std::move(so_data), handle, call_func};
    std::cout << "Deserialized kernel, key=" << key << ", so_size=" << temp_entries[key].so_data.size() << std::endl;
    return GRAPH_SUCCESS;
  }

  graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
    const gert::Tensor *input_x = ctx->GetInputTensor(0);
    const gert::Tensor *input_y = ctx->GetInputTensor(1);
    if (input_x == nullptr || input_y == nullptr) {
      std::cerr << "Execute: GetInputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    const int64_t n = input_x->GetShapeSize();
    const std::string key = BuildBinaryKey(n);

    auto it = kernel_entries_.find(key);
    if (it == kernel_entries_.end()) {
      std::cerr << "Execute: kernel not found for key=" << key << std::endl;
      return GRAPH_FAILED;
    }

    gert::Tensor *output_z =
        ctx->MallocOutputTensor(0, input_x->GetShape(), input_x->GetFormat(), input_x->GetDataType());
    if (output_z == nullptr) {
      std::cerr << "MallocOutputTensor failed" << std::endl;
      return GRAPH_FAILED;
    }

    void *stream = ctx->GetStream();
    it->second.call_func(const_cast<void *>(input_x->GetAddr()), const_cast<void *>(input_y->GetAddr()),
                         output_z->GetAddr(), stream);
    return GRAPH_SUCCESS;
  }

  graphStatus InferShape(gert::InferShapeContext *ctx) override {
    const auto *input_shape = ctx->GetInputShape(0);
    auto *output_shape = ctx->GetOutputShape(0);
    if (input_shape == nullptr || output_shape == nullptr) {
      return GRAPH_FAILED;
    }
    output_shape->SetDimNum(input_shape->GetDimNum());
    for (size_t i = 0; i < input_shape->GetDimNum(); ++i) {
      output_shape->SetDim(i, input_shape->GetDim(i));
    }
    return GRAPH_SUCCESS;
  }

  graphStatus InferDataType(gert::InferDataTypeContext *ctx) override {
    return ctx->SetOutputDataType(0, ctx->GetInputDataType(0));
  }

 private:
  std::mutex mutex_;
  KernelEntryMap kernel_entries_;
};

REG_AUTO_MAPPING_OP(AddCustomOffline);
