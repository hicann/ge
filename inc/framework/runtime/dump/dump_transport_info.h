/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef GE_FRAMEWORK_RUNTIME_DUMP_DUMP_TRANSPORT_INFO_H_
#define GE_FRAMEWORK_RUNTIME_DUMP_DUMP_TRANSPORT_INFO_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ge {
namespace dump {
enum class DumpTransStatus : uint32_t {
  kOk = 0,
  kInvalidParam,
  kFieldAbsent,
  kInvalidFormat,
  kUnsupportedVersion,
  kLengthOverflow,
  kNoMemory
};
enum class DumpTransDumpData : uint32_t { kTensor = 0, kStats = 1 };
enum class DumpTransTaskType : uint32_t { kAiCore = 0, kAiCpu = 1, kDebug = 2, kSdma = 3, kFftsPlus = 4, kDsa = 5 };
enum class DumpTransAddressType : uint32_t {
  kTraditional = 0,
  kNoTiling = 1,
  kRaw = 2,
  kNanoIo = 3,
  kNanoWeight = 4,
  kNanoWork = 5
};
enum class DumpTransWorkspaceType : uint32_t { kLog = 0 };
enum class DumpTransBufferType : uint32_t { kL1 = 0 };

namespace dump_wire_detail {
template <typename T>
struct Field {
  bool present = false;
  T value{};
};
struct Access {
  template <typename Record, typename Visitor>
  static void Visit(Record &record, Visitor &visitor) {
    record.Visit(visitor);
  }
};
}  // namespace dump_wire_detail

// Presence is private to the object; Get leaves its output unchanged when absent.
#define DUMP_TRANS_FIELD(Type, Name)             \
 public:                                         \
  void Set##Name(const Type &value) {            \
    Name##_.value = value;                       \
    Name##_.present = true;                      \
  }                                              \
  DumpTransStatus Get##Name(Type &value) const { \
    if (!Name##_.present) {                      \
      return DumpTransStatus::kFieldAbsent;      \
    }                                            \
    value = Name##_.value;                       \
    return DumpTransStatus::kOk;                 \
  }                                              \
                                                 \
 protected:                                      \
  dump_wire_detail::Field<Type> Name##_;

#define DUMP_TRANS_COLLECTION(Type, Name)                             \
 public:                                                              \
  Type &Add##Name() {                                                 \
    Name##_.emplace_back();                                           \
    return Name##_.back();                                            \
  }                                                                   \
  size_t Get##Name##Count() const {                                   \
    return Name##_.size();                                            \
  }                                                                   \
  DumpTransStatus Get##Name(size_t index, const Type *&value) const { \
    if (index >= Name##_.size()) {                                    \
      return DumpTransStatus::kInvalidParam;                          \
    }                                                                 \
    value = &Name##_[index];                                          \
    return DumpTransStatus::kOk;                                      \
  }                                                                   \
                                                                      \
 private:                                                             \
  std::vector<Type> Name##_;

class DumpTransTensorInfo {
  DUMP_TRANS_FIELD(int32_t, DataType)
  DUMP_TRANS_FIELD(int32_t, Format)
  DUMP_TRANS_FIELD(std::vector<uint64_t>, Shape)
  DUMP_TRANS_FIELD(uint64_t, Address)
  DUMP_TRANS_FIELD(uint64_t, Size)
  DUMP_TRANS_FIELD(std::vector<uint64_t>, OriginShape)
  DUMP_TRANS_FIELD(DumpTransAddressType, AddrType)
  DUMP_TRANS_FIELD(uint64_t, Offset)
};

class DumpTransModelInfo {
  DUMP_TRANS_FIELD(std::string, DumpPath)
  DUMP_TRANS_FIELD(std::string, ModelName)
  DUMP_TRANS_FIELD(uint32_t, ModelId)
  DUMP_TRANS_FIELD(uint64_t, StepIdAddr)
  DUMP_TRANS_FIELD(uint64_t, IterationsPerLoopAddr)
  DUMP_TRANS_FIELD(uint64_t, LoopCondAddr)
  DUMP_TRANS_FIELD(uint32_t, Flag)
  DUMP_TRANS_FIELD(std::string, DumpStep)
  DUMP_TRANS_FIELD(DumpTransDumpData, DumpData)
  DUMP_TRANS_FIELD(uint64_t, DumpSwitchAddr)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0101U, DumpPath_);
    visitor(0x0102U, ModelName_);
    visitor(0x0103U, ModelId_);
    visitor(0x0104U, StepIdAddr_);
    visitor(0x0105U, IterationsPerLoopAddr_);
    visitor(0x0106U, LoopCondAddr_);
    visitor(0x0107U, Flag_);
    visitor(0x0108U, DumpStep_);
    visitor(0x0109U, DumpData_);
    visitor(0x010aU, DumpSwitchAddr_);
  }
};

class DumpTransDimRangeInfo {
  DUMP_TRANS_FIELD(uint64_t, DimStart)
  DUMP_TRANS_FIELD(uint64_t, DimEnd)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0a01U, DimStart_);
    visitor(0x0a02U, DimEnd_);
  }
};

class DumpTransInputInfo : public DumpTransTensorInfo {
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0301U, DataType_);
    visitor(0x0302U, Format_);
    visitor(0x0303U, Shape_);
    visitor(0x0304U, Address_);
    visitor(0x0305U, Size_);
    visitor(0x0306U, OriginShape_);
    visitor(0x0307U, AddrType_);
    visitor(0x0308U, Offset_);
  }
};

class DumpTransOutputInfo : public DumpTransTensorInfo {
  DUMP_TRANS_FIELD(std::string, OriginalName)
  DUMP_TRANS_FIELD(int32_t, OriginalOutputIndex)
  DUMP_TRANS_FIELD(int32_t, OriginalOutputDataType)
  DUMP_TRANS_FIELD(int32_t, OriginalOutputFormat)
  DUMP_TRANS_COLLECTION(DumpTransDimRangeInfo, DimRange)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0401U, DataType_);
    visitor(0x0402U, Format_);
    visitor(0x0403U, Shape_);
    visitor(0x0404U, Address_);
    visitor(0x0405U, OriginalName_);
    visitor(0x0406U, OriginalOutputIndex_);
    visitor(0x0407U, OriginalOutputDataType_);
    visitor(0x0408U, OriginalOutputFormat_);
    visitor(0x0409U, Size_);
    visitor(0x040aU, OriginShape_);
    visitor(0x040bU, AddrType_);
    visitor(0x040cU, Offset_);
    visitor(0x0018U, DimRange_);
  }
};

class DumpTransWorkspaceInfo {
  DUMP_TRANS_FIELD(DumpTransWorkspaceType, Type)
  DUMP_TRANS_FIELD(uint64_t, DataAddr)
  DUMP_TRANS_FIELD(uint64_t, Size)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0501U, Type_);
    visitor(0x0502U, DataAddr_);
    visitor(0x0503U, Size_);
  }
};

class DumpTransBufferInfo {
  DUMP_TRANS_FIELD(DumpTransBufferType, Type)
  DUMP_TRANS_FIELD(uint64_t, Address)
  DUMP_TRANS_FIELD(uint64_t, Size)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0601U, Type_);
    visitor(0x0602U, Address_);
    visitor(0x0603U, Size_);
  }
};

class DumpTransAttrInfo {
  DUMP_TRANS_FIELD(std::string, Name)
  DUMP_TRANS_FIELD(std::string, Value)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0701U, Name_);
    visitor(0x0702U, Value_);
  }
};

class DumpTransAddrSizeInfo {
  DUMP_TRANS_FIELD(uint64_t, Address)
  DUMP_TRANS_FIELD(uint64_t, Size)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0901U, Address_);
    visitor(0x0902U, Size_);
  }
};

class DumpTransContextInfo {
  DUMP_TRANS_FIELD(uint32_t, ContextId)
  DUMP_TRANS_FIELD(uint32_t, ThreadId)
  DUMP_TRANS_COLLECTION(DumpTransAddrSizeInfo, Input)
  DUMP_TRANS_COLLECTION(DumpTransAddrSizeInfo, Output)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0801U, ContextId_);
    visitor(0x0802U, ThreadId_);
    visitor(0x0016U, Input_);
    visitor(0x0017U, Output_);
  }
};

class DumpTransTaskInfo {
  DUMP_TRANS_FIELD(uint32_t, TaskId)
  DUMP_TRANS_FIELD(uint32_t, StreamId)
  DUMP_TRANS_FIELD(std::string, OpName)
  DUMP_TRANS_FIELD(std::string, OpType)
  DUMP_TRANS_FIELD(bool, EndGraph)
  DUMP_TRANS_FIELD(DumpTransTaskType, TaskType)
  DUMP_TRANS_FIELD(uint32_t, ContextId)
  DUMP_TRANS_FIELD(uint32_t, ThreadId)
  DUMP_TRANS_COLLECTION(DumpTransInputInfo, Input)
  DUMP_TRANS_COLLECTION(DumpTransOutputInfo, Output)
  DUMP_TRANS_COLLECTION(DumpTransWorkspaceInfo, Workspace)
  DUMP_TRANS_COLLECTION(DumpTransBufferInfo, Buffer)
  DUMP_TRANS_COLLECTION(DumpTransAttrInfo, Attr)
  DUMP_TRANS_COLLECTION(DumpTransContextInfo, Context)
 private:
  friend struct dump_wire_detail::Access;
  template <typename Visitor>
  void Visit(Visitor &visitor) {
    visitor(0x0201U, TaskId_);
    visitor(0x0202U, StreamId_);
    visitor(0x0203U, OpName_);
    visitor(0x0204U, OpType_);
    visitor(0x0205U, EndGraph_);
    visitor(0x0206U, TaskType_);
    visitor(0x0207U, ContextId_);
    visitor(0x0208U, ThreadId_);
    visitor(0x0010U, Input_);
    visitor(0x0011U, Output_);
    visitor(0x0012U, Workspace_);
    visitor(0x0013U, Buffer_);
    visitor(0x0014U, Attr_);
    visitor(0x0015U, Context_);
  }
};

#undef DUMP_TRANS_FIELD
#undef DUMP_TRANS_COLLECTION

namespace dump_wire_detail {
constexpr uint32_t kMagic = 0x544C5601U;
constexpr uint16_t kVersion = 1U;
constexpr size_t kHeaderSize = 16U;
constexpr size_t kRecordHeaderSize = 6U;

inline bool IsKnownTag(uint16_t tag) {
  switch (tag) {
    case 0x0001U:
    case 0x0002U:
    case 0x0010U:
    case 0x0011U:
    case 0x0012U:
    case 0x0013U:
    case 0x0014U:
    case 0x0015U:
    case 0x0016U:
    case 0x0017U:
    case 0x0018U:
    case 0x0101U:
    case 0x0102U:
    case 0x0103U:
    case 0x0104U:
    case 0x0105U:
    case 0x0106U:
    case 0x0107U:
    case 0x0108U:
    case 0x0109U:
    case 0x010aU:
    case 0x0201U:
    case 0x0202U:
    case 0x0203U:
    case 0x0204U:
    case 0x0205U:
    case 0x0206U:
    case 0x0207U:
    case 0x0208U:
    case 0x0301U:
    case 0x0302U:
    case 0x0303U:
    case 0x0304U:
    case 0x0305U:
    case 0x0306U:
    case 0x0307U:
    case 0x0308U:
    case 0x0401U:
    case 0x0402U:
    case 0x0403U:
    case 0x0404U:
    case 0x0405U:
    case 0x0406U:
    case 0x0407U:
    case 0x0408U:
    case 0x0409U:
    case 0x040aU:
    case 0x040bU:
    case 0x040cU:
    case 0x0501U:
    case 0x0502U:
    case 0x0503U:
    case 0x0601U:
    case 0x0602U:
    case 0x0603U:
    case 0x0701U:
    case 0x0702U:
    case 0x0801U:
    case 0x0802U:
    case 0x0901U:
    case 0x0902U:
    case 0x0a01U:
    case 0x0a02U:
      return true;
    default:
      return false;
  }
}

inline uint64_t ReadLe(const uint8_t *data, size_t width) {
  uint64_t value = 0U;
  for (size_t i = 0U; i < width; ++i) {
    value |= static_cast<uint64_t>(data[i]) << (i * 8U);
  }
  return value;
}

inline void WriteLe(uint8_t *data, uint64_t value, size_t width) {
  for (size_t i = 0U; i < width; ++i) {
    data[i] = static_cast<uint8_t>(value >> (i * 8U));
  }
}

template <typename Handler>
DumpTransStatus ForEachRecord(const uint8_t *data, size_t size, Handler handler) {
  size_t pos = 0U;
  while (pos < size) {
    if (size - pos < kRecordHeaderSize) {
      return DumpTransStatus::kInvalidFormat;
    }
    const auto tag = static_cast<uint16_t>(ReadLe(data + pos, 2U));
    const auto length = static_cast<size_t>(ReadLe(data + pos + 2U, 4U));
    pos += kRecordHeaderSize;
    if (length > size - pos) {
      return DumpTransStatus::kInvalidFormat;
    }
    const auto status = handler(tag, data + pos, length);
    if (status != DumpTransStatus::kOk) {
      return status;
    }
    pos += length;
  }
  return DumpTransStatus::kOk;
}

class Encoder {
 public:
  std::vector<uint8_t> bytes;
  DumpTransStatus status = DumpTransStatus::kOk;

  void Append(uint64_t value, size_t width) {
    const size_t pos = bytes.size();
    if (Grow(width)) {
      WriteLe(bytes.data() + pos, value, width);
    }
  }

  template <typename T>
  void Record(uint16_t tag, T &record) {
    const size_t pos = Begin(tag);
    Access::Visit(record, *this);
    End(pos);
  }

  template <typename T>
  void operator()(uint16_t tag, const Field<T> &field) {
    if (field.present && status == DumpTransStatus::kOk) {
      const size_t pos = Begin(tag);
      Value(field.value);
      End(pos);
    }
  }

  template <typename T>
  void operator()(uint16_t tag, std::vector<T> &records) {
    for (auto &record : records) {
      if (status != DumpTransStatus::kOk) {
        break;
      }
      Record(tag, record);
    }
  }

 private:
  bool Grow(size_t count) {
    if (status != DumpTransStatus::kOk) {
      return false;
    }
    if (count > UINT32_MAX - bytes.size()) {
      status = DumpTransStatus::kLengthOverflow;
      return false;
    }
    bytes.resize(bytes.size() + count);
    return true;
  }

  size_t Begin(uint16_t tag) {
    const size_t pos = bytes.size();
    Append(tag, 2U);
    Append(0U, 4U);
    return pos;
  }

  void End(size_t pos) {
    if (status == DumpTransStatus::kOk) {
      WriteLe(bytes.data() + pos + 2U, bytes.size() - pos - kRecordHeaderSize, 4U);
    }
  }

  template <typename T>
  void Value(T value) {
    Append(static_cast<uint64_t>(value), sizeof(T));
  }

  void Value(bool value) {
    Append(value ? 1U : 0U, 1U);
  }

  void Value(const std::string &value) {
    const size_t pos = bytes.size();
    if (Grow(value.size())) {
      std::copy(value.begin(), value.end(), bytes.begin() + pos);
    }
  }

  void Value(const std::vector<uint64_t> &value) {
    if (value.size() > (UINT32_MAX - 4U) / 8U) {
      status = DumpTransStatus::kLengthOverflow;
      return;
    }
    Append(value.size(), 4U);
    for (const auto dim : value) {
      Append(dim, 8U);
      if (status != DumpTransStatus::kOk) {
        break;
      }
    }
  }
};

template <typename Record>
DumpTransStatus DecodeRecord(const uint8_t *data, size_t size, Record &record);

class Decoder {
 public:
  Decoder(uint16_t tag, const uint8_t *data, size_t size) : tag_(tag), data_(data), size_(size) {}
  bool matched = false;
  DumpTransStatus status = DumpTransStatus::kOk;

  template <typename T>
  void operator()(uint16_t tag, Field<T> &field) {
    if (tag == tag_) {
      matched = true;
      status = Value(field.value);
      field.present = (status == DumpTransStatus::kOk);
    }
  }

  template <typename T>
  void operator()(uint16_t tag, std::vector<T> &records) {
    if (tag == tag_) {
      matched = true;
      records.emplace_back();
      status = DecodeRecord(data_, size_, records.back());
    }
  }

 private:
  template <typename T>
  DumpTransStatus Value(T &value) const {
    if (size_ != sizeof(T)) {
      return DumpTransStatus::kInvalidFormat;
    }
    value = static_cast<T>(ReadLe(data_, size_));
    return DumpTransStatus::kOk;
  }

  DumpTransStatus Value(int32_t &value) const {
    if (size_ != 4U) {
      return DumpTransStatus::kInvalidFormat;
    }
    const uint32_t raw = static_cast<uint32_t>(ReadLe(data_, size_));
    value = raw <= INT32_MAX ? static_cast<int32_t>(raw) : -1 - static_cast<int32_t>(UINT32_MAX - raw);
    return DumpTransStatus::kOk;
  }

  DumpTransStatus Value(bool &value) const {
    if (size_ != 1U || data_[0] > 1U) {
      return DumpTransStatus::kInvalidFormat;
    }
    value = data_[0] != 0U;
    return DumpTransStatus::kOk;
  }

  DumpTransStatus Value(std::string &value) const {
    value.assign(reinterpret_cast<const char *>(data_), size_);
    return DumpTransStatus::kOk;
  }

  DumpTransStatus Value(std::vector<uint64_t> &value) const {
    if (size_ < 4U) {
      return DumpTransStatus::kInvalidFormat;
    }
    const uint64_t count = ReadLe(data_, 4U);
    if (count > (UINT32_MAX - 4U) / 8U) {
      return DumpTransStatus::kLengthOverflow;
    }
    if (size_ != 4U + count * 8U) {
      return DumpTransStatus::kInvalidFormat;
    }
    value.clear();
    value.reserve(static_cast<size_t>(count));
    for (size_t i = 0U; i < count; ++i) {
      value.push_back(ReadLe(data_ + 4U + i * 8U, 8U));
    }
    return DumpTransStatus::kOk;
  }

  uint16_t tag_;
  const uint8_t *data_;
  size_t size_;
};

template <typename Record>
DumpTransStatus DecodeRecord(const uint8_t *data, size_t size, Record &record) {
  return ForEachRecord(data, size, [&record](uint16_t tag, const uint8_t *payload, size_t payload_size) {
    Decoder decoder(tag, payload, payload_size);
    Access::Visit(record, decoder);
    if (decoder.status != DumpTransStatus::kOk) {
      return decoder.status;
    }
    if (!decoder.matched && IsKnownTag(tag)) {
      return DumpTransStatus::kInvalidFormat;
    }
    return DumpTransStatus::kOk;
  });
}
}  // namespace dump_wire_detail

class DumpTransportInfo {
 public:
  uint16_t GetVersion() const {
    return dump_wire_detail::kVersion;
  }
  DumpTransModelInfo &MutableModel() {
    return model_;
  }
  const DumpTransModelInfo &GetModel() const {
    return model_;
  }
  DumpTransTaskInfo &AddTask() {
    tasks_.emplace_back();
    return tasks_.back();
  }
  size_t GetTaskCount() const {
    return tasks_.size();
  }
  DumpTransStatus GetTask(size_t index, const DumpTransTaskInfo *&task) const {
    if (index >= tasks_.size()) {
      return DumpTransStatus::kInvalidParam;
    }
    task = &tasks_[index];
    return DumpTransStatus::kOk;
  }
  void Clear() {
    model_ = DumpTransModelInfo{};
    tasks_.clear();
  }

  DumpTransStatus Serialize(std::vector<uint8_t> &buffer) {
    using namespace dump_wire_detail;
    if (tasks_.size() >= UINT32_MAX) {
      return DumpTransStatus::kLengthOverflow;
    }
    try {
      Encoder encoder;
      encoder.Append(kMagic, 4U);
      encoder.Append(kVersion, 2U);
      encoder.Append(kHeaderSize, 2U);
      encoder.Append(0U, 4U);
      encoder.Append(tasks_.size() + 1U, 4U);
      encoder.Record(0x0001U, model_);
      encoder(0x0002U, tasks_);
      if (encoder.status != DumpTransStatus::kOk) {
        return encoder.status;
      }
      WriteLe(encoder.bytes.data() + 8U, encoder.bytes.size(), 4U);
      buffer.swap(encoder.bytes);
      return DumpTransStatus::kOk;
    } catch (const std::bad_alloc &) {
      return DumpTransStatus::kNoMemory;
    } catch (const std::length_error &) {
      return DumpTransStatus::kLengthOverflow;
    }
  }

  DumpTransStatus Deserialize(const uint8_t *data, size_t size) {
    using namespace dump_wire_detail;
    if (data == nullptr) {
      return DumpTransStatus::kInvalidParam;
    }
    if (size < kHeaderSize || ReadLe(data, 4U) != kMagic) {
      return DumpTransStatus::kInvalidFormat;
    }
    if (ReadLe(data + 4U, 2U) != kVersion) {
      return DumpTransStatus::kUnsupportedVersion;
    }
    if (ReadLe(data + 6U, 2U) != kHeaderSize || ReadLe(data + 8U, 4U) != size) {
      return DumpTransStatus::kInvalidFormat;
    }
    try {
      DumpTransportInfo parsed;
      uint64_t count = 0U;
      bool has_model = false;
      const auto status =
          ForEachRecord(data + kHeaderSize, size - kHeaderSize,
                        [&parsed, &count, &has_model](uint16_t tag, const uint8_t *payload, size_t payload_size) {
                          ++count;
                          if (tag == 0x0001U) {
                            if (has_model) {
                              return DumpTransStatus::kInvalidFormat;
                            }
                            has_model = true;
                            return DecodeRecord(payload, payload_size, parsed.model_);
                          }
                          if (tag == 0x0002U) {
                            return DecodeRecord(payload, payload_size, parsed.AddTask());
                          }
                          if (IsKnownTag(tag)) {
                            return DumpTransStatus::kInvalidFormat;
                          }
                          return DumpTransStatus::kOk;
                        });
      if (status != DumpTransStatus::kOk) {
        return status;
      }
      if (!has_model || count != ReadLe(data + 12U, 4U)) {
        return DumpTransStatus::kInvalidFormat;
      }
      std::swap(model_, parsed.model_);
      tasks_.swap(parsed.tasks_);
      return DumpTransStatus::kOk;
    } catch (const std::bad_alloc &) {
      return DumpTransStatus::kNoMemory;
    } catch (const std::length_error &) {
      return DumpTransStatus::kLengthOverflow;
    }
  }

 private:
  DumpTransModelInfo model_;
  std::vector<DumpTransTaskInfo> tasks_;
};
}  // namespace dump
}  // namespace ge
#endif  // GE_FRAMEWORK_RUNTIME_DUMP_DUMP_TRANSPORT_INFO_H_
