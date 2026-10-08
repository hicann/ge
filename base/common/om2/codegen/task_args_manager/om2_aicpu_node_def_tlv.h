/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License"). Please refer to the License for details.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND.
 */
#ifndef AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_ARGS_MANAGER_OM2_AICPU_NODE_DEF_TLV_H_
#define AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_ARGS_MANAGER_OM2_AICPU_NODE_DEF_TLV_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "aicpu_task_struct.h"
#include "common/checker.h"
#include "common/ge_common/debug/ge_log.h"
#include "graph/utils/math_util.h"
#include "proto/aicpu/cpu_attr.pb.h"
#include "proto/aicpu/cpu_node_def.pb.h"
#include "proto/aicpu/cpu_tensor.pb.h"
#include "proto/aicpu/cpu_tensor_shape.pb.h"
#include "proto/task.pb.h"

namespace ge {
namespace om2 {
namespace aicpu_node_def_tlv {

constexpr uint32_t kMagic = 0x544C5601U;
constexpr uint16_t kVersion = 1U;
constexpr uint16_t kHeaderSize = 16U;
constexpr uint32_t kRecordHeaderSize = 6U;

// Provisional spike tags. These are not a frozen public contract.
constexpr uint16_t kTagNodeRoot = 0x1000U;
constexpr uint16_t kTagNodeOp = 0x1001U;
constexpr uint16_t kTagNodeInput = 0x1002U;
constexpr uint16_t kTagNodeOutput = 0x1003U;
constexpr uint16_t kTagNodeAttr = 0x1004U;
constexpr uint16_t kTagTensorShape = 0x1011U;
constexpr uint16_t kTagTensorType = 0x1012U;
constexpr uint16_t kTagTensorName = 0x1013U;
constexpr uint16_t kTagTensorDataPtr = 0x1014U;
constexpr uint16_t kTagTensorDataSize = 0x1015U;
constexpr uint16_t kTagShapeDims = 0x1021U;
constexpr uint16_t kTagShapeUnknownRank = 0x1022U;
constexpr uint16_t kTagShapeFormat = 0x1023U;
constexpr uint16_t kTagAttrName = 0x1031U;
constexpr uint16_t kTagAttrValue = 0x1032U;

enum class AttrWireType : uint16_t {
  kEmpty = 0,
  kString = 1,
  kInt = 2,
  kFloat = 3,
  kBool = 4,
  kType = 5,
  kShape = 6,
  kTensor = 7,
  kListString = 8,
  kListInt = 9,
  kListFloat = 10,
  kListBool = 11,
  kListType = 12,
  kListShape = 13,
  kListTensor = 14,
  kListListInt = 15,
};

inline void AppendLe16(std::vector<uint8_t> &out, const uint16_t v) {
  out.push_back(static_cast<uint8_t>(v & 0xFFU));
  out.push_back(static_cast<uint8_t>((v >> 8U) & 0xFFU));
}

inline void AppendLe32(std::vector<uint8_t> &out, const uint32_t v) {
  for (uint32_t i = 0U; i < 4U; ++i) {
    out.push_back(static_cast<uint8_t>((v >> (i * 8U)) & 0xFFU));
  }
}

inline void AppendLe64(std::vector<uint8_t> &out, const uint64_t v) {
  for (uint32_t i = 0U; i < 8U; ++i) {
    out.push_back(static_cast<uint8_t>((v >> (i * 8U)) & 0xFFU));
  }
}

inline void AppendRecord(std::vector<uint8_t> &out, const uint16_t tag, const std::vector<uint8_t> &payload,
                         uint32_t &record_count) {
  AppendLe16(out, tag);
  AppendLe32(out, static_cast<uint32_t>(payload.size()));
  out.insert(out.end(), payload.begin(), payload.end());
  ++record_count;
}

inline void AppendStringPayload(std::vector<uint8_t> &out, const std::string &v) {
  out.insert(out.end(), v.begin(), v.end());
}

template <typename T>
inline void AppendScalarPayload(std::vector<uint8_t> &out, const T &v) {
  const auto *p = reinterpret_cast<const uint8_t *>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}

inline std::vector<uint8_t> Frame(const uint16_t root_tag, const std::vector<uint8_t> &root_payload) {
  std::vector<uint8_t> body;
  uint32_t record_count = 0U;
  AppendRecord(body, root_tag, root_payload, record_count);
  std::vector<uint8_t> frame;
  frame.reserve(kHeaderSize + body.size());
  AppendLe32(frame, kMagic);
  AppendLe16(frame, kVersion);
  AppendLe16(frame, kHeaderSize);
  AppendLe32(frame, static_cast<uint32_t>(kHeaderSize + body.size()));
  AppendLe32(frame, record_count);
  frame.insert(frame.end(), body.begin(), body.end());
  return frame;
}

inline void AppendDims(std::vector<uint8_t> &payload, const aicpuops::TensorShape &shape) {
  AppendLe32(payload, static_cast<uint32_t>(shape.dim_size()));
  for (int32_t i = 0; i < shape.dim_size(); ++i) {
    AppendLe64(payload, static_cast<uint64_t>(shape.dim(i).size()));
  }
}

inline std::vector<uint8_t> EncodeShape(const aicpuops::TensorShape &shape) {
  std::vector<uint8_t> out;
  uint32_t record_count = 0U;
  std::vector<uint8_t> dims;
  AppendDims(dims, shape);
  AppendRecord(out, kTagShapeDims, dims, record_count);
  std::vector<uint8_t> unknown_rank;
  unknown_rank.push_back(shape.unknown_rank() ? 1U : 0U);
  AppendRecord(out, kTagShapeUnknownRank, unknown_rank, record_count);
  std::vector<uint8_t> format;
  AppendScalarPayload(format, shape.data_format());
  AppendRecord(out, kTagShapeFormat, format, record_count);
  return out;
}

inline std::vector<uint8_t> EncodeTensor(const aicpuops::Tensor &tensor) {
  std::vector<uint8_t> out;
  uint32_t record_count = 0U;
  AppendRecord(out, kTagTensorShape, EncodeShape(tensor.tensor_shape()), record_count);
  std::vector<uint8_t> type;
  AppendScalarPayload(type, tensor.tensor_type());
  AppendRecord(out, kTagTensorType, type, record_count);
  std::vector<uint8_t> name;
  AppendStringPayload(name, tensor.name());
  AppendRecord(out, kTagTensorName, name, record_count);
  std::vector<uint8_t> data_ptr;
  AppendLe64(data_ptr, tensor.data_ptr());
  AppendRecord(out, kTagTensorDataPtr, data_ptr, record_count);
  std::vector<uint8_t> data_size;
  AppendLe64(data_size, tensor.data_size());
  AppendRecord(out, kTagTensorDataSize, data_size, record_count);
  return out;
}

inline void AppendStringList(std::vector<uint8_t> &out, const google::protobuf::RepeatedPtrField<std::string> &list) {
  AppendLe32(out, static_cast<uint32_t>(list.size()));
  for (const auto &v : list) {
    AppendLe32(out, static_cast<uint32_t>(v.size()));
    out.insert(out.end(), v.begin(), v.end());
  }
}

inline std::vector<uint8_t> EncodeAttrValue(const aicpuops::AttrValue &attr) {
  std::vector<uint8_t> out;
  auto append_type = [&out](const AttrWireType type) { AppendLe16(out, static_cast<uint16_t>(type)); };
  switch (attr.value_case()) {
    case aicpuops::AttrValue::kS:
      append_type(AttrWireType::kString);
      out.insert(out.end(), attr.s().begin(), attr.s().end());
      break;
    case aicpuops::AttrValue::kI:
      append_type(AttrWireType::kInt);
      AppendLe64(out, static_cast<uint64_t>(attr.i()));
      break;
    case aicpuops::AttrValue::kF:
      append_type(AttrWireType::kFloat);
      AppendScalarPayload(out, attr.f());
      break;
    case aicpuops::AttrValue::kB:
      append_type(AttrWireType::kBool);
      out.push_back(attr.b() ? 1U : 0U);
      break;
    case aicpuops::AttrValue::kType:
      append_type(AttrWireType::kType);
      AppendScalarPayload(out, attr.type());
      break;
    case aicpuops::AttrValue::kShape: {
      append_type(AttrWireType::kShape);
      auto shape = EncodeShape(attr.shape());
      out.insert(out.end(), shape.begin(), shape.end());
      break;
    }
    case aicpuops::AttrValue::kTensor: {
      append_type(AttrWireType::kTensor);
      auto tensor = EncodeTensor(attr.tensor());
      out.insert(out.end(), tensor.begin(), tensor.end());
      break;
    }
    case aicpuops::AttrValue::kArray: {
      const auto &array = attr.array();
      if (array.s_size() > 0) {
        append_type(AttrWireType::kListString);
        AppendStringList(out, array.s());
      } else if (array.i_size() > 0) {
        append_type(AttrWireType::kListInt);
        AppendLe32(out, static_cast<uint32_t>(array.i_size()));
        for (const auto v : array.i()) { AppendLe64(out, static_cast<uint64_t>(v)); }
      } else if (array.f_size() > 0) {
        append_type(AttrWireType::kListFloat);
        AppendLe32(out, static_cast<uint32_t>(array.f_size()));
        for (const auto v : array.f()) { AppendScalarPayload(out, v); }
      } else if (array.b_size() > 0) {
        append_type(AttrWireType::kListBool);
        AppendLe32(out, static_cast<uint32_t>(array.b_size()));
        for (const auto v : array.b()) { out.push_back(v ? 1U : 0U); }
      } else if (array.type_size() > 0) {
        append_type(AttrWireType::kListType);
        AppendLe32(out, static_cast<uint32_t>(array.type_size()));
        for (const auto v : array.type()) { AppendScalarPayload(out, v); }
      } else if (array.shape_size() > 0) {
        append_type(AttrWireType::kListShape);
        AppendLe32(out, static_cast<uint32_t>(array.shape_size()));
        for (const auto &v : array.shape()) {
          const auto item = EncodeShape(v);
          AppendLe32(out, static_cast<uint32_t>(item.size()));
          out.insert(out.end(), item.begin(), item.end());
        }
      } else if (array.tensor_size() > 0) {
        append_type(AttrWireType::kListTensor);
        AppendLe32(out, static_cast<uint32_t>(array.tensor_size()));
        for (const auto &v : array.tensor()) {
          const auto item = EncodeTensor(v);
          AppendLe32(out, static_cast<uint32_t>(item.size()));
          out.insert(out.end(), item.begin(), item.end());
        }
      } else {
        append_type(AttrWireType::kEmpty);
      }
      break;
    }
    case aicpuops::AttrValue::kListListInt:
      append_type(AttrWireType::kListListInt);
      AppendLe32(out, static_cast<uint32_t>(attr.list_list_int().list_list_i_size()));
      for (const auto &row : attr.list_list_int().list_list_i()) {
        AppendLe32(out, static_cast<uint32_t>(row.list_i_size()));
        for (const auto v : row.list_i()) { AppendLe64(out, static_cast<uint64_t>(v)); }
      }
      break;
    case aicpuops::AttrValue::VALUE_NOT_SET:
    default:
      append_type(AttrWireType::kEmpty);
      break;
  }
  return out;
}

inline std::vector<uint8_t> EncodeAttr(const std::string &name, const aicpuops::AttrValue &attr) {
  std::vector<uint8_t> out;
  uint32_t record_count = 0U;
  std::vector<uint8_t> name_payload;
  AppendStringPayload(name_payload, name);
  AppendRecord(out, kTagAttrName, name_payload, record_count);
  AppendRecord(out, kTagAttrValue, EncodeAttrValue(attr), record_count);
  return out;
}

inline Status EncodeNodeDef(const aicpuops::NodeDef &node_def, std::string &tlv) {
  std::vector<uint8_t> root;
  uint32_t record_count = 0U;
  std::vector<uint8_t> op;
  AppendStringPayload(op, node_def.op());
  AppendRecord(root, kTagNodeOp, op, record_count);
  for (const auto &input : node_def.inputs()) {
    AppendRecord(root, kTagNodeInput, EncodeTensor(input), record_count);
  }
  for (const auto &output : node_def.outputs()) {
    AppendRecord(root, kTagNodeOutput, EncodeTensor(output), record_count);
  }
  for (const auto &attr : node_def.attrs()) {
    AppendRecord(root, kTagNodeAttr, EncodeAttr(attr.first, attr.second), record_count);
  }
  const auto frame = Frame(kTagNodeRoot, root);
  tlv.assign(reinterpret_cast<const char *>(frame.data()), frame.size());
  return SUCCESS;
}

inline bool IsDefaultRunCpuKernelTask(const domi::KernelDef &kernel_def) {
  constexpr uint32_t kAiCpuKernelType = 6U;
  return kernel_def.so_name() == "libcpu_kernels.so" && kernel_def.kernel_name() == "RunCpuKernel" &&
         kernel_def.has_context() && (kernel_def.context().kernel_type() == kAiCpuKernelType);
}

inline Status RewriteKernelArgsNodeDefToTlv(domi::KernelDef &kernel_def) {
  if (!IsDefaultRunCpuKernelTask(kernel_def)) {
    return SUCCESS;
  }
  const std::string &args = kernel_def.args();
  if (args.size() < sizeof(aicpu::AicpuParamHead)) {
    GELOGE(PARAM_INVALID, "[OM2][AICPU][TLV] args too small: %zu", args.size());
    return PARAM_INVALID;
  }
  aicpu::AicpuParamHead head{};
  GE_ASSERT_EOK(memcpy_s(&head, sizeof(head), args.data(), sizeof(head)));
  if (head.length != args.size()) {
    GELOGE(PARAM_INVALID, "[OM2][AICPU][TLV] head.length %u != args size %zu", head.length, args.size());
    return PARAM_INVALID;
  }
  size_t io_bytes = 0U;
  GE_ASSERT_TRUE(!MulOverflow(static_cast<size_t>(head.ioAddrNum), sizeof(uint64_t), io_bytes));
  size_t node_len_offset = 0U;
  GE_ASSERT_TRUE(!AddOverflow(sizeof(aicpu::AicpuParamHead), io_bytes, node_len_offset));
  size_t node_data_offset = 0U;
  GE_ASSERT_TRUE(!AddOverflow(node_len_offset, sizeof(uint32_t), node_data_offset));
  GE_ASSERT_TRUE(node_data_offset <= args.size(), "[OM2][AICPU][TLV] missing NodeDef length field");
  uint32_t node_def_len = 0U;
  GE_ASSERT_EOK(memcpy_s(&node_def_len, sizeof(node_def_len), args.data() + node_len_offset, sizeof(node_def_len)));
  size_t end_offset = 0U;
  GE_ASSERT_TRUE(!AddOverflow(node_data_offset, static_cast<size_t>(node_def_len), end_offset));
  GE_ASSERT_TRUE(end_offset == args.size(), "[OM2][AICPU][TLV] NodeDef length %u does not match args size %zu",
                 node_def_len, args.size());

  aicpuops::NodeDef node_def;
  if (!node_def.ParseFromArray(args.data() + node_data_offset, static_cast<int>(node_def_len))) {
    GELOGE(PARAM_INVALID, "[OM2][AICPU][TLV] parse protobuf NodeDef failed");
    return PARAM_INVALID;
  }
  std::string tlv;
  GE_ASSERT_SUCCESS(EncodeNodeDef(node_def, tlv));
  GE_ASSERT_TRUE(tlv.size() <= static_cast<size_t>(UINT32_MAX));
  const uint32_t tlv_len = static_cast<uint32_t>(tlv.size());
  size_t new_args_size = 0U;
  GE_ASSERT_TRUE(!AddOverflow(node_data_offset, static_cast<size_t>(tlv_len), new_args_size));
  GE_ASSERT_TRUE(new_args_size <= static_cast<size_t>(UINT32_MAX));
  std::string new_args(args.data(), node_len_offset);
  head.length = static_cast<uint32_t>(new_args_size);
  GE_ASSERT_EOK(memcpy_s(&new_args[0], new_args.size(), &head, sizeof(head)));
  new_args.append(reinterpret_cast<const char *>(&tlv_len), sizeof(tlv_len));
  new_args.append(tlv);
  kernel_def.set_args(new_args.data(), new_args.size());
  kernel_def.set_args_size(static_cast<uint32_t>(new_args.size()));
  return SUCCESS;
}

inline Status RewriteTaskNodeDefToTlv(domi::TaskDef &task_def) {
  if (!task_def.has_kernel()) {
    return SUCCESS;
  }
  return RewriteKernelArgsNodeDefToTlv(*task_def.mutable_kernel());
}

}  // namespace aicpu_node_def_tlv
}  // namespace om2
}  // namespace ge
#endif  // AIR_CXX_BASE_COMMON_OM2_CODEGEN_TASK_ARGS_MANAGER_OM2_AICPU_NODE_DEF_TLV_H_
