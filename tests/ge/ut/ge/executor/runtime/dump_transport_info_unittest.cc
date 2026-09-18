/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "framework/runtime/dump/dump_transport_info.h"

namespace ge {
namespace dump {
namespace {
using Bytes = std::vector<uint8_t>;
using Status = DumpTransStatus;

// Independent fixture writer: never use the production codec to build golden inputs.
Bytes Le(uint64_t value, size_t width) {
  Bytes bytes;
  for (size_t i = 0U; i < width; ++i) {
    bytes.push_back(static_cast<uint8_t>(value & 255U));
    value >>= 8U;
  }
  return bytes;
}
void Append(Bytes &to, const Bytes &from) {
  to.insert(to.end(), from.begin(), from.end());
}
Bytes Tlv(uint16_t tag, const Bytes &payload) {
  Bytes result = Le(tag, 2U);
  Append(result, Le(payload.size(), 4U));
  Append(result, payload);
  return result;
}
Bytes Wire(const Bytes &records, uint32_t count) {
  Bytes result = {0x01, 0x56, 0x4c, 0x54, 1, 0, 16, 0};
  Append(result, Le(16U + records.size(), 4U));
  Append(result, Le(count, 4U));
  Append(result, records);
  return result;
}
Bytes WithTask(const Bytes &task) {
  Bytes records = Tlv(1U, {});
  Append(records, Tlv(2U, task));
  return Wire(records, 2U);
}

TEST(DumpTransportInfoTest, EmptyGoldenAndPresence) {
  const Bytes golden = {1, 0x56, 0x4c, 0x54, 1, 0, 16, 0, 22, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  DumpTransportInfo info;
  Bytes bytes{99};
  ASSERT_EQ(info.Serialize(bytes), Status::kOk);
  EXPECT_EQ(bytes, golden);
  ASSERT_EQ(info.Deserialize(golden.data(), golden.size()), Status::kOk);
  EXPECT_EQ(info.GetVersion(), 1U);
  EXPECT_EQ(info.GetTaskCount(), 0U);
  uint64_t address = 123U;
  EXPECT_EQ(info.GetModel().GetDumpSwitchAddr(address), Status::kFieldAbsent);
  EXPECT_EQ(address, 123U);
  info.MutableModel().SetDumpSwitchAddr(0U);
  EXPECT_EQ(info.GetModel().GetDumpSwitchAddr(address), Status::kOk);
  EXPECT_EQ(address, 0U);
  const DumpTransTaskInfo *task = nullptr;
  EXPECT_EQ(info.GetTask(0U, task), Status::kInvalidParam);
  EXPECT_EQ(task, nullptr);
  info.Clear();
  EXPECT_EQ(info.GetModel().GetDumpSwitchAddr(address), Status::kFieldAbsent);
}

TEST(DumpTransportInfoTest, CanonicalGoldenAndSignedValues) {
  DumpTransportInfo info;
  info.MutableModel().SetModelId(0x12345678U);
  info.MutableModel().SetModelName("");
  auto &task = info.AddTask();
  task.SetTaskType(DumpTransTaskType::kAiCore);
  task.SetEndGraph(false);
  auto &input = task.AddInput();
  input.SetShape({UINT64_MAX, 2U});
  input.SetDataType(-1);
  Bytes model = Tlv(0x0102, {});
  Append(model, Tlv(0x0103, {0x78, 0x56, 0x34, 0x12}));
  Bytes task_bytes = Tlv(0x0205, {0});
  Append(task_bytes, Tlv(0x0206, {0, 0, 0, 0}));
  Bytes input_bytes = Tlv(0x0301, {255, 255, 255, 255});
  Bytes shape = {2, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 2, 0, 0, 0, 0, 0, 0, 0};
  Append(input_bytes, Tlv(0x0303, shape));
  Append(task_bytes, Tlv(0x0010, input_bytes));
  Bytes records = Tlv(1, model);
  Append(records, Tlv(2, task_bytes));
  Bytes encoded;
  ASSERT_EQ(info.Serialize(encoded), Status::kOk);
  EXPECT_EQ(encoded, Wire(records, 2));
  DumpTransportInfo decoded;
  ASSERT_EQ(decoded.Deserialize(encoded.data(), encoded.size()), Status::kOk);
  const DumpTransTaskInfo *decoded_task = nullptr;
  ASSERT_EQ(decoded.GetTask(0, decoded_task), Status::kOk);
  const DumpTransInputInfo *decoded_input = nullptr;
  ASSERT_EQ(decoded_task->GetInput(0, decoded_input), Status::kOk);
  int32_t dtype = 0;
  EXPECT_EQ(decoded_input->GetDataType(dtype), Status::kOk);
  EXPECT_EQ(dtype, -1);
  std::vector<uint64_t> dims;
  EXPECT_EQ(decoded_input->GetShape(dims), Status::kOk);
  EXPECT_EQ(dims, (std::vector<uint64_t>{UINT64_MAX, 2}));
}

void FillModel(DumpTransModelInfo &record) {
  record.SetDumpPath(std::string("a\0b", 3));
  record.SetModelName(std::string("a\0b", 3));
  record.SetModelId(259U);
  record.SetStepIdAddr(260U);
  record.SetIterationsPerLoopAddr(261U);
  record.SetLoopCondAddr(262U);
  record.SetFlag(263U);
  record.SetDumpStep(std::string("a\0b", 3));
  record.SetDumpData(static_cast<DumpTransDumpData>(99U));
  record.SetDumpSwitchAddr(266U);
}
void CheckModel(const DumpTransModelInfo &record) {
  {
    std::string value{};
    EXPECT_EQ(record.GetDumpPath(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
  {
    std::string value{};
    EXPECT_EQ(record.GetModelName(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
  {
    uint32_t value{};
    EXPECT_EQ(record.GetModelId(value), Status::kOk);
    EXPECT_EQ(value, 259U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetStepIdAddr(value), Status::kOk);
    EXPECT_EQ(value, 260U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetIterationsPerLoopAddr(value), Status::kOk);
    EXPECT_EQ(value, 261U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetLoopCondAddr(value), Status::kOk);
    EXPECT_EQ(value, 262U);
  }
  {
    uint32_t value{};
    EXPECT_EQ(record.GetFlag(value), Status::kOk);
    EXPECT_EQ(value, 263U);
  }
  {
    std::string value{};
    EXPECT_EQ(record.GetDumpStep(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
  {
    DumpTransDumpData value{};
    EXPECT_EQ(record.GetDumpData(value), Status::kOk);
    EXPECT_EQ(value, static_cast<DumpTransDumpData>(99U));
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetDumpSwitchAddr(value), Status::kOk);
    EXPECT_EQ(value, 266U);
  }
}

void FillDimRange(DumpTransDimRangeInfo &record) {
  record.SetDimStart(2561U);
  record.SetDimEnd(2562U);
}
void CheckDimRange(const DumpTransDimRangeInfo &record) {
  {
    uint64_t value{};
    EXPECT_EQ(record.GetDimStart(value), Status::kOk);
    EXPECT_EQ(value, 2561U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetDimEnd(value), Status::kOk);
    EXPECT_EQ(value, 2562U);
  }
}

void FillInput(DumpTransInputInfo &record) {
  record.SetDataType(-123);
  record.SetFormat(-123);
  record.SetShape(std::vector<uint64_t>{0U, UINT64_MAX, 42U});
  record.SetAddress(772U);
  record.SetSize(773U);
  record.SetOriginShape(std::vector<uint64_t>{0U, UINT64_MAX, 42U});
  record.SetAddrType(static_cast<DumpTransAddressType>(99U));
  record.SetOffset(776U);
}
void CheckInput(const DumpTransInputInfo &record) {
  {
    int32_t value{};
    EXPECT_EQ(record.GetDataType(value), Status::kOk);
    EXPECT_EQ(value, -123);
  }
  {
    int32_t value{};
    EXPECT_EQ(record.GetFormat(value), Status::kOk);
    EXPECT_EQ(value, -123);
  }
  {
    std::vector<uint64_t> value{};
    EXPECT_EQ(record.GetShape(value), Status::kOk);
    EXPECT_EQ(value, (std::vector<uint64_t>{0U, UINT64_MAX, 42U}));
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetAddress(value), Status::kOk);
    EXPECT_EQ(value, 772U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetSize(value), Status::kOk);
    EXPECT_EQ(value, 773U);
  }
  {
    std::vector<uint64_t> value{};
    EXPECT_EQ(record.GetOriginShape(value), Status::kOk);
    EXPECT_EQ(value, (std::vector<uint64_t>{0U, UINT64_MAX, 42U}));
  }
  {
    DumpTransAddressType value{};
    EXPECT_EQ(record.GetAddrType(value), Status::kOk);
    EXPECT_EQ(value, static_cast<DumpTransAddressType>(99U));
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetOffset(value), Status::kOk);
    EXPECT_EQ(value, 776U);
  }
}

void FillOutput(DumpTransOutputInfo &record) {
  record.SetDataType(-123);
  record.SetFormat(-123);
  record.SetShape(std::vector<uint64_t>{0U, UINT64_MAX, 42U});
  record.SetAddress(1028U);
  record.SetOriginalName(std::string("a\0b", 3));
  record.SetOriginalOutputIndex(-123);
  record.SetOriginalOutputDataType(-123);
  record.SetOriginalOutputFormat(-123);
  record.SetSize(1033U);
  record.SetOriginShape(std::vector<uint64_t>{0U, UINT64_MAX, 42U});
  record.SetAddrType(static_cast<DumpTransAddressType>(99U));
  record.SetOffset(1036U);
}
void CheckOutput(const DumpTransOutputInfo &record) {
  {
    int32_t value{};
    EXPECT_EQ(record.GetDataType(value), Status::kOk);
    EXPECT_EQ(value, -123);
  }
  {
    int32_t value{};
    EXPECT_EQ(record.GetFormat(value), Status::kOk);
    EXPECT_EQ(value, -123);
  }
  {
    std::vector<uint64_t> value{};
    EXPECT_EQ(record.GetShape(value), Status::kOk);
    EXPECT_EQ(value, (std::vector<uint64_t>{0U, UINT64_MAX, 42U}));
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetAddress(value), Status::kOk);
    EXPECT_EQ(value, 1028U);
  }
  {
    std::string value{};
    EXPECT_EQ(record.GetOriginalName(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
  {
    int32_t value{};
    EXPECT_EQ(record.GetOriginalOutputIndex(value), Status::kOk);
    EXPECT_EQ(value, -123);
  }
  {
    int32_t value{};
    EXPECT_EQ(record.GetOriginalOutputDataType(value), Status::kOk);
    EXPECT_EQ(value, -123);
  }
  {
    int32_t value{};
    EXPECT_EQ(record.GetOriginalOutputFormat(value), Status::kOk);
    EXPECT_EQ(value, -123);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetSize(value), Status::kOk);
    EXPECT_EQ(value, 1033U);
  }
  {
    std::vector<uint64_t> value{};
    EXPECT_EQ(record.GetOriginShape(value), Status::kOk);
    EXPECT_EQ(value, (std::vector<uint64_t>{0U, UINT64_MAX, 42U}));
  }
  {
    DumpTransAddressType value{};
    EXPECT_EQ(record.GetAddrType(value), Status::kOk);
    EXPECT_EQ(value, static_cast<DumpTransAddressType>(99U));
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetOffset(value), Status::kOk);
    EXPECT_EQ(value, 1036U);
  }
}

void FillWorkspace(DumpTransWorkspaceInfo &record) {
  record.SetType(static_cast<DumpTransWorkspaceType>(99U));
  record.SetDataAddr(1282U);
  record.SetSize(1283U);
}
void CheckWorkspace(const DumpTransWorkspaceInfo &record) {
  {
    DumpTransWorkspaceType value{};
    EXPECT_EQ(record.GetType(value), Status::kOk);
    EXPECT_EQ(value, static_cast<DumpTransWorkspaceType>(99U));
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetDataAddr(value), Status::kOk);
    EXPECT_EQ(value, 1282U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetSize(value), Status::kOk);
    EXPECT_EQ(value, 1283U);
  }
}

void FillBuffer(DumpTransBufferInfo &record) {
  record.SetType(static_cast<DumpTransBufferType>(99U));
  record.SetAddress(1538U);
  record.SetSize(1539U);
}
void CheckBuffer(const DumpTransBufferInfo &record) {
  {
    DumpTransBufferType value{};
    EXPECT_EQ(record.GetType(value), Status::kOk);
    EXPECT_EQ(value, static_cast<DumpTransBufferType>(99U));
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetAddress(value), Status::kOk);
    EXPECT_EQ(value, 1538U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetSize(value), Status::kOk);
    EXPECT_EQ(value, 1539U);
  }
}

void FillAttr(DumpTransAttrInfo &record) {
  record.SetName(std::string("a\0b", 3));
  record.SetValue(std::string("a\0b", 3));
}
void CheckAttr(const DumpTransAttrInfo &record) {
  {
    std::string value{};
    EXPECT_EQ(record.GetName(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
  {
    std::string value{};
    EXPECT_EQ(record.GetValue(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
}

void FillAddrSize(DumpTransAddrSizeInfo &record) {
  record.SetAddress(2305U);
  record.SetSize(2306U);
}
void CheckAddrSize(const DumpTransAddrSizeInfo &record) {
  {
    uint64_t value{};
    EXPECT_EQ(record.GetAddress(value), Status::kOk);
    EXPECT_EQ(value, 2305U);
  }
  {
    uint64_t value{};
    EXPECT_EQ(record.GetSize(value), Status::kOk);
    EXPECT_EQ(value, 2306U);
  }
}

void FillContext(DumpTransContextInfo &record) {
  record.SetContextId(2049U);
  record.SetThreadId(2050U);
}
void CheckContext(const DumpTransContextInfo &record) {
  {
    uint32_t value{};
    EXPECT_EQ(record.GetContextId(value), Status::kOk);
    EXPECT_EQ(value, 2049U);
  }
  {
    uint32_t value{};
    EXPECT_EQ(record.GetThreadId(value), Status::kOk);
    EXPECT_EQ(value, 2050U);
  }
}

void FillTask(DumpTransTaskInfo &record) {
  record.SetTaskId(513U);
  record.SetStreamId(514U);
  record.SetOpName(std::string("a\0b", 3));
  record.SetOpType(std::string("a\0b", 3));
  record.SetEndGraph(true);
  record.SetTaskType(static_cast<DumpTransTaskType>(99U));
  record.SetContextId(519U);
  record.SetThreadId(520U);
}
void CheckTask(const DumpTransTaskInfo &record) {
  {
    uint32_t value{};
    EXPECT_EQ(record.GetTaskId(value), Status::kOk);
    EXPECT_EQ(value, 513U);
  }
  {
    uint32_t value{};
    EXPECT_EQ(record.GetStreamId(value), Status::kOk);
    EXPECT_EQ(value, 514U);
  }
  {
    std::string value{};
    EXPECT_EQ(record.GetOpName(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
  {
    std::string value{};
    EXPECT_EQ(record.GetOpType(value), Status::kOk);
    EXPECT_EQ(value, std::string("a\0b", 3));
  }
  {
    bool value{};
    EXPECT_EQ(record.GetEndGraph(value), Status::kOk);
    EXPECT_EQ(value, true);
  }
  {
    DumpTransTaskType value{};
    EXPECT_EQ(record.GetTaskType(value), Status::kOk);
    EXPECT_EQ(value, static_cast<DumpTransTaskType>(99U));
  }
  {
    uint32_t value{};
    EXPECT_EQ(record.GetContextId(value), Status::kOk);
    EXPECT_EQ(value, 519U);
  }
  {
    uint32_t value{};
    EXPECT_EQ(record.GetThreadId(value), Status::kOk);
    EXPECT_EQ(value, 520U);
  }
}

TEST(DumpTransportInfoTest, AllFieldsCollectionsAndUnknownEnumsRoundTrip) {
  DumpTransportInfo info;
  FillModel(info.MutableModel());
  auto &task = info.AddTask();
  FillTask(task);
  FillInput(task.AddInput());
  auto &output = task.AddOutput();
  FillOutput(output);
  FillDimRange(output.AddDimRange());
  output.AddDimRange().SetDimStart(99U);
  FillWorkspace(task.AddWorkspace());
  FillBuffer(task.AddBuffer());
  FillAttr(task.AddAttr());
  auto &context = task.AddContext();
  FillContext(context);
  FillAddrSize(context.AddInput());
  FillAddrSize(context.AddOutput());
  info.AddTask().SetTaskId(999U);
  Bytes wire;
  ASSERT_EQ(info.Serialize(wire), Status::kOk);
  DumpTransportInfo decoded;
  ASSERT_EQ(decoded.Deserialize(wire.data(), wire.size()), Status::kOk);
  CheckModel(decoded.GetModel());
  EXPECT_EQ(decoded.GetTaskCount(), 2U);
  const DumpTransTaskInfo *read_task = nullptr;
  ASSERT_EQ(decoded.GetTask(0, read_task), Status::kOk);
  CheckTask(*read_task);
  EXPECT_EQ(read_task->GetInputCount(), 1U);
  const DumpTransInputInfo *read_Input = nullptr;
  ASSERT_EQ(read_task->GetInput(0, read_Input), Status::kOk);
  CheckInput(*read_Input);
  EXPECT_EQ(read_task->GetInput(1, read_Input), Status::kInvalidParam);
  EXPECT_EQ(read_task->GetOutputCount(), 1U);
  const DumpTransOutputInfo *read_Output = nullptr;
  ASSERT_EQ(read_task->GetOutput(0, read_Output), Status::kOk);
  CheckOutput(*read_Output);
  EXPECT_EQ(read_task->GetOutput(1, read_Output), Status::kInvalidParam);
  EXPECT_EQ(read_task->GetWorkspaceCount(), 1U);
  const DumpTransWorkspaceInfo *read_Workspace = nullptr;
  ASSERT_EQ(read_task->GetWorkspace(0, read_Workspace), Status::kOk);
  CheckWorkspace(*read_Workspace);
  EXPECT_EQ(read_task->GetWorkspace(1, read_Workspace), Status::kInvalidParam);
  EXPECT_EQ(read_task->GetBufferCount(), 1U);
  const DumpTransBufferInfo *read_Buffer = nullptr;
  ASSERT_EQ(read_task->GetBuffer(0, read_Buffer), Status::kOk);
  CheckBuffer(*read_Buffer);
  EXPECT_EQ(read_task->GetBuffer(1, read_Buffer), Status::kInvalidParam);
  EXPECT_EQ(read_task->GetAttrCount(), 1U);
  const DumpTransAttrInfo *read_Attr = nullptr;
  ASSERT_EQ(read_task->GetAttr(0, read_Attr), Status::kOk);
  CheckAttr(*read_Attr);
  EXPECT_EQ(read_task->GetAttr(1, read_Attr), Status::kInvalidParam);
  EXPECT_EQ(read_task->GetContextCount(), 1U);
  const DumpTransContextInfo *read_Context = nullptr;
  ASSERT_EQ(read_task->GetContext(0, read_Context), Status::kOk);
  CheckContext(*read_Context);
  EXPECT_EQ(read_task->GetContext(1, read_Context), Status::kInvalidParam);
  EXPECT_EQ(read_Output->GetDimRangeCount(), 2U);
  const DumpTransDimRangeInfo *range = nullptr;
  ASSERT_EQ(read_Output->GetDimRange(0, range), Status::kOk);
  CheckDimRange(*range);
  ASSERT_EQ(read_Output->GetDimRange(1, range), Status::kOk);
  uint64_t start = 0;
  EXPECT_EQ(range->GetDimStart(start), Status::kOk);
  EXPECT_EQ(start, 99U);
  EXPECT_EQ(range->GetDimEnd(start), Status::kFieldAbsent);
  const DumpTransAddrSizeInfo *address = nullptr;
  EXPECT_EQ(read_Context->GetInputCount(), 1U);
  EXPECT_EQ(read_Context->GetOutputCount(), 1U);
  ASSERT_EQ(read_Context->GetInput(0, address), Status::kOk);
  CheckAddrSize(*address);
  ASSERT_EQ(read_Context->GetOutput(0, address), Status::kOk);
  CheckAddrSize(*address);
  ASSERT_EQ(decoded.GetTask(1, read_task), Status::kOk);
  uint32_t id = 0;
  EXPECT_EQ(read_task->GetTaskId(id), Status::kOk);
  EXPECT_EQ(id, 999U);
  Bytes reencoded;
  ASSERT_EQ(decoded.Serialize(reencoded), Status::kOk);
  EXPECT_EQ(wire, reencoded);
}

TEST(DumpTransportInfoTest, UnknownTagsReorderingAndLastScalarWins) {
  Bytes model = Tlv(0x0103, Le(7, 4));
  Append(model, Tlv(0xffff, {1, 2, 3}));
  Append(model, Tlv(0x0103, Le(9, 4)));
  Bytes records = Tlv(2, {});
  Append(records, Tlv(0xfffe, {0, 1}));
  Append(records, Tlv(1, model));
  const auto wire = Wire(records, 3);
  DumpTransportInfo info;
  ASSERT_EQ(info.Deserialize(wire.data(), wire.size()), Status::kOk);
  uint32_t id = 0;
  EXPECT_EQ(info.GetModel().GetModelId(id), Status::kOk);
  EXPECT_EQ(id, 9U);
  EXPECT_EQ(info.GetTaskCount(), 1U);
  Bytes encoded;
  ASSERT_EQ(info.Serialize(encoded), Status::kOk);
  Bytes expected = Tlv(1, Tlv(0x0103, Le(9, 4)));
  Append(expected, Tlv(2, {}));
  EXPECT_EQ(encoded, Wire(expected, 2));
}

TEST(DumpTransportInfoTest, RejectMalformedWithoutPartialCommit) {
  const Bytes valid = Wire(Tlv(1, {}), 1);
  DumpTransportInfo info;
  info.MutableModel().SetModelId(77U);
  Bytes unchanged;
  ASSERT_EQ(info.Serialize(unchanged), Status::kOk);
  const auto check = [&](const Bytes &wire, Status expected) {
    EXPECT_EQ(info.Deserialize(wire.data(), wire.size()), expected);
    Bytes after;
    ASSERT_EQ(info.Serialize(after), Status::kOk);
    EXPECT_EQ(after, unchanged);
  };
  EXPECT_EQ(info.Deserialize(nullptr, 16), Status::kInvalidParam);
  for (size_t i = 1; i < valid.size(); ++i) {
    check(Bytes(valid.begin(), valid.begin() + i), Status::kInvalidFormat);
  }
  for (const auto offset : {0U, 6U, 8U, 12U, 18U}) {
    Bytes bad = valid;
    bad[offset] ^= 1U;
    check(bad, Status::kInvalidFormat);
  }
  Bytes bad_version = valid;
  bad_version[4] = 2;
  check(bad_version, Status::kUnsupportedVersion);
  check(Wire({}, 0), Status::kInvalidFormat);
  Bytes duplicate_model = Tlv(1, {});
  Append(duplicate_model, Tlv(1, {}));
  check(Wire(duplicate_model, 2), Status::kInvalidFormat);
  check(Wire(Tlv(0x10, {}), 1), Status::kInvalidFormat);
  check(Wire(Tlv(1, Tlv(0x0201, Le(1, 4))), 1), Status::kInvalidFormat);
  check(WithTask(Tlv(0x0205, {2})), Status::kInvalidFormat);
  check(WithTask(Tlv(0x0201, {1})), Status::kInvalidFormat);
  check(WithTask(Tlv(0x0010, Tlv(0x0303, Le(1, 4)))), Status::kInvalidFormat);
  check(WithTask(Tlv(0x0010, Tlv(0x0303, Le(UINT32_MAX, 4)))), Status::kLengthOverflow);
  check(WithTask(Tlv(0x0010, Tlv(0x0018, {}))), Status::kInvalidFormat);
  check(WithTask(Tlv(0x0011, Tlv(0x0018, Tlv(0x0a01, {0})))), Status::kInvalidFormat);
  check(WithTask({0x10, 0, 255, 255, 255, 255}), Status::kInvalidFormat);
}

}  // namespace
}  // namespace dump
}  // namespace ge
