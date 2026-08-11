/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "graph/op_desc.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "op_tiling.h"

using namespace std;
using namespace ge;

namespace optiling {

class RegisterOpTilingPyCovUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

bool op_tiling_stub_py_v1(const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) {
  return true;
}

bool op_tiling_stub_py_v2(const Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  return true;
}

void *op_parse_stub_py_v3(const Operator &op, const ge::AscendString &compile_info_str) {
  static int32_t dummy = 0;
  return &dummy;
}

void *op_parse_stub_py_v4(const Operator &op, const ge::AscendString &compile_info_str) {
  static int32_t dummy = 0;
  return &dummy;
}

bool op_tiling_stub_py_v3(const Operator &op, const void *compile_info, OpRunInfoV2 &run_info) {
  return true;
}

bool op_tiling_stub_py_v4(const Operator &op, const CompileInfoPtr &compile_info, OpRunInfoV2 &run_info) {
  return true;
}

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info, const char *compile_info_hash,
                                      const char *inputs, const char *outputs, const char *attrs, char *run_info_json,
                                      size_t run_info_len, uint64_t *elapse);
extern "C" int TbeOpTilingPyInterfaceEx2(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse);
extern "C" int OpTilingForCompile(const char *optype, const char *compile_info, const char *compile_info_hash,
                                  const char *inputs, const char *outputs, const char *attrs, char *run_info_json,
                                  size_t run_info_len, uint64_t *elapse, const char *extra_info);
extern "C" Status TbeLoadSoAndSaveToRegistry(const char *so_path);

extern "C" const char *DoOpTilingForCompile(const char *optype, const char *compile_info, const char *compile_info_hash,
                                            const char *inputs, const char *outputs, const char *attrs,
                                            char *run_info_json, size_t run_info_len, uint64_t *elapse,
                                            const char *extra_info);
extern "C" int TbeOpTilingPyInterfaceEx3(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse,
                                         const OpTilingFuncV3 &tiling_func, const OpParseFuncV3 &parse_func,
                                         const char *attrs);
extern "C" int TbeOpTilingPyInterfaceEx4(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse,
                                         const OpTilingFuncV4 &tiling_func, const OpParseFuncV4 &parse_func,
                                         const char *attrs);
extern "C" int TbeOpTilingPyInterfaceEx2New(const char *optype, const char *compile_info, const char *inputs,
                                            const char *outputs, char *run_info_json, size_t run_info_len,
                                            const char *compile_info_hash, uint64_t *elapse,
                                            const OpTilingFuncV2 &tiling_func, const char *attrs);
extern "C" int TbeOpTilingPyInterfaceEx2BackUp(const char *optype, const char *compile_info, const char *inputs,
                                               const char *outputs, char *run_info_json, size_t run_info_len,
                                               const char *compile_info_hash, uint64_t *elapse,
                                               const OpTilingFunc &tiling_func);

CompileInfoPtr op_parse_stub_py_v4_ptr(const Operator &op, const ge::AscendString &compile_info_str) {
  return std::make_shared<CompileInfoBase>();
}

REGISTER_OP_TILING(ReluPyV1, op_tiling_stub_py_v1);
REGISTER_OP_TILING_V2(ReluPyV2, op_tiling_stub_py_v2);
REGISTER_OP_TILING_V3(ReluPyV3, op_tiling_stub_py_v3, op_parse_stub_py_v3);
REGISTER_OP_TILING_V4(ReluPyV4, op_tiling_stub_py_v4, op_parse_stub_py_v4_ptr);

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_NullOptype) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterface(nullptr, "", "", "", "", nullptr, run_info_json, sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_NullParams) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterface("Relu", nullptr, "", "", "", nullptr, run_info_json, sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_InvalidJson) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", "invalid_json", "invalid_json", nullptr,
                                   run_info_json, sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_EmptyInputsOutputs) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", "[]", "[]", nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ValidInputsOutputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ConstTensorInput) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"int32",)"
                       R"("const_value":[1,2,3,4],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ConstTensorFloat16) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float16",)"
                       R"("const_value":[1.0,2.0,3.0,4.0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float16"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ConstTensorBF16) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"bfloat16",)"
                       R"("const_value":[1.0,2.0,3.0,4.0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"bfloat16"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ConstTensorUnknownDtype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"unknown_dtype",)"
      R"("const_value":[1,2,3,4],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_AutoTiling) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface(OP_TYPE_AUTO_TILING.c_str(), "compile_info", "hash", inputs, outputs, nullptr,
                                   run_info_json, sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterfaceEx2_NullParams) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx2("Relu", nullptr, "", "", run_info_json, sizeof(run_info_json), "hash", elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterfaceEx2_InvalidJson) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx2("Relu", "compile_info", "invalid", "invalid", run_info_json,
                                      sizeof(run_info_json), "hash", elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_NullOptype) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = OpTilingForCompile(nullptr, "", "", "", "", nullptr, run_info_json, sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_NullCompileInfo) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret =
      OpTilingForCompile("Relu", nullptr, "", "", "", nullptr, run_info_json, sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_NullInputs) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = OpTilingForCompile("Relu", "compile_info", "", nullptr, "", nullptr, run_info_json, sizeof(run_info_json),
                               elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_NullOutputs) {
  char run_info_json[1024] = {0};
  uint64_t elapse[2] = {0};
  int ret = OpTilingForCompile("Relu", "compile_info", "", "", nullptr, nullptr, run_info_json, sizeof(run_info_json),
                               elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_AutoTiling) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = OpTilingForCompile(OP_TYPE_AUTO_TILING.c_str(), "compile_info", "hash", inputs, outputs, nullptr,
                               run_info_json, sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_NormalOp) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithExtraInfo) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = R"({"device_id":"0"})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithAttrs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"int","value":42}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithBoolAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"bool","value":true}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithStrAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"str","value":"test_value"}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithListAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"list_int","value":[1,2,3]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithListBoolAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"list_bool","value":[true,false]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithListStrAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"list_str","value":["a","b"]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithFloatAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"float","value":3.14}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithFloatAttrNullDesc) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"float","value":3.14,"value_null_desc":"inf"}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithListFloatAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"list_float","value":[1.1,2.2]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithListListIntAttr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"list_list_int","value":[[1,2],[3,4]]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithListListInt64Attr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"list_list_int64","value":[[1,2],[3,4]]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithUnknownAttrType) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"unknown_type","value":42}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithConstTensorAllDtypes) {
  char run_info_json[8192] = {0};
  uint64_t elapse[2] = {0};
  const vector<string> dtypes = {"int8",  "uint8",  "int16",   "uint16", "int32",   "uint32",
                                 "int64", "uint64", "float32", "double", "float16", "bfloat16"};
  for (const auto &dtype : dtypes) {
    std::string inputs_str = R"([{"shape":[2],"ori_shape":[2],"format":"NCHW","ori_format":"NCHW","dtype":")" + dtype +
                             R"(","const_value":[1,2],"name":"x"}])";
    std::string outputs_str =
        R"([{"shape":[2],"ori_shape":[2],"format":"NCHW","ori_format":"NCHW","dtype":")" + dtype + R"("}])";
    int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs_str.c_str(), outputs_str.c_str(), nullptr,
                                 run_info_json, sizeof(run_info_json), elapse, nullptr);
    EXPECT_EQ(ret, 0);
  }
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithConstTensorNullDesc) {
  char run_info_json[8192] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[2],"ori_shape":[2],"format":"NCHW","ori_format":"NCHW","dtype":"float32",)"
                       R"("const_value":[null,2.0],"const_value_null_desc":["inf",null],"name":"x"}])";
  const char *outputs = R"([{"shape":[2],"ori_shape":[2],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithExtraInfoDeterministic) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = R"({"deterministic":1,"deterministic_level":2})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithExtraInfoAicoreNum) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = R"({"_op_aicore_num":"8","_op_vectorcore_num":"4"})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithExtraInfoHcomTopo) {
  char run_info_json[8192] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info =
      R"({"hcom_topo_info":{"rank_size":8,"local_window_size":4,)"
      R"("topo_level_descs":[{"comm_sets":"0,1,2,3","rank_size":4},{"comm_sets":"0,1","rank_size":2}]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithExtraInfoInvalidJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = "invalid_json";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithCompileInfoDeviceId) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *compile_info = R"({"device_id":"0","_cube_vector_core_type":"VecCore"})";
  int ret = OpTilingForCompile(OP_TYPE_AUTO_TILING.c_str(), compile_info, "hash", inputs, outputs, nullptr,
                               run_info_json, sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, OpTilingForCompile_WithCompileInfoCoreType) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *compile_info = R"({"_cube_vector_core_type":null,"_core_type_vec":"VecCore"})";
  int ret = OpTilingForCompile(OP_TYPE_AUTO_TILING.c_str(), compile_info, "hash", inputs, outputs, nullptr,
                               run_info_json, sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeLoadSoAndSaveToRegistry_NullPath) {
  Status ret = TbeLoadSoAndSaveToRegistry(nullptr);
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(RegisterOpTilingPyCovUT, TbeLoadSoAndSaveToRegistry_NonExistSo) {
  Status ret = TbeLoadSoAndSaveToRegistry("/nonexist/path/libtest.so");
  EXPECT_NE(ret, SUCCESS);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ArrayInputOutput) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([[{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}]])";
  const char *outputs =
      R"([[{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}]])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ConstTensorNoName) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"int32","const_value":[1,2]}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ConstTensorWithShapeOnly) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"dtype":"int32","const_value":[1,2,3,4],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_NullDescInf) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"dtype":"float32","const_value":[null],"const_value_null_desc":["inf"],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_NullDescNegInf) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"dtype":"float32","const_value":[null],"const_value_null_desc":["-inf"],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_NullDescNan) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"dtype":"float32","const_value":[null],"const_value_null_desc":["nan"],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_NullDescInvalid) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"dtype":"float32","const_value":[null],"const_value_null_desc":["invalid"],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_FloatAttrWithNullDesc) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"float","value":null,"value_null_desc":"nan"}})";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_FloatAttrWithNonNullValueAndNullDesc) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"float","value":3.14,"value_null_desc":"nan"}})";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_ListFloatAttrWithNullDesc) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"({"attr_name":{"type":"list_float","value":[null,2.0],"value_null_desc":["inf",null]}})";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_AttrWithInvalidJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "invalid_json";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_WithElapse) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_SmallRunInfoBuffer) {
  char run_info_json[4] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_NullInputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", nullptr, nullptr, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_EmptyInputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = "[]";
  const char *outputs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_InvalidInputsJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = "invalid_json";
  const char *outputs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_InvalidOutputsJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = "invalid_json";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_UnknownDtype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"unknown_type"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_WithAttrsListListInt) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"list_list_int","value":[[1,2],[3,4]]}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, TbeOpTilingPyInterface_WithAttrsListListInt64) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"list_list_int64","value":[[1,2],[3,4]]}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_EmptyInputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = "[]";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_EmptyOutputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = "[]";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_InvalidInputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = "invalid_json";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_NullCompileInfo) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", nullptr, "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_UnknownOpType) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("UnknownOpType_0724", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_BFloat16Dtype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"bfloat16"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"bfloat16"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_Float16Dtype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float16"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float16"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_Int8Dtype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"int8"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"int8"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_UnsupportedDtype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"invalid_dtype"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithConstInputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"},)"
      R"({"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32","const_value":[1.0,2.0,3.0,4.0]}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2_Basic) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx2("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      "hash", elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2_WithExtraInfo) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx2("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      "hash", elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2_NullExtraInfo) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx2("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      nullptr, elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2_InvalidExtraInfo) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"invalid_dtype"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx2("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      "hash", elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsInt) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"int","value":42}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsFloat) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"float","value":3.14}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsListFloat) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"list_float","value":[1.0,2.0,3.0]}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsListInt) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"list_int","value":[1,2,3]}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsStr) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"str","value":"hello"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsBool) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"bool","value":true}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithInvalidAttrsDtype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"invalid_dtype","value":42}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithScalarShape) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[],"ori_shape":[],"format":"ND","ori_format":"ND","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[],"ori_shape":[],"format":"ND","ori_format":"ND","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithMultipleInputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"},)"
                       R"({"shape":[2,4],"ori_shape":[2,4],"format":"NCHW","ori_format":"NCHW","dtype":"float16"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_NoCompileInfo) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret =
      TbeOpTilingPyInterface("Relu", "", "hash", inputs, outputs, attrs, run_info_json, sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithNDFormat) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"ND","ori_format":"ND","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"ND","ori_format":"ND","dtype":"float32"}])";
  const char *attrs = "[]";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_Basic) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = "[]";
  const char *extra_info = R"({"op_type":"Relu","compile_info":"compile_info"})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_NullPtr) {
  int ret = OpTilingForCompile(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_DoOpTilingForCompile_NullOptype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *result =
      DoOpTilingForCompile(nullptr, "", "", "", "", nullptr, run_info_json, sizeof(run_info_json), elapse, nullptr);
  EXPECT_NE(result, nullptr);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_DoOpTilingForCompile_AutoTiling) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *result = DoOpTilingForCompile(OP_TYPE_AUTO_TILING.c_str(), "compile_info", "hash", inputs, outputs,
                                            nullptr, run_info_json, sizeof(run_info_json), elapse, nullptr);
  EXPECT_NE(result, nullptr);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_DoOpTilingForCompile_NormalOp) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *result = DoOpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                            sizeof(run_info_json), elapse, nullptr);
  EXPECT_NE(result, nullptr);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_DoOpTilingForCompile_WithExtraInfo) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = R"({"op_name":"test_op"})";
  const char *result = DoOpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                            sizeof(run_info_json), elapse, extra_info);
  EXPECT_NE(result, nullptr);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx3_Basic) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx3("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      "hash", elapse, op_tiling_stub_py_v3, op_parse_stub_py_v3, nullptr);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx3_NullParams) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx3(nullptr, nullptr, nullptr, nullptr, run_info_json, sizeof(run_info_json), nullptr,
                                      elapse, op_tiling_stub_py_v3, op_parse_stub_py_v3, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx3_InvalidJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret =
      TbeOpTilingPyInterfaceEx3("Relu", "compile_info", "invalid", "invalid", run_info_json, sizeof(run_info_json),
                                nullptr, elapse, op_tiling_stub_py_v3, op_parse_stub_py_v3, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx3_NullHash) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx3("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      nullptr, elapse, op_tiling_stub_py_v3, op_parse_stub_py_v3, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx4_Basic) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx4("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      "hash", elapse, op_tiling_stub_py_v4, op_parse_stub_py_v4_ptr, nullptr);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx4_NullParams) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx4(nullptr, nullptr, nullptr, nullptr, run_info_json, sizeof(run_info_json), nullptr,
                                      elapse, op_tiling_stub_py_v4, op_parse_stub_py_v4_ptr, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx4_InvalidJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret =
      TbeOpTilingPyInterfaceEx4("Relu", "compile_info", "invalid", "invalid", run_info_json, sizeof(run_info_json),
                                nullptr, elapse, op_tiling_stub_py_v4, op_parse_stub_py_v4_ptr, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx4_NullHash) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx4("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                      nullptr, elapse, op_tiling_stub_py_v4, op_parse_stub_py_v4_ptr, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2New_Basic) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx2New("Relu", "compile_info", inputs, outputs, run_info_json, sizeof(run_info_json),
                                         "hash", elapse, op_tiling_stub_py_v2, nullptr);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2New_NullParams) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx2New(nullptr, nullptr, nullptr, nullptr, run_info_json, sizeof(run_info_json),
                                         nullptr, elapse, op_tiling_stub_py_v2, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2New_InvalidJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx2New("Relu", "compile_info", "invalid", "invalid", run_info_json,
                                         sizeof(run_info_json), "hash", elapse, op_tiling_stub_py_v2, nullptr);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2BackUp_Basic) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterfaceEx2BackUp("Relu", "compile_info", inputs, outputs, run_info_json,
                                            sizeof(run_info_json), "hash", elapse, op_tiling_stub_py_v1);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2BackUp_NullParams) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx2BackUp(nullptr, nullptr, nullptr, nullptr, run_info_json, sizeof(run_info_json),
                                            nullptr, elapse, op_tiling_stub_py_v1);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2BackUp_InvalidJson) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx2BackUp("Relu", "compile_info", "invalid", "invalid", run_info_json,
                                            sizeof(run_info_json), "hash", elapse, op_tiling_stub_py_v1);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V1Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV1", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V2Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV2", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V3Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV3", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V4Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV4", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_NullOutput) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([null])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_OptionalInput) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"},null])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_ArrayOutputWithNull) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs =
      R"([[{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"},null]])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_RankSizeCompat) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = R"({"rank_size":8})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_TopoErrorPath) {
  char run_info_json[8192] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = R"({"hcom_topo_info":{"rank_size":8,"local_window_size":4,"topo_level_descs":[]}})";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_ExtraInfoArray) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *extra_info = R"([{"op_name":"op1"},{"op_name":"op2"}])";
  int ret = OpTilingForCompile("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, extra_info);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_ConstTensorDouble) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"double",)"
                       R"("const_value":[1.0,2.0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"double"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_ConstTensorBool) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"bool",)"
                       R"("const_value":[1,0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"bool"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_OutputWithSubFormat) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NC1HWC0","ori_format":"NCHW","dtype":"float32","sub_format":1}])";
  const char *outputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NC1HWC0","ori_format":"NCHW","dtype":"float32","sub_format":1}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_ConstTensorNoNameV2) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs =
      R"([{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"int32","const_value":[10,20]}])";
  const char *outputs = R"([{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV2", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsNoName) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"dtype":"int","value":42}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrsMissingValue) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"int"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V1WithConstAndElapse) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32",)"
                       R"("const_value":[1.0,2.0,3.0,4.0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV1", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V2WithConstAndElapse) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32",)"
                       R"("const_value":[1.0,2.0,3.0,4.0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV2", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V3WithConstAndElapse) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32",)"
                       R"("const_value":[1.0,2.0,3.0,4.0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV3", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V4WithConstAndElapse) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32",)"
                       R"("const_value":[1.0,2.0,3.0,4.0],"name":"x"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV4", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V1SmallRunInfoBuffer) {
  char run_info_json[4] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV1", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V2WithAttrs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"int","value":42}])";
  int ret = TbeOpTilingPyInterface("ReluPyV2", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_V1Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = OpTilingForCompile("ReluPyV1", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_V2Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = OpTilingForCompile("ReluPyV2", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_V3Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = OpTilingForCompile("ReluPyV3", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_OpTilingForCompile_V4Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = OpTilingForCompile("ReluPyV4", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                               sizeof(run_info_json), elapse, nullptr);
  EXPECT_EQ(ret, 1);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_DoOpTilingForCompile_V1Registered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *result = DoOpTilingForCompile("ReluPyV1", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                            sizeof(run_info_json), elapse, nullptr);
  EXPECT_NE(result, nullptr);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterfaceEx2_NullOptype) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterfaceEx2(nullptr, "compile_info", "inputs", "outputs", run_info_json,
                                      sizeof(run_info_json), "hash", elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_ConstTensorListInput) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([[{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"int32",)"
                       R"("const_value":[1,2],"name":"x"}]])";
  const char *outputs =
      R"([[{"shape":[1,2],"ori_shape":[1,2],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}]])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_EmptyArrayInput) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([[]])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_OutputWithIsNullOutput) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs =
      R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32","is_null_output":true}])";
  int ret = TbeOpTilingPyInterface("Relu", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V1TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("ReluPyV1Fail");
  OpTilingFuncInfo info("ReluPyV1Fail");
  OpTilingFunc v1_func = [](const TeOpParas &op_paras, const OpCompileInfo &compile_info, OpRunInfo &run_info) -> bool {
    return false;
  };
  info.SetOpTilingFunc(v1_func);
  func_map.emplace("ReluPyV1Fail", info);

  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV1Fail", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
  func_map.erase("ReluPyV1Fail");
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V2TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("ReluPyV2Fail");
  OpTilingFuncInfo info("ReluPyV2Fail");
  OpTilingFuncV2 v2_func = [](const ge::Operator &op, const OpCompileInfoV2 &compile_info,
                              OpRunInfoV2 &run_info) -> bool { return false; };
  info.SetOpTilingFuncV2(v2_func);
  func_map.emplace("ReluPyV2Fail", info);

  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV2Fail", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
  func_map.erase("ReluPyV2Fail");
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V3TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("ReluPyV3Fail");
  OpTilingFuncInfo info("ReluPyV3Fail");
  OpTilingFuncV3 v3_func = [](const ge::Operator &op, const void *compile_info, OpRunInfoV2 &run_info) -> bool {
    return false;
  };
  OpParseFuncV3 p3_func = op_parse_stub_py_v3;
  info.SetOpTilingFuncV3(v3_func, p3_func);
  func_map.emplace("ReluPyV3Fail", info);

  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV3Fail", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
  func_map.erase("ReluPyV3Fail");
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_V4TilingFail) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("ReluPyV4Fail");
  OpTilingFuncInfo info("ReluPyV4Fail");
  OpTilingFuncV4 v4_func = [](const ge::Operator &op, const CompileInfoPtr compile_info,
                              OpRunInfoV2 &run_info) -> bool { return false; };
  OpParseFuncV4 p4_func = op_parse_stub_py_v4_ptr;
  info.SetOpTilingFuncV4(v4_func, p4_func);
  func_map.emplace("ReluPyV4Fail", info);

  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyV4Fail", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
  func_map.erase("ReluPyV4Fail");
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_EmptyFuncInfo) {
  auto &func_map = OpTilingFuncRegistry::RegisteredOpFuncInfo();
  func_map.erase("ReluPyEmpty");
  OpTilingFuncInfo info("ReluPyEmpty");
  func_map.emplace("ReluPyEmpty", info);

  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("ReluPyEmpty", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
  func_map.erase("ReluPyEmpty");
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_NotRegistered) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("NonExistentOp", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_NullInputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  int ret = TbeOpTilingPyInterface("NonExistentOp", "compile_info", "hash", nullptr, nullptr, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_SmallBuffer) {
  char run_info_json[10] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  int ret = TbeOpTilingPyInterface("NonExistentOp", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_WithAttrs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *attrs = R"([{"name":"attr1","dtype":"int64","value":1}])";
  int ret = TbeOpTilingPyInterface("NonExistentOp", "compile_info", "hash", inputs, outputs, attrs, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}

TEST_F(RegisterOpTilingPyCovUT, IncCov_TbeOpTilingPyInterface_InvalidOutputs) {
  char run_info_json[4096] = {0};
  uint64_t elapse[2] = {0};
  const char *inputs = R"([{"shape":[1,4],"ori_shape":[1,4],"format":"NCHW","ori_format":"NCHW","dtype":"float32"}])";
  const char *outputs = "invalid_json";
  int ret = TbeOpTilingPyInterface("NonExistentOp", "compile_info", "hash", inputs, outputs, nullptr, run_info_json,
                                   sizeof(run_info_json), elapse);
  EXPECT_EQ(ret, 0);
}
}  // namespace optiling
