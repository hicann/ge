/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "e2e_load_abs_store.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "ascgraph_info_complete.h"
#include "ascir_utils.h"
#define private public
#include "optimize.h"
#undef private
#include "codegen.h"
#include "e2e_common.h"
#include "platform_context.h"
#include "autofuse_config/auto_fuse_config.h"

using namespace ascir;

class E2E_LoadAbsStore : public ::testing::Test {
 protected:
  optimize::Optimizer optimizer;
  codegen::Codegen codegen;

  E2E_LoadAbsStore() : optimizer(optimize::OptimizerOptions{}), codegen(codegen::CodegenOptions{.tiling_lib_path="asdf",.tiling_lib_codegen_symbol="as"}) {}

  void SetUp() override {
    ge::PlatformContext::GetInstance().Reset();
  }
};

std::string RemoveAutoFuseTilingHeadGuards(const std::string &input) {
  std::istringstream iss(input);
  std::ostringstream oss;
  std::string line;
  const std::string guard_token = "__AUTOFUSE_TILING_FUNC_COMMON_H__";

  while (std::getline(iss, line)) {
    // 如果当前行不包含 guard_token，则保留
    if (line.find(guard_token) == std::string::npos) {
      oss << line << "\n";
    }
  }

  return oss.str();
}

void CombineTilings(const std::map<std::string, std::string> &tilings, std::string &result) {
  const std::string tiling_head = "TilingHead";  // TilingHead作为开头拼接其他文件
  const std::string tiling_data = "TilingData";  // 要排除的 TilingData 子串
  result += RemoveAutoFuseTilingHeadGuards(tilings.at(tiling_head));  // 删除头文件的宏保护，cpp文件不需要
  const std::string include_str = "#include \"autofuse_tiling_func_common.h\"";

  // 遍历所有非 TilingHead 和 TilingData 的条目，去掉第一行后拼接
  for (const auto &[key, value] : tilings) {
    if (key == tiling_head || key.find(tiling_data) != std::string::npos) {
      continue;
    }

    // 查找并跳过第一行头文件行
    size_t include_pos = value.find(include_str);
    if (include_pos != std::string::npos) {
      // 找到 include 行，跳过它，并去掉后面的换行符
      size_t content_start = include_pos + include_str.length();
      while (content_start < value.size() && (value[content_start] == '\n' || value[content_start] == '\r')) {
        content_start++;
      }
      result += value.substr(content_start);
    } else {
      // 如果没有 include 行，直接拼接整个内容
      result += value;
    }

    if (!result.empty() && result.back() != '\n') {
      result += '\n';
    }
  }
}

TEST_F(E2E_LoadAbsStore, ConstructGraphWithAscir) {
  ge::AscGraph test_graph("test_load_abs_store");
  LoadAbsStore_BeforeAutofuse(test_graph);
  GTEST_SKIP() << "Compare expect graph ir info here";
}

TEST_F(E2E_LoadAbsStore, GetApiInfo) {
  ge::AscGraph expect_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(expect_graph);

  ge::AscGraph expect_optimize_graph("expect_optimize_graph");
  expect_optimize_graph.CopyFrom(expect_graph);
  LoadAbsStore_AfterGetApiInfo(expect_optimize_graph);

  ge::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);

  ge::AscGraph test_optimize_graph("test_optimize_graph");
  test_optimize_graph.CopyFrom(test_graph);
  optimize::AscGraphInfoComplete::CompleteApiInfo(test_optimize_graph);

  EXPECT_EQ(utils::DebugHintGraphStr(test_graph), utils::DebugHintGraphStr(expect_graph));
}

TEST_F(E2E_LoadAbsStore, Codegen_TilingData)
{
  ge::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  FusedScheduledResult fused_schedule_result;
  auto tiling_data_code = codegen.GenerateTilingData(fused_schedule_result);
  std::cout << tiling_data_code << std::endl;
  const std::string test_res = R"rawliteral(#ifndef __Autofuse_Tiling_Data_H__
#define __Autofuse_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
  TILING_DATA_FIELD_DEF_T(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF_T(uint32_t, corenum);
  TILING_DATA_FIELD_DEF_T(uint32_t, ub_size);
  TILING_DATA_FIELD_DEF_T(uint32_t, hbm_size);

END_TILING_DATA_DEF_T;

struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)rawliteral";
  EXPECT_EQ(tiling_data_code, test_res);
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling_With_Lambda)
{
  ge::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::string s0_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(0);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s1_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s2_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(2);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";

  std::map<std::string, std::string> shape_info = {{"s0", s0_source},
                                                   {"s1", s1_source},
                                                   {"s2", s2_source}};
  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  auto tiling_codes = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0");
  for (const auto&[key,value] : tiling_codes) {
    std::cout << key <<std::endl;
    std::cout << value <<std::endl;
  }
  std::string tiling_code;
  CombineTilings(tiling_codes, tiling_code);
  std::string expect_code = R"rawliteral(#include <stdexcept>
#include <sstream>
#include <cmath>
#include "autofuse_tiling_data.h"
#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "platform/platform_infos_def.h"
#include "platform_ascendc.h"
#endif

#include <cfloat>
#include <vector>

typedef long int (*ProfilingCallback)(void* stream, uint32_t workspaceSize, AutofuseTilingData* tiling_data, double* cost_time);
typedef long int (*ProfilingBatchCallback)(void* stream, uint32_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);
class PgoConfig {
public:
  static PgoConfig& Instance() {
    static PgoConfig instance;
    return instance;
  }
  ProfilingCallback single_callback;
  ProfilingBatchCallback batch_callback;
  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num
  bool need_change_solver_run = false;
private:
  PgoConfig() = default;
  ~PgoConfig() = default;
  PgoConfig(const PgoConfig &) = delete;
  PgoConfig &operator=(const PgoConfig &) = delete;
};

#include <iostream>
#include <fstream>
#include <cinttypes>
#include <sys/syscall.h>
#include <unistd.h>
#include "dlog_pub.h"
#define OP_LOGD(name, fmt, ...)
#define OP_LOGI(name, fmt, ...)
#define GE_MODULE_NAME static_cast<int32_t>(45)
inline uint64_t GetTid() {
     return static_cast<uint64_t>(syscall(__NR_gettid));
}
#define GELOGE(ERROR_CODE, fmt, ...)
#define OP_LOGE(name, fmt, ...)
#define OP_NAME "asc0000_autofused_abs"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define Pow(a, b) pow(a, b)
#define Rational(a, b) ((double)(a) / (double)(b))

namespace optiling {
extern "C" bool GetTiling(AutofuseTilingData& tiling_data, int32_t tilingCaseId=-1) {
  return true;
}
inline bool IsEqual(double a, double b) {return true;}
bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData &tiling_data, int32_t tilingCaseId, AutofuseTilingData* autofuseTilingData, void* stream, uint32_t workspaceSize, double& out_best_perf) {return true;} 
bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData* tiling_data, uint32_t max_block_dim=48) {return true;}
}

#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/tiling_context.h"
#endif
extern "C" size_t GetTilingDataSize()
{
  return sizeof(AutofuseTilingData);
}

uint32_t GetWorkspaceSize(const AutofuseTilingData &t) {
  using namespace optiling;
  uint32_t ws_size = 0;

  ws_size = (ws_size + 512 - 1) / 512 * 512;
  return ws_size;
}

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};
constexpr ResLimit g_no_limit_res = {1, 48, 0, 192 * 1024, {}};
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size)
{
  tiling->set_block_dim(aiv_num);
  tiling->set_ub_size(ub_size);
  if (!optiling::GetTiling(*tiling, -1)) {
      return -1;
  }
  *blockDim = tiling->get_block_dim();
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}
extern "C" int64_t PgoAutofuseTiling(const char* config_file, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr)
{
 const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  tiling->set_block_dim(limit->aiv_num);
  tiling->set_ub_size(limit->ub_size);
  if (!optiling::GetTiling(*tiling, -1)) {
    return -1;
  }
  *blockDim = tiling->get_block_dim();
  using namespace optiling;
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}

#ifndef __CCE_KT_TEST__
extern "C" bool AutofuseIsStaticShape() {
  return true;
}
extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{

  return -1;
}

namespace gert {
  class TilingSymbolEvalContext : public TilingContext {
    public:
      const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
        auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
        if (tensor == nullptr) {
          return nullptr;
        }
        return tensor;
      }
  };

  class SymbolTilingParseContext : public KernelContext {
    public:
      fe::PlatFormInfos *GetPlatFormInfos() const {
        auto platform = GetInputValue<fe::PlatFormInfos *>(0);
        if (platform == nullptr) {
          return nullptr;
        }
        return platform;
      }
  };
}
struct AfTilingParseData{
 uint32_t aiv_num;
 uint64_t ub_size;
};
extern "C" ge::graphStatus TilingParse(gert::SymbolTilingParseContext *context) {
 auto platform = context->GetPlatFormInfos();
 if (platform == nullptr) {
 return ge::GRAPH_FAILED;
 }
 auto ascendc_platform = platform_ascendc::PlatformAscendC(platform);
 uint32_t platform_core_num = ascendc_platform.GetCoreNumAiv();
 uint32_t aiv_num = 0;
 uint64_t ub_size = (184 * 1024);
 aiv_num = platform_core_num;
 ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
 auto extend_context = reinterpret_cast<gert::KernelContext *>(context);
 auto tiling_parse_data_av = extend_context->GetOutput(0);
 if (tiling_parse_data_av == nullptr) {
 return ge::GRAPH_FAILED;
 }
 auto tiling_parse_data_ptr = new (std::nothrow) uint8_t[sizeof(AfTilingParseData)];
 if (tiling_parse_data_ptr == nullptr) {
 return ge::GRAPH_FAILED;
 }
 tiling_parse_data_av->SetWithDefaultDeleter<uint8_t[]>(tiling_parse_data_ptr);
 auto tiling_parse_data = extend_context->GetOutputPointer<AfTilingParseData *>(0);
 (*tiling_parse_data)->aiv_num = aiv_num;
 ub_size -= (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95 && ub_size % 1024 == 0) ? 256 : 0;
 (*tiling_parse_data)->ub_size = ub_size;
 return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)
{
  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);
  auto input_data_num =  extend_context->GetInputValue<size_t>(0U);
  auto parse = extend_context->GetInputValue<AfTilingParseData*>(input_data_num + 1);
  auto tiling_data =  context->GetTilingData<AutofuseTilingData>();
  uint32_t workspace_size;
  uint32_t block_dim;
  static const char* config_file = "/test_graph_config.txt";
  ResLimit limit;
  limit.aiv_num = parse->aiv_num;
  limit.ub_size = (uint32_t)parse->ub_size;
  auto ret = PgoAutofuseTiling(config_file, tiling_data, &workspace_size, &block_dim, &limit);
  context->SetBlockDim(block_dim);
  *context->GetWorkspaceSizes(1) = workspace_size;

  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
  return ret;
}

extern "C" ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context)
{
  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);
  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);
  if (symbol_src_vec == nullptr) {
    return ge::GRAPH_FAILED;
  }

  symbol_src_vec->SetSize(0);
  return ge::GRAPH_SUCCESS;
}
extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;

  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
#endif

std::string tiling_data_const_gen_result;
AutofuseTilingData TilingDataValue;

void replaceSubstring(std::string& ori_str, const std::string& old_sub_str, const std::string& new_sub_str) {
  size_t pos = ori_str.find(old_sub_str);
  if (pos != std::string::npos) {
    ori_str.replace(pos, old_sub_str.length(), new_sub_str);
  }
}

std::string GenTilingDataFieldConstDefFunc(const std::string &f_name, uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << "const uint32_t ";
  ss_mid << f_name << " = " << std::to_string(value) << ";" << std::endl;
  return ss_mid.str();
}

std::string GenTilingDataFieldConstValueFunc(uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << std::to_string(value) << std::endl;
  return ss_mid.str();
}


extern "C" const char* GenConstTilingData(char* config_file, int aiv_num, int ub_size) {
  uint32_t workspace_size;
  uint32_t block_dim;
  ResLimit limit;
  limit.aiv_num = aiv_num;
  limit.ub_size = ub_size - 256;
  (void)PgoAutofuseTiling(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit);
  std::string GenTilingDataValue_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.block_dim);
  std::string GenTilingDataValue_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.corenum);
  std::string GenTilingDataValue_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.ub_size);
  std::string GenTilingDataValue_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.hbm_size);
  std::string GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("graph0_tiling_key", TilingDataValue.graph0_tiling_key);

  tiling_data_const_gen_result = R"(#ifndef __Autofuse_Tiling_Data_H__
#define __Autofuse_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
  GenTilingDataValue_block_dim_field_DeclareFunc_def
  GenTilingDataValue_corenum_field_DeclareFunc_def
  GenTilingDataValue_ub_size_field_DeclareFunc_def
  GenTilingDataValue_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def
END_TILING_DATA_DEF_T;

struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)";
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_block_dim_field_DeclareFunc_def",GenTilingDataValue_block_dim_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_corenum_field_DeclareFunc_def",GenTilingDataValue_corenum_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_ub_size_field_DeclareFunc_def",GenTilingDataValue_ub_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_hbm_size_field_DeclareFunc_def",GenTilingDataValue_hbm_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def",GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def);

  return tiling_data_const_gen_result.c_str();
}


#ifndef __CCE_KT_TEST__
std::string kernel_type;
extern "C" const char* GetTilingKeyKernelTypeForStatic()
{
  const std::map<int64_t, std::string> kernel_type_map = {
  };

  auto tiling_key = FindBestTilingKey(TilingDataValue);
  auto it = kernel_type_map.find(tiling_key);
  if (it != kernel_type_map.end()) {
    kernel_type = it->second;
  }
  return kernel_type.c_str();
}
#endif
)rawliteral";
  EXPECT_EQ(tiling_code, expect_code);
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling_With_Set_Vector_Core_Num)
{
  ge::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::string s0_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(0);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s1_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s2_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(2);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";

  std::map<std::string, std::string> shape_info = {{"s0", s0_source},
                                                   {"s1", s1_source},
                                                   {"s2", s2_source}};
  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  auto tiling_codes = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "20");
}

TEST_F(E2E_LoadAbsStore, Codegen_PGO_Code)
{
  ge::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  codegen::Codegen codegen(codegen::CodegenOptions{.tiling_lib_path="asdf",.tiling_lib_codegen_symbol="as"});
  std::string pgo_codes = codegen.GeneratorPgo(fused_schedule_result, "", "48", "196608", "0");
  unsetenv("AUTOFUSE_FLAGS");
  std::string expect_code = R"rawliteral(#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <unistd.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/wait.h>
#include <cerrno>
#include <thread>
#include <chrono>
#include <cfloat>
#include <cstring>
#include <fstream>
#include <dlfcn.h>

#include "acl/acl.h"
#include "autofuse_tiling_data.h"
#include "runtime/rt.h"
#include "mspti.h"
#include "tiling/platform/platform_ascendc.h"
namespace {
std::vector<uint32_t> g_mix_graph0_tiling_keys = {
};
bool IsMixTiling(const AutofuseTilingData& t) {
  if (!g_mix_graph0_tiling_keys.empty() && std::find(g_mix_graph0_tiling_keys.begin(), g_mix_graph0_tiling_keys.end(), t.graph0_tiling_key) != g_mix_graph0_tiling_keys.end()) {
    return true;
  }
  return false;
}
const bool g_is_mix_operator = false;
static bool g_is_static_kernel = false;
const char *config_file = "/test_graph_config.txt";
const char *search_file = "/test_graph_search.txt";
const char *kernel_file = "/libtest_graph.so";
const char *npu_lock_file = "/npu_lock_0.lock";
#define SUCCESS 0
#define FAILED 1
#undef DLOG
static bool debug = true;
#define DLOG() if (debug) std::cerr << "[PGO] "

class CardLock {
public:
  CardLock(const char* path) {
    fd_ = open(path, O_RDWR | O_CREAT, 0666);
    if (fd_ == -1) {
      DLOG() << "open lock file: " << std::strerror(errno) << std::endl;
      std::exit(1);
    }
    if (flock(fd_, LOCK_EX) == -1) {
      DLOG() << "flock LOCK_EX: " << std::strerror(errno) << std::endl;
      std::exit(1);
    }
  }

  ~CardLock() {
    if (fd_ != -1) {
      if (flock(fd_, LOCK_UN) == -1) {
        DLOG() << "flock LOCK_UN: " << std::strerror(errno) << std::endl;
      }
      close(fd_);
    }
  }

  CardLock(const CardLock&) = delete;
  CardLock& operator=(const CardLock&) = delete;

private:
  int fd_{-1};
};

void AppendPgoSearchTilingData(AutofuseTilingData& tiling_data, double best_perf, std::ios::openmode mode = std::ios::app) {
  DLOG() << "AppendPgoSearchTilingData to file: " << search_file << std::endl;
  std::ofstream out_file(search_file, mode);
  if (!out_file.is_open()) {
    DLOG() << "Failed to open file:" << search_file << std::endl;
    return;
  }
  auto it = &tiling_data;
  out_file << sizeof(it->block_dim) << " " << it->block_dim << ";";
  out_file << sizeof(it->corenum) << " " << it->corenum << ";";
  out_file << sizeof(it->ub_size) << " " << it->ub_size << ";";
  out_file << sizeof(it->hbm_size) << " " << it->hbm_size << ";";
    out_file << sizeof(it->graph0_tiling_key) << " " << it->graph0_tiling_key << ";";
  out_file << " # " << best_perf;
  out_file << std::endl;
  out_file.close();

  int fd = ::open(search_file, O_WRONLY);
  if (fd < 0) {
    DLOG() << "Failed to open file:" << search_file << std::endl;
    return;
  }
  if (::fsync(fd) < 0) {
    DLOG() << "Failed to fsync file:" << search_file << std::endl;
  }
  ::close(fd);

  return;
}
struct AivKernelLaunchOpArgs {
  uint64_t workspace_addr;
  uint64_t tiling_addr;
  AutofuseTilingData tiling_data;
};
struct MixKernelLaunchOpArgs {
  uint64_t ffts;
  uint64_t workspace_addr;
  uint64_t tiling_addr;
  AutofuseTilingData tiling_data;
};
static void *handle = nullptr;
static bool initialized = false;

__attribute__((constructor)) void Init() {
  if (initialized) return;
  handle = dlopen(kernel_file, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    DLOG() << "Failed to load " << kernel_file << ": " << dlerror() << std::endl;
    return;
  }
  DLOG() << "Kernel api lib " << kernel_file << " load succeed" << std::endl;
  initialized = true;
}

__attribute__((destructor)) void DeInit() {
  if (handle) {
    dlclose(handle);
    handle = nullptr;
  }
  initialized = false;
}

inline void *GetFunc(const char *func_name) {
  if (handle == nullptr) {
    return nullptr;
  }
  void *func = dlsym(handle, func_name);
  if (func == nullptr) {
    DLOG() << "Failed to load wrapper api func: " << dlerror() << std::endl;
  }
  return func;
}
aclrtStream g_stream;
uint64_t ffts;

void* g_tiling_device_addr = nullptr;
struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};
ResLimit g_res_limit = {1, {}};
inline bool IsEqual(double a, double b) {
  const double epsilon = 1e-8;
  double abs = (a > b) ? (a - b) : (b - a);
  return abs < epsilon;
}
} // namespace
typedef void (*GetKernelBinType)(std::vector<char>& kernel_bin);
GetKernelBinType get_kernel_bin_fn = reinterpret_cast<GetKernelBinType>(GetFunc("GetKernelBin"));
typedef int64_t (*FindBestTilingKeyType)(AutofuseTilingData &t);
FindBestTilingKeyType find_best_tiling_key_fn = reinterpret_cast<FindBestTilingKeyType>(GetFunc("FindBestTilingKey"));
int WrapperOnlyLaunch(uint32_t workspace_size, AutofuseTilingData *tiling_data) {
  static bool inited = false;
  static std::vector<char> kernelBin;
  static rtDevBinary_t aiv_binary = {}, mix_binary = {};
  static void *aiv_bin_handle = nullptr, *mix_bin_handle = nullptr;
  static rtTaskCfgInfo_t cfg = {};
  static AivKernelLaunchOpArgs kAivArgs;
  static MixKernelLaunchOpArgs kMixArgs;
  static rtArgsEx_t aiv_args = {}, mix_args = {};
  if (!inited) {
    if (get_kernel_bin_fn == nullptr) {
      DLOG() << "GetKernelBin func not found" << std::endl;
      return -1;
    }
    get_kernel_bin_fn(kernelBin);
    aiv_binary.version = mix_binary.version = 0U;
    aiv_binary.data = mix_binary.data = kernelBin.data();
    aiv_binary.length = mix_binary.length = kernelBin.size();
    aiv_binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
    mix_binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    auto ret = rtRegisterAllKernel(&aiv_binary, &aiv_bin_handle);
    if (ret != RT_ERROR_NONE) {
      DLOG() << "rtRegisterAllKernel failed. ERROR: " << ret << std::endl;
      return FAILED;
    }
    ret = rtRegisterAllKernel(&mix_binary, &mix_bin_handle);
    if (ret != RT_ERROR_NONE) {
      DLOG() << "rtRegisterAllKernel failed. ERROR: " << ret << std::endl;
      return FAILED;
    }
    uint32_t len = 0;
    ret = rtGetC2cCtrlAddr(&ffts, &len);
    kMixArgs.ffts = ffts;
    kAivArgs.tiling_addr = kMixArgs.tiling_addr = reinterpret_cast<uint64_t>(g_tiling_device_addr);
    aiv_args.args = reinterpret_cast<void*>(&kAivArgs);
    mix_args.args = reinterpret_cast<void*>(&kMixArgs);
    aiv_args.argsSize = sizeof(AivKernelLaunchOpArgs);
    mix_args.argsSize = sizeof(MixKernelLaunchOpArgs);
    aiv_args.tilingAddrOffset = 1 * sizeof(uint64_t);
    aiv_args.tilingDataOffset = 2 * sizeof(uint64_t);
    mix_args.tilingAddrOffset = 2 * sizeof(uint64_t);
    mix_args.tilingDataOffset = 3 * sizeof(uint64_t);
    aiv_args.hasTiling = mix_args.hasTiling = 1;
    aiv_args.isNoNeedH2DCopy = mix_args.isNoNeedH2DCopy = false;
    inited = true;
  }
  if (tiling_data == nullptr) {
    DLOG() << "test_graph tiling_data is null" << std::endl;
    return -1;
  }
  uint32_t block_dim = tiling_data->block_dim;
  auto ret = aclrtMemcpy((void *)kAivArgs.tiling_addr, sizeof(AutofuseTilingData), (void *)tiling_data, sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    DLOG() << "test_graph memcpy tiling data to device failed, ERROR: " << ret << std::endl;
  }
  ret = aclrtMemcpy((void *)kMixArgs.tiling_addr, sizeof(AutofuseTilingData), (void *)tiling_data, sizeof(AutofuseTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
  if (ret != ACL_SUCCESS) {
    DLOG() << "test_graph memcpy tiling data to device failed, ERROR: " << ret << std::endl;
  }
  kAivArgs.tiling_data = kMixArgs.tiling_data = *tiling_data;
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto ret = aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
      DLOG() << "test_graph malloc workspace failed, size: " << workspace_size << ", ERROR: " << ret << std::endl;
      return FAILED;
    }
  }
  kAivArgs.workspace_addr = kMixArgs.workspace_addr = reinterpret_cast<uint64_t>(workspace);
  aiv_args.args = reinterpret_cast<void*>(&kAivArgs);
  mix_args.args = reinterpret_cast<void*>(&kMixArgs);
  int64_t tiling_key = 0;
  if (find_best_tiling_key_fn != nullptr) {
    tiling_key = find_best_tiling_key_fn(*tiling_data);
    if (tiling_key == -1) {
      DLOG() << "test_graph find best tiling file failed" << std::endl;
      return FAILED;
    }
  } else {
    DLOG() << "find best tiling key func is null" << std::endl;
    return FAILED;
  }
  if (g_is_mix_operator) {
    if (!g_is_static_kernel || IsMixTiling(*tiling_data)) {
      ret = rtKernelLaunchWithHandleV2(mix_bin_handle, tiling_key, block_dim, &mix_args, nullptr, g_stream, &cfg);
    } else {
      ret = rtKernelLaunchWithHandleV2(aiv_bin_handle, tiling_key, block_dim, &aiv_args, nullptr, g_stream, &cfg);
    }
  } else {
    ret = rtKernelLaunchWithHandleV2(aiv_bin_handle, tiling_key, block_dim, &aiv_args, nullptr, g_stream, &cfg);
  }
  auto ret_async = aclrtSynchronizeStream(g_stream);
  if (workspace != nullptr) {
    auto ret = aclrtFree(workspace);
    if (ret != ACL_SUCCESS) {
      DLOG() << "test_graph kernel free workspace failed, size: " << workspace_size << ", ERROR: " << ret << std::endl;
    }
  }
  if (ret != ACL_SUCCESS) {
    DLOG() << "test_graph rtKernelLaunchWithHandleV2 failed, ERROR: " << ret << std::endl;
    return FAILED;
  }
  if (ret_async != ACL_SUCCESS) {
    DLOG() << "test_graph aclrtSynchronizeStream failed, ERROR: " << ret_async << std::endl;
    return FAILED;
  }
  return ret;
}

#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
    (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))
constexpr size_t group_size = 1000;
static std::map<uint64_t, msptiActivity*> g_profiling_map;
constexpr uint64_t loop = 20;
constexpr int max_flush_times = 5;
static double best_perf = DBL_MAX;

static const char* GetActivityKindString(msptiActivityKind kind) {
  static const std::unordered_map<msptiActivityKind, const char*> STRING_MAP = {
    {MSPTI_ACTIVITY_KIND_INVALID, "INVALID"},
    {MSPTI_ACTIVITY_KIND_MARKER, "MARKER"},
    {MSPTI_ACTIVITY_KIND_KERNEL, "KERNEL"},
    {MSPTI_ACTIVITY_KIND_API, "API"},
    {MSPTI_ACTIVITY_KIND_HCCL, "HCCL"},
    {MSPTI_ACTIVITY_KIND_MEMORY, "MEMORY"},
    {MSPTI_ACTIVITY_KIND_MEMSET, "MEMSET"},
    {MSPTI_ACTIVITY_KIND_MEMCPY, "MEMCPY"},
    {MSPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION, "CORRELATION"}
  };
  auto it = STRING_MAP.find(kind);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
}

static const char* GetResultCodeString(msptiResult result) {
  static const std::unordered_map<msptiResult, const char*> STRING_MAP = {
    {MSPTI_SUCCESS, "SUCCESS"},
    {MSPTI_ERROR_INVALID_PARAMETER, "ERROR_INVALID_PARAMETER"},
    {MSPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED, "MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED"},
    {MSPTI_ERROR_DEVICE_OFFLINE, "DEVICE_OFFLINE"},
    {MSPTI_ERROR_QUEUE_EMPTY, "QUEUE_EMPTY"},
    {MSPTI_ERROR_INNER, "ERROR_INNER"}
  };

  auto it = STRING_MAP.find(result);
  return it != STRING_MAP.end() ? it->second : "<unknown>";
}

void UserBufferRequest(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  DLOG() << "[mspti] UserBufferRequest..." << std::endl;
  uint8_t *pBuffer = reinterpret_cast<uint8_t *>(malloc(16 * 1024 * 1024 + ALIGN_SIZE));
  *buffer = ALIGN_BUFFER(pBuffer, ALIGN_SIZE);
  *size = 16 * 1024 * 1024;
  *maxNumRecords = 0;
}

void UserBufferComplete(uint8_t *buffer, size_t size, size_t validSize) {
  DLOG() << "[mspti] UserBufferComplete, buf addr: " << reinterpret_cast<uintptr_t>(buffer) << ", size: " << size
    << ", valid size: " << validSize << std::endl;
  if (validSize > 0) {
    msptiActivity *pRecord = NULL;
    msptiResult status = MSPTI_SUCCESS;
    do {
      status = msptiActivityGetNextRecord(buffer, validSize, &pRecord);
      if (status == MSPTI_SUCCESS) {
        if (pRecord->kind == MSPTI_ACTIVITY_KIND_KERNEL) {
          msptiActivityKernel* kernelRecord = (msptiActivityKernel*)pRecord;
          msptiActivity* pRecordCopy = (msptiActivity *)malloc(sizeof(msptiActivityKernel));
          memset(pRecordCopy, 0, sizeof(msptiActivityKernel));
          memcpy(pRecordCopy, kernelRecord, sizeof(msptiActivityKernel));
          g_profiling_map[kernelRecord->start] = pRecordCopy;

        } else {
          DLOG() << "[mspti] [" << GetActivityKindString(pRecord->kind) << "] ignored" << std::endl;
        }
      } else if (status == MSPTI_ERROR_MAX_LIMIT_REACHED) {
        break;
      } else {
        DLOG() << "[mspti] Consume data fail error is" << GetResultCodeString(status) << std::endl;
        break;
      }
    } while (1);
  }
  free(buffer);
}

void SetUpMspti(msptiSubscriberHandle* subscriber) {
  DLOG() << "[mspti] setup mspti" << std::endl;
  msptiSubscribe(subscriber, nullptr, nullptr);
  msptiActivityRegisterCallbacks(UserBufferRequest, UserBufferComplete);
  msptiActivityEnable(MSPTI_ACTIVITY_KIND_KERNEL);
}

void TearDownMspti(msptiSubscriberHandle* subscriber) {
  DLOG() << "[mspti] tear down mspti" << std::endl;
  msptiUnsubscribe(*subscriber);
  msptiActivityFlushAll(1);
}
int ProfilingBatchProcess(uint32_t workspace_size, std::vector<AutofuseTilingDataPerf>::iterator begin, std::vector<AutofuseTilingDataPerf>::iterator end) {
  uint64_t batch_size = end - begin;
  g_profiling_map.clear();
  msptiSubscriberHandle subscriber;
  SetUpMspti(&subscriber);

  static int64_t count = 0;
  count++;

  int64_t result = 0;
  for (auto it = begin; it != end; ++it) {
    it->best_perf = DBL_MAX;
    AutofuseTilingData &tiling_data = it->tiling_data;
    for (uint64_t i = 0; i < loop; ++i) {
      result = WrapperOnlyLaunch(workspace_size, &tiling_data);
      if (result != 0) {
        DLOG() << "test_graph ProfilingBatchProcess launch failed loop:" << i << std::endl;
        TearDownMspti(&subscriber);
        return -1;
      }
    }
  }

  result = aclrtSynchronizeStream(g_stream);
  TearDownMspti(&subscriber);

  int flush_count = 0;
  while (g_profiling_map.size() < batch_size * loop && flush_count < max_flush_times) {
    flush_count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));
    msptiActivityFlushAll(1);
  }

  if (g_profiling_map.size() < batch_size * loop) {
    DLOG() << "test_graph ProfilingBatchProcess g_profiling_map size " << g_profiling_map.size() << " is less than batch_size * loop " << batch_size * loop << std::endl;
    for (auto &item : g_profiling_map) {
      free(item.second);
    }
    return -1;
  }

  auto it = g_profiling_map.begin();
  for (uint64_t i = 0; i < batch_size; ++i) {
    uint64_t total_duration = 0;
    std::vector<uint64_t> durations;
    for (uint64_t j = 0; j < loop; ++j) {
      msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(it->second);
      durations.push_back(kernel->end - kernel->start);
      std::advance(it, 1);
    }
    std::sort(durations.begin(), durations.end(), std::greater<int>());
    for (size_t k = 1; k < 6; ++k) {
      total_duration += durations[k];
    }
    double average_duration = static_cast<double>(total_duration) / 5;
    (begin + i)->best_perf = average_duration;
    if (best_perf > average_duration) {
      best_perf = average_duration;
    }
    DLOG() << "average_duration:" << average_duration << " best_perf:" << best_perf << " count:" << count << " batch_size:" << batch_size << " flush_count:" << flush_count << std::endl;
  }
  for (auto &item : g_profiling_map) {
    free(item.second);
  }
  return 0;
}

extern "C" long int PGOGetProfilingBatch(void* stream, uint32_t workspace_size, std::vector<AutofuseTilingDataPerf> *profiles) {
  int case_num = profiles->size();
  DLOG() << "test_graph PGOGetProfilingBatch case_num:" << case_num << std::endl;
  int64_t result = 0;
  auto it = profiles->begin();
  while (it != profiles->end()) {
    auto end_it = (it + group_size >= profiles->end()) ? profiles->end() : it + group_size;
    size_t start_index = std::distance(profiles->begin(), it);
    for (int i = 0; i < 3; i++) {
      result = ProfilingBatchProcess(workspace_size, it, end_it);
      if (result != 0) {
        DLOG() << "test_graph ProfilingBatchProcess failed at start_index:" << start_index << " retry time:" << i << std::endl;
      } else {
        break;
      }
    }
  it = end_it;
  }
  return 0;
}

extern "C" long int PGOGetProfiling(void* stream, uint32_t workspace_size, AutofuseTilingData* tiling_data, double* outCostTime) {
  g_profiling_map.clear();
  msptiSubscriberHandle subscriber;
  SetUpMspti(&subscriber);

  int64_t result = -1;
  *outCostTime = DBL_MAX;
  static int64_t count = 0;
  count++;

  for (uint64_t j = 0; j < loop; ++j) {
    result = WrapperOnlyLaunch(workspace_size, tiling_data);
    if (result != 0) {
      DLOG() << "test_graph launch failed loop:" << j << std::endl;
      TearDownMspti(&subscriber);
      return -1;
    }
  }

  result = aclrtSynchronizeStream(g_stream);
  if (result != 0) {
    DLOG() << "test_graph sync stream failed" << std::endl;
    TearDownMspti(&subscriber);
    return -1;
  }
  TearDownMspti(&subscriber);

  int flush_count = 0;
  while (g_profiling_map.size() < loop && flush_count < max_flush_times) {
    flush_count++;
    std::this_thread::sleep_for(std::chrono::milliseconds(10 * flush_count));
    msptiActivityFlushAll(1);
  }

  if (g_profiling_map.size() != loop) {
    DLOG() << "test_graph map size " << g_profiling_map.size() << " not equals to loop " << loop << std::endl;
    for (auto &item : g_profiling_map) {
      free(item.second);
    }
    return -1;
  }

  uint64_t total_duration = 0;
  std::vector<uint64_t> durations;
  for (const auto& pair : g_profiling_map) {
    msptiActivityKernel* kernel = reinterpret_cast<msptiActivityKernel*>(pair.second);
    durations.push_back(kernel->end - kernel->start);
    DLOG() << kernel->end - kernel->start << ", ";
  }
  DLOG() << std::endl;
  std::sort(durations.begin(), durations.end(), std::greater<int>());
  for (size_t i = 1; i < 6; ++i) {
    total_duration += durations[i];
  }
  double average_duration = static_cast<double>(total_duration) / 5;
  *outCostTime = average_duration;

  if (best_perf > *outCostTime) {
    best_perf = *outCostTime;
  }
  DLOG() << "average_duration:" << *outCostTime << " best_perf:" << best_perf << " count:" << count << " flush_count:" << flush_count << std::endl;
  for (auto &item : g_profiling_map) {
    free(item.second);
  }
  return 0;
}

typedef int64_t (*PGOSearchType)(char* search_file, char* config_file, AutofuseTilingData* tiling_data, uint32_t* workspace_size, uint32_t* blockDim, void *resource_limit,void *stream, void* prof_callback, void *prof_batch_callback);
static PGOSearchType pgo_search_fn = reinterpret_cast<PGOSearchType>(GetFunc("PgoTilingSearch"));
int pgo() {
  AutofuseTilingData tiling_data = {0};
  uint32_t workspace_size = 0;
  uint32_t block_dim = 0;
  if (pgo_search_fn == nullptr) {
    DLOG() << "test_graph pgo search func not found" << std::endl;
    return -1;
  }
  int64_t result = pgo_search_fn((char*)search_file, (char *)config_file, &tiling_data, &workspace_size, &block_dim, &g_res_limit,&g_stream, reinterpret_cast<void*>(PGOGetProfiling), reinterpret_cast<void*>(PGOGetProfilingBatch));
  if (result != 0) {
    DLOG() << "test_graph pgo search failed. ERROR: " << result << std::endl;
    return -1;
  }
  return 0;
}

typedef int64_t (*PgoAutofuseTilingType)(const char* config_file, AutofuseTilingData* tiling, uint32_t* workspace_size, uint32_t *blockDim, ResLimit *res_limit);
static PgoAutofuseTilingType pgo_autofuse_tiling_fn = reinterpret_cast<PgoAutofuseTilingType>(GetFunc("PgoAutofuseTiling"));
int static_pgo(const char* config_file) {
  if (pgo_autofuse_tiling_fn == nullptr) {
    DLOG() << "test_graph pgo autofuse tiling func not found" << std::endl;
    return -1;
  }
  AutofuseTilingData tiling_data = {0};
  uint32_t workspace_size = 0;
  uint32_t block_dim = 0;
  int64_t result = pgo_autofuse_tiling_fn(config_file, &tiling_data, &workspace_size, &block_dim, &g_res_limit);
  if (result != 0) {
    DLOG() << "test_graph pgo autofuse tiling failed. ERROR: " << result << std::endl;
    return -1;
  }
  double out_cost = DBL_MAX;
  for (int i = 0; i < max_flush_times; i++) {
    result = PGOGetProfiling(g_stream, workspace_size, &tiling_data, &out_cost);
    if (result != 0 || IsEqual(out_cost, DBL_MAX)) {
      DLOG() << "test_graph get profiling failed." << std::endl;
    } else {
      break;
    }
  }
  AppendPgoSearchTilingData(tiling_data, out_cost);
  return 0;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    DLOG() << "Usage: " << argv[0] << " <type>" << std::endl;
    return -1;
  }
  int32_t type = static_cast<int32_t>(atoi(argv[1]));
  CardLock lock(npu_lock_file);
  int32_t device_id = 0;
  int32_t aiv_num = 48;
  int32_t ub_size = 196608;
  g_res_limit.aiv_num = aiv_num;
  g_res_limit.ub_size = ub_size;
  auto ret = aclInit(nullptr);
  if (ret != ACL_SUCCESS) {
    DLOG() << "acl init failed, ERROR: " << ret << std::endl;
    return FAILED;
  }
  ret = aclrtSetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    DLOG() << "acl set device failed, device id: " << device_id << ", ERROR: " << ret << std::endl;
    aclFinalize();
    return FAILED;
  }
  ret = aclrtCreateStream(&g_stream);
  if (ret != ACL_SUCCESS) {
    DLOG() << "acl create stream failed, ERROR: " << ret << std::endl;
    aclrtResetDevice(device_id);
    aclFinalize();
    return FAILED;
  }

  ret = aclrtMalloc(&g_tiling_device_addr, sizeof(AutofuseTilingData), ACL_MEM_MALLOC_HUGE_FIRST);
  if (ret != ACL_SUCCESS) {
    DLOG() << "acl malloc tiling data failed, ERROR: " << ret << std::endl;
    return FAILED;
  }
  if (type == 0) {
    ret = pgo();
  } else if (type == 1) {
    g_is_static_kernel = true;
    ret = static_pgo(config_file);
  } else {
    DLOG() << "Invalid type: " << type << std::endl;
    ret = -1;
  }

  if (g_tiling_device_addr != nullptr) {
    ret = aclrtFree(g_tiling_device_addr);
    if (ret != ACL_SUCCESS) {
      DLOG() << "acl free tiling data failed, ERROR: " << ret << std::endl;
      return FAILED;
    }
    g_tiling_device_addr = nullptr;
  }
  ret = aclrtDestroyStream(g_stream);
  if (ret != ACL_SUCCESS) {
    DLOG() << "acl destroy stream failed, ERROR: " << ret << std::endl;
    return FAILED;
  }
  ret = aclrtResetDevice(device_id);
  if (ret != ACL_SUCCESS) {
    DLOG() << "acl reset device failed, device id: " << device_id << ", ERROR: " << ret << std::endl;
    return FAILED;
  }
  ret = aclFinalize();
  if (ret != ACL_SUCCESS) {
    DLOG() << "acl finalize failed, ERROR: " << ret << std::endl;
    return FAILED;
  }
  DeInit();
  return ret;
}
)rawliteral";
  EXPECT_EQ(expect_code, pgo_codes);
}

TEST_F(E2E_LoadAbsStore, Codegen_Tiling_With_LambdaWithPGO)
{
  ge::AscGraph test_graph("test_graph");
  LoadAbsStore_BeforeAutofuse(test_graph);
  LoadAbsStore_AfterInferOutput(test_graph);

  std::vector<ge::AscGraph> test_impl_graphs = {ge::AscGraph("test_impl_graph")};
  test_impl_graphs[0].CopyFrom(test_graph);
  LoadAbsStore_AfterGetApiInfo(test_impl_graphs[0]);
  LoadAbsStore_AfterScheduler(test_impl_graphs[0]);
  LoadAbsStore_AfterQueBufAlloc(test_impl_graphs[0]);

  std::string s0_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(0);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s1_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(1);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";
  std::string s2_source = R"([&]() -> int64_t {
    auto *tensor = context->GetGraphInputTensor(2);
    if (tensor == nullptr) {
      return gert::Shape::kInvalidDimValue;
    }
    return tensor->GetOriginShape().GetDim(1);
  }())";

  std::map<std::string, std::string> shape_info = {{"s0", s0_source},
                                                   {"s1", s1_source},
                                                   {"s2", s2_source}};
  FusedScheduledResult fused_schedule_result;
  fused_schedule_result.fused_graph_name = ge::AscendString(test_graph.GetName().c_str());
  std::vector<ScheduledResult> schedule_results;
  fused_schedule_result.node_idx_to_scheduled_results.push_back(schedule_results);
  setenv("AUTOFUSE_FLAGS", "--autofuse_enable_pgo=true", 1);
  att::AutoFuseConfig::MutablePgoStrategyConfig().is_first_init = true;
  ASSERT_EQ(att::AutoFuseConfig::MutablePgoStrategyConfig().Init(), ge::SUCCESS);
  EXPECT_EQ(att::AutoFuseConfig::GetPgoStrategyConfig().enable_autofuse_pgo, "true");
  auto tiling_codes = codegen.GenerateTiling(fused_schedule_result, shape_info, "", "0");
  for (const auto&[key,value] : tiling_codes) {
    std::cout << key <<std::endl;
    std::cout << value <<std::endl;
  }
  std::string tiling_code;
  CombineTilings(tiling_codes, tiling_code);
  unsetenv("AUTOFUSE_FLAGS");
  std::string expect_code = R"rawliteral(#include <stdexcept>
#include <sstream>
#include <cmath>
#include "autofuse_tiling_data.h"
#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "platform/platform_infos_def.h"
#include "platform_ascendc.h"
#endif

#include <cfloat>
#include <vector>

typedef long int (*ProfilingCallback)(void* stream, uint32_t workspaceSize, AutofuseTilingData* tiling_data, double* cost_time);
typedef long int (*ProfilingBatchCallback)(void* stream, uint32_t workspaceSize, std::vector<AutofuseTilingDataPerf> *profiles);
class PgoConfig {
public:
  static PgoConfig& Instance() {
    static PgoConfig instance;
    return instance;
  }
  ProfilingCallback single_callback;
  ProfilingBatchCallback batch_callback;
  int32_t pgo_algorithm = 1; // 0 for pruning, 1 for core num
  bool need_change_solver_run = false;
private:
  PgoConfig() = default;
  ~PgoConfig() = default;
  PgoConfig(const PgoConfig &) = delete;
  PgoConfig &operator=(const PgoConfig &) = delete;
};

#include <iostream>
#include <fstream>
#include <cinttypes>
#include <sys/syscall.h>
#include <unistd.h>
#include "dlog_pub.h"
#define OP_LOGD(name, fmt, ...)
#define OP_LOGI(name, fmt, ...)
#define GE_MODULE_NAME static_cast<int32_t>(45)
inline uint64_t GetTid() {
     return static_cast<uint64_t>(syscall(__NR_gettid));
}
#define GELOGE(ERROR_CODE, fmt, ...)
#define OP_LOGE(name, fmt, ...)
#define OP_NAME "asc0000_autofused_abs"
#define Max(a, b) ((double)(a) > (double)(b) ? (a) : (b))
#define Min(a, b) ((double)(a) < (double)(b) ? (a) : (b))
#define Log(a) (log((double)(a)))
#define Pow(a, b) pow(a, b)
#define Rational(a, b) ((double)(a) / (double)(b))

namespace optiling {
extern "C" bool GetTiling(AutofuseTilingData& tiling_data, int32_t tilingCaseId=-1) {
  return true;
}
inline bool IsEqual(double a, double b) {return true;}
bool PGOSearchTilingKey(std::vector<AutofuseTilingDataPerf>& tiling_data_list, AutofuseTilingData &tiling_data, int32_t tilingCaseId, AutofuseTilingData* autofuseTilingData, void* stream, uint32_t workspaceSize, double& out_best_perf) {return true;} 
bool PGOByCoreNumSearchTilingKey(std::vector<AutofuseTilingData>& tiling_data_list, AutofuseTilingData* tiling_data, uint32_t max_block_dim=48) {return true;}
}

#ifndef __CCE_KT_TEST__
#include "exe_graph/runtime/tiling_context.h"
#endif
extern "C" size_t GetTilingDataSize()
{
  return sizeof(AutofuseTilingData);
}

uint32_t GetWorkspaceSize(const AutofuseTilingData &t) {
  using namespace optiling;
  uint32_t ws_size = 0;

  ws_size = (ws_size + 512 - 1) / 512 * 512;
  return ws_size;
}

struct ResLimit {
  uint32_t valid_num = 0;
  uint32_t aiv_num = 0;
  uint32_t aic_num = 0;
  uint32_t ub_size = 0;
  uint32_t resv[10];
};
constexpr ResLimit g_no_limit_res = {1, 48, 0, 192 * 1024, {}};
extern "C" int64_t AutofuseTiling(AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, uint32_t aiv_num, uint32_t ub_size)
{
  tiling->set_block_dim(aiv_num);
  tiling->set_ub_size(ub_size);
  if (!optiling::GetTiling(*tiling, -1)) {
      return -1;
  }
  *blockDim = tiling->get_block_dim();
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}
bool PGOGetTilingKey(const char* config_file, AutofuseTilingData &tiling_data) {
  OP_LOGD(OP_NAME, "PGOGetTilingKey from file:%s.", config_file);
  static int best_config = 0;
  static AutofuseTilingData best_tiling;
  if (best_config == 0) {
    std::ifstream inFile(config_file);
    if (!inFile.is_open()) {
      OP_LOGD(OP_NAME, "failed to open or not exist: %s.", config_file);
      return false;
    }
    OP_LOGD(OP_NAME, "[Start to use tiling result]: %s.", config_file);
    std::string line;
    // first line: 0:read everytime; 1:read first time
    std::getline(inFile, line);
    std::istringstream iss0(line);
    int flag = -1;
    iss0 >> flag;
    OP_LOGD(OP_NAME, "best_config %d.", flag);
    // second line: byte_size value;
    std::getline(inFile, line);
    if (line.find('#') != std::string::npos) {
        line = line.substr(0, line.find('#'));
    }
    std::istringstream iss1(line);
    std::string byte_size, value;
    char* ptr = (char*)&tiling_data;
    while (std::getline(iss1, byte_size, ' ') && std::getline(iss1, value, ';')) {
      int size = std::stoi(byte_size);
      uint64_t number = std::stoull(value);
      std::memcpy(ptr, &number, size);
      ptr += size;
    }
    inFile.close();
    if (flag == 1) {
      best_tiling = tiling_data;
      best_config = flag;
    }
  } else {
    tiling_data = best_tiling;
  }
  return true;
}

extern "C" int64_t PgoAutofuseTiling(const char* config_file, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr)
{
 const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  tiling->set_block_dim(limit->aiv_num);
  tiling->set_ub_size(limit->ub_size);
  if (!PGOGetTilingKey(config_file, *tiling)) {
    if (!optiling::GetTiling(*tiling, -1)) {
      return -1;
    }
  }
  *blockDim = tiling->get_block_dim();
  using namespace optiling;
  *workspaceSize = GetWorkspaceSize(*tiling);
  *workspaceSize += 16 * 1024 * 1024;

  return 0;
}
void SavePGOSearchTilingData(char* search_file, std::vector<AutofuseTilingDataPerf>& tiling_data_list, std::ios::openmode mode = std::ios::out) {
  OP_LOGI(OP_NAME, "SavePGOSearchTilingData to file:%s.", search_file);
  std::ofstream out_file(search_file, mode);
  if (!out_file.is_open()) {
    OP_LOGE(OP_NAME, "Failed to open file:%s.", search_file);
    return;
  }
  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {
    auto it = &item->tiling_data;
    out_file << sizeof(it->block_dim) << " " << it->block_dim << ";";
    out_file << sizeof(it->corenum) << " " << it->corenum << ";";
    out_file << sizeof(it->ub_size) << " " << it->ub_size << ";";
    out_file << sizeof(it->hbm_size) << " " << it->hbm_size << ";";
    out_file << sizeof(it->graph0_tiling_key) << " " << it->graph0_tiling_key << ";";
    out_file << " # " << item->best_perf;
    out_file << std::endl;
  }
  out_file.close();

  return;
}
void SavePGOConfigTilingData(char* file, std::vector<AutofuseTilingDataPerf>& tiling_data_list, double best_perf, std::ios::openmode mode = std::ios::out) {
  OP_LOGI(OP_NAME, "SavePGOConfigTilingData to file:%s.", file);
  std::ofstream out_file(file, mode);
  if (!out_file.is_open()) {
    OP_LOGE(OP_NAME, "Failed to open file:%s.", file);
    return;
  }
  for (auto item : tiling_data_list) {
      if (item.best_perf < best_perf) { 
        best_perf = item.best_perf;
      }
  }
  out_file << "1" << std::endl;
  for (auto item = tiling_data_list.rbegin(); item != tiling_data_list.rend(); ++item) {
     auto it = &item->tiling_data;
     if (optiling::IsEqual(item->best_perf, best_perf)) { 
    out_file << sizeof(it->block_dim) << " " << it->block_dim << ";";
    out_file << sizeof(it->corenum) << " " << it->corenum << ";";
    out_file << sizeof(it->ub_size) << " " << it->ub_size << ";";
    out_file << sizeof(it->hbm_size) << " " << it->hbm_size << ";";
    out_file << sizeof(it->graph0_tiling_key) << " " << it->graph0_tiling_key << ";";
    out_file << " # " << best_perf;
    out_file << std::endl;
    break;
    }
  }
  out_file.close();

  return;
}
extern "C" int64_t PgoTilingSearchByCoreNum(char* search_file, char* config_file, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, void* stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback prof_batch_callback=nullptr)
{
  PgoConfig::Instance().single_callback = prof_callback;
  const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  double best_perf = DBL_MAX;
  uint32_t max_block_dim = limit->aiv_num;
  const char* var = std::getenv("EXPERIMENTAL_AUTOFUSE_PGO_MAX_CORENUM");
  if (var != nullptr) {
    auto max_core_num = std::stoi(var);
    tiling->set_block_dim(max_core_num);
    max_block_dim = max_core_num;
  }
  using namespace optiling;
  std::vector<AutofuseTilingData> tiling_data_list;
  std::vector<AutofuseTilingDataPerf> tiling_data_perf_list;
  double axeorder_cost = DBL_MAX;
  AutofuseTiling(tiling, workspaceSize, blockDim, limit->aiv_num, limit->ub_size - 256);
  PgoConfig::Instance().single_callback(stream, *workspaceSize, tiling, &axeorder_cost);
  AutofuseTilingDataPerf tiling_data_axereorder_perf;
  tiling_data_axereorder_perf.tiling_data = *tiling;
  tiling_data_axereorder_perf.best_perf = axeorder_cost;
  tiling_data_perf_list.push_back(tiling_data_axereorder_perf);
  PgoConfig::Instance().need_change_solver_run = true;
  if (!optiling::PGOByCoreNumSearchTilingKey(tiling_data_list, tiling, max_block_dim)) {
    return -1;
  }
  double out_cost = DBL_MAX;
  *workspaceSize = 0;
  for (const auto &tiling_data_item : tiling_data_list) {
    *workspaceSize = std::max(GetWorkspaceSize(tiling_data_item), *workspaceSize);
    AutofuseTilingDataPerf tiling_data_perf;
    tiling_data_perf.tiling_data = tiling_data_item;
    tiling_data_perf.best_perf = DBL_MAX;
    tiling_data_perf_list.push_back(tiling_data_perf);
  }
  *workspaceSize += 16 * 1024 * 1024;
  PgoConfig::Instance().batch_callback(stream, *workspaceSize, &tiling_data_perf_list);
  best_perf = DBL_MAX;
  SavePGOSearchTilingData(search_file, tiling_data_perf_list);
  SavePGOConfigTilingData(config_file, tiling_data_perf_list, best_perf);
  return 0;
}
extern "C" int64_t PgoTilingSearchPGO(char* search_file, char* config_file, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, void* stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback prof_batch_callback=nullptr)
{
 const ResLimit *limit = (res_limit == nullptr) ? &g_no_limit_res : res_limit;
  std::vector<AutofuseTilingDataPerf> tiling_data_list;
  tiling->set_block_dim(limit->aiv_num);
  double best_perf = DBL_MAX;
  uint32_t max_block_dim = limit->aiv_num;
  const char* var = std::getenv("EXPERIMENTAL_AUTOFUSE_PGO_MAX_CORENUM");
  if (var != nullptr) {
    auto max_core_num = std::stoi(var);
    tiling->set_block_dim(max_core_num);
    max_block_dim = max_core_num;
  }
  AutofuseTiling(tiling, workspaceSize, blockDim, limit->aiv_num, limit->ub_size - 256);
  PgoConfig::Instance().single_callback = prof_callback;
  PgoConfig::Instance().batch_callback = prof_batch_callback;
  PgoConfig::Instance().single_callback(stream, *workspaceSize, tiling, &best_perf);
  if (optiling::IsEqual(best_perf, DBL_MAX)) {
    OP_LOGE(OP_NAME, "axesreorder solution get perf failed %lf", best_perf);
    return -1;
  }
  AutofuseTilingDataPerf tiling_perf;
  tiling_perf.tiling_data = *tiling;
  tiling_perf.best_perf = best_perf;
  tiling_data_list.push_back(tiling_perf);
  OP_LOGD(OP_NAME, "axesreorder solution base perf is %lf", best_perf);
  if (!optiling::PGOSearchTilingKey(tiling_data_list, *tiling, -1, tiling, stream, *workspaceSize, best_perf)) {
    return -1;
  }
  if (optiling::IsEqual(best_perf, DBL_MAX)) {
    OP_LOGE(OP_NAME, "pgo solution get perf failed %lf", best_perf);
    return -1;
  }
  SavePGOSearchTilingData(search_file, tiling_data_list);
  SavePGOConfigTilingData(config_file, tiling_data_list, best_perf);
  OP_LOGD(OP_NAME, "pgo solution best perf is %lf", best_perf);

  return 0;
}
extern "C" int64_t PgoTilingSearch(char* search_file, char* config_file, AutofuseTilingData* tiling, uint32_t* workspaceSize, uint32_t *blockDim, ResLimit *res_limit = nullptr, void* stream=nullptr, ProfilingCallback prof_callback=nullptr, ProfilingBatchCallback prof_batch_callback=nullptr)
{
  const char* var = std::getenv("EXPERIMENTAL_AUTOFUSE_PGO_BY_CORENUM");
  if (var != nullptr) {
    PgoConfig::Instance().pgo_algorithm = std::stoi(var);
  }
  PgoConfig::Instance().single_callback = prof_callback;
  PgoConfig::Instance().batch_callback = prof_batch_callback;
  if (PgoConfig::Instance().pgo_algorithm == 0) {
    PgoTilingSearchPGO(search_file, config_file,  tiling, workspaceSize, blockDim, res_limit, stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);
  } else if (PgoConfig::Instance().pgo_algorithm == 1) {
    PgoTilingSearchByCoreNum(search_file, config_file,  tiling, workspaceSize, blockDim, res_limit, stream, PgoConfig::Instance().single_callback, PgoConfig::Instance().batch_callback);
  }
  return 0;
}

#ifndef __CCE_KT_TEST__
extern "C" bool AutofuseIsStaticShape() {
  return true;
}
extern "C" int64_t FindBestTilingKey(AutofuseTilingData &t)
{

  return -1;
}

namespace gert {
  class TilingSymbolEvalContext : public TilingContext {
    public:
      const gert::Tensor *GetGraphInputTensor(size_t data_index) const {
        auto *tensor = GetInputPointer<gert::Tensor>(data_index + 1);
        if (tensor == nullptr) {
          return nullptr;
        }
        return tensor;
      }
  };

  class SymbolTilingParseContext : public KernelContext {
    public:
      fe::PlatFormInfos *GetPlatFormInfos() const {
        auto platform = GetInputValue<fe::PlatFormInfos *>(0);
        if (platform == nullptr) {
          return nullptr;
        }
        return platform;
      }
  };
}
struct AfTilingParseData{
 uint32_t aiv_num;
 uint64_t ub_size;
};
extern "C" ge::graphStatus TilingParse(gert::SymbolTilingParseContext *context) {
 auto platform = context->GetPlatFormInfos();
 if (platform == nullptr) {
 return ge::GRAPH_FAILED;
 }
 auto ascendc_platform = platform_ascendc::PlatformAscendC(platform);
 uint32_t platform_core_num = ascendc_platform.GetCoreNumAiv();
 uint32_t aiv_num = 0;
 uint64_t ub_size = (184 * 1024);
 aiv_num = platform_core_num;
 ascendc_platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
 auto extend_context = reinterpret_cast<gert::KernelContext *>(context);
 auto tiling_parse_data_av = extend_context->GetOutput(0);
 if (tiling_parse_data_av == nullptr) {
 return ge::GRAPH_FAILED;
 }
 auto tiling_parse_data_ptr = new (std::nothrow) uint8_t[sizeof(AfTilingParseData)];
 if (tiling_parse_data_ptr == nullptr) {
 return ge::GRAPH_FAILED;
 }
 tiling_parse_data_av->SetWithDefaultDeleter<uint8_t[]>(tiling_parse_data_ptr);
 auto tiling_parse_data = extend_context->GetOutputPointer<AfTilingParseData *>(0);
 (*tiling_parse_data)->aiv_num = aiv_num;
 ub_size -= (ascendc_platform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95 && ub_size % 1024 == 0) ? 256 : 0;
 (*tiling_parse_data)->ub_size = ub_size;
 return ge::GRAPH_SUCCESS;
}

extern "C" ge::graphStatus TilingFunc(gert::TilingSymbolEvalContext *context)
{
  auto extend_context = reinterpret_cast<const gert::KernelContext *>(context);
  auto input_data_num =  extend_context->GetInputValue<size_t>(0U);
  auto parse = extend_context->GetInputValue<AfTilingParseData*>(input_data_num + 1);
  auto tiling_data =  context->GetTilingData<AutofuseTilingData>();
  uint32_t workspace_size;
  uint32_t block_dim;
  static const char* config_file = "/test_graph_config.txt";
  ResLimit limit;
  limit.aiv_num = parse->aiv_num;
  limit.ub_size = (uint32_t)parse->ub_size;
  auto ret = PgoAutofuseTiling(config_file, tiling_data, &workspace_size, &block_dim, &limit);
  context->SetBlockDim(block_dim);
  *context->GetWorkspaceSizes(1) = workspace_size;

  auto tiling_key = FindBestTilingKey(*tiling_data);
  if (tiling_key < 0) {
    return ge::GRAPH_FAILED;
  }
  context->SetTilingKey(static_cast<uint64_t>(tiling_key));
  return ret;
}

extern "C" ge::graphStatus GetSymbolTilingCacheKey(gert::TilingSymbolEvalContext *context)
{
  auto kernel_context = reinterpret_cast<gert::KernelContext *>(context);
  auto symbol_src_vec = kernel_context->GetOutputPointer<gert::TypedContinuousVector<int64_t>>(0U);
  if (symbol_src_vec == nullptr) {
    return ge::GRAPH_FAILED;
  }

  symbol_src_vec->SetSize(0);
  return ge::GRAPH_SUCCESS;
}
extern "C" ge::graphStatus DfxInputSymbolInfo(gert::TilingSymbolEvalContext *context, char *out_symbol_info, size_t size)
{
  if (out_symbol_info == nullptr || size == 0) {
    return ge::GRAPH_SUCCESS;
  }
  std::string symbol_info;

  if (symbol_info.empty()) {
    out_symbol_info[0] = '\0';
    return ge::GRAPH_SUCCESS;
  }
  symbol_info += ".";
  if (strncpy_s(out_symbol_info, size, symbol_info.c_str(), std::min(symbol_info.size(), size - 1)) != 0) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
#endif

std::string tiling_data_const_gen_result;
AutofuseTilingData TilingDataValue;

void replaceSubstring(std::string& ori_str, const std::string& old_sub_str, const std::string& new_sub_str) {
  size_t pos = ori_str.find(old_sub_str);
  if (pos != std::string::npos) {
    ori_str.replace(pos, old_sub_str.length(), new_sub_str);
  }
}

std::string GenTilingDataFieldConstDefFunc(const std::string &f_name, uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << "const uint32_t ";
  ss_mid << f_name << " = " << std::to_string(value) << ";" << std::endl;
  return ss_mid.str();
}

std::string GenTilingDataFieldConstValueFunc(uint32_t value) {
  std::stringstream ss_mid;
  ss_mid << std::to_string(value) << std::endl;
  return ss_mid.str();
}


extern "C" const char* GenConstTilingData(char* config_file, int aiv_num, int ub_size) {
  uint32_t workspace_size;
  uint32_t block_dim;
  ResLimit limit;
  limit.aiv_num = aiv_num;
  limit.ub_size = ub_size - 256;
  (void)PgoAutofuseTiling(config_file, &TilingDataValue, &workspace_size, &block_dim, &limit);
  std::string GenTilingDataValue_block_dim_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("block_dim", TilingDataValue.block_dim);
  std::string GenTilingDataValue_corenum_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("corenum", TilingDataValue.corenum);
  std::string GenTilingDataValue_ub_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("ub_size", TilingDataValue.ub_size);
  std::string GenTilingDataValue_hbm_size_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("hbm_size", TilingDataValue.hbm_size);
  std::string GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def = GenTilingDataFieldConstDefFunc("graph0_tiling_key", TilingDataValue.graph0_tiling_key);

  tiling_data_const_gen_result = R"(#ifndef __Autofuse_Tiling_Data_H__
#define __Autofuse_Tiling_Data_H__
#include <stdint.h>
#include "kernel_tiling/kernel_tiling.h"
#define BEGIN_TILING_DATA_DEF_T(name) struct name {
#define TILING_DATA_FIELD_DEF_T(type, name) \
  type name; \
  inline void set_##name(type value) { name = value; } \
  inline type get_##name() { return name; } \
  inline type* get_addr_##name() {return &name;}
#define END_TILING_DATA_DEF_T };
#define TILING_DATA_FIELD_DEF_T_STRUCT(struct_type, filed_name) \
  struct_type filed_name;

BEGIN_TILING_DATA_DEF_T(AutofuseTilingData)
  GenTilingDataValue_block_dim_field_DeclareFunc_def
  GenTilingDataValue_corenum_field_DeclareFunc_def
  GenTilingDataValue_ub_size_field_DeclareFunc_def
  GenTilingDataValue_hbm_size_field_DeclareFunc_def
  GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def
END_TILING_DATA_DEF_T;

struct AutofuseTilingDataPerf {
  AutofuseTilingData tiling_data;
  double best_perf;
};
#endif
)";
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_block_dim_field_DeclareFunc_def",GenTilingDataValue_block_dim_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_corenum_field_DeclareFunc_def",GenTilingDataValue_corenum_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_ub_size_field_DeclareFunc_def",GenTilingDataValue_ub_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_hbm_size_field_DeclareFunc_def",GenTilingDataValue_hbm_size_field_DeclareFunc_def);
  replaceSubstring(tiling_data_const_gen_result, "GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def",GenTilingDataValue_graph0_tiling_key_field_DeclareFunc_def);

  return tiling_data_const_gen_result.c_str();
}


#ifndef __CCE_KT_TEST__
std::string kernel_type;
extern "C" const char* GetTilingKeyKernelTypeForStatic()
{
  const std::map<int64_t, std::string> kernel_type_map = {
  };

  auto tiling_key = FindBestTilingKey(TilingDataValue);
  auto it = kernel_type_map.find(tiling_key);
  if (it != kernel_type_map.end()) {
    kernel_type = it->second;
  }
  return kernel_type.c_str();
}
#endif
)rawliteral";
 EXPECT_EQ(tiling_code, expect_code);
}