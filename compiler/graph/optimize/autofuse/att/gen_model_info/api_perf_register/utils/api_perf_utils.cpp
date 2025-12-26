/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "api_perf_utils.h"
#include "common_utils.h"
#include "api_perf_register/perf_param.h"
#include "api_perf_register/api_perf_factory.h"
#include "api_perf_register/ascendc_api_perf.h"

namespace att {
namespace {
std::string GetDefaultDataType(const std::string dtype) {
  static const std::map<std::string, std::string> kTypeToDType = {
      {"int8", "float16"},
      {"uint8", "float16"},
      {"uint16", "float16"},
      {"int16", "float16"},
      {"uint32", "float32"},
      {"int32", "float32"},
      {"uint64", "float32"},
      {"int64", "float32"},
      {"float32", "float32"},
  };
  auto it = kTypeToDType.find(dtype);
  if (it != kTypeToDType.end()) {
    return it->second;
  }
  return "float16";
}

ge::Status StringToJson(const std::string &json_str, Json &json) {
  std::stringstream ss;
  ss << json_str;
  try {
    ss >> json;
  } catch (const nlohmann::json::exception &e) {
    GELOGE(ge::PARAM_INVALID, "Failed to init json object, err = %s, json_str = %s", e.what(), json_str.c_str());
    return ge::PARAM_INVALID;
  }
  return ge::SUCCESS;
}

ge::Status LinearFunc(const std::map<std::string, float> &param_map,
                      const std::vector<Expr> &dims, const Expr &stride, Expr &res) {
  (void)stride;
  GE_ASSERT_TRUE(!dims.empty(), "Dims is empty.");
  GE_ASSERT_TRUE(param_map.find("k") != param_map.end(), "Param k not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("b") != param_map.end(), "Param b not found in param_map.");
  Expr k = CreateExpr(param_map.at("k"));
  Expr b = CreateExpr(param_map.at("b"));
  Expr dim_product =  accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  res = k * dim_product + b;
  return ge::SUCCESS;
}

ge::Status LoadStoreStrideFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                               const Expr &stride, Expr &res) {
  GE_ASSERT_TRUE(!dims.empty(), "Dims is empty.");
  GE_ASSERT_TRUE(param_map.find("k") != param_map.end(), "Param k not found in param_map.");
  Expr k = CreateExpr(param_map.at("k"));
  Expr block_count = accumulate(dims.begin(), dims.end() - 1, CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  Expr op1 = ge::sym::Mul(k, block_count);
  Expr op2 = ge::sym::Mod(stride, CreateExpr(256));
  res = ge::sym::Mul(op1, op2);
  GELOGD("stride[%s], k[%s], block_count[%s], perf_res[%s]", Str(stride).c_str(), Str(k).c_str(),
         Str(block_count).c_str(), Str(res).c_str());
  return ge::SUCCESS;
}

ge::Status LoadStoreStrideV2Func(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                                 const Expr &stride, Expr &res) {
  GE_ASSERT_TRUE(!dims.empty(), "Dims is empty.");
  GE_ASSERT_TRUE(param_map.find("k") != param_map.end(), "Param k not found in param_map.");
  Expr k = CreateExpr(param_map.at("k"));
  Expr block_count = accumulate(dims.begin(), dims.end() - 1, CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  Expr op1 = ge::sym::Mul(k, block_count);
  res = ge::sym::Mul(op1, stride);
  GELOGD("stride[%s], k[%s], block_count[%s], perf_res[%s]", Str(stride).c_str(), Str(k).c_str(),
         Str(block_count).c_str(), Str(res).c_str());
  return ge::SUCCESS;
}

ge::Status LoadStoreFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                         const Expr &stride, Expr &res, bool expand_data) {
  (void)stride;
  GE_ASSERT_TRUE(!dims.empty(), "Dims is empty.");
  GE_ASSERT_TRUE(param_map.find("h") != param_map.end(), "Param h not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("a") != param_map.end(), "Param a not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("b") != param_map.end(), "Param b not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("hl") != param_map.end(), "Param hl not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("data_type_size") != param_map.end(), "Param data_type_size not found in param_map.");
  Expr h = CreateExpr(param_map.at("h"));
  Expr a = CreateExpr(param_map.at("a"));
  Expr b = CreateExpr(param_map.at("b"));
  Expr hl = CreateExpr(param_map.at("hl"));
  Expr data_type_size = CreateExpr(static_cast<int32_t>(param_map.at("data_type_size")));
  Expr dim_product = accumulate(dims.begin(), dims.end(), CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  Expr ub_align = CreateExpr(32);
  Expr blockdim = CreateExpr("block_dim");
  Expr t = ge::sym::Add(a, ge::sym::Div(b, blockdim));
  Expr data_size = ge::sym::Mul(dim_product, data_type_size);
  if (expand_data) {
    Expr blocklen = ge::sym::Mul(dims[dims.size() - 1UL], data_type_size);
    data_size = ge::sym::Mul(ge::sym::Div(data_size, blocklen), kMte2UbStrideAlignSize);
  }
  auto cycles = ge::sym::Div(data_size, t);
  // 32B非对齐的惩罚项，hl=1时需要加该惩罚项
  // penalty = (512 / 2 - (h + 512 / t)) * hl
  Expr penalty = hl * (CreateExpr(256) - (h + ge::sym::Div(CreateExpr(512), t)));
  res = ge::sym::Add(ge::sym::Add(cycles, h), penalty);
  return ge::SUCCESS;
}

ge::Status LoadStoreFunc(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                         const Expr &stride, Expr &res) {
  return LoadStoreFunc(param_map, dims, stride, res, false);
}

ge::Status LoadUbStride(const std::map<std::string, float> &param_map, const std::vector<Expr> &dims,
                        const Expr &stride, Expr &res) {
  return LoadStoreFunc(param_map, dims, stride, res, true);
}

ge::Status StoreFunc(const std::map<std::string, float> &param_map,
                     const std::vector<Expr> &dims, const Expr &stride, Expr &res) {
  (void)stride;
  GE_ASSERT_TRUE(!dims.empty(), "Dims is empty.");
  GE_ASSERT_TRUE(param_map.find("h") != param_map.end(), "Param h not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("ak") != param_map.end(), "Param ak not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("ab") != param_map.end(), "Param ab not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("bk") != param_map.end(), "Param bk not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("bb") != param_map.end(), "Param bb not found in param_map.");
  GE_ASSERT_TRUE(param_map.find("data_type_size") != param_map.end(), "Param data_type_size not found in param_map.");
  Expr h = CreateExpr(param_map.at("h"));
  Expr ak = CreateExpr(param_map.at("ak"));
  Expr ab = CreateExpr(param_map.at("ab"));
  Expr bk = CreateExpr(param_map.at("bk"));
  Expr bb = CreateExpr(param_map.at("bb"));
  Expr data_type_size = CreateExpr(static_cast<int32_t>(param_map.at("data_type_size")));
  Expr blocklen = ge::sym::Mul(dims[dims.size() - 1UL], data_type_size);
  Expr blockcount = accumulate(dims.begin(), dims.end() - 1, CreateExpr(1), [](Expr a, Expr b) { return a * b; });
  GELOGD("Special case for store. Blocklen = [%s]Byte, Blockcount = [%s].", Str(blocklen).c_str(), Str(blockcount).c_str());
  Expr blockdim = CreateExpr("block_dim");
  Expr a = ak * blockdim + ab;
  Expr b = bk * blockdim + bb;
  res = (a * blocklen + b) * blockcount + h;
  return ge::SUCCESS;
}
const std::map<std::string, AscendCApiPerfFunc> kModelMap = {
    {"SimpleLinear", LinearFunc},
    {"LoadUbStride", LoadUbStride},
    {"LoadStoreFunc", LoadStoreFunc},
    {"LoadStoreStrideFunc", LoadStoreStrideFunc},
    {"LoadStoreStrideV2Func", LoadStoreStrideV2Func},
    {"StoreFunc", StoreFunc},
};

const PerfParamTable *GetParamPerfTable() {
  const auto asc_ir_att_impl = ascgen_utils::GetAscIrAttImpl(kDefaultApi);
  GE_ASSERT_NOTNULL(asc_ir_att_impl);
  const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
  GE_ASSERT_NOTNULL(api_perf_name);
  auto api_perf = ApiPerfFactory::Instance().Create(api_perf_name);
  GE_ASSERT_NOTNULL(api_perf);
  return api_perf->GetPerfParam();
}

void UpdateContinuousIdx(size_t idx, const std::vector<Expr> &strides, const std::vector<Expr> &repeats,
                         std::vector<bool> &is_continuous) {
  size_t cur = idx - 1UL;
  Expr cur_axis_stride = repeats[idx] * strides[idx];
  while (cur >= 1UL && strides[cur] == 0) {
    --cur;
  }
  if (strides[cur] != 0 && cur_axis_stride != strides[cur]) {
    is_continuous[idx] = false;
  }
}

void UpdateIsContinous(const std::vector<Expr> &repeats, const std::vector<Expr> &strides,
                       std::vector<bool> &is_continuous) {
  GELOGD("tensor repeats is {%s}.", GetVecString(repeats).c_str());
  GELOGD("tensor strides is {%s}.", GetVecString(strides).c_str());
  for (size_t i = (repeats.size() - 1UL); i >= 1UL; --i) {
    // 判断哪些轴是非连续轴
    if (strides[i] == 0) {
      is_continuous[i] = false;
      continue;
    }
    UpdateContinuousIdx(i, strides, repeats, is_continuous);
  }
}

void UpdateTensorDim(const std::string &op_type, const std::vector<bool> &is_continuous,
                     const std::vector<Expr> &org_dims, TensorShapeInfo &tensor) {
  for (size_t i = 1UL; i <= (tensor.repeats.size() - 1UL); ++i) {
    Expr product = tensor.dims.back();
    if (is_continuous[i]) {
      // 连续轴的dim使用合并原始轴后的两个dim乘积
      product = org_dims[i] * product;
      GELOGD("Got continuous axis[%zu], dim[%s] * [%s] = [%s]", i, ge::SymbolicUtils::ToString(org_dims[i]).c_str(),
             ge::SymbolicUtils::ToString(tensor.dims.back()).c_str(),
             ge::SymbolicUtils::ToString(product).c_str());
      tensor.dims.pop_back();
    } else {
      // 非连续轴的dim使用stride
      size_t cur = i - 1UL;
      while (cur >= 1UL && tensor.strides[cur] == 0) {
        --cur;
      }
      if (IsLoadStore(op_type)) {
        product = org_dims[i];
      } else {
        product = tensor.strides[cur];
      }
      GELOGD("Got non-continuous axis[%zu], dim[%s]", i, ge::SymbolicUtils::ToString(product).c_str());
    }
    tensor.dims.emplace_back(product);
  }
}

ge::Status GetDMAActualPerf(const NodeDetail &node_info, const Expr &swap_outer_repeat, const std::vector<Expr> &dims,
                            PerfOutputInfo &perf) {
  NodeDetail cur_node =
      NodeDetail{node_info.name, node_info.optype, {node_info.input_dtype[0]}, {node_info.output_dtype[0]}};
  cur_node.gm_stride = node_info.gm_stride;
  cur_node.ub_stride = node_info.ub_stride;
  GE_ASSERT_SUCCESS(SetDims(dims, cur_node));
  if (cur_node.optype == kMoveUbToGm) {
    ascendcperf::StorePerf(cur_node, perf);
    perf.pipe_res[PipeType::AIV_MTE3] = swap_outer_repeat * GetPipeCost(perf, PipeType::AIV_MTE3);
  } else {
    if (node_info.optype == kMoveNddma) {
      ascendcperf::NddmaPerf(cur_node, perf);
    } else {
      ascendcperf::LoadPerf(cur_node, perf);
    }
    perf.pipe_res[PipeType::AIV_MTE2] = swap_outer_repeat * GetPipeCost(perf, PipeType::AIV_MTE2);
  }
  return ge::SUCCESS;
}

ge::Status UpdateSwapPerf(const NodeDetail &node_info, const int32_t supported_max_dma_len, PerfOutputInfo &swap_perf,
                          PerfOutputInfo &non_swap_perf, PerfOutputInfo &perf_res) {
  size_t dim_size = node_info.input_dims.size();
  if (node_info.input_dims[dim_size - supported_max_dma_len].IsConstExpr() &&
      node_info.input_dims[dim_size - supported_max_dma_len - 1].IsConstExpr()) {
    if (ge::SymbolicUtils::StaticCheckLt(node_info.input_dims[dim_size - supported_max_dma_len],
                                         node_info.input_dims[dim_size - supported_max_dma_len - 1]) ==
        ge::TriBool::kTrue) {
      GE_ASSERT_SUCCESS(UpdateTenary(swap_perf, perf_res));
    } else {
      GE_ASSERT_SUCCESS(UpdateTenary(non_swap_perf, perf_res));
    }
  } else {
    Expr res;
    PipeType pipe_type = node_info.optype == kMoveUbToGm ? PipeType::AIV_MTE3 : PipeType::AIV_MTE2;
    GE_ASSERT_SUCCESS(UpdateTenary(swap_perf, perf_res));
    GE_ASSERT_SUCCESS(UpdateTenary(non_swap_perf, perf_res));
    GetPerfVar(node_info.name, res, perf_res.tenary_ops);
    perf_res.tenary_ops[res] = TenaryOp(CondType::K_LT, node_info.input_dims[dim_size - supported_max_dma_len],
                                        node_info.input_dims[dim_size - supported_max_dma_len - 1],
                                        GetPipeCost(swap_perf, pipe_type), GetPipeCost(non_swap_perf, pipe_type));
    perf_res.tenary_ops[res].SetVariable(res);
    perf_res.pipe_res[pipe_type] = res;
  }
  return ge::SUCCESS;
}

// 获取dma参数，获取外派循环轴/使用的轴，支持根据内轴大小比较进行交换
ge::Status GetDmaParams(const vector<Expr> &dims, Expr &outer_repeat, vector<Expr> &used_dims,
                        const int32_t supported_max_dma_len, bool need_swap = false) {
  used_dims.clear();
  GE_ASSERT_TRUE(!dims.empty());
  if (static_cast<int32_t>(dims.size()) <= supported_max_dma_len) {
    used_dims = dims;
    outer_repeat = ge::sym::kSymbolOne;
    GELOGD("Got outer loop %s, used dims %s", outer_repeat.Serialize().get(), GetVecString(used_dims).c_str());
  } else {
    size_t dim_size = dims.size();
    // [0, 1, 2_sw, 3_sw, 4, 5, 6]
    // dim_size = 7
    // supported_max_dma_len = 4
    // need_swap = true
    // used_dims = [3, 4, 5, 6]
    // outer_repeat = 0 * 1 * 2
    outer_repeat = accumulate(dims.begin(), (dims.end() - supported_max_dma_len - 1), CreateExpr(1),
                              [](const Expr &a, const Expr &b) { return ge::sym::Mul(a, b); });
    size_t first_dim_index;
    if (need_swap) {
      first_dim_index = dim_size - supported_max_dma_len - 1;
      used_dims.emplace_back(dims[first_dim_index]);
      outer_repeat = outer_repeat * dims[dim_size - supported_max_dma_len];
    } else {
      first_dim_index = dim_size - supported_max_dma_len;
      used_dims.emplace_back(dims[first_dim_index]);
      outer_repeat = outer_repeat * dims[first_dim_index - 1];
    }
    for (int32_t i = 1; i < supported_max_dma_len; i++) {
      used_dims.emplace_back(dims[first_dim_index + i]);
    }

    GELOGD("Got dim outer loop %s, used dims %s", outer_repeat.Serialize().get(), GetVecString(used_dims).c_str());
  }
  return ge::SUCCESS;
}
}  // namespace

PipeHeadPerfFunc GetPipeHeadPerfFunc(PipeType pipe_type) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  return param_table->GetPipeHeadPerfFunc(pipe_type);
}

ge::Status GetApiRegisterVerName(std::string &registered_key_name) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  registered_key_name = param_table->GetApiRegisterVerName();
  return ge::SUCCESS;
}

ge::Status GetOpHeadCost(Expr &head_cost) {
  const auto param_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(param_table);
  head_cost = param_table->GetOpHeadCost();
  return ge::SUCCESS;
}

std::unique_ptr<ApiPerf> GetApiPerf(const std::string &node_type) {
  GELOGD("Begin to get api name for node[%s]", node_type.c_str());
  const auto asc_ir_att_impl = ascgen_utils::GetAscIrAttImpl(node_type);
  GE_ASSERT_NOTNULL(asc_ir_att_impl);
  const auto api_perf_name = ge::PtrToPtr<void, ge::char_t>(asc_ir_att_impl->GetApiPerf());
  if (api_perf_name != nullptr) {
    GELOGD("Found api name for node[%s]: [%s]", node_type.c_str(), api_perf_name);
    return ApiPerfFactory::Instance().Create(api_perf_name);
  }
  return ApiPerfFactory::Instance().Create(node_type);
}

ge::Status GetPerf(const NodePerfInfo &node_perf_info, Expr &res) {
  std::string optype = node_perf_info.optype;
  std::string input_dtype = node_perf_info.input_dtype;
  std::string output_dtype = node_perf_info.output_dtype;
  GELOGD("GetPerf: optype=[%s], input_dtype=[%s], output_dtype=[%s]", optype.c_str(), input_dtype.c_str(),
         output_dtype.c_str());
  const auto perf_table = GetParamPerfTable();
  GE_ASSERT_NOTNULL(perf_table);
  const auto ascend_api_perf_table = perf_table->GetAscendCApiPerfTable();
  GE_ASSERT_NOTNULL(ascend_api_perf_table);
  Json params_data;
  GE_ASSERT_SUCCESS(StringToJson(*ascend_api_perf_table, params_data));
  const auto &opt_type_param = params_data.find(optype);
  GE_ASSERT_TRUE(opt_type_param != params_data.end(), "Optype[%s] not found! json: %s", optype.c_str(),
                 params_data.dump().c_str());
  auto op_model = opt_type_param.value();
  const auto &model_type_param = op_model.find("model_type");
  std::string model_type = model_type_param.value();
  auto model_params = op_model.at("model_params");
  std::string dtype = input_dtype + "to" + output_dtype;
  if (model_params.find(dtype) == model_params.end()) {
    GELOGW("Dtype[%s] is not registered for optype[%s]!", dtype.c_str(), optype.c_str());
    dtype = GetDefaultDataType(input_dtype) + "to" + GetDefaultDataType(output_dtype);
    GELOGD("Use default dtype[%s] for optype[%s].", dtype.c_str(), optype.c_str());
  }
  const auto used_params = model_params.find(dtype);
  GE_ASSERT_TRUE(used_params != model_params.end(), "Dtype[%s] for optype[%s] not found! json: %s", dtype.c_str(),
                 optype.c_str(), params_data.dump().c_str());
  const auto &perf_func = kModelMap.find(model_type);
  GE_ASSERT_TRUE(perf_func != kModelMap.end(), "Model type[%s] for optype[%s] not found! json: %s", model_type.c_str(),
                 optype.c_str(), params_data.dump().c_str());
  GE_ASSERT_SUCCESS(perf_func->second(used_params.value(), node_perf_info.dims, node_perf_info.gm_stride, res),
                    "Get Perf Func failed! json: %s", params_data.dump().c_str());
  return ge::SUCCESS;
}

inline void FilterStridesAndRepeats(const std::vector<Expr>& strides,
                                    const std::vector<Expr>& repeats,
                                    std::vector<Expr>& filtered_strides,  // 输出：过滤后的strides
                                    std::vector<Expr>& filtered_repeats   // 输出：过滤后的repeats
) {
  size_t loop_max = 0;
  if (!strides.empty() && !repeats.empty()) {
    loop_max = std::min(strides.size() - 1, repeats.size() - 1);
  }
  // 过滤出非尾轴的非广播维度
  for (size_t i = 0UL; i < loop_max; ++i) {
    if (!(ge::SymbolicUtils::StaticCheckEq(strides[i], ge::sym::kSymbolZero) == ge::TriBool::kTrue &&
        ge::SymbolicUtils::StaticCheckEq(repeats[i], ge::sym::kSymbolOne) == ge::TriBool::kTrue)) {
      filtered_strides.emplace_back(strides[i]);
      filtered_repeats.emplace_back(repeats[i]);
    }
  }
  // 保存尾轴
  if (!repeats.empty() && repeats.size() == strides.size()) {
    filtered_strides.emplace_back(strides.back());
    filtered_repeats.emplace_back(repeats.back());
  }
  GELOGD("Repeats, strides filter broadcast axis: filtered_strides=[%s], filtered_repeats=[%s]",
         GetVecString(filtered_strides).c_str(), GetVecString(filtered_repeats).c_str());
}

Expr CalculateStride(const std::vector<Expr> &strides, const std::vector<Expr> &repeats, const NodeDetail &node_info,
                     const int32_t supported_max_dma_len, bool need_swap) {
  GELOGD("%s(%s): strides=[%s], repeats=[%s], need_swap=%d", node_info.name.c_str(), node_info.optype.c_str(),
         GetVecString(strides).c_str(), GetVecString(repeats).c_str(), need_swap);
  std::vector<Expr> filtered_strides;
  std::vector<Expr> filtered_repeats;
  FilterStridesAndRepeats(strides, repeats, filtered_strides, filtered_repeats);
  auto filtered_dim_size = static_cast<int32_t>(filtered_repeats.size());
  int32_t supported_max_dma_idx = filtered_dim_size - supported_max_dma_len - 1;
  if ((filtered_dim_size <= supported_max_dma_len) && need_swap) {
    GELOGW("%s, can't swap stride. filtered_dim_size=%d, supported_max_dma_len=%d", node_info.ToString().c_str(), filtered_dim_size,
           supported_max_dma_len);
    return CreateExpr(0);
  }
  if (filtered_dim_size == 1 || filtered_strides.empty()) {
    bool is_stride_zero = filtered_strides.empty() || (filtered_strides.back() == 0);
    auto stride = is_stride_zero ? CreateExpr(0) : (filtered_strides.back() - CreateExpr(1)) * filtered_repeats.back();
    GELOGD("%s, total len is %d, is_stride_zero=%d, stride=%s.", node_info.ToString().c_str(), filtered_dim_size,
           is_stride_zero, Str(stride).c_str());
    return stride;
  }
  // 比如，repeat = [10, 20, 30]， stride = [20 * 30, 30, 1]，supported_max_dma_len=2
  // 若交换，repeat = [20, 10, 30]，对应stride需要交换为[30, 20 * 30, 1]
  int32_t block_count_idx = need_swap ? (filtered_dim_size - supported_max_dma_len - 1) : (filtered_dim_size - supported_max_dma_len);
  if ((block_count_idx >= static_cast<int32_t>(filtered_strides.size())) || (block_count_idx < 0)) {
    GELOGW("%s, block_count_idx is %d over size(%zu).", node_info.ToString().c_str(), block_count_idx, filtered_strides.size());
    return CreateExpr(0);
  }
  if (ge::SymbolicUtils::StaticCheckEq(filtered_strides[block_count_idx], ge::sym::kSymbolZero) == ge::TriBool::kTrue ||
      ge::SymbolicUtils::StaticCheckEq(filtered_strides[block_count_idx], ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
    GELOGW("%s, filtered_strides[%d] is 0 or 1", node_info.ToString().c_str(), block_count_idx, filtered_dim_size);
    return CreateExpr(0);
  }
  auto expr = ge::sym::Sub(filtered_strides[block_count_idx], filtered_repeats[filtered_dim_size - 1]);
  GELOGD("%s, block_count_idx=%d, stride=[%s]", node_info.ToString().c_str(), block_count_idx, Str(expr).c_str());
  return expr;
}

ge::Status SetStride(const TensorShapeInfo &shape_info, NodeDetail &node_info, const int32_t supported_max_dma_len,
                     bool need_swap) {
  node_info.gm_stride =
      CalculateStride(shape_info.gm_strides, shape_info.repeats, node_info, supported_max_dma_len, need_swap);
  node_info.ub_stride =
      CalculateStride(shape_info.strides, shape_info.origin_repeats, node_info, supported_max_dma_len, need_swap);
  GELOGD("%s: repeats=%s, origin_repeats=%s, need_swap=%d", node_info.ToString().c_str(),
         GetVecString(shape_info.repeats).c_str(), GetVecString(shape_info.origin_repeats).c_str(), need_swap);
  return ge::SUCCESS;
}

ge::Status SetNodeDetail(const std::vector<TensorShapeInfo> &input_shapes,
                         const std::vector<TensorShapeInfo> &output_shapes, NodeDetail &node_info) {
  GE_ASSERT_TRUE(!input_shapes.empty() && !output_shapes.empty());
  for (const auto &input_shape : input_shapes) {
    node_info.input_dtype.emplace_back(input_shape.data_type);
  }
  for (const auto &output_shape : output_shapes) {
    node_info.output_dtype.emplace_back(output_shape.data_type);
  }
  GE_ASSERT_SUCCESS(SetDims(input_shapes[0].dims, output_shapes[0].dims, node_info));
  return ge::SUCCESS;
}

ge::Status SetDims(const std::vector<Expr> &dims, NodeDetail &node_info) {
  node_info.input_dims.clear();
  node_info.output_dims.clear();
  for (const auto &dim : dims) {
    node_info.input_dims.emplace_back(dim);
    node_info.output_dims.emplace_back(dim);
  }
  return ge::SUCCESS;
}

ge::Status SetDims(const std::vector<Expr> &input_dims, const std::vector<Expr> &output_dims, NodeDetail &node_info) {
  node_info.input_dims.clear();
  node_info.output_dims.clear();
  for (const auto &dim : input_dims) {
    node_info.input_dims.emplace_back(dim);
  }
  for (const auto &dim : output_dims) {
    node_info.output_dims.emplace_back(dim);
  }
  return ge::SUCCESS;
}

NodeDetail GenNodeDetail(const std::string &input_dtype, const std::string &output_dtype,
                         const std::vector<Expr> &dims) {
  NodeDetail ret;
  ret.input_dtype.emplace_back(input_dtype);
  ret.output_dtype.emplace_back(output_dtype);
  if (dims.empty()) {
    ret.input_dims.emplace_back(ge::sym::kSymbolOne);
    ret.output_dims.emplace_back(ge::sym::kSymbolOne);
  } else {
    for (const auto &dim : dims) {
      ret.input_dims.emplace_back(dim);
      ret.output_dims.emplace_back(dim);
    }
  }
  return ret;
}

Expr GetPipeCost(const PerfOutputInfo &perf_res, const PipeType &pipe_type) {
  auto iter = perf_res.pipe_res.find(pipe_type);
  if (iter != perf_res.pipe_res.end()) {
    return iter->second;
  } else {
    return ge::sym::kSymbolZero;
  }
}

bool HasSmallBlockLenWithUbStride(const NodeDetail &node_info) {
  constexpr int32_t kOneBlkSize = 32;
  uint32_t ub_stride_val = 0;
  uint32_t block_len_val = kOneBlkSize;
  auto data_type_size = kDataTypeSizeMap.find(node_info.input_dtype[0]);
  GE_ASSERT_TRUE(data_type_size != kDataTypeSizeMap.end());
  Expr blocklen = ge::sym::Mul(node_info.input_dims[node_info.input_dims.size() - 1UL], data_type_size->second);
  if (blocklen.GetConstValue(block_len_val) && node_info.ub_stride.GetConstValue(ub_stride_val)) {
    if (block_len_val < static_cast<uint32_t>(kOneBlkSize) && ub_stride_val > 0U) {
      GELOGD("%s: ub stride(%u) and block len(%u) less than %d", node_info.ToString().c_str(), ub_stride_val,
             block_len_val, kOneBlkSize);
      return true;
    }
  }
  return false;
}

inline bool IsLoadStore(const std::string &op_type) {
  static const std::set<std::string> kOpType = {
      kMoveGmToUb,
      kMoveUbToGm,
      kMoveNddma
  };
  return kOpType.find(op_type) != kOpType.end();
}

/** 举例:
  repeats为{z0Tb_size, z0t_size, 2, 10,};
  strides为{(20 * z0t_size, 20, 10, 1,};
  vectorized_axis为{z0t, z2, z3};
  考虑非连续场景:
  vectorized_stride为{32, 16, 1}  非连续场景
  连续标记为{true, true, false} {跳过,    32等于16*2,        16不等于10*2}
                                        z0t_size与z1合轴  z1与z2不合轴
  考虑连续场景:
  vectorized_stride为{20, 10, 1}  连续场景
  连续标记为{true, true, true}  {跳过     20等于10*2,        10不等于1*2}
                                        z0t_size与z1合轴  z1与z2合轴
  reduce场景:reduce轴会断掉连续轴的合并
 **/
ge::Status MergeTensorContinuousDims(const std::string &op_type, const std::string &tensor_name,
                                     TensorShapeInfo &tensor) {
  Expr axis_size;
  std::vector<Expr> org_dims; // 原始的dim
  GE_ASSERT_TRUE(!tensor.repeats.empty());
  GE_ASSERT_TRUE(tensor.repeats.size() == tensor.strides.size());
  // 使用repeat和stride重新初始化dims
  tensor.dims.clear();
  size_t vectorized_size = tensor.repeats.size();
  std::vector<bool> is_continuous(vectorized_size, true);
  for (size_t i = 0UL; i < vectorized_size; ++i) {
    org_dims.emplace_back(tensor.repeats[i]);
  }
  if (org_dims.empty()) {
    org_dims.emplace_back(ge::sym::kSymbolOne);
  } else {
    UpdateIsContinous(tensor.repeats, tensor.strides, is_continuous);
    if (IsLoadStore(op_type)) {
      UpdateIsContinous(tensor.repeats, tensor.gm_strides, is_continuous);
    }
    tensor.dims.emplace_back(org_dims[0]);
    UpdateTensorDim(op_type, is_continuous, org_dims, tensor);
  }
  GELOGD("Tensor[%s] shape info: merge dims:%s, org dims:%s, continuous:%s", tensor_name.c_str(),
         tensor.GetDimExpr().c_str(), GetVecString(org_dims).c_str(), ge::ToString(is_continuous).c_str());
  return ge::SUCCESS;
}

ge::Status GetOuterParams(const vector<Expr> &dims, Expr &outer_repeat, vector<Expr> &used_dims,
                          const uint32_t dma_max_len) {
  // 若dma_max_len为2:
  // (2, 2, 2)
  //     used_dims = {2, 2}
  // repeat=2
  GE_ASSERT_TRUE(!dims.empty());
  if (dims.size() == 1) {
    used_dims = {dims[0]};
    outer_repeat = ge::sym::kSymbolOne;
  } else {
    size_t dim_size = dims.size();
    for (int64_t i = 0L; i < static_cast<int64_t>(dma_max_len); ++i) {
      used_dims.emplace_back(dims[dim_size - static_cast<int64_t>(dma_max_len) + i]);
    }
    outer_repeat = accumulate(dims.begin(), (dims.end() - dma_max_len), CreateExpr(1),
                              [](Expr a, Expr b) { return ge::sym::Mul(a, b); });
    GELOGD("Got outer loop %s, used dims %s", outer_repeat.Serialize().get(), GetVecString(used_dims).c_str());
  }
  return ge::SUCCESS;
}

ge::Status UpdateTenary(PerfOutputInfo &perf_res, PerfOutputInfo &output_res) {
  Expr cur_var;
  std::vector<std::pair<Expr, Expr>> replace_vars;
  for (const auto &pair : perf_res.tenary_ops) {
    std::string cur_name = Str(pair.first);
    GetPerfVar(cur_name, cur_var, output_res.tenary_ops);
    replace_vars.emplace_back(std::make_pair(pair.first, cur_var));
    output_res.tenary_ops[cur_var] = pair.second;
    output_res.tenary_ops[cur_var].SetVariable(cur_var);
  }
  for (const auto &pair : perf_res.pipe_res) {
    output_res.pipe_res[pair.first] = pair.second.Replace(replace_vars);
  }
  perf_res.pipe_res = output_res.pipe_res;
  return ge::SUCCESS;
}

ge::Status GetDmaPerf(const TensorShapeInfo &tensor_info, NodeDetail &node_info, PerfOutputInfo &perf_res,
                      int32_t supported_max_dma_len, bool need_swap) {
  PerfOutputInfo non_swap_perf;
  vector<Expr> used_dims;
  PerfOutputInfo swap_perf;
  Expr non_swap_outer_repeat;
  Expr swap_outer_repeat;
  size_t dim_size = node_info.input_dims.size();
  if (static_cast<int32_t>(dim_size) <= supported_max_dma_len) {
    GELOGD("Dim size [%zu] without dma optimize, node name[%s].", dim_size, node_info.name.c_str());
    GE_ASSERT_SUCCESS(SetStride(tensor_info, node_info, supported_max_dma_len));
    GE_ASSERT_SUCCESS(GetDMAActualPerf(node_info, ge::sym::kSymbolOne, node_info.input_dims, non_swap_perf));
    GE_ASSERT_SUCCESS(UpdateTenary(non_swap_perf, perf_res));
  } else {
    GELOGD("Dim size [%zu] with dma optimize, node name[%s].", dim_size, node_info.name.c_str());
    GE_ASSERT_SUCCESS(GetDmaParams(node_info.input_dims, non_swap_outer_repeat, used_dims, supported_max_dma_len));
    GE_ASSERT_SUCCESS(SetStride(tensor_info, node_info, supported_max_dma_len));
    GE_ASSERT_SUCCESS(GetDMAActualPerf(node_info, non_swap_outer_repeat, used_dims, non_swap_perf));
    GELOGD("Perf without swap is [%s]", non_swap_perf.ToString().c_str());
    if (need_swap) {
      GE_ASSERT_SUCCESS(GetDmaParams(node_info.input_dims, swap_outer_repeat, used_dims, supported_max_dma_len, true));
      GE_ASSERT_SUCCESS(SetStride(tensor_info, node_info, supported_max_dma_len, true));
      GE_ASSERT_SUCCESS(GetDMAActualPerf(node_info, swap_outer_repeat, used_dims, swap_perf));
      GELOGD("Perf with swap is [%s]", swap_perf.ToString().c_str());
    }
    GELOGD("The input dim is [%s]", GetVecString(node_info.input_dims).c_str());
    GE_ASSERT_SUCCESS(UpdateTenary(swap_perf, perf_res), "Update swap perf failed, node=%s",
                      node_info.ToString().c_str());
    if (need_swap) {
      GE_ASSERT_SUCCESS(UpdateSwapPerf(node_info, supported_max_dma_len, swap_perf, non_swap_perf, perf_res),
                        "Update swap perf failed, node=%s", node_info.ToString().c_str());
    }
  }
  return ge::SUCCESS;
}
}
