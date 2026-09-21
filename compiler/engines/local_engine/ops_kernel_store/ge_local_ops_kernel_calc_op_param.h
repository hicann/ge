/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_CALC_OP_PARAM_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_CALC_OP_PARAM_H_

#include <string>
#include <unordered_set>

#include "ge/ge_api_error_codes.h"
#include "graph/node.h"
#include "external/ge_common/ge_common_api_types.h"

namespace ge {
namespace ge_local {
// ReuseInput 类算子类型列表：输出地址复用输入地址的零拷贝形状变换算子。
// 供 GeLocalGraphOptimizer::OptimizeWholeGraph（前置设置属性）与
// CalcNodeOffsetByReuseInput（Build 阶段兜底）共同引用，新增/删除类型时只维护本列表。
// 使用 inline 变量：libge_local_engine.so 与 libge_local_opskernel_builder.so 分别编译，
// 外部变量会在 -fvisibility=hidden 且插件 dlopen 顺序不确定时引发跨 so 符号解析失败。
inline const std::unordered_set<std::string> kReuseInputOpTypes = {
    "Bitcast",   "Flatten",   "FlattenV2", "ExpandDims",  "ReFormat",   "Squeeze",
    "Unsqueeze", "SqueezeV2", "SqueezeV3", "UnsqueezeV2", "UnsqueezeV3"};

class GE_FUNC_VISIBILITY GeLocalOpsKernelBuilderCalcOpParam {
 public:
  static graphStatus CalcPhonyConcatNodeOffset(const Node &node);
  static graphStatus CalcPhonySplitNodeOffset(const Node &node);
  static graphStatus CalcNodeOffsetByReuseInput(const Node &node);

 private:
  static Status CheckDimAttrValid(const std::vector<int64_t> &dim_list, const std::vector<int64_t> &num_list,
                                  const uint32_t anchors_size);
  static Status CheckDimAttrMatchTensor(std::vector<int64_t> &dim_list, const GeTensorDescPtr &tensor_desc);

  static Status CheckPhonyConcatInputShapeSame(const Node &node);
  static Status CheckPhonySplitOutputShapeSame(const Node &node);

  static Status CheckPhonyConcatInputIs32Align(const ConstOpDescPtr &op_desc);

  static int64_t GetSplitOffset(const GeTensorDescPtr &tensor_desc, uint32_t node_idx,
                                const std::vector<int64_t> &split_dim_list, const std::vector<int64_t> &split_num_list);
  static int64_t GetConcatOffset(const GeTensorDescPtr &tensor_desc, uint32_t node_idx,
                                 const std::vector<int64_t> &concat_dim_list,
                                 const std::vector<int64_t> &concat_num_list);
  static graphStatus HasBufferFusionOffset(const Node &node, bool &has_fusion_offset);
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_CALC_OP_PARAM_H_
