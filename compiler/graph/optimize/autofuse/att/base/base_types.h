/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed unde the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the license is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ATT_BASIC_BASIC_TYPE_H_
#define ATT_BASIC_BASIC_TYPE_H_
#include <limits>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <map>
#include <algorithm>
#include "graph/def_types.h"
#include "ge_common/ge_api_types.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/expression/const_values.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace att {
using Expr = ge::Expression;
enum SolverType : uint32_t { L0_TILE = 0, L2_TILE, SEARCH_TILE, ERROR };

enum class HardwareDef {
  GM = 0,
  L1,
  L2,
  L0A,
  L0B,
  L0C,
  UB,
  BTBUF,
  CORENUM,
  HARDWAREERR = std::numeric_limits<int32_t>::max()
};

enum class PipeType {
  AIC_MTE1 = 0,
  AIC_MTE2,
  AIC_FIXPIPE,
  AIC_MAC,
  AIV_MTE2,
  AIV_MTE3,
  AIV_VEC,
  AICORE_MTE1,
  AICORE_MTE2,
  AICORE_MTE3,
  AICORE_CUBE,
  AICORE_VEC,
  PIPE_NONE = std::numeric_limits<int32_t>::max()
};

enum class AxisPosition {
  ORIGIN = 0,
  INNER,
  OUTER,
  MERGED,
  POSERR = std::numeric_limits<int32_t>::max()
};

enum class SocVersion {
  ASCEND910B2 = 0,
  ASCEND910B4,
  UNKNOWN = -1
};

enum class TilingDataType {
  // Axis params
  AXIS_ALIGNED_SIZE = 0,
  AXIS_LOOP_NUM,
  AXIS_TAIL_SIZE,
  SPLIT_OUTER_AXIS_TAIL_LOOP_NUM,
  SPLIT_OUTER_AXIS_TAIL_TAIL_SIZE,
  // genera params
  USED_BLOCK_DIM,
  // buffer params
  BUFFER_SIZE,
  TENSOR_SIZE,
  TILING_DATA_TYPE_ALL,
  TILING_DATA_TYPE_ERR = std::numeric_limits<int32_t>::max(),
};

enum class TilingScenarioType : int32_t {
  ATT_TOOLS,
  CANN_AUTOFUSED,
  SCENARIO_INVALID,
};

enum class MicroApiType : int32_t {
  MICRO_API_ADD = 0,
  MICRO_API_SUB,
  MICRO_API_INVALID,
};

struct AxisTilingData {
  TilingDataType arg_type;
  std::string arg_name;
  std::string arg_expr;
};

struct TensorInfo {
  std::string name;
  Expr variable;
  Expr expr;
  TensorInfo(const std::string &name, const Expr &variable, const Expr &expr) : name(name), variable(variable), expr(expr) {}
};

struct NodeApiTilingCode {
  std::string function_invoke; // 高阶API的函数调用
  std::string function_impl; // 高阶API的函数实现
  std::string head_files; // 高阶API依赖的头文件
};

struct VfInstructPerf {
  std::vector<std::string> support_data_types;
  int32_t latency{0};
  int32_t throughput{0};
};

struct TensorShapeInfo {
  inline std::string GetDimExpr() const
  {
    std::stringstream ss;
    for (size_t i = 0U; i < dims.size(); i++) {
      if (i == (dims.size() - 1U)) {
        ss << dims[i];
      } else {
        ss << dims[i] << ",";
      }
    }
    return ss.str();
  }
  uint32_t data_type_size;
  std::string data_type;
  HardwareDef loc;
  std::vector<Expr> dims;
  std::vector<Expr> repeats;  // tensor的repeat
  std::vector<Expr> strides;  // tensor的stride
  std::vector<Expr> gm_strides;  // tensor的stride
  std::vector<Expr> origin_repeats;  // 和codegen逻辑一致，tail切分时保留原始的repeats信息
};

inline std::string Str(const Expr& e) {
  if (!e.IsValid()) {
    return "";
  }
  if (e.Str() == nullptr) {
    return "";
  }
  return std::string(e.Str().get());
}

inline bool IsValid(const Expr &e) {
  Expr tmp;
  if (e.IsValid() && (e != tmp)) {
    return true;
  }
  return false;
}

inline std::string GetSymbolName(const Expr &e) {
  if (!e.IsValid()) {
    return "";
  }
  const ge::Symbol *sym = ge::PtrToPtr<Expr, const ge::Symbol>(&e);
  if (sym->GetName() == nullptr) {
    return "";
  }
  return std::string(sym->GetName().get());
}


struct ExprCmp {
  bool operator()(const Expr &lhs, const Expr &rhs) const
  {
    if (!lhs.IsValid()) {
      return true;
    }
    if (!rhs.IsValid()) {
      return false;
    }
    std::string lhs_str = ge::SymbolicUtils::ToString(lhs);
    std::string rhs_str = ge::SymbolicUtils::ToString(rhs);
    auto cmp_res = lhs_str.compare(rhs_str);
    if (cmp_res < 0) {
      return true;
    }
    return false;
  }
};

inline std::string GetVecString(const std::vector<Expr> &dims) {
  std::string output;
  for (const auto &dim : dims) {
    output.append(ge::SymbolicUtils::ToString(dim)).append(",");
  }
  return output;
}

template <typename T>
inline std::string DebugString(const std::vector<T> &strs) {
  std::string s = "[";
  for (auto &str : strs) {
    s += str;
    s += ",";
  }
  s += "]";
  return s;
}

using ExprExprMap = std::map<Expr, Expr, ExprCmp>;
using ExprUintMap = std::map<Expr, uint32_t, ExprCmp>;

template<typename T>
inline Expr CreateExpr(T value) {
  return ge::Symbol(value);
}

using AscendCApiPerfFunc = ge::Status (*)(const std::map<std::string, float> &param_map,
                                          const std::vector<Expr> &dims, const Expr &gm_stride, Expr &res);
}  // namespace

#endif  // ATT_BASIC_BASIC_TYPE_H_
