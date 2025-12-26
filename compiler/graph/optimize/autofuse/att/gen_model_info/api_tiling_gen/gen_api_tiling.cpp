/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024 All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "gen_api_tiling.h"
#include <set>
#include <vector>
#include "graph/node.h"
#include "common/checker.h"
#include "base/base_types.h"
#include "api_tiling_gen/api_tiling_gen_register.h"

namespace att {
const std::string kTilingDataName = "tiling_data";

ApiTilingMgr &ApiTilingMgr::Instance() {
  static ApiTilingMgr api_tiling_mgr;
  return api_tiling_mgr;
}

std::string GetCallFunc(const std::string &func_name,
                        const std::vector<std::pair<std::string, std::string>> &params) {
  std::string call_func = "\t" + func_name + "(";
  for (const auto &param : params) {
    call_func += param.first;
    call_func += ", ";
  }
  if (!call_func.empty()) {
    call_func.erase(call_func.size() - 1);
    call_func.erase(call_func.size() - 1);
  }
  call_func += ");\n";
  return call_func;
}

std::string GetApiTilingCode(const std::vector<std::pair<std::string, std::string>> &params) {
  std::string api_tiling_code;
  for (const auto &param : params) {
    api_tiling_code += param.second;
  }
  return api_tiling_code;
}

std::string GetNodeValue(const ge::AscGraph& graph, const ge::AscNodePtr& node, const std::string &tensor_name) {
  // 找到node的input tensor对应的节点为图的第几个输入
  auto idx = node->GetOpDesc()->GetInputIndexByName(tensor_name);
  auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor, "Node [%s] input tensor [%d] is null.", node->GetName().c_str(), idx);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "Node [%s] input tensor [%d] peer out is null.", node->GetName().c_str(), idx);
  auto peer_out_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_out_node, "Node [%s] input tensor [%d] peer out node is null.", node->GetName().c_str(), idx);
  uint32_t input_idx = 0u;
  bool found_input_in_graph = false;
  for (const auto &input_node : graph.GetInputNodes()) {
    if (input_node == nullptr) {
      continue;
    }
    if (input_node->GetName() == peer_out_node->GetName()) {
      found_input_in_graph = true;
      break;
    }
    input_idx++;
  }
  // 生成代码。从图的输入中获取tensor的值
  if (found_input_in_graph) {
    return ("\t int32_t " + tensor_name + "_value = context->GetInputTensor(" + std::to_string(idx) + ")->GetData();\n");
  }
  return "return false;\n";
}

std::string GetNodeAttr(const ge::AscNodePtr &node, const std::string &attr_name, bool attr_value)
{
  ge::AttrUtils::GetBool(node->GetOpDesc(), attr_name, attr_value);
  if (attr_value) {
    return "true";
  }
  return "false";
}

std::string GetNodeAttr(const ge::AscNodePtr &node, const std::string &attr_name, int attr_value)
{
  ge::AttrUtils::GetInt(node->GetOpDesc(), attr_name, attr_value);
  return std::to_string(attr_value);
}

std::string GetInputTensorDataTypeSize(const ge::AscNodePtr &node, const std::string &tensor_name) {
  auto tensor_desc = node->GetOpDesc()->GetInputDescPtr(tensor_name);
  GE_ASSERT_NOTNULL(tensor_desc, "Node [%s] input tensor [%s] is null.", node->GetName().c_str(), tensor_name.c_str());
  auto datatype_size = ge::GetSizeByDataType(tensor_desc->GetDataType());
  return std::to_string(datatype_size);
}

ge::graphStatus GetNodeInputTensorView(const ge::AscNodePtr &node, const ge::AscGraph &graph,
                                       const std::string &tensor_name, ge::AscTensorAttr &tensor_attr) {
  auto idx = node->GetOpDesc()->GetInputIndexByName(tensor_name);
  auto in_anchor = node->GetInDataAnchor(idx);
  GE_ASSERT_NOTNULL(in_anchor, "Node [%s] input tensor [%d] is null.", node->GetName().c_str(), idx);
  auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor, "Node [%s] input tensor [%d] peer out is null.", node->GetName().c_str(), idx);
  auto peer_out_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_out_node, "Node [%s] input tensor [%d] peer out node is null.", node->GetName().c_str(), idx);
  auto peer_out_idx = peer_out_anchor->GetIdx();
  const auto node_view = graph.FindNode(peer_out_node->GetName().c_str());
  GE_ASSERT_NOTNULL(node_view, "Node [%s] can't find.", node->GetName().c_str());
  tensor_attr = node_view->outputs[peer_out_idx].attr;
  return ge::SUCCESS;
}

std::vector<std::vector<Expr>> GetAxisGroup(const std::map<int64_t, ge::AxisPtr> &axes,
  const std::map<int64_t, Expr> &valid_axis_size,
  const std::vector<int64_t> &loop_inside_axis_ids) {
  (void)axes;
  std::vector<std::vector<Expr>> axis_groups;
  for (const auto &axisid : loop_inside_axis_ids) {
    std::vector<Expr> axis_group;
    if (valid_axis_size.find(axisid) == valid_axis_size.end()) {
      continue;
    }
    axis_group.emplace_back(valid_axis_size.at(axisid));
    axis_groups.emplace_back(axis_group);
  }
  return axis_groups;
}

inline std::string GetNodeTilingName(const ge::AscNodePtr &node) {
  return (node->GetName() + "_tiling");
}

std::vector<Expr> GetNodeInputTensorShape(const ge::AscNodePtr &node, const ge::AscGraph &graph,
                                          const std::string &tensor_name) {
  std::map<int64_t, ge::AxisPtr> axes;
  std::vector<Expr> axis_group_dims;
  for (const auto &ax : graph.GetAllAxis()) {
    if (ax == nullptr) {
      continue;
    }
    axes[ax->id] = ax;
  }
  ge::AscTensorAttr tensor_attr;
  GE_ASSERT_SUCCESS(GetNodeInputTensorView(node, graph, tensor_name, tensor_attr));
  const auto node_ptr = graph.FindNode(node->GetName().c_str());
  if (node_ptr == nullptr) {
    return axis_group_dims;
  }
  auto loop_axis_id = node_ptr->attr.sched.loop_axis;
  std::map<int64_t, Expr> valid_axis_size;
  for (size_t i = 0u; i < tensor_attr.axis.size(); ++i) {
    if (tensor_attr.strides[i] == 0) {
      continue;
    }
    valid_axis_size.emplace(tensor_attr.axis[i], tensor_attr.repeats[i]);
  }
  std::vector<int64_t> loop_inside_axis_ids;
  for (auto &axis : tensor_attr.vectorized_axis) {
    if (axis == loop_axis_id) {
      loop_inside_axis_ids.clear();
      continue;
    }
    loop_inside_axis_ids.emplace_back(axis);
  }
  // 通过axis的from分组，每个组对应一个axis
  auto axis_groups = GetAxisGroup(axes, valid_axis_size, loop_inside_axis_ids);
  for (auto &group : axis_groups) {
    Expr axis_dim_expr;
    for (auto &sub_dim_size : group) {
      if (!axis_dim_expr.IsValid()) {
        axis_dim_expr = sub_dim_size;
      } else {
        axis_dim_expr = ge::sym::Mul(axis_dim_expr, sub_dim_size);
      }
    }
    axis_group_dims.emplace_back(axis_dim_expr);
  }
  return axis_group_dims;
}

std::string GetAxisizeStr(const Expr& size_var) {
  return (kTilingDataName + "." + Str(size_var));
}

const std::map<ge::Position, std::string> kPosition2Str = {
  {ge::Position::kPositionGM, "TPosition::GM"},
  {ge::Position::kPositionVecIn, "TPosition::VECIN"},
  {ge::Position::kPositionVecOut, "TPosition::VECOUT"},
};

std::string GetInputTensorPosition(const ge::AscNodePtr &node, const std::string &tensor_name) {
  GE_ASSERT_NOTNULL(node, "node is null");
  GE_ASSERT_NOTNULL(node->GetOpDesc(), "node opdesc is null");
  auto idx = node->GetOpDesc()->GetInputIndexByName(tensor_name);
  auto &peer_output = node->inputs[idx].attr;
  if (kPosition2Str.find(peer_output.mem.position) != kPosition2Str.end()) {
    return kPosition2Str.at(peer_output.mem.position);
  }
  return kPosition2Str.at(ge::Position::kPositionGM);
}

std::string GetOutputTensorPosition(const ge::AscNodePtr &node, const std::string &tensor_name) {
  GE_ASSERT_NOTNULL(node, "node is null");
  GE_ASSERT_NOTNULL(node->GetOpDesc(), "node opdesc is null");
  auto idx = node->GetOpDesc()->GetOutputIndexByName(tensor_name);
  auto &tensor_view = node->outputs[idx].attr;
  if (kPosition2Str.find(tensor_view.mem.position) != kPosition2Str.end()) {
    return kPosition2Str.at(tensor_view.mem.position);
  }
  return kPosition2Str.at(ge::Position::kPositionGM);
}

const std::map<ge::DataType, std::string> kDataType2Str = {
  {ge::DT_FLOAT16, "matmul_tiling::DataType::DT_FLOAT16"},
  {ge::DT_FLOAT, "matmul_tiling::DataType::DT_FLOAT"}
};

std::string GetInTensorDtypeStr(const ge::AscNodePtr &node, const std::string &tensor_name) {
  GE_ASSERT_NOTNULL(node, "node is null");
  GE_ASSERT_NOTNULL(node->GetOpDesc(), "node opdesc is null");
  auto tensor_desc = node->GetOpDesc()->GetInputDescPtr(tensor_name);
  GE_ASSERT_NOTNULL(tensor_desc, "Node [%s] input tensor [%s] is null.", node->GetName().c_str(), tensor_name.c_str());
  if (kDataType2Str.find(tensor_desc->GetDataType()) != kDataType2Str.end()) {
    return kDataType2Str.at(tensor_desc->GetDataType());
  }
  return "matmul_tiling::DataType::DT_FLOAT16";
}

std::string GetOutTensorDtypeStr(const ge::AscNodePtr &node, const std::string &tensor_name) {
  GE_ASSERT_NOTNULL(node, "node is null");
  GE_ASSERT_NOTNULL(node->GetOpDesc(), "node opdesc is null");
  auto tensor_desc = node->GetOpDesc()->GetOutputDesc(tensor_name);
  if (kDataType2Str.find(tensor_desc.GetDataType()) != kDataType2Str.end()) {
    return kDataType2Str.at(tensor_desc.GetDataType());
  }
  return "matmul_tiling::DataType::DT_FLOAT16";
}

std::string GetTensorOrigShapeStr(const std::vector<Expr>& shape) {
  std::string tensor_shape = "{";
  for (auto dim : shape) {
    tensor_shape += GetAxisizeStr(dim);
    tensor_shape += ",";
  }
  if (tensor_shape != "{") {
    tensor_shape.erase(tensor_shape.size() - 1);
  }
  tensor_shape += "}";
  return tensor_shape;
}

std::string GetTensorShapeStr(const std::vector<Expr>& shape) {
  std::string tensor_shape = "{";
  for (auto dim : shape) {
    tensor_shape += "(" + GetAxisizeStr(dim) + " + 15) / 16";
    tensor_shape += ",";
  }
  if (tensor_shape != "{") {
    tensor_shape.erase(tensor_shape.size() - 1);
  }
  tensor_shape += "}";
  return tensor_shape;
}

std::string GetTensorSizeStr(const std::vector<Expr>& shape) {
  std::string tensor_size = "(";
  for (auto dim : shape) {
    tensor_size += GetAxisizeStr(dim);
    tensor_size += " * ";
  }
  if (tensor_size != "(") {
    tensor_size.erase(tensor_size.size() - 1);
    tensor_size.erase(tensor_size.size() - 1);
  }
  tensor_size += ")";
  return tensor_size;
}

class ApiTiling {
 public:
  ApiTiling(const ge::AscGraph &graph, const ge::AscNodePtr &node) : graph_(graph), node_(node) {}
  virtual ~ApiTiling() = default;

  void SetModelId(const uint32_t id) {
    model_id_ = id;
  }

  virtual void GetPlatform() {
    std::string platform_code = "\tauto ascendcPlatform = plat_ascendc::PlatformAscendC(context->GetPlatformInfo());\n";
    tiling_vars_to_values_.emplace_back(std::make_pair("ascendcPlatform", platform_code));
  }

  virtual void GetTensorShape(const std::string &tensor_name) {
    auto tensor_shape = GetNodeInputTensorShape(node_, graph_, tensor_name);
    std::string tensor_shape_name = tensor_name + "_shape";
    std::string tensor_shape_str = GetTensorShapeStr(tensor_shape);
    tensor_shape_str = "\tge::Shape " + tensor_shape_name + "= ge::Shape(" + tensor_shape_str + ");\n";
    tiling_vars_to_values_.emplace_back(std::make_pair(tensor_shape_name, tensor_shape_str));
  }

  virtual void GetTensorOrigShape(const std::string &tensor_name) {
    auto tensor_shape = GetNodeInputTensorShape(node_, graph_, tensor_name);
    std::string tensor_orig_shape = tensor_name + "_orig_shape";
    std::string tensor_orig_shape_str = GetTensorOrigShapeStr(tensor_shape);
    tensor_orig_shape_str = "\tge::Shape " + tensor_orig_shape + "= ge::Shape(" + tensor_orig_shape_str + ");\n";
    tiling_vars_to_values_.emplace_back(std::make_pair(tensor_orig_shape, tensor_orig_shape_str));
  }

  virtual void GetTensorSize(const std::string &tensor_name) {
    auto tensor_shape = GetNodeInputTensorShape(node_, graph_, tensor_name);
    std::string tensor_size = tensor_name + "_size";
    std::string tensor_size_str = GetTensorSizeStr(tensor_shape);
    auto tensor_dtype_size = GetInputTensorDataTypeSize(node_, tensor_name);
    tensor_size_str = "(" + tensor_size_str + " * " + tensor_dtype_size + ")";
    tensor_size_str = "\tuint32_t " + tensor_size + " = " + tensor_size_str + ";\n";
    tiling_vars_to_values_.emplace_back(std::make_pair(tensor_size, tensor_size_str));
  }
  
  virtual void GetTensorAttrBool(const std::string &attr_name) {
    bool attr{false};
    std::string node_attr_name = node_->GetName() + attr_name;
    std::string attr_str = "\tbool " +  node_attr_name + " = "
                           + GetNodeAttr(node_, "ascir::" + node_->GetType() + "::ATTR_" + attr_name, attr) + ";\n";
    tiling_vars_to_values_.emplace_back(std::make_pair(node_attr_name, attr_str));
  }

  virtual void GetTensorDtypeSize(const std::string &tensor_name) {
    auto tensor_dtype_size = GetInputTensorDataTypeSize(node_, tensor_name);
    auto tensor_dtype_size_str = "\tuint32_t " + tensor_name + "_dtype_size = " + tensor_dtype_size + ";\n";
    std::string tensor_datatype_size(tensor_name + "_dtype_size");
    tiling_vars_to_values_.emplace_back(std::make_pair(tensor_datatype_size, tensor_dtype_size_str));
  }

  virtual void GetTilingData(const std::string &tiling_data_type_name) {
    std::string tiling_define_code;
    // std::string tiling_define_code(tiling_data_type_name);
    ApiTilingMgr::Instance().SetApiTilingDataType(model_id_, node_->GetName(),
      std::make_pair(GetNodeTilingName(node_), tiling_data_type_name));
    //tiling_define_code += " tiling_data;\n";
    tiling_vars_to_values_.emplace_back(std::make_pair((kTilingDataName + "." + GetNodeTilingName(node_)), tiling_define_code));
  }

  virtual void GetTilingFunc(const std::string &tiling_func_name) {
    std::string call_func = GetCallFunc(tiling_func_name, tiling_vars_to_values_);
    tiling_vars_to_values_.emplace_back(std::make_pair("call_func", call_func));
  }

  virtual std::string Run() {
    return GetApiTilingCode(tiling_vars_to_values_);
  }
 protected:
  uint32_t model_id_{0u};
  const ge::AscGraph &graph_;
  const ge::AscNodePtr &node_;
  std::vector<std::pair<std::string, std::string>> tiling_vars_to_values_;
};

class MatmulApiTilng : public ApiTiling {
public:
  MatmulApiTilng(const ge::AscGraph &graph, const ge::AscNodePtr &node) : ApiTiling(graph, node) {}
  ~MatmulApiTilng() override = default;
  void GetApiTilingDef() {
    std::string api_tiling_def = "\tMatmulApiTiling tiling(ascendcPlatform);\n";
    tiling_vars_to_values_.emplace_back(std::make_pair("api_tiling_def", api_tiling_def));
  }

  void GetTensorInfo(const std::string &tensor_a_name, const std::string &tensor_b_name,
    const std::string &tensor_c_name, std::string bias_name = "") {
    auto tensor_a_position = GetInputTensorPosition(node_, tensor_a_name);
    auto tensor_b_position = GetInputTensorPosition(node_, tensor_b_name);
    auto tensor_c_position = GetOutputTensorPosition(node_, tensor_c_name);
    auto tensor_a_dtype = GetInTensorDtypeStr(node_, tensor_a_name);
    auto tensor_b_dtype = GetInTensorDtypeStr(node_, tensor_b_name);
    auto tensor_c_dtype = GetOutTensorDtypeStr(node_, tensor_c_name);
    auto a_info = "\ttiling.SetAType(" + tensor_a_position + ", CubeFormat::ND, " + tensor_a_dtype + ");\n";
    auto b_info = "\ttiling.SetBType(" + tensor_b_position + ", CubeFormat::ND, " + tensor_b_dtype + ");\n";
    auto c_info = "\ttiling.SetCType(" + tensor_c_position + ", CubeFormat::ND, " + tensor_c_dtype + ");\n";
    std::string bias_info;
    if (!bias_name.empty()) {
      auto tensor_bias_position = GetInputTensorPosition(node_, bias_name);
      auto tensor_bias_dtype = GetInTensorDtypeStr(node_, bias_name);
      bias_info = "\ttiling.SetBiasType(" + tensor_a_position + ", CubeFormat::ND, " + tensor_a_dtype + ");\n";
    }
    std::string tensor_desc = a_info;
    tensor_desc += b_info;
    tensor_desc += c_info;
    tensor_desc += bias_info;
    tiling_vars_to_values_.emplace_back(std::make_pair("tensor_desc", tensor_desc));
  }

  void GetTensorShapes() {
    auto tensor_a_shape = GetNodeInputTensorShape(node_, graph_, "x1");
    Expr m;
    Expr ka;
    Expr n;
    Expr kb;
    if (tensor_a_shape.size() == 2u) {
      // [M, K]
      size_t m_dim_idx_in_mm = 0u;
      size_t ka_dim_idx_in_mm = 1u;
      m = tensor_a_shape[m_dim_idx_in_mm];
      ka = tensor_a_shape[ka_dim_idx_in_mm];
    } else if (tensor_a_shape.size() == 3u) {
      // [B, M, K]
      size_t m_dim_idx_in_bmm = 1u;
      size_t ka_dim_idx_in_bmm = 2u;
      m = tensor_a_shape[m_dim_idx_in_bmm];
      ka = tensor_a_shape[ka_dim_idx_in_bmm];
    }
    auto tensor_b_shape = GetNodeInputTensorShape(node_, graph_, "x2");
    if (tensor_b_shape.size() == 2u) {
      // [M, K]
      size_t kb_dim_idx_in_mm = 0u;
      size_t n_dim_idx_in_mm = 1u;
      kb = tensor_b_shape[kb_dim_idx_in_mm];
      n = tensor_b_shape[n_dim_idx_in_mm];
    } else if (tensor_b_shape.size() == 3u) {
      // [B, M, K]
      size_t kb_dim_idx_in_bmm = 1u;
      size_t n_dim_idx_in_bmm = 2u;
      kb = tensor_a_shape[kb_dim_idx_in_bmm];
      n = tensor_a_shape[n_dim_idx_in_bmm];
    }
    std::string shape_str;
    if ((m.IsValid()) && (ka.IsValid()) && (n.IsValid()) && (kb.IsValid())) {
      if (ka == kb) {
        shape_str = GetAxisizeStr(m) + ", " + GetAxisizeStr(n) + ", " + GetAxisizeStr(ka);
      } else {
        shape_str = GetAxisizeStr(m) + ", " + GetAxisizeStr(n) + ", " + GetAxisizeStr(ka) + "," + GetAxisizeStr(kb);
      }
    } else {
      shape_str = "-1, -1, -1";
    }
    std::string shape_settting = "\ttiling.SetShape(" + shape_str + ");\n";
    shape_settting += "\ttiling.SetOriShape(" + shape_str + ");\n";
    tiling_vars_to_values_.emplace_back(std::make_pair("shape_settting", shape_settting));
  }

  void GetUsableSpace() {
    std::string spapce_str("\ttiling.SetBufferSpace(-1, -1, -1);\n");
    tiling_vars_to_values_.emplace_back(std::make_pair("space", spapce_str));
  }

  void GetPreferedBase() {
    std::string spapce_str("\ttiling.SetFixSplit(-1, -1, -1);\n");
    tiling_vars_to_values_.emplace_back(std::make_pair("base", spapce_str));
  }

  void GetTilingData(const std::string &tiling_data_type_name) override {
    std::string tiling_define_code(tiling_data_type_name);
    ApiTilingMgr::Instance().SetApiTilingDataType(model_id_, node_->GetName(),
      std::make_pair(GetNodeTilingName(node_), tiling_data_type_name));
  }

  void GetTilingFunc(const std::string &tiling_func_name) override {
    std::string call_func = "\ttiling." + tiling_func_name 
                            + "(" + kTilingDataName + "." + GetNodeTilingName(node_) +");\n";
    tiling_vars_to_values_.emplace_back(std::make_pair("call_func", call_func));
  }
};
/**
 * auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
 * MatmulApiTiling tiling(ascendcPlatform); 
 * tiling.SetAType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
 * tiling.SetBType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT16);   
 * tiling.SetCType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
 * tiling.SetBiasType(TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);   
 * tiling.SetShape(1024, 1024, 1024);   
 * tiling.SetOrgShape(1024, 1024, 1024); //或Ka,Kb不等长，如tiling.SetOrgShape(1024, 1024, 1024, 1280)   
 * tiling.SetBias(true);   
 * tiling.SetBufferSpace(-1, -1, -1);  // 设定允许使用的空间，缺省使用该AI处理器所有空间
 * optiling::TCubeTiling tilingData;   
 * int ret = tiling.GetTiling(tilingData);    // if ret = -1, get tiling failed
 */
std::string GetMatmulTiling(const ge::AscGraph &graph, const ge::AscNodePtr &node, const uint32_t model_id = 0u) {
  MatmulApiTilng tiling_gen(graph, node);
  tiling_gen.SetModelId(model_id);
  tiling_gen.GetPlatform();
  tiling_gen.GetApiTilingDef();
  tiling_gen.GetTensorInfo("x1", "x2", "y");
  tiling_gen.GetTensorShapes();
  tiling_gen.GetUsableSpace();
  tiling_gen.GetPreferedBase();
  tiling_gen.GetTilingData("TCubeTiling");
  tiling_gen.GetTilingFunc("GetTiling");
  return tiling_gen.Run();
}

/**
 * void SoftMaxFlashV2TilingFunc(const ge::Shape& srcShape, const uint32_t dataTypeSize1, const uint32_t dataTypeSize2,
 *  const uint32_t localWorkSpaceSize, optiling::SoftMaxTiling& softmaxTiling, bool isUpdate, bool isBasicBlock = false)
 */
std::string GetSoftmaxTiling(const ge::AscGraph &graph, const ge::AscNodePtr &node, const uint32_t model_id = 0u) {
  ApiTiling tiling_gen(graph, node);
  tiling_gen.SetModelId(model_id);
  tiling_gen.GetTensorOrigShape("x1");
  tiling_gen.GetTensorDtypeSize("x1");
  tiling_gen.GetTensorDtypeSize("x2");
  tiling_gen.GetTensorSize("x3");
  tiling_gen.GetTilingData("SoftMaxTiling");
  tiling_gen.GetTensorAttrBool("isUpdate");
  tiling_gen.GetTensorAttrBool("isBasicBlock");
  tiling_gen.GetTilingFunc("SoftMaxFlashV2TilingFunc");
  return tiling_gen.Run();
}
}  // namespace att

namespace att {
namespace {
// for att tools
const std::string kMatMulType = "MatMul";
const std::string kFlashSoftmaxType = "FlashSoftmax";
}
using AttToolsApiTilingDataGenerator =
    std::function<std::string(const ge::AscGraph &graph, const ge::AscNodePtr &node, const uint32_t model_id)>;
const std::map<std::string, AttToolsApiTilingDataGenerator> kTilingOpsForAttTools = {
    {kMatMulType, GetMatmulTiling},
    {kFlashSoftmaxType, GetSoftmaxTiling},
};

inline bool NeedTiling(const std::string &node_type) {
  return kTilingOpsForAttTools.find(node_type) != kTilingOpsForAttTools.end();
}

std::string GetApiTilingFunc(const ge::AscGraph &graph, const ge::AscNodePtr &node, const uint32_t model_id = 0u) {
  auto node_type = node->GetType();
  if (node_type == kFlashSoftmaxType) {
    return GetSoftmaxTiling(graph, node, model_id);
  }
  if (node_type == kMatMulType) {
    return GetMatmulTiling(graph, node, model_id);
  }
  return "";
}

void GenApiTilingFuncCode(NodeApiTilingParams &node_param, const std::string &tiling_data_name = "TilingData",
                          uint32_t model_id = 0u, bool need_context = false) {
  const auto &node = node_param.node;
  std::string main_func = GetApiTilingFunc(node_param.api_tiling_params.graph, node, model_id);
  std::string code("void Get" + node->GetName() + "Tiling(" + tiling_data_name + " &tiling_data) {\n");
  if (need_context) {
    code = "void Get" + node->GetName() + "Tiling(" + tiling_data_name +
           " &tiling_data, gert::TilingContext *context) {\n";
  }
  code += main_func;
  code += "}\n";
  node_param.api_tiling_code.function_invoke = "Get" + node->GetName() + "Tiling(tiling_data";
  node_param.api_tiling_code.function_invoke += (need_context ? ", context)" : ")");
  node_param.api_tiling_code.function_impl = code;
}

ge::Status GetApiTilingInfo(const uint32_t tiling_case_id, const ApiTilingParams &params,
                            std::map<std::string, NodeApiTilingCode> &node_name_to_api_code) {
  // 遍历图上的节点，判断节点是否需要tiling
  for (const auto &node : params.graph.GetAllNodes()) {
    // 判断节点是否需要tiling
    const auto node_type = node->GetType();
    GE_ASSERT_NOTNULL(node, "Get graph node failed.");
    // 优先注册自动融合场景的高阶API
    if (ApiTilingGenRegistry::Instance().IsApiTilingRegistered(node_type) &&
        (params.type == TilingScenarioType::CANN_AUTOFUSED)) {
      AutofuseApiTilingGenerator generator(params.graph, node, params.tiling_data_type, tiling_case_id);
      GE_ASSERT_SUCCESS(generator.Generate(),
                        "Generate api tiling code failed, graph[%s], node[%s] tiling data type[%s]",
                        params.graph.GetName().c_str(), node->GetName().c_str(), params.tiling_data_type.c_str());
      node_name_to_api_code[node->GetName()].function_invoke = generator.GetFuncInvoke();
      node_name_to_api_code[node->GetName()].function_impl = generator.GetFuncImpl();
      node_name_to_api_code[node->GetName()].head_files = generator.GetHeadFiles();
      continue;
    }
    if (NeedTiling(node_type)) {
      // 如果需要tiling，则生成tiling的代码
      NodeApiTilingParams node_api_params{params, {}, node};
      bool need_ctx = (node_type == kMatMulType);
      GenApiTilingFuncCode(node_api_params, params.tiling_data_type, tiling_case_id, need_ctx);
      std::string api_name("Get" + node->GetName() + "Tiling(tiling_data)");
      if (need_ctx) {
        api_name.erase(api_name.size() - 1);
        api_name += ", context)";
      }
      ApiTilingMgr::Instance().SetApiTilingFunc(tiling_case_id, api_name,
                                                node_api_params.api_tiling_code.function_impl);
    }
  }
  return ge::SUCCESS;
}
}  // namespace att
