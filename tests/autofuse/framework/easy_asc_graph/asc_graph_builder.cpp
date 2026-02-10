/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ===================================================================================================================*/

#include "asc_graph_builder.h"

#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/symbolizer/symbolic.h"
#include "graph/symbolizer/symbolic_utils.h"

namespace ge::testing {
namespace {
void ComputeStrides(const std::vector<Expression> &repeats,
                    std::vector<Expression> &strides) {
  strides.clear();
  Expression stride = sym::kSymbolOne;
  for (auto iter = repeats.rbegin(); iter != repeats.rend(); ++iter) {
    if (SymbolicUtils::StaticCheckEq(*iter, ge::sym::kSymbolOne) == ge::TriBool::kTrue) {
      strides.push_back(ge::sym::kSymbolZero);
    } else {
      strides.push_back(stride);
      stride = stride * (*iter);
    }
  }
  std::reverse(strides.begin(), strides.end());
}
}

AscGraphBuilder::AscGraphBuilder(const std::string &name)
  : impl_(std::make_unique<Impl>(name)) {
}

AscGraphBuilder::~AscGraphBuilder() = default;

AscGraphBuilder &AscGraphBuilder::Loops(std::initializer_list<int64_t> sizes) {
  std::vector<Expression> expr_sizes;
  for (auto s: sizes) {
    expr_sizes.push_back(Symbol(s));
  }
  return Loops(expr_sizes);
}

AscGraphBuilder &AscGraphBuilder::Loops(std::initializer_list<Expression> sizes) {
  return Loops(std::vector<Expression>(sizes));
}

AscGraphBuilder &AscGraphBuilder::Loops(const std::vector<Expression> &sizes) {
  for (size_t i = 0; i < sizes.size(); ++i) {
    auto axis = impl_->graph_.CreateAxis("z" + std::to_string(i), sizes[i]);
    impl_->axis_ids_.push_back(axis.id);
    impl_->loop_repeats_.push_back(sizes[i]);
  }
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Data(const std::string &name, int64_t index, DataType dtype) {
  ascir_op::Data data_op(name.c_str(), impl_->graph_);
  auto node = impl_->graph_.FindNode(name.c_str());
  data_op.ir_attr.SetIndex(index);
  data_op.y.dtype = dtype;
  if (node != nullptr) {
    impl_->nodes_[name] = node;
  }
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Data(const std::string &name, int64_t index,
                                        const std::vector<Expression> &shape,
                                        const std::vector<Expression> &strides,
                                        DataType dtype) {
  ascir_op::Data data_op(name.c_str(), impl_->graph_);
  auto node = impl_->graph_.FindNode(name.c_str());
  data_op.ir_attr.SetIndex(index);
  data_op.y.dtype = dtype;
  *data_op.y.axis = impl_->axis_ids_;
  *data_op.y.repeats = shape;
  *data_op.y.strides = strides;
  if (node != nullptr) {
    impl_->nodes_[name] = node;
  }
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Scalar(const std::string &name, const std::string &value, DataType dtype) {
  ascir_op::Scalar scalar_op(name.c_str(), impl_->graph_);
  scalar_op.ir_attr.SetValue(value);
  scalar_op.y.dtype = dtype;
  // Scalar: repeat 全1，stride 全0, 其实可以不设置，以防codegen有bug
  std::vector<Expression> scalar_repeats(impl_->loop_repeats_.size(), sym::kSymbolOne);
  *scalar_op.y.repeats = scalar_repeats;
  auto node = impl_->graph_.FindNode(name.c_str());
  if (node != nullptr) {
    impl_->nodes_[name] = node;
  }
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Output(const std::string &name, const std::string &input, int64_t index,
                                         DataType dtype) {
  ascir_op::Output output_op(name.c_str());
  auto node = impl_->graph_.AddNode(output_op);
  output_op.ir_attr.SetIndex(index);
  output_op.y.dtype = dtype;

  impl_->nodes_[name] = node;
  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
  }
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Workspace(const std::string &name, const std::string &input, DataType dtype) {
  ascir_op::Workspace workspace_op(name.c_str());
  auto node = impl_->graph_.AddNode(workspace_op);
  workspace_op.y.dtype = dtype;

  impl_->nodes_[name] = node;
  if (!input.empty()) {
    auto it = impl_->nodes_.find(input);
    if (it != impl_->nodes_.end()) {
      GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                          node->GetInDataAnchor(0));
    }
  }
  return *this;
}

AscGraphBuilder &AscGraphBuilder::Load(const std::string &name, const std::string &input) {
  ascir_op::Load load_op(name.c_str());
  auto node = impl_->graph_.AddNode(load_op);
  *load_op.y.axis = impl_->axis_ids_;

  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
  }

  // 默认使用 loop_repeats_
  *load_op.y.repeats = impl_->loop_repeats_;
  load_op.y.dtype = it->second->outputs()[0]->attr.dtype;

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Load(const std::string &name, const std::string &input,
                                       const std::vector<Expression> &shape,
                                       const std::vector<Expression> &strides) {
  ascir_op::Load load_op(name.c_str());
  auto node = impl_->graph_.AddNode(load_op);
  *load_op.y.axis = impl_->axis_ids_;

  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
  }

  *load_op.y.repeats = shape;
  *load_op.y.strides = strides;
  load_op.y.dtype = it->second->outputs()[0]->attr.dtype;

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Store(const std::string &name, const std::string &input) {
  ascir_op::Store store_op(name.c_str());
  auto node = impl_->graph_.AddNode(store_op);
  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    *store_op.y.axis = it->second->outputs()[0]->attr.axis;
    *store_op.y.repeats = it->second->outputs()[0]->attr.repeats;
    store_op.y.dtype = it->second->outputs()[0]->attr.dtype;
  }

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Store(const std::string &name, const std::string &input,
                                        const std::vector<Expression> &shape,
                                        const std::vector<Expression> &strides) {
  ascir_op::Store store_op(name.c_str());
  auto node = impl_->graph_.AddNode(store_op);
  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    *store_op.y.axis = it->second->outputs()[0]->attr.axis;
    store_op.y.dtype = it->second->outputs()[0]->attr.dtype;
  }

  *store_op.y.repeats = shape;
  *store_op.y.strides = strides;

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Broadcast(const std::string &name, const std::string &input,
                                            const std::vector<int64_t> &brc_axes) {
  ascir_op::Broadcast brc_op(name.c_str());
  auto node = impl_->graph_.AddNode(brc_op);
  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    *brc_op.y.axis = it->second->outputs()[0]->attr.axis;
    brc_op.y.dtype = it->second->outputs()[0]->attr.dtype;

    std::vector<Expression> output_shape = it->second->outputs()[0]->attr.repeats;
    for (int64_t axis: brc_axes) {
      if (axis >= 0 && axis < static_cast<int64_t>(output_shape.size()) &&
          axis < static_cast<int64_t>(impl_->loop_repeats_.size())) {
        output_shape[axis] = impl_->loop_repeats_[axis];
      }
    }
    *brc_op.y.repeats = output_shape;
  }

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Broadcast(const std::string &name, const std::string &input,
                                            std::initializer_list<int64_t> brc_axes) {
  // 转换为 vector 调用另一个重载
  std::vector<int64_t> axes_vec(brc_axes);
  return Broadcast(name, input, axes_vec);
}

AscGraphBuilder &AscGraphBuilder::Broadcast(const std::string &name, const std::string &input,
                                            const std::vector<Expression> &shape) {
  ascir_op::Broadcast brc_op(name.c_str());
  auto node = impl_->graph_.AddNode(brc_op);
  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    *brc_op.y.axis = it->second->outputs()[0]->attr.axis;
    brc_op.y.dtype = it->second->outputs()[0]->attr.dtype;
    *brc_op.y.repeats = shape;
  }

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Transpose(const std::string &name, const std::string &input,
                                            const std::vector<int64_t> &axes) {
  ascir_op::Transpose transpose_op(name.c_str());
  auto node = impl_->graph_.AddNode(transpose_op);
  impl_->nodes_[name] = node;

  // 获取输入节点的输出 shape
  std::vector<Expression> input_shape;
  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end() && !it->second->outputs().empty()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    transpose_op.y.dtype = it->second->outputs()[0]->attr.dtype;
    input_shape = it->second->outputs()[0]->attr.repeats;
  } else {
    input_shape = impl_->loop_repeats_;
  }

  // 在输入 shape 的基础上做 transpose
  std::vector<Expression> output_shape;
  std::vector<AxisId> output_axis;
  for (int64_t axis_idx: axes) {
    if (axis_idx >= 0 && axis_idx < static_cast<int64_t>(input_shape.size())) {
      output_shape.push_back(input_shape[axis_idx]);
    }
    if (axis_idx >= 0 && axis_idx < static_cast<int64_t>(impl_->axis_ids_.size())) {
      output_axis.push_back(impl_->axis_ids_[axis_idx]);
    }
  }

  *transpose_op.y.repeats = output_shape;
  *transpose_op.y.axis = output_axis;

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Concat(const std::string &name, const std::vector<std::string> &inputs) {
  ascir_op::Concat concat_op(name.c_str());
  auto node = impl_->graph_.AddNode(concat_op);
  impl_->nodes_[name] = node;

  // 添加所有输入的边（动态输入）
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto it = impl_->nodes_.find(inputs[i]);
    if (it != impl_->nodes_.end()) {
      GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                          node->GetInDataAnchor(i));
    }
  }

  // 输出属性：repeat 为 loop_repeat，其他跟随 input[0]
  if (!inputs.empty()) {
    auto it = impl_->nodes_.find(inputs[0]);
    if (it != impl_->nodes_.end() && !it->second->outputs().empty()) {
      *concat_op.y.axis = it->second->outputs()[0]->attr.axis;
      concat_op.y.dtype = it->second->outputs()[0]->attr.dtype;
    }
  }
  *concat_op.y.repeats = impl_->loop_repeats_;

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Max(const std::string &name, const std::string &input,
                                      const std::vector<size_t> &reduce_axes) {
  ascir_op::Max max_op(name.c_str());
  auto node = impl_->graph_.AddNode(max_op);
  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    std::vector<Expression> output_shape = it->second->outputs()[0]->attr.repeats;
    std::set<size_t> axes_set(reduce_axes.begin(), reduce_axes.end());
    for (size_t i = 0; i < output_shape.size(); ++i) {
      if (axes_set.count(i) > 0) {
        output_shape[i] = sym::kSymbolOne;
      }
    }
    *max_op.y.axis = it->second->outputs()[0]->attr.axis;
    max_op.y.dtype = it->second->outputs()[0]->attr.dtype;
    *max_op.y.repeats = output_shape;
  }

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Sum(const std::string &name, const std::string &input,
                                      const std::vector<size_t> &reduce_axes) {
  ascir_op::Sum sum_op(name.c_str());
  auto node = impl_->graph_.AddNode(sum_op);
  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    std::vector<Expression> output_shape = it->second->outputs()[0]->attr.repeats;
    std::set<size_t> axes_set(reduce_axes.begin(), reduce_axes.end());
    for (size_t i = 0; i < output_shape.size(); ++i) {
      if (axes_set.count(i) > 0) {
        output_shape[i] = sym::kSymbolOne;
      }
    }
    *sum_op.y.axis = it->second->outputs()[0]->attr.axis;
    sum_op.y.dtype = it->second->outputs()[0]->attr.dtype;
    *sum_op.y.repeats = output_shape;
  }

  return *this;
}

AscGraphBuilder &AscGraphBuilder::Cast(const std::string &name, const std::string &input, DataType dtype) {
  ascir_op::Cast cast_op(name.c_str());
  auto node = impl_->graph_.AddNode(cast_op);
  cast_op.y.dtype = dtype;
  impl_->nodes_[name] = node;

  auto it = impl_->nodes_.find(input);
  if (it != impl_->nodes_.end()) {
    GraphUtils::AddEdge(it->second->GetOutDataAnchor(0),
                        node->GetInDataAnchor(0));
    *cast_op.y.axis = it->second->outputs()[0]->attr.axis;
    *cast_op.y.repeats = it->second->outputs()[0]->attr.repeats;
  }
  return *this;
}

AscGraph AscGraphBuilder::Build() {
  for (const auto &[name, node]: impl_->nodes_) {
    if (node->attr.api.type == ge::ApiType::kAPITypeBuffer) {
      continue;
    }
    node->attr.sched.axis = impl_->axis_ids_;

    for (auto &output: node->outputs()) {
      if (output->attr.axis.empty()) {
        output->attr.axis = impl_->axis_ids_;
      }

      if (output->attr.repeats.empty()) {
        output->attr.repeats = impl_->loop_repeats_;
      }

      // 设置 strides（如果还没设置）
      if (output->attr.strides.empty()) {
        ComputeStrides(output->attr.repeats, output->attr.strides);
      }
    }
  }

  return impl_->graph_;
}
} // namespace ge::testing
