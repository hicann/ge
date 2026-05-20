/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_EXTERNAL_GRAPH_NAMED_IO_NODE_BUILDER_H_
#define INC_EXTERNAL_GRAPH_NAMED_IO_NODE_BUILDER_H_

#include <memory>
#include "graph/attr_value.h"
#include "graph/gnode.h"

namespace ge {

/**
 * @brief 基于输入、输出和属性名称构造图节点的 Builder。
 *
 * NamedIoNodeBuilder 面向直接按照节点输入/输出名称构图的场景。
 * 与 CompliantNodeBuilder 要求调用方直接按照 IR 定义实例化节点不同，
 * 本 Builder 允许调用方指定节点实例的输入名、输出名、属性名及相关描述信息，
 * 并根据节点类型匹配已注册的算子 IR 定义，校验输入/输出实例名称及顺序
 * 是否与该定义兼容，最终创建对应的图节点。
 *
 * 约束：
 *   - 节点类型对应的算子 IR 需已通过 REG_OP 完成注册，否则 Build 返回 nullptr
 *   - Builder 对象 Build 成功后不应再次使用
 *
 */
class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY NamedIoNodeBuilder {
 public:
  explicit NamedIoNodeBuilder(Graph &graph);
  ~NamedIoNodeBuilder();

  NamedIoNodeBuilder(const NamedIoNodeBuilder &) = delete;
  NamedIoNodeBuilder &operator=(const NamedIoNodeBuilder &) = delete;

  /**
   * @brief 设置节点类型（必填）
   * @param type 算子类型，如 "Add"、"MatMul" 等，对应算子 IR 需已通过 REG_OP 完成注册
   * @return 构建器引用，支持链式调用
   */
  NamedIoNodeBuilder &Type(const char_t *type);

  /**
   * @brief 设置节点名称（选填）
   * @param name 节点名称
   * @return 构建器引用，支持链式调用
   */
  NamedIoNodeBuilder &Name(const char_t *name);

  /**
   * @brief 添加输入实例端口
   * @param name 输入实例名称；普通/可选输入使用 IR 名，动态输入使用 ir_name0, ir_name1, ...
   * 默认 TensorDesc 为 DT_FLOAT + FORMAT_ND，若需指定数据类型或格式请使用带 desc 参数的重载
   * @return 构建器引用，支持链式调用
   *
   * @note：
   *  * 可选输入（OPTIONAL_INPUT）的使用方式：
   *   假设算子 IR 定义为 INPUT(x), INPUT(w), OPTIONAL_INPUT(bias), OUTPUT(y)：
   *   1. 不传递可选输入：仅添加必选输入，跳过 bias 即可
   *      builder.AddInput("x").AddInput("w").AddOutput("y")
   *   2. 传递有效可选输入：按 IR 顺序在对应位置添加，使用默认 TensorDesc
   *      builder.AddInput("x").AddInput("w").AddInput("bias").AddOutput("y")
   *   3. 传递占位可选输入：使用带 TensorDesc 的重载，传入 DT_UNDEFINED / FORMAT_RESERVED
   *      TensorDesc placeholder(Shape(), FORMAT_RESERVED, DT_UNDEFINED);
   *      builder.AddInput("x").AddInput("w").AddInput("bias", placeholder).AddOutput("y")
   *   无论哪种方式，可选输入最多只能提供一个实例，重复添加会导致校验失败。
   *  * 动态输入（DYNAMIC_INPUT）的使用方式：
   *   若传递 DYNAMIC_INPUT(x) 实例，调用方需按 x0, x1, ... 连续添加，
   *   不能跳过中间序号。
   */
  NamedIoNodeBuilder &AddInput(const char_t *name);

  /**
   * @brief 添加带描述的输入实例端口
   * @param name 输入实例名称；普通/可选输入使用 IR 名，动态输入使用 ir_name0, ir_name1, ...
   * @param desc 输入张量描述
   * @return 构建器引用，支持链式调用
   */
  NamedIoNodeBuilder &AddInput(const char_t *name, const TensorDesc &desc);

  /**
   * @brief 添加输出实例端口
   * @param name 输出实例名称；普通输出使用 IR 名，动态输出使用 ir_name0, ir_name1, ...
   * 默认 TensorDesc 为 DT_FLOAT + FORMAT_ND，若需指定数据类型或格式请使用带 desc 参数的重载
   * @return 构建器引用，支持链式调用
   *
   * @note：
   *  * 动态输出（DYNAMIC_OUTPUT）的使用方式：
   *   若传递 DYNAMIC_OUTPUT(y) 实例，调用方需按 y0, y1, ... 连续添加，
   *   不能跳过中间序号。
   */
  NamedIoNodeBuilder &AddOutput(const char_t *name);

  /**
   * @brief 添加带描述的输出实例端口
   * @param name 输出实例名称；普通输出使用 IR 名，动态输出使用 ir_name0, ir_name1, ...
   * @param desc 输出张量描述
   * @return 构建器引用，支持链式调用
   */
  NamedIoNodeBuilder &AddOutput(const char_t *name, const TensorDesc &desc);

  /**
   * @brief 设置节点属性
   * @param name 属性名称
   * @param value 属性值，通过 AttrValue::SetAttrValue 构造
   *
   * 示例：
   *   AttrValue axis;
   *   axis.SetAttrValue(static_cast<int64_t>(1));
   *   builder.Attr("axis", axis);
   *
   * Build 时用户设置的属性优先，已注册 IR 定义中的默认属性仅用于补全，不覆盖用户设置值。
   *
   * @return 构建器引用，支持链式调用
   */
  NamedIoNodeBuilder &Attr(const char_t *name, const AttrValue &value);

  /**
   * @brief 构建图节点并添加到 Graph。
   *
   * Build 会基于已设置的输入、输出和属性信息创建 GNode，并校验输入/输出
   * 实例名称及顺序是否与已注册的算子 IR 定义兼容。
   *
   * @param[out] error_message 构建失败时的错误信息
   * @return 成功返回 unique_ptr<GNode>（已被添加到 Graph，error_message 被清空）；失败返回 nullptr
   */
  std::unique_ptr<GNode> Build(AscendString &error_message);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_NAMED_IO_NODE_BUILDER_H_
