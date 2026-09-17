#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""parse_node 与 decompose 回调配合的可运行示例。

ThresholdedRelu 没有一对一的现成算子，需要用已有算子组合表达，两个回调接力：

- parse_node 在 ONNX 节点转换为目标算子时执行：把 alpha 属性按名取出写入目标
  算子，并为动态 IO 的目标算子 PartitionedCall 注册端口（parser 连线需要端口名）；
- decompose 收到的 source 就是 parse_node 产出的算子（读出的 alpha 即中转结果），
  用已有算子构建 Threshold x Mul 子图返回，GE 将原节点原地展开（1:N），因此无需
  提供新的设备 kernel。
"""

from ge.es import GraphBuilder
from ge.es.math import Mul
from ge.es.nn import Threshold
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin


thresholded_relu = onnx_plugin(
    source="ThresholdedRelu",
    domain="example.domain",
    opsets=(1,),
    target="PartitionedCall",
)


@thresholded_relu.parse_node
def parse_thresholded_relu(node: OnnxNode, target: Operator) -> None:
    """把 ONNX 节点的 alpha 属性按名取出写入目标算子，并注册端口供 parser 连线。"""
    target.set_attr("alpha", node.attrs.get("alpha", 1.0))
    target.register_input("x")
    target.register_output("y")


@thresholded_relu.decompose
def decompose_thresholded_relu(source):
    """用已有算子构建 x * (x > alpha) 子图，替换整个 ThresholdedRelu 节点。"""
    alpha = float(source.get_attr("alpha"))
    builder = GraphBuilder("thresholded_relu_decomposition")
    x = builder.create_input(0)
    mask = Threshold(x, threshold=alpha)
    output = Mul(x, mask)
    graph = builder.build_and_reset([output])
    # 内省打印分解产物清单：枚举构建出的子图的真实节点类型，图中不存在的算子不会被打印
    op_types = sorted({node.type for node in graph.get_all_nodes()})
    print(f"[Plugin] ThresholdedRelu is decomposed into operators: {op_types}")
    return graph
