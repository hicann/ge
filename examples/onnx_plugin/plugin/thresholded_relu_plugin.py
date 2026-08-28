#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A runnable ONNX Python plugin example.

The custom node is decomposed into existing GE operators, so the generated
model does not need a separate device kernel for ThresholdedRelu.

Both callbacks cooperate in this example:

- ``parse_node`` runs when the ONNX node is converted to the target operator.
  It relays the ``alpha`` attribute and registers the ports of the dynamic-IO
  target ``PartitionedCall``. The ``decompose`` callback reads ``alpha`` from
  that operator, so it depends on this attribute relay.
- ``decompose`` receives the parsed operator as ``source`` and builds the
  replacement graph with existing GE/ES operators.
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
    """Relay the ONNX attribute and register ports for the target operator.

    The parsed operator is later handed to ``decompose`` as ``source``, so the
    attribute written here is what ``decompose`` reads. Port registration is
    required because ``PartitionedCall`` has dynamic IO and the parser needs
    the port names to wire the graph.
    """
    target.set_attr("alpha", node.attrs.get("alpha", 1.0))
    target.register_input("x")
    target.register_output("y")


@thresholded_relu.decompose
def decompose_thresholded_relu(source):
    """Build x * (x > alpha) with existing GE/ES operators.

    ``source`` is the operator produced by ``parse_node`` above; the ``alpha``
    read here was relayed by that callback.
    """
    alpha = float(source.get_attr("alpha"))
    builder = GraphBuilder("thresholded_relu_decomposition")
    x = builder.create_input(0)
    mask = Threshold(x, threshold=alpha)
    output = Mul(x, mask)
    return builder.build_and_reset([output])
