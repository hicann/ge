#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
生成包含 Data + Transpose 结构的 ONNX 模型。
模型结构:
  input1(NCHW) -> Transpose(perm=[0,2,3,1]) -> Add -> output(NHWC)
  input2(NCHW) -> Transpose(perm=[0,2,3,1]) /
"""

import os
import onnx
from onnx import TensorProto, helper


def convert():
    input1 = helper.make_tensor_value_info(
        "input1", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    input2 = helper.make_tensor_value_info(
        "input2", TensorProto.FLOAT, [1, 3, 224, 224]
    )
    output = helper.make_tensor_value_info(
        "output", TensorProto.FLOAT, [1, 224, 224, 3]
    )

    transpose1 = helper.make_node(
        "Transpose", ["input1"], ["transposed1"], perm=[0, 2, 3, 1], name="transpose1"
    )
    transpose2 = helper.make_node(
        "Transpose", ["input2"], ["transposed2"], perm=[0, 2, 3, 1], name="transpose2"
    )
    add = helper.make_node(
        "Add", ["transposed1", "transposed2"], ["output"], name="add"
    )

    graph = helper.make_graph(
        [transpose1, transpose2, add],
        "test_graph",
        [input1, input2],
        [output],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.checker.check_model(model)

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")
    onnx.save(model, output_path)


if __name__ == "__main__":
    convert()
