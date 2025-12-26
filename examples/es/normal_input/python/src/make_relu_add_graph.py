#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import numpy as np

from ge.es.graph_builder import GraphBuilder, TensorHolder
from ge.graph import Tensor
from ge.graph.types import DataType, Format
from ge.graph import Graph, DumpFormat
from ge.ge_global import GeApi
from ge.session import Session
from ge.es.all import Add, Relu


def build_input_graph():
    # 1、创建图构建器
    builder = GraphBuilder("MakeMatMulGraph")
    # 2、创建图输入节点
    input_tensor_holder1 = builder.create_input(
        index=0,
        name="input",
        data_type=DataType.DT_FLOAT,
        shape=[2, 3]
    )
    relu1 = Relu(input_tensor_holder1)
    relu2 = Relu(relu1)
    add_tensor_holder = Add(input_tensor_holder1, relu2)
    # 3、设置图输出节点
    builder.set_graph_output(add_tensor_holder, 0)
    # 4、构建图
    return builder.build_and_reset()


def dump_input_graph(graph):
    graph.dump_to_file(format=DumpFormat.kOnnx, suffix="make_relu_add_graph")


graph = build_input_graph()
dump_input_graph(graph)
