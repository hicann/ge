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

"""parse_operator 回调的可运行示例。

MyElu 与 GE 已有算子 Elu 一一对应，接入只需转写属性，因此只绑定 parse_operator
一个回调，无需分解。

parse_operator 收到 (source, target) 两个算子：source 是框架把 ONNX 节点先行
物化成的算子，全部节点属性被打包成一个 JSON 串存于 "attribute" 键（与 C++
by-operator 解析器读取 GetAttr("attribute") 的惯例一致）；回调解析出 alpha
后转写给 target。Elu 是静态 IR 算子，无需注册端口。

parse_node 与 parse_operator 同属参数解析阶段、二选一（同时绑定时只有
parse_operator 生效）：parse_node 按名直取原始 ONNX 节点属性，适合新写插件；
parse_operator 适合整体搬运属性（无需预知属性名）或迁移 C++
ParseParamsByOperatorFn 插件。节点携带 tensor/子图等复杂属性时只能用
parse_operator（node.attrs 仅支持 int/float/string/同构列表）。
"""

import json

from ge.onnx_plugin import onnx_plugin

my_elu = onnx_plugin(
    source="MyElu",
    domain="example.domain",
    opsets=(1,),
    target="Elu",
)


@my_elu.parse_operator
def parse_my_elu(source, target) -> None:
    """从 source 算子的 JSON 属性串解析 alpha，转写给 Elu 目标算子。"""
    attrs = json.loads(source.get_attr("attribute"))
    alpha = 1.0
    for attr in attrs.get("attribute", []):
        if attr.get("name") == "alpha":
            alpha = float(attr.get("f", 1.0))
    target.set_attr("alpha", alpha)
    print(f"[Plugin] MyElu is mapped to Elu by parse_operator with alpha={alpha}")
