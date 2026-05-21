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

"""Python PatternFusionPass sample equivalent to C++ MatmulAddFusionPass matcher-config sample."""

from __future__ import annotations

from ge.es.graph_builder import GraphBuilder
from ge.passes import (
    PassStage,
    PatternFusionPass,
    PatternMatcherConfigBuilder,
    create_pattern,
    create_replacement,
    register_fusion_pass,
)

try:
    from ge.es.math import Add, GEMM, MatMul
except ImportError:
    try:
        from ge.es.all import Add, GEMM, MatMul
    except ImportError:
        Add = None
        GEMM = None
        MatMul = None


def _require_es_apis() -> None:
    pairs = [
        ("MatMul", MatMul),
        ("Add", Add),
        ("GEMM", GEMM),
    ]
    missing = [name for name, obj in pairs if obj is None]
    if missing:
        raise RuntimeError(
            "未找到 ES API: "
            + ", ".join(missing)
            + "。请先 source CANN 环境；如仍缺失，请参考 README 的“ES API 缺失时处理（可选）”生成并加载 es_all 后重新执行。"
        )


@register_fusion_pass(name="PythonMatmulAddFusionPass", stage=PassStage.BEFORE_INFER_SHAPE)
class PythonMatmulAddFusionPass(PatternFusionPass):

    def __init__(self):
        super().__init__(
            PatternMatcherConfigBuilder()
            .enable_const_value_match()
            .enable_ir_attr_match()
            .build()
        )

    def patterns(self):
        print("Define pattern for MatmulAddFusionPass in matcher config sample")
        _require_es_apis()

        graph_builder = GraphBuilder("pattern")
        x, y = graph_builder.create_inputs(2)
        z = graph_builder.create_const_float([0.1, 0.1, 0.1, 0.1], [2, 2])
        add = MatMul(x, y, None, transpose_x1=False, transpose_x2=False) + z

        return [create_pattern(graph_builder.build_and_reset([add]))]

    def replacement(self, match_result):
        print("Define replacement for MatmulAddFusionPass in matcher config sample")
        _require_es_apis()

        replace_graph_builder = GraphBuilder("replacement")
        r_a, r_b = replace_graph_builder.create_inputs(2)
        r_c = replace_graph_builder.create_const_float([0.1, 0.1, 0.1, 0.1], [2, 2])
        alpha_const = replace_graph_builder.create_scalar_float(1.0)
        beta_const = replace_graph_builder.create_scalar_float(1.0)
        gemm = GEMM(r_a, r_b, r_c, alpha_const, beta_const)
        return create_replacement(replace_graph_builder.build_and_reset([gemm]))


if __name__ == "__main__":
    print("PythonMatmulAddFusionPass 已注册。")
    print("请通过 ASCEND_GE_PY_PASS_PATH 指向本文件，例如：")
    print("  export ASCEND_GE_PY_PASS_PATH=$PWD/src/python_fuse_matmul_add_pass.py")
