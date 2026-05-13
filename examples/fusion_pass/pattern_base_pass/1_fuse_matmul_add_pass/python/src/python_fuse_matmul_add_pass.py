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

"""Python PatternFusionPass equivalent to C++ FuseMatMulAndAddPass (MatMul+Add / BatchMatMulV2+Add -> GEMM)."""

from __future__ import annotations

from ge.es.graph_builder import GraphBuilder
from ge.passes import (
    PassStage,
    PatternFusionPass,
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
        MatMul = None
        Add = None
        GEMM = None

try:
    from ge.es.nn import BatchMatMulV2
except ImportError:
    try:
        from ge.es.all import BatchMatMulV2
    except ImportError:
        BatchMatMulV2 = None


def _require_es_apis() -> None:
    pairs = [
        ("MatMul", MatMul),
        ("Add", Add),
        ("GEMM", GEMM),
        ("BatchMatMulV2", BatchMatMulV2),
    ]
    missing = [name for name, obj in pairs if obj is None]
    if missing:
        raise RuntimeError(
            "未找到 ES API: "
            + ", ".join(missing)
            + "。请先 source CANN 环境，并确认 run 包中的 es_math / es_nn（或 es_all）已正确安装。"
        )


@register_fusion_pass(name="PythonFuseMatMulAndAddPass", stage=PassStage.BEFORE_INFER_SHAPE)
class PythonFuseMatMulAndAddPass(PatternFusionPass):

    def patterns(self):
        print("Define pattern for FuseMatMulAndAddPass")
        _require_es_apis()

        pattern_builder0 = GraphBuilder("pattern0")
        a0, b0, c0 = pattern_builder0.create_inputs(3)
        add0 = MatMul(a0, b0) + c0

        pat0 = create_pattern(pattern_builder0.build_and_reset([add0]))

        pattern_builder1 = GraphBuilder("pattern1")
        a1, b1, c1 = pattern_builder1.create_inputs(3)
        add1 = BatchMatMulV2(a1, b1) + c1
        pat1 = create_pattern(pattern_builder1.build_and_reset([add1]))

        return [pat0, pat1]

    def meet_requirements(self, match_result):
        return True

    def replacement(self, match_result):
        print("Define replacement for FuseMatMulAndAddPass")
        _require_es_apis()
        replace_builder = GraphBuilder("replacement")
        r_a, r_b, r_c = replace_builder.create_inputs(3)
        alpha_const = replace_builder.create_scalar_float(1.0)
        beta_const = replace_builder.create_scalar_float(1.0)
        gemm = GEMM(r_a, r_b, r_c, alpha_const, beta_const)
        return create_replacement(replace_builder.build_and_reset([gemm]))


if __name__ == "__main__":
    print("PythonFuseMatMulAndAddPass 已注册。")
    print("请通过 ASCEND_GE_PY_PASS_PATH 指向本文件，例如：")
    print("  export ASCEND_GE_PY_PASS_PATH=$(pwd)/src/python_fuse_matmul_add_pass.py")
