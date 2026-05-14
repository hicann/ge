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
"""Python PatternFusionPass aligned with C++ FuseMatMulAndAddPass (capture tensor sample)."""
from __future__ import annotations
from typing import Optional
from ge.es.graph_builder import GraphBuilder
from ge.graph.types import DataType
from ge.graph.node import Node
from ge.passes import (
    PassStage,
    PatternFusionPass,
    capture_tensor,
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
_K_MATMUL_CAPTURE_IDX = 0
_K_ADD_CAPTURE_IDX = 1


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
            "Missing ES APIs: "
            + ", ".join(missing)
            + ". Source the CANN environment and ensure es_math / es_nn (or es_all) is installed."
        )


def _as_datatype(value: object) -> Optional[DataType]:
    if isinstance(value, DataType):
        return value
    if isinstance(value, int):
        try:
            return DataType(value)
        except ValueError:
            return None
    return None


def _dtype_from_producer_output(node: Node, _out_port: int) -> Optional[DataType]:
    """Best-effort dtype for the tensor produced at (node, out_port).
    Python ``Node`` does not expose ``GetInputDesc`` / ``GetOutputDesc`` like C++; this
    walks ``Data`` producers (and one level through ``MatMul`` / ``BatchMatMulV2``) and
    reads the ``data_type`` graph attribute on ``Data`` nodes, matching the intent of the
    C++ sample's ``MeetRequirements`` check.
    """
    if node.type == "Data":
        try:
            return _as_datatype(node.get_attr("data_type"))
        except RuntimeError:
            return None
    if node.type in ("MatMul", "BatchMatMulV2"):
        dtypes: list[DataType] = []
        for i in range(node.get_inputs_size()):
            try:
                pred, pred_out = node.get_in_data_nodes_and_port_indexes(i)
            except RuntimeError:
                return None
            dt = _dtype_from_producer_output(pred, pred_out)
            if dt is None:
                return None
            dtypes.append(dt)
        if dtypes and all(d == dtypes[0] for d in dtypes):
            return dtypes[0]
    return None


def _dtype_at_add_input(add_node: Node, input_index: int) -> Optional[DataType]:
    try:
        pred, pred_out = add_node.get_in_data_nodes_and_port_indexes(input_index)
    except RuntimeError:
        return None
    return _dtype_from_producer_output(pred, pred_out)


@register_fusion_pass(
    name="PythonFuseMatMulAndAddCaptureTensorPass",
    stage=PassStage.BEFORE_INFER_SHAPE,
)
class PythonFuseMatMulAndAddCaptureTensorPass(PatternFusionPass):

    def patterns(self):
        print("Define pattern for FuseMatMulAndAddPass in capture tensor sample")
        _require_es_apis()
        pattern_builder0 = GraphBuilder("pattern0")
        a0, b0, c0 = pattern_builder0.create_inputs(3)
        matmul0 = MatMul(a0, b0)
        add0 = Add(matmul0, c0)
        pat0 = create_pattern(pattern_builder0.build_and_reset([add0]))
        pat0.capture_tensor(capture_tensor(matmul0)).capture_tensor(capture_tensor(add0))
        pattern_builder1 = GraphBuilder("pattern1")
        a1, b1, c1 = pattern_builder1.create_inputs(3)
        matmul1 = BatchMatMulV2(a1, b1)
        add1 = Add(matmul1, c1)
        pat1 = create_pattern(pattern_builder1.build_and_reset([add1]))
        pat1.capture_tensor(capture_tensor(matmul1)).capture_tensor(capture_tensor(add1))
        return [pat0, pat1]

    def meet_requirements(self, match_result):
        print("Define MeetRequirements for FuseMatMulAndAddPass in capture tensor sample")
        add_io = match_result.get_captured_tensor(_K_ADD_CAPTURE_IDX)
        add_node = add_io.node
        for idx in (0, 1):
            dt = _dtype_at_add_input(add_node, idx)
            if dt is None:
                continue
            if dt != DataType.DT_FLOAT:
                print("Only support Add inputs are fp32")
                return False
        return True

    def replacement(self, match_result):
        print("Define replacement for FuseMatMulAndAddPass in capture tensor sample")
        _require_es_apis()
        matmul_io = match_result.get_captured_tensor(_K_MATMUL_CAPTURE_IDX)
        mnode = matmul_io.node
        transpose_a = False
        transpose_b = False
        try:
            transpose_a = bool(mnode.get_attr("transpose_x1"))
        except RuntimeError:
            pass
        try:
            transpose_b = bool(mnode.get_attr("transpose_x2"))
        except RuntimeError:
            pass
        replace_builder = GraphBuilder("replacement")
        r_a, r_b, r_c = replace_builder.create_inputs(3)
        alpha_const = replace_builder.create_scalar_float(1.0)
        beta_const = replace_builder.create_scalar_float(1.0)
        gemm = GEMM(
            r_a,
            r_b,
            r_c,
            alpha_const,
            beta_const,
            transpose_a=transpose_a,
            transpose_b=transpose_b,
        )
        return create_replacement(replace_builder.build_and_reset([gemm]))


if __name__ == "__main__":
    print("PythonFuseMatMulAndAddCaptureTensorPass 已注册。")
    print("请通过 ASCEND_GE_PY_PASS_PATH 指向本文件，例如：")
    print("  export ASCEND_GE_PY_PASS_PATH=$(pwd)/src/python_fuse_matmul_add_capture_tensor_pass.py")
