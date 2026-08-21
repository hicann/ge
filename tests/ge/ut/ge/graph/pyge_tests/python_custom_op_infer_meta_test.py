#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Pytest coverage for Python custom op infer_meta callback."""

import importlib
from typing import List, Optional, Tuple

import pytest

proto = importlib.import_module("ge.custom_op.proto")
bridge = importlib.import_module("ge.custom_op._bridge")
runtime = importlib.import_module("ge.runtime")
runtime_native = importlib.import_module("ge.runtime._native")
graph = importlib.import_module("ge.graph")

DataType = graph.DataType
TensorDesc = runtime.TensorDesc
InputType = proto.InputType
OutputType = proto.OutputType


@pytest.fixture(autouse=True)
def clear_registries():
    proto.clear_registered_op_protos()
    yield
    proto.clear_registered_op_protos()


class FakeInferMetaContext:
    """Minimal fake of native InferMetaContext for Python-side testing."""

    def __init__(self, ir_inputs, ir_outputs, ir_attrs=None):
        self._ir_inputs = ir_inputs
        self._ir_outputs = ir_outputs
        self._ir_attrs = ir_attrs or []
        self._input_shapes = []
        self._input_dtypes = []
        self._dynamic_input_counts = []
        self._dynamic_output_counts = []
        self._invalidated = False

    def set_input_shape(self, ir_index, shape, dtype):
        self._input_shapes.insert(ir_index, shape)
        self._input_dtypes.insert(ir_index, dtype)

    def set_dynamic_input(self, ir_index, shapes, dtypes):
        while len(self._dynamic_input_counts) <= ir_index:
            self._dynamic_input_counts.append(0)
        self._dynamic_input_counts[ir_index] = len(shapes)
        for s, d in zip(shapes, dtypes):
            self._input_shapes.append(s)
            self._input_dtypes.append(d)

    def set_dynamic_output_count(self, ir_index, count):
        while len(self._dynamic_output_counts) <= ir_index:
            self._dynamic_output_counts.append(0)
        self._dynamic_output_counts[ir_index] = count

    def get_required_input_tensor(self, ir_index):
        return TensorDesc(
            self._input_shapes[ir_index], DataType(self._input_dtypes[ir_index])
        )

    def get_optional_input_tensor(self, ir_index):
        if ir_index >= len(self._input_shapes) or self._input_shapes[ir_index] is None:
            return None
        return TensorDesc(
            self._input_shapes[ir_index], DataType(self._input_dtypes[ir_index])
        )

    def get_dynamic_input_num(self, ir_index):
        return (
            self._dynamic_input_counts[ir_index]
            if ir_index < len(self._dynamic_input_counts)
            else 0
        )

    def get_dynamic_input_tensor(self, ir_index, relative_index):
        start = sum(self._dynamic_input_counts[:ir_index]) if ir_index > 0 else 0
        return TensorDesc(
            self._input_shapes[start + relative_index],
            DataType(self._input_dtypes[start + relative_index]),
        )

    def get_attrs(self):
        return FakeRuntimeAttrs(self._ir_attrs)

    def get_dynamic_output_num(self, ir_index):
        return (
            self._dynamic_output_counts[ir_index]
            if ir_index < len(self._dynamic_output_counts)
            else 1
        )

    def _invalidate(self):
        self._invalidated = True


class FakeRuntimeAttrs:
    def __init__(self, ir_attrs):
        self._ir_attrs = ir_attrs
        self._values = {}
        for i, attr in enumerate(ir_attrs):
            self._values[i] = attr.get("default", None)

    def get_int(self, index):
        return self._values.get(index, 0)

    def get_float(self, index):
        return self._values.get(index, 0.0)

    def get_bool(self, index):
        return self._values.get(index, False)

    def get_str(self, index):
        return self._values.get(index, "")

    def get_data_type(self, index):
        return self._values.get(index, DataType.DT_FLOAT)

    def get_tensor(self, index):
        return self._values.get(index, None)

    def get_list_int(self, index):
        return self._values.get(index, [])

    def get_list_float(self, index):
        return self._values.get(index, [])

    def get_list_bool(self, index):
        return self._values.get(index, [])

    def get_list_str(self, index):
        return self._values.get(index, [])

    def get_list_data_type(self, index):
        return self._values.get(index, [])

    def get_list_list_int(self, index):
        return self._values.get(index, [])


def _make_ir_meta(inputs, outputs, attrs=None):
    return {
        "inputs": inputs,
        "outputs": outputs,
        "attrs": attrs or [],
    }


def _make_ir_input(name, kind):
    return {"name": name, "kind": int(kind)}


def _make_ir_output(name, kind):
    return {"name": name, "kind": int(kind)}


def _make_ir_attr(name, type_str, default=None):
    return {"name": name, "type": type_str, "default": default}


def _output_dtypes(outputs):
    return [output[2] for output in outputs]


# ---------------------------------------------------------------------------
# Basic infer_meta: single required input, single required output
# ---------------------------------------------------------------------------


def test_call_infer_meta_required_input_output():
    @proto.register_op(op_type="InferMetaBasic")
    def infer_meta(x: TensorDesc) -> TensorDesc:
        return x

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [_make_ir_output("output0", OutputType.REQUIRED)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([2, 3], [2, 3]), int(DataType.DT_FLOAT)
    )

    outputs = bridge.call_infer_meta("InferMetaBasic", ir_meta, ctx)
    assert _output_dtypes(outputs) == [int(DataType.DT_FLOAT)]
    assert outputs[0][1] == [2, 3]


def test_call_infer_meta_with_attrs():
    @proto.register_op(op_type="InferMetaAttrs")
    def infer_meta(x: TensorDesc, *, axis: int, scale: float = 2.0) -> TensorDesc:
        return x

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [_make_ir_output("output0", OutputType.REQUIRED)],
        [
            _make_ir_attr("axis", "VT_INT"),
            _make_ir_attr("scale", "VT_FLOAT", 2.0),
        ],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([4], [4]), int(DataType.DT_FLOAT16)
    )

    outputs = bridge.call_infer_meta("InferMetaAttrs", ir_meta, ctx)
    assert _output_dtypes(outputs) == [int(DataType.DT_FLOAT16)]


def test_call_infer_meta_zero_output():
    @proto.register_op(op_type="InferMetaZeroOutput")
    def infer_meta(x: TensorDesc) -> None:
        return None

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([1], [1]), int(DataType.DT_FLOAT)
    )

    outputs = bridge.call_infer_meta("InferMetaZeroOutput", ir_meta, ctx)
    assert outputs == []


def test_call_infer_meta_multiple_outputs():
    @proto.register_op(op_type="InferMetaMultiOutput")
    def infer_meta(x: TensorDesc) -> Tuple[TensorDesc, TensorDesc]:
        return x, x

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [
            _make_ir_output("output0", OutputType.REQUIRED),
            _make_ir_output("output1", OutputType.REQUIRED),
        ],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([2, 2], [2, 2]), int(DataType.DT_INT32)
    )

    outputs = bridge.call_infer_meta("InferMetaMultiOutput", ir_meta, ctx)
    assert _output_dtypes(outputs) == [int(DataType.DT_INT32), int(DataType.DT_INT32)]
    assert outputs[0][1] == [2, 2]
    assert outputs[1][1] == [2, 2]


def test_call_infer_meta_dynamic_input():
    @proto.register_op(op_type="InferMetaDynInput")
    def infer_meta(x: TensorDesc, ys: List[TensorDesc]) -> TensorDesc:
        return x

    ir_meta = _make_ir_meta(
        [
            _make_ir_input("x", InputType.REQUIRED),
            _make_ir_input("ys", InputType.DYNAMIC),
        ],
        [_make_ir_output("output0", OutputType.REQUIRED)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([3], [3]), int(DataType.DT_FLOAT)
    )
    ctx.set_dynamic_input(
        1,
        [
            runtime_native.StorageShape([1], [1]),
            runtime_native.StorageShape([2], [2]),
        ],
        [int(DataType.DT_FLOAT), int(DataType.DT_FLOAT)],
    )

    outputs = bridge.call_infer_meta("InferMetaDynInput", ir_meta, ctx)
    assert _output_dtypes(outputs) == [int(DataType.DT_FLOAT)]


def test_call_infer_meta_optional_input():
    @proto.register_op(op_type="InferMetaOptInput")
    def infer_meta(x: TensorDesc, y: Optional[TensorDesc]) -> TensorDesc:
        return x

    ir_meta = _make_ir_meta(
        [
            _make_ir_input("x", InputType.REQUIRED),
            _make_ir_input("y", InputType.OPTIONAL),
        ],
        [_make_ir_output("output0", OutputType.REQUIRED)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(0, runtime_native.StorageShape([5], [5]), int(DataType.DT_BOOL))
    ctx.set_input_shape(1, None, None)

    outputs = bridge.call_infer_meta("InferMetaOptInput", ir_meta, ctx)
    assert _output_dtypes(outputs) == [int(DataType.DT_BOOL)]


def test_call_infer_meta_dynamic_output():
    @proto.register_op(op_type="InferMetaDynOutput")
    def infer_meta(x: TensorDesc) -> List[TensorDesc]:
        return [x, x]

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [_make_ir_output("output0", OutputType.DYNAMIC)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([8], [8]), int(DataType.DT_FLOAT16)
    )
    ctx.set_dynamic_output_count(0, 2)

    outputs = bridge.call_infer_meta("InferMetaDynOutput", ir_meta, ctx)
    assert _output_dtypes(outputs) == [
        int(DataType.DT_FLOAT16),
        int(DataType.DT_FLOAT16),
    ]


def test_call_infer_meta_dynamic_output_followed_by_required_output():
    @proto.register_op(op_type="InferMetaDynamicThenRequired")
    def infer_meta(x: TensorDesc) -> Tuple[List[TensorDesc], TensorDesc]:
        return [x, x], x

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [
            _make_ir_output("dynamic", OutputType.DYNAMIC),
            _make_ir_output("required", OutputType.REQUIRED),
        ],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([8], [8]), int(DataType.DT_FLOAT16)
    )
    ctx.set_dynamic_output_count(0, 2)

    outputs = bridge.call_infer_meta("InferMetaDynamicThenRequired", ir_meta, ctx)
    assert _output_dtypes(outputs) == [int(DataType.DT_FLOAT16)] * 3
    assert len(outputs) == 3


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_call_infer_meta_wrong_return_type():
    @proto.register_op(op_type="InferMetaBadReturn")
    def infer_meta(x: TensorDesc) -> TensorDesc:
        return 42

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [_make_ir_output("output0", OutputType.REQUIRED)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([1], [1]), int(DataType.DT_FLOAT)
    )

    with pytest.raises(TypeError, match="must be TensorDesc"):
        bridge.call_infer_meta("InferMetaBadReturn", ir_meta, ctx)


def test_call_infer_meta_return_count_mismatch():
    @proto.register_op(op_type="InferMetaCountMismatch")
    def infer_meta(x: TensorDesc) -> Tuple[TensorDesc, TensorDesc]:
        return x

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [
            _make_ir_output("output0", OutputType.REQUIRED),
            _make_ir_output("output1", OutputType.REQUIRED),
        ],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([1], [1]), int(DataType.DT_FLOAT)
    )

    with pytest.raises(TypeError, match="must return list/tuple for multiple outputs"):
        bridge.call_infer_meta("InferMetaCountMismatch", ir_meta, ctx)


def test_call_infer_meta_dynamic_output_count_mismatch():
    @proto.register_op(op_type="InferMetaDynOutputMismatch")
    def infer_meta(x: TensorDesc) -> List[TensorDesc]:
        return [x, x, x]

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [_make_ir_output("output0", OutputType.DYNAMIC)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([1], [1]), int(DataType.DT_FLOAT)
    )
    ctx.set_dynamic_output_count(0, 2)

    with pytest.raises(TypeError, match="instance count mismatch"):
        bridge.call_infer_meta("InferMetaDynOutputMismatch", ir_meta, ctx)


def test_call_infer_meta_context_invalidated_after_call():
    @proto.register_op(op_type="InferMetaInvalidate")
    def infer_meta(x: TensorDesc) -> TensorDesc:
        return x

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [_make_ir_output("output0", OutputType.REQUIRED)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([1], [1]), int(DataType.DT_FLOAT)
    )

    bridge.call_infer_meta("InferMetaInvalidate", ir_meta, ctx)
    assert ctx._invalidated is True


def test_get_registered_op_proto_by_op_type():
    @proto.register_op(op_type="ProtoByOpType")
    def infer_meta() -> None:
        return None

    found = proto.get_registered_op_proto_by_op_type("ProtoByOpType")
    assert found is not None
    assert found.op_type == "ProtoByOpType"

    missing = proto.get_registered_op_proto_by_op_type("NonExistent")
    assert missing is None


def test_call_infer_meta_changes_dtype():
    @proto.register_op(op_type="InferMetaChangeDtype")
    def infer_meta(x: TensorDesc) -> TensorDesc:
        result = TensorDesc(x.shape, DataType.DT_INT32)
        return result

    ir_meta = _make_ir_meta(
        [_make_ir_input("x", InputType.REQUIRED)],
        [_make_ir_output("output0", OutputType.REQUIRED)],
    )
    ctx = FakeInferMetaContext(ir_meta["inputs"], ir_meta["outputs"])
    ctx.set_input_shape(
        0, runtime_native.StorageShape([2, 3], [2, 3]), int(DataType.DT_FLOAT)
    )

    outputs = bridge.call_infer_meta("InferMetaChangeDtype", ir_meta, ctx)
    assert _output_dtypes(outputs) == [int(DataType.DT_INT32)]
