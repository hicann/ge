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

"""Pytest coverage for Python custom op prototype contracts."""

import gc
import importlib
import os
from typing import List, Optional, Tuple

import pytest

bootstrap = importlib.import_module("ge.custom_op.bootstrap")
proto = importlib.import_module("ge.custom_op.proto")
runtime = importlib.import_module("ge.runtime")
runtime_native = importlib.import_module("ge.runtime._native")
graph = importlib.import_module("ge.graph")

DataType = graph.DataType
Format = graph.Format
StorageShape = runtime_native.StorageShape
Tensor = runtime_native.Tensor
TensorDesc = runtime.TensorDesc

_SKIP_NATIVE_EXCEPTION_WITH_INCOMPLETE_ASAN_PRELOAD = pytest.mark.skipif(
    "ASAN_OPTIONS" in os.environ
    and "libstdc++" not in os.environ.get("LD_PRELOAD", ""),
    reason="ASan cannot intercept C++ exceptions without preloaded libstdc++",
)


@pytest.fixture(autouse=True)
def clear_python_custom_op_protos():
    proto.clear_registered_op_protos()
    yield
    proto.clear_registered_op_protos()


def _shape_dims(storage_shape):
    return (
        storage_shape.origin_shape.dims,
        storage_shape.storage_shape.dims,
    )


def _assert_register_error(
    op_type, fn, *, error_type=TypeError, match=None, mutates_args=()
):
    with pytest.raises(error_type, match=match) as error:
        proto.register_op(op_type=op_type, mutates_args=mutates_args)(fn)

    assert f"register_op op_type '{op_type}' failed" in str(error.value)


def _make_all_types_infer_meta():
    def infer_meta(
        x: TensorDesc,
        optional_y: Optional[TensorDesc],
        dynamic_z: List[TensorDesc],
        *,
        tensor_attr: Tensor,
        int_attr: int = 1,
        float_attr: float = 1.5,
        bool_attr: bool = False,
        str_attr: str = "value",
        data_type_attr: DataType = DataType.DT_FLOAT16,
        list_int_attr: List[int],
        list_float_attr: List[float],
        list_bool_attr: List[bool],
        list_str_attr: List[str],
        list_data_type_attr: List[DataType],
        list_list_int_attr: List[List[int]],
    ) -> Tuple[TensorDesc, List[TensorDesc]]:
        del optional_y, dynamic_z, tensor_attr
        return x, []

    infer_meta.__kwdefaults__.update(
        {
            "list_int_attr": [1, 2],
            "list_float_attr": [1.0, 2.0],
            "list_bool_attr": [True, False],
            "list_str_attr": ["x", "y"],
            "list_data_type_attr": [DataType.DT_INT32],
            "list_list_int_attr": [[1, 2], [3]],
        }
    )
    return infer_meta


def test_tensor_desc_is_exported_from_runtime():
    assert "TensorDesc" in runtime.__all__
    assert runtime.TensorDesc is runtime_native.TensorDesc


def test_tensor_desc_owns_shape_and_data_type():
    source = StorageShape([1, 2], [1, 1, 2])
    desc = TensorDesc(source, DataType.DT_FLOAT16)

    assert _shape_dims(desc.shape) == ([1, 2], [1, 1, 2])
    assert desc.data_type == DataType.DT_FLOAT16

    source.origin_shape.set_dim(0, 9)
    source.storage_shape.set_dim(0, 9)
    assert _shape_dims(desc.shape) == ([1, 2], [1, 1, 2])

    desc.shape = [3, -1]
    assert _shape_dims(desc.shape) == ([3, -1], [3, -1])

    replacement = StorageShape([4], [2, 2])
    desc.shape = replacement
    replacement.origin_shape.set_dim(0, 8)
    assert _shape_dims(desc.shape) == ([4], [2, 2])

    desc.shape = None
    desc.data_type = DataType.DT_UNDEFINED
    assert _shape_dims(desc.shape) == ([], [])
    assert desc.data_type == DataType.DT_UNDEFINED


@pytest.mark.parametrize("shape", [(1, 2), "1,2", [True], [1.0]])
@_SKIP_NATIVE_EXCEPTION_WITH_INCOMPLETE_ASAN_PRELOAD
def test_tensor_desc_rejects_invalid_shape(shape):
    with pytest.raises(TypeError, match="StorageShape, list of integers, or None"):
        TensorDesc(shape, DataType.DT_FLOAT)


@_SKIP_NATIVE_EXCEPTION_WITH_INCOMPLETE_ASAN_PRELOAD
def test_tensor_desc_rejects_too_many_dimensions():
    with pytest.raises(ValueError, match="maximum dimension count"):
        TensorDesc([1] * 1024, DataType.DT_FLOAT)


@pytest.mark.parametrize(
    "data_type",
    [0, True, Format.FORMAT_NCHW, "DT_FLOAT"],
)
@_SKIP_NATIVE_EXCEPTION_WITH_INCOMPLETE_ASAN_PRELOAD
def test_tensor_desc_rejects_invalid_data_type(data_type):
    with pytest.raises(TypeError, match="data_type must be a DataType"):
        TensorDesc(None, data_type)


@_SKIP_NATIVE_EXCEPTION_WITH_INCOMPLETE_ASAN_PRELOAD
def test_tensor_desc_rejects_data_type_boundary():
    with pytest.raises(ValueError, match="less than DataType.DT_MAX"):
        TensorDesc(None, DataType.DT_MAX)


@_SKIP_NATIVE_EXCEPTION_WITH_INCOMPLETE_ASAN_PRELOAD
def test_tensor_desc_shape_view_expires_with_owner():
    desc = TensorDesc([1], DataType.DT_FLOAT)
    shape = desc.shape

    del desc
    gc.collect()

    with pytest.raises(RuntimeError, match="handle has expired"):
        _ = shape.origin_shape.dims


def test_register_op_parses_supported_signature_and_defaults():
    infer_meta = _make_all_types_infer_meta()
    infer_meta = proto.register_op(op_type="AllTypesCustom", mutates_args=("x",))(
        infer_meta
    )
    descriptor = infer_meta.__ge_op_proto_descriptor__

    assert [(item.name, item.index, item.kind) for item in descriptor.inputs] == [
        ("x", 0, proto.InputType.REQUIRED),
        ("optional_y", 1, proto.InputType.OPTIONAL),
        ("dynamic_z", 2, proto.InputType.DYNAMIC),
    ]
    assert [(item.name, item.type, item.is_required) for item in descriptor.attrs] == [
        ("tensor_attr", "VT_TENSOR", True),
        ("int_attr", "VT_INT", False),
        ("float_attr", "VT_FLOAT", False),
        ("bool_attr", "VT_BOOL", False),
        ("str_attr", "VT_STRING", False),
        ("data_type_attr", "VT_DATA_TYPE", False),
        ("list_int_attr", "VT_LIST_INT", False),
        ("list_float_attr", "VT_LIST_FLOAT", False),
        ("list_bool_attr", "VT_LIST_BOOL", False),
        ("list_str_attr", "VT_LIST_STRING", False),
        ("list_data_type_attr", "VT_LIST_DATA_TYPE", False),
        ("list_list_int_attr", "VT_LIST_LIST_INT", False),
    ]
    assert [(item.name, item.index, item.kind) for item in descriptor.outputs] == [
        ("x", 0, proto.OutputType.REQUIRED),
        ("output1", 1, proto.OutputType.DYNAMIC),
    ]

    bridge_descriptor = descriptor.to_bridge_dict()
    assert bridge_descriptor["attrs"][0] == {
        "name": "tensor_attr",
        "type": "VT_TENSOR",
        "is_required": True,
        "default": None,
    }
    assert bridge_descriptor["attrs"][6]["default"] == [1, 2]
    assert bridge_descriptor["attrs"][11]["default"] == [[1, 2], [3]]
    assert bridge_descriptor["outputs"] == [
        {"name": "x", "kind": int(proto.OutputType.REQUIRED)},
        {"name": "output1", "kind": int(proto.OutputType.DYNAMIC)},
    ]


def test_register_op_supports_zero_args_and_zero_outputs():
    @proto.register_op(op_type="NoArgCustom")
    def infer_meta() -> None:
        return None

    descriptor = infer_meta.__ge_op_proto_descriptor__
    assert descriptor.inputs == ()
    assert descriptor.attrs == ()
    assert descriptor.outputs == ()


def test_register_op_is_exported_from_custom_op():
    custom_op = importlib.import_module("ge.custom_op")

    @custom_op.register_op(op_type="PublicRegisterCustom")
    def infer_meta() -> None:
        pass

    assert "register_op" in custom_op.__all__
    assert custom_op.register_op is proto.register_op
    assert infer_meta.__ge_op_proto_descriptor__.op_type == "PublicRegisterCustom"


@pytest.mark.parametrize("op_type", [None, "", 1])
def test_register_op_rejects_invalid_op_type(op_type):
    with pytest.raises(TypeError) as error:
        proto.register_op(op_type=op_type)

    assert "register_op op_type must be a non-empty string" in str(error.value)
    assert repr(op_type) in str(error.value)


def test_register_op_rejects_non_function_with_op_type():
    _assert_register_error(
        "NonFunctionCustom",
        object(),
        match="register_op expects a Python function",
    )


def test_direct_dataclass_construction_freezes_mutable_containers():
    default = [[1, 2], [3]]
    op_input = proto.OpInput("x", 0, proto.InputType.REQUIRED)
    op_attr = proto.OpAttr("axes", 0, "VT_LIST_LIST_INT", False, default)
    op_output = proto.OpOutput("output0", 0, proto.OutputType.REQUIRED)
    inputs = [op_input]
    attrs = [op_attr]
    outputs = [op_output]

    descriptor = proto.OpProtoDescriptor(
        descriptor_key="direct_source:infer_meta:DirectCustom",
        op_type="DirectCustom",
        module_name="direct_source",
        func_name="infer_meta",
        inputs=inputs,
        attrs=attrs,
        outputs=outputs,
        infer_func=lambda: None,
    )

    default[0][0] = 99
    default.append([4])
    inputs.clear()
    attrs.clear()
    outputs.clear()

    assert op_attr.default == ((1, 2), (3,))
    assert all(isinstance(item, tuple) for item in (op_attr.default, *op_attr.default))
    assert descriptor.inputs == (op_input,)
    assert descriptor.attrs == (op_attr,)
    assert descriptor.outputs == (op_output,)
    assert all(
        isinstance(items, tuple)
        for items in (descriptor.inputs, descriptor.attrs, descriptor.outputs)
    )


def test_register_op_supports_explicit_mutation_and_unique_fresh_names():
    @proto.register_op(op_type="ExplicitMutateCustom", mutates_args=(("state", 2),))
    def infer_meta(
        output0: TensorDesc, state: TensorDesc
    ) -> Tuple[TensorDesc, List[TensorDesc], TensorDesc]:
        return output0, [], state

    assert [
        (item.name, item.kind) for item in infer_meta.__ge_op_proto_descriptor__.outputs
    ] == [
        ("output0_1", proto.OutputType.REQUIRED),
        ("output1", proto.OutputType.DYNAMIC),
        ("state", proto.OutputType.REQUIRED),
    ]


@pytest.mark.parametrize(
    ("annotation", "default"),
    [
        (int, True),
        (int, DataType.DT_FLOAT),
        (float, 1),
        (bool, 1),
        (str, 1),
        (DataType, 0),
        (DataType, DataType.DT_MAX),
        (List[int], [True]),
        (List[float], [1]),
        (List[bool], [1]),
        (List[str], [1]),
        (List[DataType], [0]),
        (List[DataType], [DataType.DT_MAX]),
        (List[List[int]], [[True]]),
        (List[int], (1, 2)),
        (int, None),
    ],
)
def test_register_op_rejects_default_value_type_mismatch(annotation, default):
    def infer_meta(*, value=default) -> None:
        del value

    infer_meta.__annotations__["value"] = annotation

    _assert_register_error(
        "BadDefaultCustom",
        infer_meta,
        match="default value for attr 'value'",
    )


def test_register_op_rejects_tensor_attr_default():
    def infer_meta(*, value=None) -> None:
        del value

    infer_meta.__annotations__["value"] = Tensor

    _assert_register_error(
        "TensorDefaultCustom",
        infer_meta,
        match="Tensor attr 'value' does not support a default",
    )


def test_register_op_rejects_invalid_parameter_and_return_annotations():
    def missing_input_annotation(x) -> TensorDesc:
        return x

    _assert_register_error(
        "MissingInputAnnotation",
        missing_input_annotation,
        match="input 'x' must have a type annotation",
    )

    def primitive_input(x: int) -> TensorDesc:
        return x

    _assert_register_error(
        "PrimitiveInput",
        primitive_input,
        match="unsupported input annotation for 'x'",
    )

    def default_input(x: TensorDesc = None) -> TensorDesc:
        return x

    _assert_register_error(
        "DefaultInput",
        default_input,
        match="input 'x' must not have a default value",
    )

    def keyword_tensor(*, x: TensorDesc) -> TensorDesc:
        return x

    _assert_register_error(
        "KeywordTensor",
        keyword_tensor,
        match="unsupported attr annotation for 'x'",
    )

    def variadic_input(*args: TensorDesc) -> TensorDesc:
        return args[0]

    _assert_register_error(
        "VariadicInput",
        variadic_input,
        match="does not support variadic parameter 'args'",
    )

    def missing_return(x: TensorDesc):
        return x

    _assert_register_error(
        "MissingReturn",
        missing_return,
        match="return type annotation is required",
    )

    def invalid_return(x: TensorDesc) -> Optional[TensorDesc]:
        return x

    _assert_register_error(
        "InvalidReturn",
        invalid_return,
        match="unsupported return annotation",
    )


@pytest.mark.parametrize(
    "mutates_args",
    [
        "x",
        ("x", ("y", 1)),
        ("missing",),
        ("x", "x"),
        (("x", 2),),
        (("x", True),),
    ],
)
def test_register_op_rejects_invalid_mutates_args(mutates_args):
    def infer_meta(x: TensorDesc, y: TensorDesc) -> Tuple[TensorDesc, TensorDesc]:
        return x, y

    _assert_register_error(
        "BadMutatesCustom",
        infer_meta,
        error_type=(TypeError, ValueError),
        match="mutates_args",
        mutates_args=mutates_args,
    )


def _make_infer_function(module_name, output_annotation=TensorDesc):
    def infer_meta(x: TensorDesc) -> TensorDesc:
        return x

    infer_meta.__module__ = module_name
    infer_meta.__qualname__ = "infer_meta"
    infer_meta.__annotations__["return"] = output_annotation
    return infer_meta


def _make_nan_default_infer_function(module_name):
    default = float("nan")

    def infer_meta(*, value: float = default) -> None:
        del value

    infer_meta.__module__ = module_name
    infer_meta.__qualname__ = "infer_meta"
    return infer_meta


def test_register_op_is_idempotent_for_same_source_and_content():
    first = _make_infer_function("same_source")
    second = _make_infer_function("same_source")

    proto.register_op(op_type="IdempotentCustom")(first)
    proto.register_op(op_type="IdempotentCustom")(second)

    descriptors = proto.get_registered_op_protos()
    assert len(descriptors) == 1
    assert descriptors[0].infer_func is first
    assert first.__ge_op_proto_descriptor__.descriptor_key == (
        second.__ge_op_proto_descriptor__.descriptor_key
    )


def test_register_op_is_idempotent_for_distinct_nan_defaults():
    first = _make_nan_default_infer_function("nan_source")
    second = _make_nan_default_infer_function("nan_source")

    assert first is not second
    proto.register_op(op_type="NanDefaultCustom")(first)
    first_descriptor = first.__ge_op_proto_descriptor__
    proto.register_op(op_type="NanDefaultCustom")(second)

    descriptors = proto.get_registered_op_protos()
    assert len(descriptors) == 1
    assert descriptors[0] is first_descriptor
    assert descriptors[0].infer_func is first
    assert second.__ge_op_proto_descriptor__ is first_descriptor


def test_register_op_reports_invalid_output_index():
    def infer_meta(x: TensorDesc) -> Tuple[TensorDesc, int]:
        return x, 1

    _assert_register_error(
        "InvalidOutputIndexCustom", infer_meta, match="output index 1"
    )


def test_register_op_rejects_changed_content_and_different_source():
    proto.register_op(op_type="ConflictCustom")(_make_infer_function("first_source"))

    with pytest.raises(ValueError) as changed_error:
        proto.register_op(op_type="ConflictCustom")(
            _make_infer_function("first_source", List[TensorDesc])
        )

    changed_message = str(changed_error.value)
    same_source = (
        "module_name.func_name 'first_source.infer_meta', "
        "descriptor_key 'first_source:infer_meta:ConflictCustom'"
    )
    assert "register_op op_type 'ConflictCustom' failed" in changed_message
    assert "descriptor content changed" in changed_message
    assert f"existing source: {same_source}" in changed_message
    assert f"current source: {same_source}" in changed_message

    with pytest.raises(ValueError) as different_source_error:
        proto.register_op(op_type="ConflictCustom")(
            _make_infer_function("second_source")
        )

    different_source_message = str(different_source_error.value)
    assert "register_op op_type 'ConflictCustom' failed" in different_source_message
    assert "op type 'ConflictCustom' already registered" in different_source_message
    assert (
        "existing source: module_name.func_name 'first_source.infer_meta', "
        "descriptor_key 'first_source:infer_meta:ConflictCustom'"
        in different_source_message
    )
    assert (
        "current source: module_name.func_name 'second_source.infer_meta', "
        "descriptor_key 'second_source:infer_meta:ConflictCustom'"
        in different_source_message
    )


def test_proto_bridge_dicts_are_isolated_and_bootstrap_visible():
    def infer_meta(*, axes: List[int]) -> None:
        del axes

    infer_meta.__kwdefaults__ = {"axes": [1, 2]}
    infer_meta = proto.register_op(op_type="BridgeProtoCustom")(infer_meta)
    first = bootstrap.get_registered_op_protos()
    first[0]["attrs"][0]["default"][0] = 99
    second = bootstrap.get_registered_op_protos()

    assert second[0]["attrs"][0]["default"] == [1, 2]
    assert second[0]["func_name"].endswith("infer_meta")
    assert second[0]["descriptor_key"] == (
        infer_meta.__ge_op_proto_descriptor__.descriptor_key
    )
