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

"""Pytest coverage for Python custom op API and bridge descriptor protocol."""

import contextvars
import importlib
import textwrap
from abc import ABC, abstractmethod
from pathlib import Path

import pytest

try:
    bootstrap = importlib.import_module("ge.custom_op.bootstrap")
    bridge = importlib.import_module("ge.custom_op._bridge")
    custom_op = importlib.import_module("ge.custom_op")
    ir_types = importlib.import_module("ge.custom_op._ir_types")
    proto = importlib.import_module("ge.custom_op.proto")
    runtime = importlib.import_module("ge.runtime")
    from ge.runtime import Tensor
except ImportError as exc:
    pytest.skip(f"无法导入 Python custom op 相关模块: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def clear_python_custom_op_runtime(monkeypatch):
    monkeypatch.delenv(bootstrap.ENV_PY_CUSTOM_OP_PATH, raising=False)
    custom_op.clear_registered_op_impls()
    proto.clear_registered_op_protos()
    bridge.clear_op_impl_holders()
    bridge.clear_loaded_op_impl_modules()
    yield
    bridge.clear_op_impl_holders()
    bridge.clear_loaded_op_impl_modules()
    custom_op.clear_registered_op_impls()
    proto.clear_registered_op_protos()


def _write_custom_op_module(
    dir_path: Path, module_name: str, class_name: str, op_type: str
) -> Path:
    file_path = dir_path / f"{module_name}.py"
    file_path.write_text(
        textwrap.dedent(f"""
        from ge.custom_op import register_op_impl

        @register_op_impl(op_type="{op_type}")
        class {class_name}:
            def execute(self) -> None:
                pass
    """).strip()
        + "\n",
        encoding="utf-8",
    )
    return file_path


class _FakeEagerContext:
    def __init__(self, inputs=None):
        self.inputs = list(inputs or [])
        self.invalidated = False
        self.stream = "fake_stream"
        self.input_calls = []
        self.runtime_attrs = _FakeRuntimeAttrs()
        self.attrs_requested = False

    def get_required_input_tensor(self, ir_index):
        self.input_calls.append(("required", ir_index))
        return ("required", ir_index)

    def get_optional_input_tensor(self, ir_index):
        self.input_calls.append(("optional", ir_index))
        return ("optional", ir_index)

    def get_dynamic_input_num(self, ir_index):
        self.input_calls.append(("dynamic_num", ir_index))
        return 2

    def get_dynamic_input_tensor(self, ir_index, relative_index):
        self.input_calls.append(("dynamic", ir_index, relative_index))
        return ("dynamic", ir_index, relative_index)

    def get_attrs(self):
        self.attrs_requested = True
        return self.runtime_attrs

    def get_input_num(self):
        return len(self.inputs)

    def get_input_tensor(self, index):
        return self.inputs[index]

    def get_stream(self):
        return self.stream

    def _invalidate(self):
        self.invalidated = True


class _FakeRuntimeAttrs:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("get_"):
            raise AttributeError(name)

        def getter(index):
            self.calls.append((name, index))
            return (name, index)

        return getter


def _create_holder(instance_id: str, op_type: str) -> None:
    descriptors = {
        item["op_type"]: item for item in bridge.load_and_get_op_impl_descriptors()
    }
    assert bridge.create_op_impl_holder(
        instance_id, descriptors[op_type]["descriptor_key"]
    )


def _get_descriptor_key(op_type: str) -> str:
    descriptors = {
        item["op_type"]: item for item in bridge.load_and_get_op_impl_descriptors()
    }
    return descriptors[op_type]["descriptor_key"]


def test_bridge_validate_op_impl_descriptor_rejects_unknown_descriptor():
    with pytest.raises(KeyError, match="descriptor_key not found"):
        bridge.validate_op_impl_descriptor("missing-descriptor", None)


def test_bridge_descriptor_snapshot_contains_proto_and_impl():
    @proto.register_op(op_type="SnapshotCustom")
    def infer_meta(x: runtime.TensorDesc, *, alpha: float = 1.0) -> runtime.TensorDesc:
        raise AssertionError("prototype registration must not execute infer_meta")

    @custom_op.register_op_impl(op_type="SnapshotCustom")
    class SnapshotCustom:
        def execute(self, x: Tensor, *, alpha: float) -> None:
            pass

    snapshot = bridge.load_and_get_op_descriptors()

    assert [item["op_type"] for item in snapshot["protos"]] == ["SnapshotCustom"]
    assert [item["op_type"] for item in snapshot["impls"]] == ["SnapshotCustom"]


def test_bridge_descriptor_snapshot_supports_proto_only():
    @proto.register_op(op_type="ProtoOnlyCustom")
    def infer_meta(x: runtime.TensorDesc) -> runtime.TensorDesc:
        raise AssertionError("prototype registration must not execute infer_meta")

    snapshot = bridge.load_and_get_op_descriptors()

    assert [item["op_type"] for item in snapshot["protos"]] == ["ProtoOnlyCustom"]
    assert snapshot["impls"] == []


def test_register_op_impl_exports_descriptor_dict():
    @custom_op.register_op_impl(op_type="AddCustom")
    class AddCustom:
        def execute(self) -> None:
            pass

    descriptors = custom_op.get_registered_op_impl_dicts()

    assert len(descriptors) == 1
    assert descriptors[0]["op_type"] == "AddCustom"
    assert descriptors[0]["class_name"] == "AddCustom"
    assert descriptors[0]["interfaces"] == ["eager_execute"]
    assert AddCustom.__ge_op_impl_descriptor__.to_bridge_dict() == descriptors[0]


def test_schema_bound_execute_class_definition_is_supported():
    @custom_op.register_op_impl(op_type="AutoExecuteCustom")
    class AutoExecuteCustom:
        def __init__(self):
            self.seen_args = None

        def execute(self, x, optional_y, dynamic_z, *, alpha) -> None:
            self.seen_args = (x, optional_y, dynamic_z, alpha)

    instance = AutoExecuteCustom()
    instance.execute("x", None, ["z0", "z1"], alpha=1.0)

    assert instance.seen_args == ("x", None, ["z0", "z1"], 1.0)


def test_get_execute_ctx_is_unavailable_outside_schema_execute():
    with pytest.raises(
        RuntimeError,
        match="only available inside schema-bound execute",
    ):
        custom_op.get_execute_ctx()


def test_bridge_validates_descriptor_signature_without_constructing_instance():
    constructed = []

    @custom_op.register_op_impl(op_type="RegistrationValidatedCustom")
    class RegistrationValidatedCustom:
        def __init__(self):
            constructed.append(True)

        def execute(self, x: Tensor, *, alpha: float) -> None:
            pass

    assert (
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("RegistrationValidatedCustom"),
            {
                "op_type": "RegistrationValidatedCustom",
                "inputs": [{"name": "x", "kind": ir_types.InputType.REQUIRED}],
                "attrs": [{"name": "alpha", "type": ir_types.AttrType.FLOAT}],
                "outputs": [],
            },
        )
        is True
    )
    assert constructed == []


def test_bridge_rejects_execute_without_return_annotation():
    @custom_op.register_op_impl(op_type="MissingExecuteReturnAnnotationCustom")
    class MissingExecuteReturnAnnotationCustom:
        def execute(self, x):
            pass

    with pytest.raises(
        TypeError,
        match=r"invalid execute signature.*missing return annotation",
    ):
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("MissingExecuteReturnAnnotationCustom"),
            {
                "op_type": "MissingExecuteReturnAnnotationCustom",
                "inputs": [{"name": "x", "kind": ir_types.InputType.REQUIRED}],
                "attrs": [],
                "outputs": [],
            },
        )


def test_bridge_rejects_execute_without_canonical_ir():
    @custom_op.register_op_impl(op_type="RegistrationWithoutIrCustom")
    class RegistrationWithoutIrCustom:
        def execute(self) -> None:
            pass

    with pytest.raises(RuntimeError, match="canonical IR not found"):
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("RegistrationWithoutIrCustom"), None
        )


def test_bridge_requires_canonical_ir_to_validate_schema_execute():
    @custom_op.register_op_impl(op_type="MissingRegistrationSchemaCustom")
    class MissingRegistrationSchemaCustom:
        def execute(self, x) -> None:
            pass

    with pytest.raises(
        RuntimeError,
        match="canonical IR not found for schema-bound execute",
    ):
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("MissingRegistrationSchemaCustom"), None
        )


def test_register_op_impl_rejects_duplicate_op_type():
    @custom_op.register_op_impl(op_type="AddCustom")
    class AddCustom:
        def execute(self) -> None:
            pass

    with pytest.raises(ValueError, match="op impl type already exists"):

        @custom_op.register_op_impl(op_type="AddCustom")
        class AddCustomAgain:
            def execute(self) -> None:
                pass


def test_register_op_impl_rejects_invalid_op_type():
    with pytest.raises(TypeError, match="non-empty string"):
        custom_op.register_op_impl(op_type="")

    with pytest.raises(TypeError, match="non-empty string"):
        custom_op.register_op_impl(op_type=None)


def test_register_op_impl_supports_plain_class_with_execute():
    @custom_op.register_op_impl(op_type="PlainCustom")
    class PlainCustom:
        def execute(self) -> None:
            self.called = True

    descriptors = custom_op.get_registered_op_impl_dicts()

    assert descriptors[0]["op_type"] == "PlainCustom"
    assert descriptors[0]["interfaces"] == ["eager_execute"]
    instance = PlainCustom()
    instance.execute()
    assert instance.called is True


def test_register_op_impl_rejects_class_without_supported_method():
    with pytest.raises(
        TypeError,
        match=r"BaseOnlyCustom' must implement at least one supported method: execute, compile, declare_launch_args",
    ):

        @custom_op.register_op_impl(op_type="BaseOnlyCustom")
        class BaseOnlyCustom:
            pass


def test_register_op_impl_rejects_non_class():
    with pytest.raises(TypeError, match="register_op_impl expects a class"):
        custom_op.register_op_impl(op_type="NonClassCustom")(lambda: None)


def test_register_op_impl_rejects_abstract_class():
    with pytest.raises(TypeError, match="register_op_impl expects a concrete class"):

        class AbstractCustom(ABC):
            @abstractmethod
            def execute(self) -> None:
                pass

        custom_op.register_op_impl(op_type="AbstractCustom")(AbstractCustom)


def test_register_op_impl_keeps_inherited_execute_method():
    class CustomOpBase:
        def execute(self) -> None:
            pass

    @custom_op.register_op_impl(op_type="InheritedExecuteCustom")
    class InheritedExecuteCustom(CustomOpBase):
        pass

    assert custom_op.get_registered_op_impl_dicts()[0]["interfaces"] == [
        "eager_execute"
    ]


def test_bridge_custom_op_holder_and_execute():
    @custom_op.register_op_impl(op_type="AddCustom")
    class AddCustom:
        def __init__(self):
            self.called_ctx = None

        def execute(self, x, *, alpha) -> None:
            self.called_ctx = (x, alpha)

    descriptors = bridge.load_and_get_op_impl_descriptors()
    assert len(descriptors) == 1

    instance_id = "AddCustom#1"
    descriptor_key = descriptors[0]["descriptor_key"]
    ctx = _FakeEagerContext()
    assert bridge.create_op_impl_holder(instance_id, descriptor_key) is True
    ir_meta = {
        "op_type": "AddCustom",
        "inputs": [{"name": "x", "kind": 0}],
        "attrs": [{"name": "alpha", "type": "VT_FLOAT"}],
        "outputs": [{"name": "y", "kind": 0}],
    }
    assert bridge.call_execute(instance_id, ir_meta, ctx) is None
    assert ctx.invalidated is True
    assert bridge.destroy_op_impl_holder(instance_id) is True


def test_bridge_call_execute_rejects_return_value():
    @custom_op.register_op_impl(op_type="ReturnCustom")
    class ReturnCustom:
        def execute(self) -> None:
            return True

    descriptor_key = bridge.load_and_get_op_impl_descriptors()[0]["descriptor_key"]
    instance_id = "ReturnCustom#1"
    ctx = _FakeEagerContext()
    assert bridge.create_op_impl_holder(instance_id, descriptor_key) is True
    with pytest.raises(TypeError, match="execute must return None"):
        bridge.call_execute(
            instance_id,
            {"op_type": "ReturnCustom", "inputs": [], "attrs": [], "outputs": []},
            ctx,
        )
    assert ctx.invalidated is True
    assert bridge.destroy_op_impl_holder(instance_id) is True


@pytest.mark.parametrize("method_kind", ["staticmethod", "classmethod"])
def test_bridge_call_execute_binds_schema_inputs_and_attrs(method_kind):
    called = []

    @custom_op.register_op_impl(op_type="SchemaBoundCustom")
    class SchemaBoundCustom:
        if method_kind == "staticmethod":

            @staticmethod
            def execute(x, optional_y, dynamic_z, *, alpha, axes) -> None:
                called.append((x, optional_y, dynamic_z, alpha, axes))
                return None
        else:

            @classmethod
            def execute(cls, x, optional_y, dynamic_z, *, alpha, axes) -> None:
                called.append((x, optional_y, dynamic_z, alpha, axes))
                return None

    descriptor_key = bridge.load_and_get_op_impl_descriptors()[0]["descriptor_key"]
    instance_id = "SchemaBoundCustom#1"
    ctx = _FakeEagerContext()
    ir_meta = {
        "op_type": "SchemaBoundCustom",
        "inputs": [
            {"name": "x", "kind": 0},
            {"name": "optional_y", "kind": 1},
            {"name": "dynamic_z", "kind": 2},
        ],
        "attrs": [
            {"name": "alpha", "type": "VT_FLOAT"},
            {"name": "axes", "type": "VT_LIST_INT"},
        ],
        "outputs": [],
    }

    assert bridge.validate_op_impl_descriptor(descriptor_key, ir_meta) is True
    assert bridge.create_op_impl_holder(instance_id, descriptor_key) is True
    assert bridge.call_execute(instance_id, ir_meta, ctx) is None
    assert called == [
        (
            ("required", 0),
            ("optional", 1),
            [("dynamic", 2, 0), ("dynamic", 2, 1)],
            ("get_float", 0),
            ("get_list_int", 1),
        )
    ]
    assert ctx.input_calls == [
        ("required", 0),
        ("optional", 1),
        ("dynamic_num", 2),
        ("dynamic", 2, 0),
        ("dynamic", 2, 1),
    ]
    assert ctx.runtime_attrs.calls == [
        ("get_float", 0),
        ("get_list_int", 1),
    ]
    assert ctx.attrs_requested is True
    assert ctx.invalidated is True


def test_bridge_validate_op_impl_descriptor_rejects_mismatched_attr_name():
    @custom_op.register_op_impl(op_type="WrongExecuteAttrNameCustom")
    class WrongExecuteAttrNameCustom:
        def execute(self, x, *, beta) -> None:
            pass

    with pytest.raises(
        TypeError,
        match=r"invalid execute signature.*expected attr name alpha.*actual attr name beta",
    ):
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("WrongExecuteAttrNameCustom"),
            {
                "op_type": "WrongExecuteAttrNameCustom",
                "inputs": [{"name": "x", "kind": ir_types.InputType.REQUIRED}],
                "attrs": [{"name": "alpha", "type": ir_types.AttrType.FLOAT}],
                "outputs": [],
            },
        )


def test_bridge_validate_op_impl_descriptor_rejects_mismatched_input_annotation():
    @custom_op.register_op_impl(op_type="WrongExecuteAnnotationCustom")
    class WrongExecuteAnnotationCustom:
        def execute(self, x: list[Tensor], *, alpha: float) -> None:
            pass

    with pytest.raises(
        TypeError,
        match=r"invalid execute signature.*input parameter at index 0 annotation",
    ):
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("WrongExecuteAnnotationCustom"),
            {
                "op_type": "WrongExecuteAnnotationCustom",
                "inputs": [{"name": "x", "kind": ir_types.InputType.REQUIRED}],
                "attrs": [{"name": "alpha", "type": ir_types.AttrType.FLOAT}],
                "outputs": [],
            },
        )


def test_bridge_validate_op_impl_descriptor_rejects_execute_return_annotation():
    @custom_op.register_op_impl(op_type="WrongExecuteReturnAnnotationCustom")
    class WrongExecuteReturnAnnotationCustom:
        def execute(self, x) -> int:
            return 0

    with pytest.raises(
        TypeError,
        match=r"invalid execute signature.*None return annotation",
    ):
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("WrongExecuteReturnAnnotationCustom"),
            {
                "op_type": "WrongExecuteReturnAnnotationCustom",
                "inputs": [{"name": "x", "kind": ir_types.InputType.REQUIRED}],
                "attrs": [],
                "outputs": [],
            },
        )


def test_bridge_call_execute_supports_ir_outputs():
    called = []

    @custom_op.register_op_impl(op_type="CompatibleExecuteSignatureCustom")
    class CompatibleExecuteSignatureCustom:
        def execute(self, x: Tensor, *, alpha: float) -> None:
            called.append((x, alpha))
            return None

    instance_id = "CompatibleExecuteSignatureCustom#1"
    ctx = _FakeEagerContext()
    _create_holder(instance_id, "CompatibleExecuteSignatureCustom")

    assert (
        bridge.call_execute(
            instance_id,
            {
                "op_type": "CompatibleExecuteSignatureCustom",
                "inputs": [{"name": "x", "kind": ir_types.InputType.REQUIRED}],
                "attrs": [{"name": "alpha", "type": ir_types.AttrType.FLOAT}],
                "outputs": [{"name": "y", "kind": ir_types.OutputType.REQUIRED}],
            },
            ctx,
        )
        is None
    )

    assert called == [(("required", 0), ("get_float", 0))]
    assert ctx.invalidated is True


def test_bridge_call_execute_does_not_validate_signature_at_runtime(monkeypatch):
    execute_calls = []

    @custom_op.register_op_impl(op_type="CachedExecuteSignatureCustom")
    class CachedExecuteSignatureCustom:
        def execute(self, x: Tensor) -> None:
            execute_calls.append(x)

    descriptor_key = _get_descriptor_key("CachedExecuteSignatureCustom")
    instance_id = "CachedExecuteSignatureCustom#1"
    ir_meta = {
        "op_type": "CachedExecuteSignatureCustom",
        "inputs": [{"name": "x", "kind": ir_types.InputType.REQUIRED}],
        "attrs": [],
        "outputs": [],
    }

    assert bridge.validate_op_impl_descriptor(descriptor_key, ir_meta) is True
    assert bridge.create_op_impl_holder(instance_id, descriptor_key) is True

    def fail_on_runtime_validation(*args, **kwargs):
        _ = (args, kwargs)
        pytest.fail("schema execute must not validate its signature at runtime")

    monkeypatch.setattr(bridge, "_validate_args_signature", fail_on_runtime_validation)
    bridge.call_execute(instance_id, ir_meta, _FakeEagerContext())

    assert execute_calls == [("required", 0)]


def test_schema_bound_execute_can_get_current_context():
    called = []

    @custom_op.register_op_impl(op_type="SchemaContextCustom")
    class SchemaContextCustom:
        def execute(self, x) -> None:
            current_ctx = custom_op.get_execute_ctx()
            called.append((x, current_ctx, current_ctx.get_stream()))

    instance_id = "SchemaContextCustom#1"
    ctx = _FakeEagerContext()
    _create_holder(instance_id, "SchemaContextCustom")

    bridge.call_execute(
        instance_id,
        {
            "op_type": "SchemaContextCustom",
            "inputs": [{"name": "x", "kind": 0}],
            "attrs": [],
            "outputs": [],
        },
        ctx,
    )

    assert called == [(("required", 0), ctx, "fake_stream")]
    assert ctx.invalidated is True
    with pytest.raises(
        RuntimeError,
        match="only available inside schema-bound execute",
    ):
        custom_op.get_execute_ctx()


def test_schema_execute_context_is_deactivated_after_exception():
    copied_contexts = []

    @custom_op.register_op_impl(op_type="FailingSchemaContextCustom")
    class FailingSchemaContextCustom:
        def execute(self, x) -> None:
            assert custom_op.get_execute_ctx() is not None
            copied_contexts.append(contextvars.copy_context())
            raise ValueError("schema execute failed")

    instance_id = "FailingSchemaContextCustom#1"
    ctx = _FakeEagerContext()
    _create_holder(instance_id, "FailingSchemaContextCustom")

    with pytest.raises(ValueError, match="schema execute failed"):
        bridge.call_execute(
            instance_id,
            {
                "op_type": "FailingSchemaContextCustom",
                "inputs": [{"name": "x", "kind": 0}],
                "attrs": [],
                "outputs": [],
            },
            ctx,
        )

    assert ctx.invalidated is True
    with pytest.raises(
        RuntimeError,
        match="only available inside schema-bound execute",
    ):
        copied_contexts[0].run(custom_op.get_execute_ctx)


def test_schema_execute_context_restores_outer_binding_after_nested_call():
    called = []
    outer_ctx = _FakeEagerContext()
    inner_ctx = _FakeEagerContext()

    @custom_op.register_op_impl(op_type="InnerSchemaContextCustom")
    class InnerSchemaContextCustom:
        def execute(self, x) -> None:
            called.append(("inner", custom_op.get_execute_ctx()))

    @custom_op.register_op_impl(op_type="OuterSchemaContextCustom")
    class OuterSchemaContextCustom:
        def execute(self, x) -> None:
            called.append(("outer_before", custom_op.get_execute_ctx()))
            bridge.call_execute(
                "InnerSchemaContextCustom#1",
                {
                    "op_type": "InnerSchemaContextCustom",
                    "inputs": [{"name": "x", "kind": 0}],
                    "attrs": [],
                    "outputs": [],
                },
                inner_ctx,
            )
            called.append(("outer_after", custom_op.get_execute_ctx()))

    _create_holder("InnerSchemaContextCustom#1", "InnerSchemaContextCustom")
    _create_holder("OuterSchemaContextCustom#1", "OuterSchemaContextCustom")

    bridge.call_execute(
        "OuterSchemaContextCustom#1",
        {
            "op_type": "OuterSchemaContextCustom",
            "inputs": [{"name": "x", "kind": 0}],
            "attrs": [],
            "outputs": [],
        },
        outer_ctx,
    )

    assert called == [
        ("outer_before", outer_ctx),
        ("inner", inner_ctx),
        ("outer_after", outer_ctx),
    ]
    assert inner_ctx.invalidated is True
    assert outer_ctx.invalidated is True


def test_bridge_call_execute_supports_zero_input_and_zero_attr_schema():
    called = []

    @custom_op.register_op_impl(op_type="NoArgCustom")
    class NoArgCustom:
        def execute(self) -> None:
            called.append(True)

    descriptor_key = bridge.load_and_get_op_impl_descriptors()[0]["descriptor_key"]
    instance_id = "NoArgCustom#1"
    ctx = _FakeEagerContext()
    assert bridge.create_op_impl_holder(instance_id, descriptor_key) is True

    assert (
        bridge.call_execute(
            instance_id,
            {"op_type": "NoArgCustom", "inputs": [], "attrs": [], "outputs": []},
            ctx,
        )
        is None
    )
    assert called == [True]
    assert ctx.input_calls == []
    assert ctx.runtime_attrs.calls == []
    assert ctx.attrs_requested is False
    assert ctx.invalidated is True


@pytest.mark.parametrize(
    ("ir_type", "getter_name"),
    [
        (ir_types.AttrType.INT, "get_int"),
        (ir_types.AttrType.FLOAT, "get_float"),
        (ir_types.AttrType.BOOL, "get_bool"),
        (ir_types.AttrType.STRING, "get_str"),
        (ir_types.AttrType.DATA_TYPE, "get_data_type"),
        (ir_types.AttrType.TENSOR, "get_tensor"),
        (ir_types.AttrType.LIST_INT, "get_list_int"),
        (ir_types.AttrType.LIST_FLOAT, "get_list_float"),
        (ir_types.AttrType.LIST_BOOL, "get_list_bool"),
        (ir_types.AttrType.LIST_STRING, "get_list_str"),
        (ir_types.AttrType.LIST_DATA_TYPE, "get_list_data_type"),
        (ir_types.AttrType.LIST_LIST_INT, "get_list_list_int"),
    ],
)
def test_bridge_reads_runtime_attr_by_canonical_type(ir_type, getter_name):
    attrs = _FakeRuntimeAttrs()

    assert bridge._read_runtime_attr(attrs, 3, ir_type) == (getter_name, 3)
    assert attrs.calls == [(getter_name, 3)]


def test_bridge_rejects_schema_bound_execute_without_ir_meta():
    @custom_op.register_op_impl(op_type="MissingSchemaCustom")
    class MissingSchemaCustom:
        def execute(self, inputs) -> None:
            pass

    descriptor_key = bridge.load_and_get_op_impl_descriptors()[0]["descriptor_key"]
    instance_id = "MissingSchemaCustom#1"
    ctx = _FakeEagerContext()
    assert bridge.create_op_impl_holder(instance_id, descriptor_key) is True

    with pytest.raises(
        RuntimeError,
        match="canonical IR not found for schema-bound execute: MissingSchemaCustom",
    ):
        bridge.call_execute(instance_id, None, ctx)

    assert ctx.invalidated is True


def test_native_context_exposes_execute_binding_views():
    native_module = importlib.import_module("ge.custom_op._native")._native
    assert hasattr(native_module, "RuntimeAttrs")
    for method_name in (
        "get_int",
        "get_float",
        "get_bool",
        "get_str",
        "get_data_type",
        "get_tensor",
        "get_list_int",
        "get_list_float",
        "get_list_bool",
        "get_list_str",
        "get_list_data_type",
        "get_list_list_int",
        "get_attr_num",
    ):
        assert hasattr(native_module.RuntimeAttrs, method_name)
    assert hasattr(native_module.EagerOpExecutionContext, "get_dynamic_input_num")
    assert hasattr(native_module.EagerOpExecutionContext, "get_attrs")


def test_bridge_rejects_unknown_descriptor_key():
    with pytest.raises(KeyError, match="descriptor_key not found"):
        bridge.create_op_impl_holder("unknown#1", "not-found")


def test_bridge_rejects_holder_without_callable_execute():
    instance_id = "InvalidExecute#1"
    bridge._OP_IMPL_HOLDERS[instance_id] = bridge._OpImplHolder(
        descriptor_key="invalid",
        instance_id=instance_id,
        instance=object(),
    )

    with pytest.raises(
        TypeError,
        match="python op impl does not implement callable execute",
    ):
        bridge._get_eager_execute_holder(instance_id)


def test_bridge_loads_custom_op_plugins_from_env_path(tmp_path, monkeypatch):
    module_path = _write_custom_op_module(
        tmp_path, "env_custom_op", "EnvCustomOp", "EnvCustom"
    )
    monkeypatch.setenv(bootstrap.ENV_PY_CUSTOM_OP_PATH, str(module_path))

    descriptors = bridge.load_and_get_op_impl_descriptors()

    assert len(descriptors) == 1
    assert descriptors[0]["op_type"] == "EnvCustom"
    assert descriptors[0]["class_name"] == "EnvCustomOp"
    assert descriptors[0]["interfaces"] == ["eager_execute"]


def test_load_custom_op_plugins_rejects_missing_path(monkeypatch):
    monkeypatch.setenv(bootstrap.ENV_PY_CUSTOM_OP_PATH, "/path/not/exist/custom_op.py")

    with pytest.raises(FileNotFoundError, match="python custom op path does not exist"):
        bridge.load_and_get_op_impl_descriptors()
