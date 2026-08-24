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

"""Pytest coverage for schema-bound Python declare_launch_args callbacks."""

import contextvars
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional

import pytest

try:
    bridge = importlib.import_module("ge.custom_op._bridge")
    context = importlib.import_module("ge.custom_op.context")
    custom_op = importlib.import_module("ge.custom_op")
    from ge.graph import DataType
    from ge.runtime import Tensor
except ImportError as exc:
    pytest.skip(f"无法导入 Python custom op 相关模块: {exc}", allow_module_level=True)


@pytest.fixture(autouse=True)
def clear_python_custom_op_runtime():
    custom_op.clear_registered_op_impls()
    bridge.clear_op_impl_holders()
    yield
    bridge.clear_op_impl_holders()
    custom_op.clear_registered_op_impls()


class _FakeRuntimeAttrs:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if not name.startswith("get_"):
            raise AttributeError(name)
        return lambda index: self._record(name, index)

    def _record(self, name, index):
        self.calls.append((name, index))
        return (name, index)


class _FakeDlaContext:
    def __init__(self):
        self.invalidated = False
        self.attrs_requested = False
        self.attrs = _FakeRuntimeAttrs()
        self.calls = []

    def _get_required_input_tensor(self, ir_index):
        self.calls.append(("required_input", ir_index))
        return ("required_input", ir_index)

    def _get_optional_input_tensor(self, ir_index):
        self.calls.append(("optional_input", ir_index))
        return None

    def _get_dynamic_input_num(self, ir_index):
        self.calls.append(("dynamic_input_num", ir_index))
        return 2

    def _get_dynamic_input_tensor(self, ir_index, relative_index):
        self.calls.append(("dynamic_input", ir_index, relative_index))
        return ("dynamic_input", ir_index, relative_index)

    def _get_required_output_tensor(self, ir_index):
        self.calls.append(("required_output", ir_index))
        return ("required_output", ir_index)

    def _get_dynamic_output_num(self, ir_index):
        self.calls.append(("dynamic_output_num", ir_index))
        return 2

    def _get_dynamic_output_tensor(self, ir_index, relative_index):
        self.calls.append(("dynamic_output", ir_index, relative_index))
        return ("dynamic_output", ir_index, relative_index)

    def _get_attrs(self):
        self.attrs_requested = True
        return self.attrs

    def _invalidate(self):
        self.invalidated = True


def _create_holder(instance_id: str, op_type: str) -> None:
    descriptors = {
        item["op_type"]: item for item in bridge.load_and_get_op_impl_descriptors()
    }
    assert bridge.create_op_impl_holder(
        instance_id, descriptors[op_type]["descriptor_key"]
    )


def _get_descriptor_key(op_type: str) -> str:
    return next(
        item["descriptor_key"]
        for item in bridge.load_and_get_op_impl_descriptors()
        if item["op_type"] == op_type
    )


def _full_ir_meta(op_type: str) -> dict:
    return {
        "op_type": op_type,
        "inputs": [
            {"name": "first", "kind": 0},
            {"name": "maybe", "kind": 1},
            {"name": "many", "kind": 2},
        ],
        "attrs": [{"name": "alpha", "type": "VT_INT"}],
        "outputs": [
            {"name": "result", "kind": 0},
            {"name": "result_many", "kind": 1},
        ],
    }


def test_plain_class_declares_annotated_args_capability():
    @custom_op.register_op_impl(op_type="DlaOnlyCustom")
    class DlaOnlyCustom:
        def declare_launch_args(self) -> None:
            pass

    assert DlaOnlyCustom.__ge_op_impl_descriptor__.interfaces == ["annotated_args"]


def test_plain_execute_and_declare_registers_both_capabilities():
    @custom_op.register_op_impl(op_type="PlainExecuteAndDlaCustom")
    class PlainExecuteAndDlaCustom:
        def execute(self) -> None:
            pass

        def declare_launch_args(self) -> None:
            pass

    assert PlainExecuteAndDlaCustom.__ge_op_impl_descriptor__.interfaces == [
        "eager_execute",
        "annotated_args",
    ]


def test_dual_capability_is_registered_in_stable_order():
    @custom_op.register_op_impl(op_type="DualCapabilityCustom")
    class DualCapabilityCustom:
        def execute(self) -> None:
            pass

        def declare_launch_args(self) -> None:
            pass

    assert DualCapabilityCustom.__ge_op_impl_descriptor__.interfaces == [
        "eager_execute",
        "annotated_args",
    ]


def test_non_callable_declare_launch_args_does_not_satisfy_registration():
    with pytest.raises(TypeError, match="must implement at least one supported method"):

        @custom_op.register_op_impl(op_type="BadDlaCustom")
        class BadDlaCustom:
            declare_launch_args = 1


def test_non_callable_declare_launch_args_is_ignored_with_valid_execute():
    @custom_op.register_op_impl(op_type="ExecuteWithBadDlaCustom")
    class ExecuteWithBadDlaCustom:
        declare_launch_args = 1

        def execute(self) -> None:
            pass

    assert ExecuteWithBadDlaCustom.__ge_op_impl_descriptor__.interfaces == [
        "eager_execute"
    ]


def test_get_declare_launch_args_ctx_is_unavailable_outside_callback():
    with pytest.raises(RuntimeError, match="only available inside declare_launch_args"):
        custom_op.get_declare_launch_args_ctx()


def test_declare_launch_args_context_scope_restores_and_invalidates():
    outer = object()
    inner = object()
    copied_contexts = []
    context_scope = getattr(context, "_declare_launch_args_ctx_scope")

    with context_scope(outer):
        assert custom_op.get_declare_launch_args_ctx() is outer
        with context_scope(inner):
            assert custom_op.get_declare_launch_args_ctx() is inner
            copied_contexts.append(contextvars.copy_context())
        assert custom_op.get_declare_launch_args_ctx() is outer

    with pytest.raises(RuntimeError, match="only available inside declare_launch_args"):
        custom_op.get_declare_launch_args_ctx()
    with pytest.raises(RuntimeError, match="only available inside declare_launch_args"):
        copied_contexts[0].run(custom_op.get_declare_launch_args_ctx)


def test_call_declare_launch_args_binds_schema_arguments_and_context():
    seen = []

    @custom_op.register_op_impl(op_type="SchemaDlaCustom")
    class SchemaDlaCustom:
        def declare_launch_args(
            self,
            first: Tensor,
            maybe: Optional[Tensor],
            many: List[Tensor],
            result: Tensor,
            result_many: list[Tensor],
            *,
            alpha: int,
        ) -> None:
            _ = self
            seen.append(
                (
                    first,
                    maybe,
                    many,
                    result,
                    result_many,
                    alpha,
                    custom_op.get_declare_launch_args_ctx(),
                )
            )

    ctx = _FakeDlaContext()
    descriptor_key = _get_descriptor_key("SchemaDlaCustom")
    assert (
        bridge.validate_op_impl_descriptor(
            descriptor_key, _full_ir_meta("SchemaDlaCustom")
        )
        is True
    )
    _create_holder("SchemaDlaCustom#1", "SchemaDlaCustom")

    assert (
        bridge.call_declare_launch_args(
            "SchemaDlaCustom#1", _full_ir_meta("SchemaDlaCustom"), ctx
        )
        is None
    )
    assert seen == [
        (
            ("required_input", 0),
            None,
            [("dynamic_input", 2, 0), ("dynamic_input", 2, 1)],
            ("required_output", 0),
            [("dynamic_output", 1, 0), ("dynamic_output", 1, 1)],
            ("get_int", 0),
            ctx,
        )
    ]
    assert ctx.calls == [
        ("required_input", 0),
        ("optional_input", 1),
        ("dynamic_input_num", 2),
        ("dynamic_input", 2, 0),
        ("dynamic_input", 2, 1),
        ("required_output", 0),
        ("dynamic_output_num", 1),
        ("dynamic_output", 1, 0),
        ("dynamic_output", 1, 1),
    ]
    assert ctx.attrs.calls == [("get_int", 0)]
    assert ctx.invalidated is True


@pytest.mark.parametrize(
    "method_body, expected",
    [
        (
            "def declare_launch_args(self, first, *, alpha) -> None: pass",
            "expected 5 positional",
        ),
        (
            "def declare_launch_args(self, first, maybe, many, result, result_many, alpha) -> None: pass",
            "keyword-only",
        ),
        (
            "def declare_launch_args(self, first, maybe, many, result, result_many, *, beta) -> None: pass",
            "expected attr name",
        ),
        (
            "def declare_launch_args(self, first, maybe, many, result, result_many, *args, alpha) -> None: pass",
            "variadic",
        ),
        (
            "def declare_launch_args(self, first, maybe, many, result, result_many, *, alpha) -> int: pass",
            "expected None",
        ),
    ],
)
def test_validate_op_impl_descriptor_rejects_invalid_declare_signature(
    method_body, expected
):
    namespace = {}
    exec(method_body, namespace)
    method = namespace.get("declare_launch_args")
    assert method is not None

    @custom_op.register_op_impl(op_type=f"InvalidSignature{abs(hash(method_body))}")
    class InvalidSignature:
        declare_launch_args = method

    op_type = InvalidSignature.__ge_op_impl_descriptor__.op_type
    descriptor_key = _get_descriptor_key(op_type)

    with pytest.raises(TypeError) as exc_info:
        bridge.validate_op_impl_descriptor(descriptor_key, _full_ir_meta(op_type))

    message = str(exc_info.value)
    assert op_type in message
    assert InvalidSignature.__ge_op_impl_descriptor__.descriptor_key in message
    assert "declare_launch_args" in message
    assert "expected" in message
    assert "actual" in message
    assert expected in message


def test_call_declare_launch_args_does_not_validate_signature_at_runtime(
    monkeypatch,
):
    called = []

    @custom_op.register_op_impl(op_type="RegistrationValidatedDlaCustom")
    class RegistrationValidatedDlaCustom:
        @staticmethod
        def declare_launch_args(
            first: Tensor,
            maybe: Optional[Tensor],
            many: List[Tensor],
            result: Tensor,
            result_many: list[Tensor],
            *,
            alpha: int,
        ) -> None:
            called.append((first, maybe, many, result, result_many, alpha))

    op_type = "RegistrationValidatedDlaCustom"
    descriptor_key = _get_descriptor_key(op_type)
    ir_meta = _full_ir_meta(op_type)
    assert bridge.validate_op_impl_descriptor(descriptor_key, ir_meta) is True
    assert bridge.create_op_impl_holder(f"{op_type}#1", descriptor_key) is True

    def fail_on_runtime_validation(*args, **kwargs):
        _ = (args, kwargs)
        pytest.fail("declare_launch_args must not validate its signature at runtime")

    monkeypatch.setattr(bridge, "_validate_args_signature", fail_on_runtime_validation)
    ctx = _FakeDlaContext()
    bridge.call_declare_launch_args(f"{op_type}#1", ir_meta, ctx)

    assert len(called) == 1
    assert ctx.invalidated is True


def test_validate_op_impl_descriptor_requires_canonical_ir_for_declare():
    @custom_op.register_op_impl(op_type="MissingRegistrationDlaSchemaCustom")
    class MissingRegistrationDlaSchemaCustom:
        def declare_launch_args(self, x: Tensor) -> None:
            pass

    with pytest.raises(
        RuntimeError,
        match="canonical IR not found for schema-bound declare_launch_args",
    ):
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key("MissingRegistrationDlaSchemaCustom"), None
        )


def test_call_declare_launch_args_rejects_missing_schema_and_non_none_result():
    @custom_op.register_op_impl(op_type="MissingSchemaDlaCustom")
    class MissingSchemaDlaCustom:
        def declare_launch_args(self) -> None:
            _ = self
            return True

    ctx = _FakeDlaContext()
    _create_holder("MissingSchemaDlaCustom#1", "MissingSchemaDlaCustom")
    with pytest.raises(
        RuntimeError, match="canonical IR not found.*declare_launch_args"
    ):
        bridge.call_declare_launch_args("MissingSchemaDlaCustom#1", None, ctx)
    assert ctx.invalidated is True


@pytest.mark.parametrize(
    ("ir_type", "annotation"),
    [
        ("VT_INT", int),
        ("VT_FLOAT", float),
        ("VT_BOOL", bool),
        ("VT_STRING", str),
        ("VT_DATA_TYPE", DataType),
        ("VT_TENSOR", Tensor),
        ("VT_LIST_INT", List[int]),
        ("VT_LIST_FLOAT", list[float]),
        ("VT_LIST_BOOL", List[bool]),
        ("VT_LIST_STRING", list[str]),
        ("VT_LIST_DATA_TYPE", List[DataType]),
        ("VT_LIST_LIST_INT", list[list[int]]),
    ],
)
def test_signature_accepts_canonical_attr_annotations(ir_type, annotation):
    def declare_launch_args(*, alpha) -> None:
        pass

    declare_launch_args.__annotations__ = {"alpha": annotation, "return": None}
    descriptor = SimpleNamespace(
        op_type="AttrAnnotationCustom", descriptor_key="attr-annotation"
    )
    validate_args_signature = getattr(bridge, "_validate_args_signature")
    validate_args_signature(
        declare_launch_args,
        {
            "op_type": "AttrAnnotationCustom",
            "inputs": [],
            "attrs": [{"name": "alpha", "type": ir_type}],
            "outputs": [],
        },
        descriptor,
        method_name="declare_launch_args",
    )


def test_signature_rejects_noncanonical_tensor_annotation():
    @custom_op.register_op_impl(op_type="WrongTensorAnnotationCustom")
    class WrongTensorAnnotationCustom:
        def declare_launch_args(self, x: list[Tensor]) -> None:
            pass

    descriptor_key = _get_descriptor_key("WrongTensorAnnotationCustom")
    with pytest.raises(TypeError) as exc_info:
        bridge.validate_op_impl_descriptor(
            descriptor_key,
            {
                "op_type": "WrongTensorAnnotationCustom",
                "inputs": [{"name": "x", "kind": 0}],
                "attrs": [],
                "outputs": [],
            },
        )
    assert "WrongTensorAnnotationCustom" in str(exc_info.value)
    assert "expected" in str(exc_info.value)
    assert "actual" in str(exc_info.value)

    @custom_op.register_op_impl(op_type="NonNoneResultDlaCustom")
    class NonNoneResultDlaCustom:
        def declare_launch_args(self) -> None:
            _ = self
            return True

    ctx = _FakeDlaContext()
    _create_holder("NonNoneResultDlaCustom#1", "NonNoneResultDlaCustom")
    with pytest.raises(TypeError, match="declare_launch_args must return None"):
        bridge.call_declare_launch_args(
            "NonNoneResultDlaCustom#1",
            {
                "op_type": "NonNoneResultDlaCustom",
                "inputs": [],
                "attrs": [],
                "outputs": [],
            },
            ctx,
        )
    assert ctx.invalidated is True


def test_native_module_exposes_annotated_args_types():
    native_module = getattr(importlib.import_module("ge.custom_op._native"), "_native")
    for type_name in (
        "AnnotatedArgsContext",
        "AnnotatedKernelArgs",
        "AnnotatedKernelLaunchInfo",
        "InferMetaContext",
        "WorkspaceAddr",
    ):
        assert hasattr(native_module, type_name)
    with pytest.raises(TypeError):
        native_module.AnnotatedArgsContext()
    with pytest.raises(TypeError):
        native_module.AnnotatedKernelArgs()
    with pytest.raises(TypeError):
        native_module.WorkspaceAddr()


def test_public_stub_hides_bridge_private_attr_readers():
    stub_path = Path(custom_op.__file__).with_name("_ge_custom_op_native.pyi")
    stub_text = stub_path.read_text(encoding="utf-8")
    annotated_context_stub = stub_text.split("class AnnotatedArgsContext:", 1)[1].split(
        "class InferMetaContext:", 1
    )[0]
    assert "def _get_attrs" not in annotated_context_stub
    assert "def get_attrs" not in annotated_context_stub
    assert "def get_attr" not in annotated_context_stub
    assert "get_attrs" not in custom_op.__all__


def test_python_artifact_uses_bridge_abi_v1():
    artifact_utils = importlib.import_module("ge.custom_op._artifact_utils")
    assert artifact_utils.BRIDGE_ABI_VERSION == 1


def test_launch_info_owns_and_validates_fields():
    info = custom_op.AnnotatedKernelLaunchInfo(
        kernel_name="add_custom",
        kernel_bin=b"\x01\x02",
        block_dim=1,
        stream_id=0,
    )
    assert info is not None
    with pytest.raises(ValueError, match="kernel_name"):
        custom_op.AnnotatedKernelLaunchInfo(
            kernel_name="", kernel_bin=b"\x01", block_dim=1, stream_id=0
        )
