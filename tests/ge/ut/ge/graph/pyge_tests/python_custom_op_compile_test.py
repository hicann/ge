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

"""Pytest coverage for schema-bound Python custom-op compile callbacks."""

import contextvars
import importlib
from typing import List, Optional

import pytest

try:
    bridge = importlib.import_module("ge.custom_op._bridge")
    context = importlib.import_module("ge.custom_op.context")
    custom_op = importlib.import_module("ge.custom_op")
    from ge.runtime import Tensor
except (ImportError, OSError) as exc:
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

        def getter(index):
            self.calls.append((name, index))
            return (name, index)

        return getter


class _FakeCompilePlatformInfo:
    def get_soc_version(self):
        return "Ascend910B"


class _FakeCompileContext:
    def __init__(self):
        self.invalidated = False
        self.attrs = _FakeRuntimeAttrs()
        self.platform_info = _FakeCompilePlatformInfo()
        self.attrs_requested = False
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

    def _get_platform_info(self):
        return self.platform_info

    def _invalidate(self):
        self.invalidated = True


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


def _create_holder(instance_id: str, op_type: str) -> None:
    descriptors = {}
    for item in bridge.load_and_get_op_impl_descriptors():
        descriptors[item["op_type"]] = item
    assert bridge.create_op_impl_holder(
        instance_id, descriptors[op_type]["descriptor_key"]
    )


def _get_descriptor_key(op_type: str) -> str:
    return next(
        item["descriptor_key"]
        for item in bridge.load_and_get_op_impl_descriptors()
        if item["op_type"] == op_type
    )


def test_callable_compile_declares_capability():
    @custom_op.register_op_impl(op_type="CompilableOnly")
    class CompilableOnly:
        def compile(self) -> None:
            pass

    assert CompilableOnly.__ge_op_impl_descriptor__.interfaces == ["compilable"]


@pytest.mark.parametrize("method_kind", ["inherited", "staticmethod", "classmethod"])
def test_compile_supports_inherited_static_and_class_methods(method_kind):
    called = []

    if method_kind == "inherited":

        class BaseCompile:
            def compile(self) -> None:
                called.append(method_kind)

        class CompileImpl(BaseCompile):
            pass
    elif method_kind == "staticmethod":

        class CompileImpl:
            @staticmethod
            def compile() -> None:
                called.append(method_kind)
    else:

        class CompileImpl:
            @classmethod
            def compile(cls) -> None:
                called.append(method_kind)

    op_type = f"CompileMethod{method_kind}"
    compile_impl = custom_op.register_op_impl(op_type=op_type)(CompileImpl)
    assert compile_impl.__ge_op_impl_descriptor__.interfaces == ["compilable"]

    ctx = _FakeCompileContext()
    ir_meta = {"op_type": op_type, "inputs": [], "attrs": [], "outputs": []}
    assert bridge.validate_op_impl_descriptor(_get_descriptor_key(op_type), ir_meta)
    _create_holder(f"{op_type}#1", op_type)
    assert bridge.call_compile(f"{op_type}#1", ir_meta, ctx) is None
    assert called == [method_kind]
    assert ctx.invalidated is True


@pytest.mark.parametrize("method_name", ["compile"])
@pytest.mark.parametrize("method_value", [None, 1])
def test_non_callable_interface_method_is_rejected_at_registration(
    method_name, method_value
):
    class BadInterface:
        pass

    setattr(BadInterface, method_name, method_value)
    with pytest.raises(TypeError, match=f"{method_name} must be callable"):
        custom_op.register_op_impl(op_type=f"Bad{method_name}")(BadInterface)


def test_compile_context_scope_is_nested_and_invalidated():
    outer = object()
    inner = object()
    copied_contexts = []

    with context._compile_ctx_scope(outer):
        assert custom_op.get_compile_ctx() is outer
        with context._compile_ctx_scope(inner):
            assert custom_op.get_compile_ctx() is inner
            copied_contexts.append(contextvars.copy_context())
        assert custom_op.get_compile_ctx() is outer

    with pytest.raises(
        RuntimeError, match="only available inside schema-bound compile"
    ):
        custom_op.get_compile_ctx()
    with pytest.raises(
        RuntimeError, match="only available inside schema-bound compile"
    ):
        copied_contexts[0].run(custom_op.get_compile_ctx)


def test_get_compile_ctx_is_unavailable_outside_callback():
    with pytest.raises(
        RuntimeError, match="only available inside schema-bound compile"
    ):
        custom_op.get_compile_ctx()


def test_compile_platform_info_is_separate_from_compile_context():
    ctx = _FakeCompileContext()

    with context._compile_ctx_scope(ctx):
        assert custom_op.get_compile_platform_info() is ctx.platform_info
        assert not hasattr(custom_op.get_compile_ctx(), "get_soc_version")


def test_call_compile_binds_inputs_outputs_attrs_and_context():
    seen = []

    @custom_op.register_op_impl(op_type="SchemaCompile")
    class SchemaCompile:
        def compile(
            self,
            first: Tensor,
            maybe: Optional[Tensor],
            many: List[Tensor],
            result: Tensor,
            result_many: list[Tensor],
            *,
            alpha: int,
        ) -> None:
            seen.append(
                (
                    first,
                    maybe,
                    many,
                    result,
                    result_many,
                    alpha,
                    custom_op.get_compile_ctx(),
                )
            )

    ctx = _FakeCompileContext()
    assert bridge.validate_op_impl_descriptor(
        _get_descriptor_key("SchemaCompile"), _full_ir_meta("SchemaCompile")
    )
    _create_holder("SchemaCompile#1", "SchemaCompile")
    assert (
        bridge.call_compile("SchemaCompile#1", _full_ir_meta("SchemaCompile"), ctx)
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
    assert ctx.invalidated is True
    assert ctx.attrs.calls == [("get_int", 0)]


def test_call_compile_binds_all_canonical_attr_types():
    seen = []

    @custom_op.register_op_impl(op_type="AllAttrCompile")
    class AllAttrCompile:
        def compile(
            self,
            *,
            int_attr,
            float_attr,
            bool_attr,
            string_attr,
            data_type_attr,
            tensor_attr,
            list_int_attr,
            list_float_attr,
            list_bool_attr,
            list_string_attr,
            list_data_type_attr,
            list_list_int_attr,
        ) -> None:
            seen.append(
                [
                    int_attr,
                    float_attr,
                    bool_attr,
                    string_attr,
                    data_type_attr,
                    tensor_attr,
                    list_int_attr,
                    list_float_attr,
                    list_bool_attr,
                    list_string_attr,
                    list_data_type_attr,
                    list_list_int_attr,
                ]
            )

    attr_specs = [
        ("int_attr", "VT_INT", "get_int"),
        ("float_attr", "VT_FLOAT", "get_float"),
        ("bool_attr", "VT_BOOL", "get_bool"),
        ("string_attr", "VT_STRING", "get_str"),
        ("data_type_attr", "VT_DATA_TYPE", "get_data_type"),
        ("tensor_attr", "VT_TENSOR", "get_tensor"),
        ("list_int_attr", "VT_LIST_INT", "get_list_int"),
        ("list_float_attr", "VT_LIST_FLOAT", "get_list_float"),
        ("list_bool_attr", "VT_LIST_BOOL", "get_list_bool"),
        ("list_string_attr", "VT_LIST_STRING", "get_list_str"),
        ("list_data_type_attr", "VT_LIST_DATA_TYPE", "get_list_data_type"),
        ("list_list_int_attr", "VT_LIST_LIST_INT", "get_list_list_int"),
    ]
    ir_meta = {
        "op_type": "AllAttrCompile",
        "inputs": [],
        "attrs": [{"name": name, "type": ir_type} for name, ir_type, _ in attr_specs],
        "outputs": [],
    }
    ctx = _FakeCompileContext()
    _create_holder("AllAttrCompile#1", "AllAttrCompile")
    assert bridge.call_compile("AllAttrCompile#1", ir_meta, ctx) is None

    expected_values = []
    for index, (_, _, getter_name) in enumerate(attr_specs):
        expected_values.append((getter_name, index))
    assert seen == [expected_values]
    assert ctx.attrs.calls == expected_values
    assert ctx.invalidated is True


@pytest.mark.parametrize(
    ("method_body", "expected"),
    [
        ("def compile(self, first, *, alpha) -> None: pass", "expected 5 positional"),
        (
            "def compile(self, first, maybe, many, result, result_many, alpha) -> None: pass",
            "keyword-only",
        ),
        (
            "def compile(self, first, maybe, many, result, result_many, *, beta) -> None: pass",
            "expected attr name",
        ),
        (
            "def compile(self, first, maybe, many, result, result_many, *args, alpha) -> None: pass",
            "variadic",
        ),
        (
            "def compile(self, first, maybe, many, result, result_many, *, alpha) -> int: pass",
            "expected None",
        ),
    ],
)
def test_call_compile_rejects_invalid_signature(method_body, expected):
    namespace = {}
    exec(method_body, namespace)
    method = namespace["compile"]

    op_type = f"InvalidCompile{abs(hash((method_body, expected)))}"

    @custom_op.register_op_impl(op_type=op_type)
    class InvalidCompile:
        compile = method

    with pytest.raises(TypeError) as exc_info:
        bridge.validate_op_impl_descriptor(
            _get_descriptor_key(op_type), _full_ir_meta(op_type)
        )

    message = str(exc_info.value)
    assert op_type in message
    assert "compile" in message
    assert "expected" in message
    assert "actual" in message
    assert expected in message


def test_call_compile_rejects_missing_schema_and_non_none_result():
    @custom_op.register_op_impl(op_type="MissingSchemaCompile")
    class MissingSchemaCompile:
        def compile(self) -> None:
            return None

    ctx = _FakeCompileContext()
    _create_holder("MissingSchemaCompile#1", "MissingSchemaCompile")
    with pytest.raises(RuntimeError, match="canonical IR not found.*compile"):
        bridge.call_compile("MissingSchemaCompile#1", None, ctx)
    assert ctx.invalidated is True


def test_compile_descriptor_validation_requires_schema():
    @custom_op.register_op_impl(op_type="OfflineCompileOnly")
    class OfflineCompileOnly:
        def compile(self, first, *, alpha) -> None:
            pass

    descriptor_key = _get_descriptor_key("OfflineCompileOnly")
    with pytest.raises(RuntimeError, match="canonical IR not found.*compile"):
        bridge.validate_op_impl_descriptor(descriptor_key, None)

    @custom_op.register_op_impl(op_type="NonNoneCompile")
    class NonNoneCompile:
        def compile(self) -> None:
            return True

    ctx = _FakeCompileContext()
    _create_holder("NonNoneCompile#1", "NonNoneCompile")
    with pytest.raises(TypeError, match="compile must return None"):
        bridge.call_compile(
            "NonNoneCompile#1",
            {"op_type": "NonNoneCompile", "inputs": [], "attrs": [], "outputs": []},
            ctx,
        )
    assert ctx.invalidated is True


def test_legacy_compile_context_signature_is_rejected():
    @custom_op.register_op_impl(op_type="LegacyCompile")
    class LegacyCompile:
        def compile(self, ctx) -> None:
            pass

    fake = _FakeCompileContext()
    _create_holder("LegacyCompile#1", "LegacyCompile")
    with pytest.raises(TypeError, match="expected 0 positional"):
        bridge.call_compile(
            "LegacyCompile#1",
            {"op_type": "LegacyCompile", "inputs": [], "attrs": [], "outputs": []},
            fake,
        )
    assert fake.invalidated is True


def test_compile_context_is_deactivated_after_exception():
    copied_contexts = []

    @custom_op.register_op_impl(op_type="FailingCompile")
    class FailingCompile:
        def compile(self) -> None:
            assert custom_op.get_compile_ctx() is not None
            copied_contexts.append(contextvars.copy_context())
            raise ValueError("compile failed")

    fake = _FakeCompileContext()
    _create_holder("FailingCompile#1", "FailingCompile")
    with pytest.raises(ValueError, match="compile failed"):
        bridge.call_compile(
            "FailingCompile#1",
            {"op_type": "FailingCompile", "inputs": [], "attrs": [], "outputs": []},
            fake,
        )
    assert fake.invalidated is True
    with pytest.raises(
        RuntimeError, match="only available inside schema-bound compile"
    ):
        copied_contexts[0].run(custom_op.get_compile_ctx)


def test_native_module_exposes_compile_context_type_when_available():
    native_module = importlib.import_module("ge.custom_op._native")._native
    assert hasattr(native_module, "OpCompileContext")
    assert hasattr(native_module, "CompilePlatformInfo")
    assert not hasattr(native_module.OpCompileContext, "get_soc_version")
    assert hasattr(native_module, "_borrow_op_compile_context")
