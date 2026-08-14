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

"""Contract tests for ONNX Plugin descriptors and Python registry."""

from dataclasses import FrozenInstanceError
import pytest

from ge.onnx_plugin import OnnxPlugin, onnx_plugin
from ge.onnx_plugin.registry import (
    clear_registered_onnx_plugins,
    get_registered_onnx_plugin_by_origin_type,
    get_registered_onnx_plugin_dicts,
    get_registered_onnx_plugins,
)


@pytest.fixture(autouse=True)
def clear_registry():
    clear_registered_onnx_plugins()
    yield
    clear_registered_onnx_plugins()


def test_onnx_plugin_binds_parse_node_and_expands_normalized_opsets():
    plugin = onnx_plugin(
        source="Elu",
        domain="ai.onnx",
        opsets=[13, 11, 13],
        target="EluTarget",
    )

    assert isinstance(plugin, OnnxPlugin)
    assert get_registered_onnx_plugins() == []

    def parse_elu(node, target):
        del node, target

    decorated = plugin.parse_node(parse_elu)

    assert decorated is parse_elu
    descriptor = parse_elu.__ge_onnx_plugin_descriptor__
    assert descriptor.source == "Elu"
    assert descriptor.domain == "ai.onnx"
    assert descriptor.opsets == (11, 13)
    assert descriptor.target == "EluTarget"
    assert descriptor.origin_types == (
        "ai.onnx::11::Elu",
        "ai.onnx::13::Elu",
    )
    assert descriptor.parser_node is parse_elu
    assert get_registered_onnx_plugins() == [descriptor]
    assert get_registered_onnx_plugin_dicts() == [descriptor.to_bridge_dict()]
    assert descriptor.to_bridge_dict() == {
        "descriptor_key": descriptor.descriptor_key,
        "source": "Elu",
        "domain": "ai.onnx",
        "opsets": [11, 13],
        "target": "EluTarget",
        "origin_types": ["ai.onnx::11::Elu", "ai.onnx::13::Elu"],
        "module_name": __name__,
    }


def test_descriptor_is_frozen():
    plugin = onnx_plugin(
        source="Source",
        domain="test.domain",
        opsets=(1,),
        target="Target",
    )

    @plugin.parse_node
    def parse_source(node, target):
        del node, target

    with pytest.raises(FrozenInstanceError):
        parse_source.__ge_onnx_plugin_descriptor__.target = "Changed"


def test_disjoint_opsets_can_share_source_domain_and_target():
    first = onnx_plugin(
        source="Source",
        domain="test.domain",
        opsets=(1, 2),
        target="Target",
    )
    second = onnx_plugin(
        source="Source",
        domain="test.domain",
        opsets=(3,),
        target="Target",
    )

    @first.parse_node
    def parse_legacy(node, target):
        del node, target

    @second.parse_node
    def parse_modern(node, target):
        del node, target

    assert len(get_registered_onnx_plugins()) == 2
    assert (
        get_registered_onnx_plugin_by_origin_type("test.domain::3::Source").parser_node
        is parse_modern
    )


def test_overlapping_origin_registration_is_atomic():
    first = onnx_plugin(
        source="Source",
        domain="test.domain",
        opsets=(2,),
        target="TargetA",
    )
    overlapping = onnx_plugin(
        source="Source",
        domain="test.domain",
        opsets=(1, 2),
        target="TargetB",
    )

    @first.parse_node
    def parse_first(node, target):
        del node, target

    with pytest.raises(
        ValueError, match="origin type already exists.*test.domain::2::Source"
    ):

        @overlapping.parse_node
        def parse_overlapping(node, target):
            del node, target

    assert len(get_registered_onnx_plugins()) == 1
    assert get_registered_onnx_plugin_by_origin_type("test.domain::1::Source") is None
    assert (
        get_registered_onnx_plugin_by_origin_type("test.domain::2::Source").parser_node
        is parse_first
    )


def test_parse_node_can_only_be_bound_once():
    plugin = onnx_plugin(
        source="Source",
        domain="test.domain",
        opsets=(1,),
        target="Target",
    )

    @plugin.parse_node
    def parse_first(node, target):
        del node, target

    with pytest.raises(ValueError, match="parse_node is already bound"):

        @plugin.parse_node
        def parse_second(node, target):
            del node, target

    assert len(get_registered_onnx_plugins()) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", None),
        ("source", ""),
        ("source", "ai.onnx::Elu"),
        ("domain", None),
        ("domain", ""),
        ("domain", "bad::domain"),
        ("target", None),
        ("target", ""),
    ],
)
def test_onnx_plugin_rejects_invalid_string_fields(field, value):
    arguments = {
        "source": "Source",
        "domain": "test.domain",
        "opsets": (1,),
        "target": "Target",
    }
    arguments[field] = value

    with pytest.raises(TypeError, match=field):
        onnx_plugin(**arguments)


@pytest.mark.parametrize(
    ("opsets", "exception"),
    [
        ((), ValueError),
        ((True,), TypeError),
        ((0,), ValueError),
        ((-1,), ValueError),
        ((1.0,), TypeError),
        ("1", TypeError),
        ((item for item in (1, 2)), TypeError),
    ],
)
def test_onnx_plugin_rejects_invalid_opsets(opsets, exception):
    with pytest.raises(exception, match="opsets"):
        onnx_plugin(
            source="Source",
            domain="test.domain",
            opsets=opsets,
            target="Target",
        )


def test_parse_node_rejects_non_function_without_registering():
    plugin = onnx_plugin(
        source="Source",
        domain="test.domain",
        opsets=(1,),
        target="Target",
    )

    with pytest.raises(TypeError, match="expects a Python function"):
        plugin.parse_node(object())

    assert get_registered_onnx_plugins() == []
