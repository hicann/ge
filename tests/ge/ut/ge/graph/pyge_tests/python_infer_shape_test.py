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

"""Pytest coverage for replacement graph shape inference."""

import pytest

try:
    from ge.es.graph_builder import GraphBuilder
    from es_ut_test import Add
    from ge.graph import DataType, Format, Graph, Node
except ImportError as exc:
    pytest.skip(
        f"Cannot import infer shape test dependencies: {exc}", allow_module_level=True
    )

from ge import passes as passes

infer_shape = passes.infer_shape


def _build_target_add_graph():
    builder = GraphBuilder("infer_shape_target")
    lhs = builder.create_input(
        0,
        name="target_input_0",
        data_type=DataType.DT_FLOAT16,
        format=Format.FORMAT_NCHW,
        shape=[2, 3],
    )
    rhs = builder.create_input(
        1,
        name="target_input_1",
        data_type=DataType.DT_FLOAT,
        format=Format.FORMAT_NHWC,
        shape=[4, 5],
    )
    add = Add(lhs, rhs)
    graph = builder.build_and_reset([add])
    add_node = next(node for node in graph.get_direct_nodes() if node.type == "Add")
    assert isinstance(add_node, Node)
    return graph, add_node


def _build_replacement_graph():
    builder = GraphBuilder("infer_shape_replacement")
    input0 = builder.create_input(
        0,
        name="replacement_input_0",
        data_type=DataType.DT_INT32,
        format=Format.FORMAT_ND,
        shape=[1],
    )
    input1 = builder.create_input(
        1,
        name="replacement_input_1",
        data_type=DataType.DT_INT32,
        format=Format.FORMAT_ND,
        shape=[1],
    )
    graph = builder.build_and_reset([input0, input1])
    data_nodes = sorted(
        (node for node in graph.get_direct_nodes() if node.type == "Data"),
        key=lambda node: node.name,
    )
    assert len(data_nodes) == 2
    return graph, data_nodes


def _data_descs(data_nodes):
    return [
        (
            list(node.get_output_desc(0).get_shape()),
            node.get_output_desc(0).get_data_type(),
            node.get_output_desc(0).get_format(),
        )
        for node in data_nodes
    ]


def test_infer_shape_from_node_updates_replacement_data_descs_in_place():
    target_graph, add_node = _build_target_add_graph()
    replacement_graph, replacement_data_nodes = _build_replacement_graph()
    assert _data_descs(replacement_data_nodes) == [
        ([1], DataType.DT_INT32, Format.FORMAT_ND),
        ([1], DataType.DT_INT32, Format.FORMAT_ND),
    ]

    result = infer_shape(replacement_graph, add_node)

    assert result is None
    assert target_graph is not None
    assert _data_descs(replacement_data_nodes) == [
        ([2, 3], DataType.DT_FLOAT16, Format.FORMAT_NCHW),
        ([4, 5], DataType.DT_FLOAT, Format.FORMAT_NHWC),
    ]


def test_infer_shape_from_explicit_boundary_updates_replacement_data_descs():
    target_graph, add_node = _build_target_add_graph()
    boundary = passes.SubgraphBoundary()
    assert boundary.add_input(0, passes.SubgraphInput([(add_node, 0)])) == 0
    assert boundary.add_input(1, passes.SubgraphInput([(add_node, 1)])) == 0
    replacement_graph, replacement_data_nodes = _build_replacement_graph()

    result = infer_shape(replacement_graph, boundary)

    assert result is None
    assert target_graph is not None
    assert _data_descs(replacement_data_nodes) == [
        ([2, 3], DataType.DT_FLOAT16, Format.FORMAT_NCHW),
        ([4, 5], DataType.DT_FLOAT, Format.FORMAT_NHWC),
    ]


def test_infer_shape_rejects_non_graph_replacement():
    target_graph, add_node = _build_target_add_graph()

    with pytest.raises(TypeError):
        infer_shape(object(), add_node)

    assert target_graph is not None


def test_infer_shape_rejects_invalid_source_without_mutating_replacement():
    replacement_graph, replacement_data_nodes = _build_replacement_graph()
    descs_before = _data_descs(replacement_data_nodes)

    with pytest.raises(TypeError):
        infer_shape(replacement_graph, object())

    assert _data_descs(replacement_data_nodes) == descs_before


def test_infer_shape_rejects_empty_replacement_graph():
    target_graph, add_node = _build_target_add_graph()
    empty_replacement = Graph("empty_infer_shape_replacement")

    with pytest.raises(RuntimeError):
        infer_shape(empty_replacement, add_node)

    assert target_graph is not None
