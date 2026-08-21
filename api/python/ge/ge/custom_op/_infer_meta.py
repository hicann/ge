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

"""Python infer_meta callback helpers for GE custom operators."""

from typing import Optional

from ._ir_types import InputType, OutputType
from ._signature import _get_runtime_attr_spec
from .proto import get_registered_op_proto_by_op_type


def _build_infer_inputs(ctx, ir_inputs: list) -> list:
    args = []
    for ir_index, item in enumerate(ir_inputs):
        kind = item["kind"]
        if kind == InputType.REQUIRED:
            args.append(ctx.get_required_input_tensor(ir_index))
        elif kind == InputType.OPTIONAL:
            args.append(ctx.get_optional_input_tensor(ir_index))
        elif kind == InputType.DYNAMIC:
            instance_num = ctx.get_dynamic_input_num(ir_index)
            descs = []
            for relative_index in range(instance_num):
                descs.append(ctx.get_dynamic_input_tensor(ir_index, relative_index))
            args.append(descs)
    return args


def _read_infer_attr(attrs, index: int, ir_type: str):
    getter_name, _ = _get_runtime_attr_spec(ir_type, index)
    return getattr(attrs, getter_name)(index)


def _build_infer_attrs(ctx, ir_attrs: list) -> dict:
    if not ir_attrs:
        return {}
    attrs = ctx.get_attrs()
    return {
        item["name"]: _read_infer_attr(attrs, index, item["type"])
        for index, item in enumerate(ir_attrs)
    }


def _validate_tensor_desc(desc, output_index: int) -> None:
    from ge.runtime import TensorDesc

    if not isinstance(desc, TensorDesc):
        raise TypeError(
            f"infer_meta output[{output_index}] must be TensorDesc, "
            f"got {type(desc).__name__}"
        )


def _flatten_infer_outputs(ir_outputs: list, result) -> tuple:
    if not ir_outputs:
        return [], []

    if len(ir_outputs) == 1:
        kind = ir_outputs[0]["kind"]
        if kind == OutputType.REQUIRED:
            _validate_tensor_desc(result, 0)
            return [result], [1]
        if kind == OutputType.DYNAMIC:
            if not isinstance(result, (list, tuple)):
                raise TypeError(
                    f"infer_meta output[0] is dynamic, must return list, "
                    f"got {type(result).__name__}"
                )
            for i, desc in enumerate(result):
                _validate_tensor_desc(desc, i)
            flattened = list(result)
            return flattened, [len(flattened)]

    if not isinstance(result, (list, tuple)):
        raise TypeError(
            "infer_meta must return list/tuple for multiple outputs, "
            f"got {type(result).__name__}"
        )
    if len(result) != len(ir_outputs):
        raise TypeError(
            f"infer_meta return count {len(result)} != output count {len(ir_outputs)}"
        )
    flattened = []
    slot_sizes = []
    for ir_index, (item, desc) in enumerate(zip(ir_outputs, result)):
        kind = item["kind"]
        if kind == OutputType.REQUIRED:
            _validate_tensor_desc(desc, ir_index)
            flattened.append(desc)
            slot_sizes.append(1)
        elif kind == OutputType.DYNAMIC:
            if not isinstance(desc, (list, tuple)):
                raise TypeError(
                    f"infer_meta output[{ir_index}] is dynamic, must return list"
                )
            for i, d in enumerate(desc):
                _validate_tensor_desc(d, ir_index)
            flattened.extend(desc)
            slot_sizes.append(len(desc))
    return flattened, slot_sizes


def call_infer_meta(op_type: str, ir_meta: Optional[dict], ctx) -> list:
    try:
        proto = get_registered_op_proto_by_op_type(op_type)
        infer_func = proto.infer_func
        args = _build_infer_inputs(ctx, ir_meta["inputs"])
        kwargs = _build_infer_attrs(ctx, ir_meta["attrs"])
        result = infer_func(*args, **kwargs)
        flattened, slot_sizes = _flatten_infer_outputs(ir_meta["outputs"], result)
        for ir_index, item in enumerate(ir_meta["outputs"]):
            kind = item["kind"]
            if kind == OutputType.DYNAMIC:
                instance_num = ctx.get_dynamic_output_num(ir_index)
                actual_num = slot_sizes[ir_index]
                if instance_num != actual_num:
                    raise TypeError(
                        f"infer_meta dynamic output[{ir_index}] instance count mismatch: "
                        f"expected {instance_num}, got {actual_num}"
                    )
        return [
            (
                list(desc.shape.origin_shape.dims),
                list(desc.shape.storage_shape.dims),
                int(desc.data_type),
            )
            for desc in flattened
        ]
    finally:
        ctx._invalidate()
