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

"""Python ONNX Plugin descriptor registry."""

import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

PARSE_NODE = "parse_node"
PARSE_OPERATOR = "parse_operator"


@dataclass(frozen=True)
class OnnxPluginDescriptor:
    """Normalized descriptor for an ONNX parser plugin."""

    descriptor_key: str
    source: str
    domain: str
    opsets: Tuple[int, ...]
    target: str
    origin_types: Tuple[str, ...]
    module_name: str
    parser_node: Optional[Callable[..., None]] = field(
        default=None, compare=False, repr=False
    )
    parser_operator: Optional[Callable[..., None]] = field(
        default=None, compare=False, repr=False
    )

    @property
    def callback_kinds(self) -> Tuple[str, ...]:
        kinds = []
        if self.parser_node is not None:
            kinds.append(PARSE_NODE)
        if self.parser_operator is not None:
            kinds.append(PARSE_OPERATOR)
        return tuple(kinds)

    @property
    def callback_kind(self) -> str:
        """Compatibility view for descriptors with one callback."""
        return (
            self.callback_kinds[0]
            if len(self.callback_kinds) == 1
            else ",".join(self.callback_kinds)
        )

    def to_bridge_dict(self) -> dict:
        descriptor = {
            "descriptor_key": self.descriptor_key,
            "source": self.source,
            "domain": self.domain,
            "opsets": list(self.opsets),
            "target": self.target,
            "origin_types": list(self.origin_types),
            "module_name": self.module_name,
        }
        if len(self.callback_kinds) == 1:
            descriptor["callback_kind"] = self.callback_kinds[0]
        else:
            descriptor["callback_kinds"] = list(self.callback_kinds)
        return descriptor


class _OnnxPluginRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._descriptor_key_to_desc: Dict[str, OnnxPluginDescriptor] = {}
        self._origin_type_to_desc: Dict[str, OnnxPluginDescriptor] = {}

    def clear(self) -> None:
        with self._lock:
            self._descriptor_key_to_desc.clear()
            self._origin_type_to_desc.clear()

    def register(self, descriptor: OnnxPluginDescriptor) -> OnnxPluginDescriptor:
        with self._lock:
            if descriptor.descriptor_key in self._descriptor_key_to_desc:
                raise ValueError(
                    "python ONNX Plugin descriptor_key already exists: "
                    f"{descriptor.descriptor_key}"
                )
            for origin_type in descriptor.origin_types:
                if origin_type in self._origin_type_to_desc:
                    raise ValueError(
                        f"python ONNX Plugin origin type already exists: {origin_type}"
                    )
            self._descriptor_key_to_desc[descriptor.descriptor_key] = descriptor
            for origin_type in descriptor.origin_types:
                self._origin_type_to_desc[origin_type] = descriptor
        return descriptor

    def get_all(self) -> List[OnnxPluginDescriptor]:
        with self._lock:
            return list(self._descriptor_key_to_desc.values())

    def get_by_origin_type(self, origin_type: str) -> Optional[OnnxPluginDescriptor]:
        # Registration finishes before parsing; parse-time lookup is read-only.
        return self._origin_type_to_desc.get(origin_type)

    def replace(self, descriptor: OnnxPluginDescriptor) -> OnnxPluginDescriptor:
        with self._lock:
            if descriptor.descriptor_key not in self._descriptor_key_to_desc:
                raise ValueError(
                    "python ONNX Plugin descriptor_key does not exist: "
                    f"{descriptor.descriptor_key}"
                )
            self._descriptor_key_to_desc[descriptor.descriptor_key] = descriptor
            for origin_type in descriptor.origin_types:
                self._origin_type_to_desc[origin_type] = descriptor
        return descriptor


_ONNX_PLUGIN_REGISTRY = _OnnxPluginRegistry()


def register_onnx_plugin(
    descriptor: OnnxPluginDescriptor,
) -> OnnxPluginDescriptor:
    return _ONNX_PLUGIN_REGISTRY.register(descriptor)


def replace_registered_onnx_plugin(
    descriptor: OnnxPluginDescriptor,
) -> OnnxPluginDescriptor:
    return _ONNX_PLUGIN_REGISTRY.replace(descriptor)


def clear_registered_onnx_plugins() -> None:
    _ONNX_PLUGIN_REGISTRY.clear()


def get_registered_onnx_plugins() -> List[OnnxPluginDescriptor]:
    return _ONNX_PLUGIN_REGISTRY.get_all()


def get_registered_onnx_plugin_dicts() -> List[dict]:
    return [item.to_bridge_dict() for item in get_registered_onnx_plugins()]


def get_registered_onnx_plugin_by_origin_type(
    origin_type: str,
) -> Optional[OnnxPluginDescriptor]:
    return _ONNX_PLUGIN_REGISTRY.get_by_origin_type(origin_type)
