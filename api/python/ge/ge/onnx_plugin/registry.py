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


@dataclass(frozen=True)
class OnnxPluginDescriptor:
    """Normalized descriptor for one parse_node callback."""

    descriptor_key: str
    source: str
    domain: str
    opsets: Tuple[int, ...]
    target: str
    origin_types: Tuple[str, ...]
    module_name: str
    parser_node: Callable[..., None] = field(compare=False, repr=False)

    def to_bridge_dict(self) -> dict:
        return {
            "descriptor_key": self.descriptor_key,
            "source": self.source,
            "domain": self.domain,
            "opsets": list(self.opsets),
            "target": self.target,
            "origin_types": list(self.origin_types),
            "module_name": self.module_name,
        }


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


_ONNX_PLUGIN_REGISTRY = _OnnxPluginRegistry()


def register_onnx_plugin(
    descriptor: OnnxPluginDescriptor,
) -> OnnxPluginDescriptor:
    return _ONNX_PLUGIN_REGISTRY.register(descriptor)


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
