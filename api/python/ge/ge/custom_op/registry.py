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

"""Python custom op implementation registry and decorators."""

import inspect
import threading
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Type

INTERFACE_EAGER_EXECUTE = "eager_execute"
INTERFACE_HOST_CPU_EXECUTE = "host_cpu_execute"
INTERFACE_COMPILABLE = "compilable"
INTERFACE_ANNOTATED_ARGS = "annotated_args"


class OpBackend(Enum):
    """Backend selector for ``register_kernel``."""

    DEVICE = "device"
    HOST = "host"


_EXECUTE_INTERFACE_BY_BACKEND = {
    OpBackend.DEVICE: INTERFACE_EAGER_EXECUTE,
    OpBackend.HOST: INTERFACE_HOST_CPU_EXECUTE,
}
_METHOD_INTERFACES = (
    (INTERFACE_COMPILABLE, "compile"),
    (INTERFACE_ANNOTATED_ARGS, "declare_launch_args"),
)
_KERNEL_COLLECTOR_ATTR = "__ge_pending_custom_op_kernels__"
_OP_IMPL_DESCRIPTOR_ATTR = "__ge_op_impl_descriptor__"


@dataclass(frozen=True)
class KernelBinding:
    """One unbound execute callback registered for a backend."""

    backend: OpBackend
    function: Callable[..., Any] = field(compare=False, repr=False)

    @property
    def source_location(self) -> str:
        function = getattr(self.function, "__func__", self.function)
        code = getattr(function, "__code__", None)
        if code is None:
            return "<unknown>:0"
        return f"{code.co_filename}:{code.co_firstlineno}"

    @classmethod
    def from_descriptor(
        cls, backend: OpBackend, descriptor: Callable[..., Any]
    ) -> "KernelBinding":
        return cls(backend=backend, function=descriptor)


@dataclass
class _KernelCollector:
    """Class-body-local backend to kernel map."""

    bindings: Dict[OpBackend, KernelBinding] = field(default_factory=dict)

    def contains_function(self, function: object) -> bool:
        return any(binding.function is function for binding in self.bindings.values())


@dataclass(frozen=True)
class OpImplDescriptor:
    """Normalized Python custom op implementation descriptor."""

    descriptor_key: str
    op_type: str
    module_name: str
    class_name: str
    interfaces: List[str] = field(default_factory=list)
    cls: Type[Any] = field(compare=False, repr=False, default=object)
    kernel_bindings: Mapping[OpBackend, KernelBinding] = field(
        default_factory=dict, compare=False, repr=False
    )

    def to_bridge_dict(self) -> dict:
        return {
            "descriptor_key": self.descriptor_key,
            "op_type": self.op_type,
            "module_name": self.module_name,
            "class_name": self.class_name,
            "interfaces": list(self.interfaces),
        }


class _OpImplRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._descriptor_key_to_desc: Dict[str, OpImplDescriptor] = {}
        self._op_type_to_desc: Dict[str, OpImplDescriptor] = {}

    def clear(self) -> None:
        with self._lock:
            self._descriptor_key_to_desc.clear()
            self._op_type_to_desc.clear()

    def register(self, descriptor: OpImplDescriptor) -> OpImplDescriptor:
        with self._lock:
            if descriptor.descriptor_key in self._descriptor_key_to_desc:
                raise ValueError(
                    f"python op impl descriptor_key already exists: {descriptor.descriptor_key}"
                )
            if descriptor.op_type in self._op_type_to_desc:
                raise ValueError(
                    f"python op impl type already exists: {descriptor.op_type}"
                )
            self._descriptor_key_to_desc[descriptor.descriptor_key] = descriptor
            self._op_type_to_desc[descriptor.op_type] = descriptor
        return descriptor

    def get_by_descriptor_key(self, descriptor_key: str) -> Optional[OpImplDescriptor]:
        with self._lock:
            return self._descriptor_key_to_desc.get(descriptor_key)

    def get_all(self) -> List[OpImplDescriptor]:
        with self._lock:
            return list(self._descriptor_key_to_desc.values())


_OP_IMPL_REGISTRY = _OpImplRegistry()


def _build_descriptor_key(module_name: str, class_name: str, op_type: str) -> str:
    return f"{module_name}:{class_name}:{op_type}"


def _normalize_op_type(op_type: str) -> str:
    if not isinstance(op_type, str) or not op_type:
        raise TypeError("register_op_impl op_type must be a non-empty string")
    return op_type


def _get_current_class_namespace() -> Dict[str, Any]:
    frame = inspect.currentframe()
    try:
        decorator_frame = frame.f_back if frame is not None else None
        class_frame = decorator_frame.f_back if decorator_frame is not None else None
        if class_frame is None:
            raise TypeError("register_kernel must be used in a class body")
        namespace = class_frame.f_locals
        if ("__module__" not in namespace) or ("__qualname__" not in namespace):
            raise TypeError("register_kernel must be used in a class body")
        return namespace
    finally:
        del frame


def _validate_kernel_backend(backend: OpBackend) -> None:
    if not isinstance(backend, OpBackend):
        raise TypeError(
            "register_kernel backend must be an OpBackend member, "
            "e.g. OpBackend.DEVICE or OpBackend.HOST"
        )


def _unwrap_kernel_descriptor(descriptor: object) -> Optional[Callable[..., Any]]:
    if isinstance(descriptor, (staticmethod, classmethod)):
        descriptor = descriptor.__func__
    if inspect.isfunction(descriptor):
        return descriptor
    return None


def _validate_kernel_descriptor(descriptor: object) -> None:
    function = _unwrap_kernel_descriptor(descriptor)
    if function is None:
        raise TypeError(
            "register_kernel expects an ordinary function, staticmethod, or classmethod"
        )
    if function.__name__ != "execute":
        raise TypeError("register_kernel can only decorate execute")


def _get_inherited_kernel_bindings(cls: Type[Any]) -> Dict[OpBackend, KernelBinding]:
    for base in cls.__mro__[1:]:
        descriptor = base.__dict__.get(_OP_IMPL_DESCRIPTOR_ATTR)
        if descriptor is not None:
            return dict(descriptor.kernel_bindings)
        inherited_execute = base.__dict__.get("execute")
        if _unwrap_kernel_descriptor(inherited_execute) is not None:
            return {
                OpBackend.DEVICE: KernelBinding.from_descriptor(
                    OpBackend.DEVICE, inherited_execute
                )
            }
    return {}


def _collect_kernel_bindings(cls: Type[Any]) -> Dict[str, KernelBinding]:
    collector = cls.__dict__.get(_KERNEL_COLLECTOR_ATTR)
    if collector is not None:
        # The temporary collector must not become part of the public class API.
        delattr(cls, _KERNEL_COLLECTOR_ATTR)
        final_execute = cls.__dict__.get("execute")
        if ("execute" not in cls.__dict__) or (
            not collector.contains_function(final_execute)
        ):
            if isinstance(final_execute, (staticmethod, classmethod)) and (
                collector.contains_function(final_execute.__func__)
            ):
                raise TypeError(
                    "register_kernel must be the outermost decorator when "
                    "combined with staticmethod or classmethod"
                )
            raise TypeError(
                "when register_kernel is used, every execute implementation "
                "must declare a backend"
            )
        return dict(collector.bindings)

    if "execute" in cls.__dict__:
        local_execute = cls.__dict__["execute"]
        if _unwrap_kernel_descriptor(local_execute) is not None:
            return {
                OpBackend.DEVICE: KernelBinding.from_descriptor(
                    OpBackend.DEVICE, local_execute
                )
            }
        return {}
    return _get_inherited_kernel_bindings(cls)


def _collect_interfaces(
    cls: Type[Any], kernel_bindings: Mapping[OpBackend, KernelBinding]
) -> List[str]:
    interfaces = []
    for backend, interface_name in _EXECUTE_INTERFACE_BY_BACKEND.items():
        has_kernel_binding = backend in kernel_bindings
        legacy_device_execute = (
            not kernel_bindings
            and backend is OpBackend.DEVICE
            and callable(getattr(cls, "execute", None))
        )
        if has_kernel_binding or legacy_device_execute:
            interfaces.append(interface_name)
    for interface_name, method_name in _METHOD_INTERFACES:
        method = getattr(cls, method_name, None)
        # Declare_launch_args keeps legacy discovery behavior.  Compile is
        # schema-bound and must reject an explicitly declared non-callable
        # callback at registration time.
        if (
            interface_name == INTERFACE_COMPILABLE
            and hasattr(cls, method_name)
            and not callable(method)
        ):
            raise TypeError(f"{method_name} must be callable")
        if callable(method):
            interfaces.append(interface_name)
    return interfaces


def _get_interfaces(
    cls: Type[Any], kernel_bindings: Mapping[str, KernelBinding]
) -> List[str]:
    interfaces = _collect_interfaces(cls, kernel_bindings)
    if not interfaces:
        supported_methods = ", ".join(
            ["execute"] + [method_name for _, method_name in _METHOD_INTERFACES]
        )
        class_name = f"{cls.__module__}.{cls.__qualname__}"
        raise TypeError(
            f"register_op_impl class '{class_name}' must implement at least one "
            f"supported method: {supported_methods}"
        )
    return interfaces


def _register_op_impl_class(cls: Type[Any], *, op_type: str) -> Type[Any]:
    module_name = cls.__module__
    class_name = cls.__name__
    kernel_bindings = _collect_kernel_bindings(cls)
    descriptor = OpImplDescriptor(
        descriptor_key=_build_descriptor_key(module_name, class_name, op_type),
        op_type=op_type,
        module_name=module_name,
        class_name=class_name,
        interfaces=_get_interfaces(cls, kernel_bindings),
        cls=cls,
        kernel_bindings=MappingProxyType(kernel_bindings),
    )
    _OP_IMPL_REGISTRY.register(descriptor)
    setattr(cls, _OP_IMPL_DESCRIPTOR_ATTR, descriptor)
    return cls


def register_op_impl(*, op_type: str) -> callable:
    """Decorator for Python custom op implementation classes."""

    normalized_op_type = _normalize_op_type(op_type)

    def decorator(cls: Type[Any]) -> Type[Any]:
        if not inspect.isclass(cls):
            raise TypeError("register_op_impl expects a class")
        if inspect.isabstract(cls):
            raise TypeError("register_op_impl expects a concrete class")
        return _register_op_impl_class(cls, op_type=normalized_op_type)

    return decorator


def clear_registered_op_impls() -> None:
    _OP_IMPL_REGISTRY.clear()


def get_registered_op_impls() -> List[OpImplDescriptor]:
    return _OP_IMPL_REGISTRY.get_all()


def get_registered_op_impl_dicts() -> List[dict]:
    return [item.to_bridge_dict() for item in get_registered_op_impls()]


def get_registered_op_impl_by_descriptor_key(
    descriptor_key: str,
) -> Optional[OpImplDescriptor]:
    return _OP_IMPL_REGISTRY.get_by_descriptor_key(descriptor_key)


def register_kernel(
    *, backend: OpBackend
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register an execute implementation for one backend."""

    _validate_kernel_backend(backend)

    def decorator(descriptor: Callable[..., Any]) -> Callable[..., Any]:
        _validate_kernel_descriptor(descriptor)
        class_namespace = _get_current_class_namespace()
        collector = class_namespace.setdefault(
            _KERNEL_COLLECTOR_ATTR, _KernelCollector()
        )

        previous_execute = class_namespace.get("execute")
        if ("execute" in class_namespace) and (
            not collector.contains_function(previous_execute)
        ):
            raise TypeError(
                "an earlier execute implementation is not decorated "
                "with register_kernel"
            )

        binding = KernelBinding.from_descriptor(backend, descriptor)
        if backend in collector.bindings:
            old_binding = collector.bindings[backend]
            raise TypeError(
                f"backend {backend.value!r} is already registered; previous declaration: "
                f"{old_binding.source_location}; current declaration: "
                f"{binding.source_location}"
            )

        collector.bindings[backend] = binding
        return descriptor

    return decorator
