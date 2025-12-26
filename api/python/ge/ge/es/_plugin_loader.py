#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime utilities for dynamically discovering and loading ES plugins."""

import importlib
import logging
import sys
from types import ModuleType
from typing import Dict, Any, Union, List

try:
    from importlib.metadata import entry_points
except ImportError:  
    from importlib_metadata import entry_points  # type: ignore

LOG = logging.getLogger(__name__)
_ENTRY_POINT_GROUP = "ge.es.plugins"


def _iter_plugin_entry_points() -> List[Any]:
    """Compatible with entry_points API changes across different Python versions."""
    try:
        # Python 3.10+
        candidates = entry_points(group=_ENTRY_POINT_GROUP)
        if isinstance(candidates, dict):
            return list(candidates.get(_ENTRY_POINT_GROUP, []))
        return list(candidates)
    except TypeError:
        # Python 3.7-3.9
        candidates = entry_points()
        if hasattr(candidates, "select"):
            return list(candidates.select(group=_ENTRY_POINT_GROUP))
        return list(candidates.get(_ENTRY_POINT_GROUP, []))


def _coerce_to_module(obj: Any, plugin_name: str) -> ModuleType:
    """Convert entry point loading result to a module object.
    
    Args:
        obj: Object returned by entry point loading
        plugin_name: Plugin name for error reporting
        
    Returns:
        ModuleType: Converted module object
        
    Raises:
        TypeError: If the object type is not supported
    """
    if isinstance(obj, ModuleType):
        return obj
    if isinstance(obj, str):
        return importlib.import_module(obj)
    if callable(obj):
        result = obj()
        return _coerce_to_module(result, plugin_name)
    raise TypeError(
        f"Plugin '{plugin_name}' returned unexpected type: {type(obj).__name__}. "
        f"Expected ModuleType, str, or callable returning module."
    )


def load_all_plugins() -> Dict[str, ModuleType]:
    """Load all plugins registered to ge.es.plugins.
    
    Returns:
        dict[str, ModuleType]: Mapping of plugin names to module objects
    """
    plugins: Dict[str, ModuleType] = {}
    
    for entry_point in _iter_plugin_entry_points():
        name = getattr(entry_point, "name", None)
        if not name:
            LOG.warning("Ignoring ES plugin entry point without name: %s", entry_point)
            continue
        
        try:
            # Load entry point
            loaded_obj = entry_point.load()
            
            # Convert to module object
            module = _coerce_to_module(loaded_obj, name)
            
            # Register to sys.modules to make import ge.es.<name> available
            fullname = f"ge.es.{name}"
            sys.modules.setdefault(fullname, module)
            
            # Add to plugin dictionary
            plugins[name] = module
            
            # Check plugin status (if plugin provides status function)
            status = "loaded"
            if hasattr(module, "is_ops_loaded"):
                ops_loaded = module.is_ops_loaded()
                status = "fully loaded" if ops_loaded else "partially loaded"
            
            LOG.info("ES plugin '%s' %s: %s", name, status, module.__name__)
            
        except AttributeError as err:
            LOG.error(
                "Failed to load ES plugin '%s': entry point '%s' missing required attribute. "
                "Ensure the plugin's __init__.py defines get_module(). Error: %s",
                name, getattr(entry_point, "value", "unknown"), err
            )
        except ImportError as err:
            LOG.error(
                "Failed to import ES plugin '%s': %s. "
                "Check if all dependencies are installed.",
                name, err
            )
        except Exception as err:
            LOG.error(
                "Unexpected error loading ES plugin '%s' (entry point: %s): %s",
                name, getattr(entry_point, "value", "unknown"), err
            )
    
    return plugins