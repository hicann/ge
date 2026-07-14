# GE Python Custom Operator Design Document

## 1. Introduction

### 1.1 Purpose

This document describes the requirements, design boundaries, runtime access methods, and external Python API of the GE Python custom operator. The intended audience includes GE Python API developers, custom operator sample maintainers, and business developers who need to write custom operator prototypes and capability implementations in Python.

### 1.2 Scope

The full positioning of the Python custom operator is to support users in describing custom operator prototypes in Python and implementing all custom operator capabilities based on `BaseCustomOp`. The current V1 release completes the minimum viable closed loop and covers only the following capabilities:

- Python users write `EagerExecuteOp` execution logic through `ge.custom_op`.
- Python plugins are discovered and imported through `ASCEND_CUSTOM_OPP_PATH`.
- GE loads Python custom ops idempotently before initialization, online compilation, and execution.
- The C++ runtime accesses the existing `CustomOpFactory` / `CustomOpRegistry` through `PythonCustomOpAdapter`.
- The Python native module `_ge_custom_op_native` provides the `EagerOpExecutionContext` borrowed view.
- `ge.runtime` provides runtime data structures required by the context for return values or input parameters, such as `Tensor`, `StorageShape`, `StorageFormat`, `Shape`, and `TensorPlacement`.

V1 does not cover the following items, but these capabilities remain within the scope of subsequent Python custom operator evolution:

- Python versions of `ShapeInferOp`, `CompilableOp`, `PortableOp`, `ArgsUpdater`, and `AnnotatedArgsOp`.
- Python operator prototype definition and op proto generation/registration.
- A more Pythonic `execute` user interface.
- Python custom operator serialization and deserialization with OM and cross-process loading.
- External encapsulation of `KernelArgs` / `MallocReadOnlyDevArgs` on the Python side.

## 2. General Overview

### 2.1 Software Overview

GE supports extending operator prototypes, compilation capabilities, and runtime execution capabilities through custom operators. Traditional C++ custom operators define operator prototypes, implement `BaseCustomOp` and its capability interfaces, and register creators to `CustomOpFactory`. They are created and invoked by GE during the compilation and execution phases.

The Python custom operator aims to progressively provide Python implementations for operator prototypes and all `BaseCustomOp` capabilities. The current V1 release only enables the `EagerExecuteOp` execution capability: users still provide operator definitions and kernels through the existing OPP / op proto mechanism, and GE wraps the runtime `EagerOpExecutionContext` as a Python borrowed view and calls back the user's Python `execute` method to complete host-side scheduling.

The current `execute(ctx)` is a direct Python counterpart of the C++ `EagerExecuteOp::Execute(gert::EagerOpExecutionContext *)`. It is primarily used to establish the bridge, lifecycle, and runtime tensor access closed loop. Subsequent releases need to provide more Pythonic interfaces on this basis, such as more natural input parameter expressions, to reduce the cost for users to directly understand C++ context details.

### 2.2 Product Environment Introduction

The Python custom op is part of the GE Python system and shares the following design principles with the Python pass:

- Python code is located in the `ge_py` package.
- Version-sensitive native capabilities are carried by artifacts corresponding to the Python minor version.
- The C++ main body does not directly expose Python object lifecycles. Python-related logic is consolidated in the bridge/native components.

The actual module boundaries are as follows:

| Module | Location | Responsibility |
|--------|----------|----------------|
| Python API | `api/python/ge/ge/custom_op/` | User base class inheritance, implementation registration, plugin discovery, bridge helper |
| Runtime types | `api/python/ge/ge/runtime/` | Runtime data structures such as `Tensor`, `StorageShape`, and `StorageFormat` |
| Native context | `api/python/ge/ge/custom_op/native_bindings/` | `_ge_custom_op_native`, binding `EagerOpExecutionContext` |
| Runtime loader | `runtime/custom_op/custom_op_loader.cc` | Unified loading of C++ custom ops and Python custom ops |
| Bridge loader | `runtime/custom_op/python_custom_op_bridge_loader.cc` | Artifact selection, loading `libge_python_custom_op_bridge.so`, and creator registration |
| Pybind bridge | `runtime/custom_op/python_custom_op_pybind_bridge.cc` | Importing the Python bridge module, creating holders, and calling back `execute` |
| Adapter | `runtime/custom_op/python_custom_op_adapter.*` | Serving as a C++ `BaseCustomOp` instance to access the existing runtime |
| Capability helper | `inc/graph_metadef/graph/custom_op/` | `CustomOpCapability` and `CustomOpCast<T>` |

### 2.3 Software Functions

V1 functions include:

- `@register_op_impl(op_type=...)` registers a Python custom operator implementation.
- `EagerExecuteOp.execute(self, ctx)` receives an `EagerOpExecutionContext`.
- `EagerOpExecutionContext` supports input and output tensor queries, output and workspace allocation, and stream retrieval.
- `ASCEND_CUSTOM_OPP_PATH` carries both the existing C++ custom op OPP paths and the Python custom op file or package paths.
- GE initialization and `GraphManager::PreRun()` idempotently load Python custom ops when needed, so that the corresponding op type is visible before `OpsKernelInfo` is refreshed.

### 2.4 Design Constraints

- The AscendIR graph structure, op proto format, and OM file format are not changed.
- Python runtime or pybind dependencies are not directly introduced in `graph_metadef/register`.
- Python entry failures affect loading only when Python custom op entries actually exist. The loading is skipped when no Python file or package is present.
- `EagerOpExecutionContext` and objects returned through it are all borrowed views and can be used only within the current `execute` callback.
- The return value of the Python `execute` method is not used as a status code. A normal return indicates success, and an exception indicates failure.
- The Python custom op currently declares only the `EagerExecuteOp` capability. Other C++ capability interfaces are retained as overrides in the adapter but are treated as unsupported.
- The current `execute(ctx)` maintains direct semantic alignment with the C++ `EagerOpExecutionContext`. The Pythonic API is a subsequent evolution goal and is not forcibly abstracted in V1.
- The Python custom op native/bridge is related to the Python ABI at build time. Cross-Python minor version compatibility is not guaranteed.

### 2.5 Assumptions and Dependencies

- The GE / ATC / Executor entry calls `GePythonRuntimeManager::EnsureReady()` before `LoadCustomOps()`. If interpreter initialization fails, the system continues with a warning according to the existing entry strategy.
- Python entries in `ASCEND_CUSTOM_OPP_PATH` are `.py` files, non-underscore-prefixed `.py` files one level deep in a directory, or package directories with `__init__.py`.
- In V1, operator prototypes and shape/dtype inference are still provided by users through the existing C++ / OPP method.
- The Python custom op sample depends on the ACL Python runtime and a Python environment that matches the run package.

## 3. Feature Requirement Analysis and Design

### 3.1 Overall Introduction

The Python custom op adds a Python execution implementation layer on top of the existing custom op framework:

```text
User Python module
  -> @register_op_impl(op_type="AddPythonCustomOp")
  -> ge.custom_op registry saves the descriptor

GE initialization / PreRun
  -> LoadCustomOps()
  -> PreProcessForCustomOp() loads C++ custom ops
  -> NeedLoadPythonCustomOps() checks whether a Python entry exists in ASCEND_CUSTOM_OPP_PATH
  -> LoadPythonCustomOps()
  -> Loads libge_python_custom_op_bridge.so and _ge_custom_op_native.so
  -> ge.custom_op._bridge.load_and_get_op_impl_descriptors()
  -> CustomOpFactory::RegisterCustomOpCreator()

Runtime execution
  -> CustomOpRegistry creates PythonCustomOpAdapter
  -> CustomOpCast<EagerExecuteOp>()
  -> PythonCustomOpAdapter::Execute(ctx)
  -> ge.custom_op._bridge.call_execute(instance_id, python_ctx)
  -> User EagerExecuteOp.execute(...)
```

### 3.2 Functional Requirements

#### 3.2.1 Python Eager Execution Interface

**Introduction**

Users inherit `ge.custom_op.EagerExecuteOp` and implement the `execute` method. The current `ctx` form directly maps to the C++ `EagerOpExecutionContext` and is a low-level interface in V1.

```python
def execute(self, ctx: EagerOpExecutionContext) -> None:
    ...
```

**Input**

- `ctx`: `EagerOpExecutionContext` borrowed view.

**Processing**

- The Python bridge calls `ctx._invalidate()` in the `finally` block to invalidate the context and its derived borrowed views.

**Output**

- A normal return indicates successful execution.
- Users should express failure by raising exceptions.

#### 3.2.2 Plugin Discovery and Registration

**Introduction**

The Python custom op uses the `@register_op_impl(op_type=...)` decorator to register implementation classes. Plugin discovery reuses `ASCEND_CUSTOM_OPP_PATH` and maintains the same configuration entry as the existing C++ OPP paths.

**Input**

- `op_type`: A non-empty string, consistent with the op type of the custom operator in the graph.
- Plugin path: `.py` files, plain directories, or Python packages in `ASCEND_CUSTOM_OPP_PATH`.

**Processing**

- `register_op_impl` verifies that the decorated object is a subclass of `BaseCustomOp`.
- The registry collects the interfaces supported by the class. Currently, only `EagerExecuteOp` is supported, and it generates `descriptor_key = module_name:class_name:op_type`.
- `ge.custom_op.bootstrap.load_custom_op_plugins()` imports Python plugins from the environment variable path through `ge._internal.plugin_loader`.
- The bridge reads `get_registered_op_impl_dicts()` and converts descriptors into data consumable by C++.

**Output**

Each descriptor contains at least:

| Field | Description |
|-------|-------------|
| `descriptor_key` | Unique key of the Python implementation |
| `op_type` | Custom operator type |
| `module_name` | Python module name |
| `class_name` | Python class name |
| `interfaces` | Capability list, currently `["eager_execute"]` |

#### 3.2.3 Native Context Interface

**Introduction**

`_ge_custom_op_native` only binds `EagerOpExecutionContext`. Types such as `Tensor`, `StorageShape`, and `StorageFormat` returned by context methods are provided by `ge.runtime`.

**Input**

The bridge layer injects the Python borrowed view at the execution entry.

**Processing**

`EagerOpExecutionContext` exposes the following public methods:

| Method | Description |
|--------|-------------|
| `get_input_tensor(index)` | Obtains an input `Tensor` by input index |
| `get_input_num()` | Obtains the number of runtime input tensors of the current compute node |
| `get_required_input_tensor(ir_index)` | Obtains a `REQUIRED_INPUT` type input `Tensor` based on the operator IR prototype definition |
| `get_optional_input_tensor(ir_index)` | Obtains an `OPTIONAL_INPUT` type input `Tensor` based on the operator IR prototype definition |
| `get_dynamic_input_tensor(ir_index, relative_index)` | Obtains a `DYNAMIC_INPUT` type input `Tensor` based on the operator IR prototype definition |
| `malloc_output_tensor(index, shape, format, dtype)` | Allocates device memory for an output tensor and initializes the basic information of the output tensor |
| `make_output_ref_input(output_index, input_index)` | Specifies that the memory address of an output references an input |
| `malloc_workspace(size)` | Allocates workspace memory with device placement and returns the address as an integer |
| `get_output_tensor(index)` | Obtains the output `Tensor` specified by index |
| `get_stream()` | Obtains the address integer of the associated execution stream |

**Output**

- Tensor-related methods return `ge.runtime.Tensor`.
- Shape and format input parameters use `ge.runtime.StorageShape` and `ge.runtime.StorageFormat`.
- dtype uses `ge.graph.DataType`.
- Stream and workspace addresses are represented as Python `int`.

#### 3.2.4 C++ Adapter and Capability Detection

**Introduction**

Existing C++ custom ops express capabilities through interface inheritance. The Python custom op uses a single `PythonCustomOpAdapter`, so a capability bitmask is needed to maintain the capability detection semantics.

**Input**

- `interfaces` in the Python descriptor.
- `CustomOpCapabilityMask` parsed by the bridge.

**Processing**

- `PythonCustomOpAdapter` inherits `EagerExecuteOp`, `CompilableOp`, `ShapeInferOp`, `PortableOp`, `ArgsUpdater`, and `CustomOpCapabilityProvider`.
- Currently, `PythonCustomOpCallbacks::IsValid()` only accepts `kEagerExecute`.
- The internal GE capability detection uses `CustomOpCast<T>()`. For regular C++ custom ops, it degrades to `dynamic_cast<T *>`. For the Python adapter, it checks the bitmask first.

**Output**

- When `kEagerExecute` is supported, `Execute(ctx)` forwards to Python.
- Unsupported `Compile`, `InferShape`, `InferDataType`, `Serialize`, `Deserialize`, and `UpdateHostArgs` return `GRAPH_FAILED` and log a message.

#### 3.2.5 Loading, Unloading, and Lifecycle

**Introduction**

Python custom op loading is managed by `runtime/custom_op` to avoid direct Python runtime dependencies in `graph_metadef/register`.

**Processing**

- `custom_op::LoadCustomOps()` first calls `OpLibRegistry::PreProcessForCustomOp()` to load C++ custom ops.
- `NeedLoadPythonCustomOps()` returns true only when Python files or packages are found under `ASCEND_CUSTOM_OPP_PATH`.
- `LoadPythonCustomOps()` resolves the loaded Python runtime key and selects the bridge/native artifact under `custom_op/python_custom_op_artifacts/<python_tag>-<platform>`.
- `libge_python_custom_op_bridge.so` exposes the C ABI through `GeGetPythonCustomOpBridgeApi()`.
- The bridge imports `_ge_custom_op_native` and `ge.custom_op._bridge`, registers descriptors, and creates Python holders for each adapter.
- `ShutdownCustomOpsForProcess()` first unloads Python custom ops, cleans up Python holders and the registry, and then shuts down the bridge.

**Output**

- Python descriptors are registered as `CustomOpFactory` creators.
- When the adapter is destructed, the Python holder is destroyed and the runtime registry entry is released.
- The runtime registry does not allow unregistration while active adapters exist.

### 3.3 Non-Functional Requirements

#### 3.3.1 Maintainability

- The Python API, native context, bridge loader, and adapter are clearly layered to avoid scattering Python logic into the graph infrastructure.
- The Python pass and Python custom op only reuse the internal artifact/plugin loader design and do not forcibly extract unstable common interfaces.
- The external API is bounded by the `__all__` of `ge.custom_op`. Internal `_bridge`, `_native`, and `_artifact_utils` are not user-facing APIs.

#### 3.3.2 Testability

- Python UT covers registration, `ctx` signature verification, holder creation and destruction, and environment variable plugin loading.
- C++ UT should cover the capability helper, adapter execute forwarding, loader skip and load paths, bridge ABI verification, and shutdown order.
- The sample `examples/custom_op/args_refresh_add_custom/python` verifies end-to-end loading, graph construction, and execution.

#### 3.3.3 Portability

- The native/bridge artifact is selected by Python tag, platform tag, and bridge ABI.
- Cross-Python minor version reuse is not currently committed. The build and runtime Python versions must match.

#### 3.3.4 Reliability

- When no Python custom op entry exists, loading is skipped and existing C++ custom ops are not affected.
- When a Python custom op entry exists but the interpreter is not loaded or not initialized, the loader returns failure to avoid continuing execution in a partially initialized state.
- Borrowed views are uniformly invalidated after `execute` completes, reducing the risk of dangling references across callbacks.

#### 3.3.5 Platform Requirements

The Python custom op does not differentiate between chips and does not introduce chip-specific branches. Device kernel capabilities are determined by the user-provided kernel and ACL/RT interfaces.

#### 3.3.6 Feature Cross Analysis

| Scenario | Applicability | Analysis |
|----------|---------------|----------|
| Static Shape | Applicable | The Python custom op executes through the existing `EagerExecuteOp` call site and does not change static shape compilation, memory planning, or DavinciModel interfaces. The execution hot path only calls back to Python after the custom operator node enters the adapter. |
| Dynamic Shape | Applicable | `EagerOpExecutionContext` provides dynamic input queries and runtime tensor metadata. The Python side reads shape, format, and dtype from the current execution context and does not add RT2 lowering data. |
| Dynamic Shape Static Subgraph | Applicable | No new `DavinciModelCreate` / `DavinciModelCreateV2` inputs are added, and the v2-to-v1 boundary data is not changed. If a custom op exists in a static subgraph, it is still called through the existing custom op registry and `EagerExecuteOp` path. |
| Offline Scenario (atc compilation) | Partially applicable | atc initialization loads custom ops, making the Python op type visible during compilation. However, the current Python implementation is not saved with the OM and cannot serve as an independently deployable offline execution implementation. |
| Online Scenario (framework adaptation) | Applicable | Both online initialization and `GraphManager::PreRun()` can load Python custom ops. The front end still needs to generate matching op types and necessary tensor descriptions. |

## 4. Performance

### 4.1 Model Compilation Duration

When no Python custom op entry exists, `NeedLoadPythonCustomOps()` only scans one level of Python files and packages under `ASCEND_CUSTOM_OPP_PATH` and then skips bridge loading. When a Python custom op exists, additional time is spent on Python plugin import, artifact selection, and bridge registration. This is the fixed initialization cost of using this capability.

### 4.2 OM Size and Memory Usage

The current implementation does not serialize Python implementations into the OM, does not add OM partitions, and does not change model file sizes. Additional in-process memory comes primarily from the Python interpreter, imported modules, bridge/native SO, and Python holders.

### 4.3 Execution Performance

The V1 Python `EagerExecuteOp.execute` path enters the Python GIL and calls back user Python code. Its performance is not equivalent to that of C++ custom ops. This low-level execution interface is primarily used to enable development convenience and host-side scheduling capabilities and is not suitable as an ultimate execution performance path. The execution hot path does not print high-frequency logs. Logs, dynamic allocation, ACL calls, and kernel args management in user Python code are controlled by the user.

## 5. Interface Design

### 5.1 New/Modified Interface Description

For the Python external API, refer to `docs/zh/api/graph_engine_api/python/ge/custom_op/`. The current public interfaces are as follows:

| Interface | Description |
|-----------|-------------|
| `BaseCustomOp` | Python custom op base class |
| `EagerExecuteOp` | Eager execution custom operator base class |
| `EagerExecuteOp.execute` | User-implemented execution entry |
| `EagerOpExecutionContext` | Execution context borrowed view |
| `register_op_impl` | Registers a Python custom op implementation class |
| `get_registered_op_impls` | Obtains the list of descriptor objects |
| `get_registered_op_impl_dicts` | Obtains the bridge dictionary list |
| `get_registered_op_impl_by_descriptor_key` | Queries a descriptor by descriptor key |
| `clear_registered_op_impls` | Clears the Python registry |

`Tensor`, `Shape`, `StorageShape`, `StorageFormat`, and `TensorPlacement` in `ge.runtime` are input and return types of the context and are not included in the `__all__` of `ge.custom_op`.

### 5.2 Interface Check Items

| Check Item | Sub-Check Item | Involved |
|------------|----------------|----------|
| Interface description | Whether review is required; review should focus on interface compatibility and interface constraints | Involved; new Python external API added |
| Interface description | Whether supplementary materials are needed | Involved; API documentation and samples have been added |
| Interface description | Whether interface prototypes, functions, and return values are clearly described | Involved; refer to the API documentation |
| Interface compatibility | Whether behavior changes before and after modification | Not involved; this is a new capability with no historical public Python custom op API |
| Interface compatibility | Whether the new interface works properly on older versions | Involved; it is unavailable when the older run package lacks the corresponding native/bridge |
| Interface constraints | Whether usage scenarios and call sequence constraints are involved | Involved; the context can be used only within the current `execute` callback |
| Interface constraints | Whether clear errors are reported when constraints are not met | Involved; the Python side raises `TypeError` / `ValueError` / `RuntimeError` |
| Interface constraints | Whether dedicated test cases need to be designed | Involved; coverage of registration, signatures, lifecycle, and native context is required |

## 6. Software Design

### 6.1 Key Data Structures

#### OpImplDescriptor

The Python registry uses `OpImplDescriptor` to describe an implementation class:

```python
@dataclass(frozen=True)
class OpImplDescriptor:
    descriptor_key: str
    op_type: str
    module_name: str
    class_name: str
    interfaces: List[str]
    cls: Type[BaseCustomOp]
```

`to_bridge_dict()` returns the stable fields required by the bridge and does not expose `cls`.

#### PythonCustomOpDescriptor

The C++ runtime uses `PythonCustomOpDescriptor`:

```cpp
struct PythonCustomOpDescriptor {
  std::string descriptor_key;
  std::string op_type;
  CustomOpCapabilityMask capabilities{0U};
};
```

#### PythonCustomOpCallbacks

The bridge registers create/destroy/execute callbacks to the runtime. Currently, `IsValid()` only accepts `kEagerExecute` and requires `create`, `destroy`, and `execute` to be non-null.

#### BorrowedEagerOpExecutionContext

The native binding holds a `gert::EagerOpExecutionContext *` and a shared validity flag. `_invalidate()` sets validity to false and causes all borrowed runtime objects derived from this context to raise errors on subsequent access.

### 6.2 Key Technologies and Algorithms

- **Plugin discovery**: Reuses `ge._internal.plugin_loader` and splits environment variables by `os.pathsep`. Files are imported by dynamic module name, and directories are imported as one-level `.py` files and packages.
- **Artifact selection**: Reuses `python_artifact_utils` and `python_bridge_loader_utils` and matches `python_custom_op_artifacts` by the loaded Python runtime key.
- **Capability filtering**: `CustomOpCast<T>()` first identifies `CustomOpCapabilityProvider` and then checks the bitmask to determine whether the target interface is supported.
- **Holder lifecycle**: The C++ adapter owns `PythonCustomOpHolder`. The Python side uses `_OP_IMPL_HOLDERS` to save instances by `instance_id`. When the adapter is destructed, the Python holder is destroyed.
- **Context invalidation**: The `call_execute` of the bridge uses `finally` to ensure context invalidation and does not rely on the user returning normally.

### 6.3 Process Design

#### Initialization Loading Process

```text
Entry EnsureReady()
  -> custom_op::LoadCustomOps()
     -> OpLibRegistry::PreProcessForCustomOp()
     -> NeedLoadPythonCustomOps()
     -> LoadPythonCustomOps()
        -> ResolveLoadedPythonRuntimeKey()
        -> BuildPrebuiltBridgeLibraryCandidates()
        -> dlopen libge_python_custom_op_bridge.so
        -> set_artifact_config(native_module_path)
        -> register_custom_ops(registrar)
        -> CustomOpFactory::RegisterCustomOpCreator()
```

#### PreRun Idempotent Reload Process

```text
GraphManager::PreRun()
  -> custom_op::LoadPythonCustomOpsIfNeeded()
  -> OpsKernelManager::RefreshOpsKernelInfo()
```

This path ensures that when the user sets the Python custom op path only after initialization, there is still one idempotent loading opportunity before ops kernel information is refreshed.

#### Execution Callback Process

```text
CustomOpRegistry::CreateOrGetCustomOp(op_type)
  -> PythonCustomOpAdapter(desc)
  -> PythonCustomOpHolder(desc)
  -> callbacks.create(desc)
  -> ge.custom_op._bridge.create_op_impl_holder(instance_id, descriptor_key)

PythonCustomOpAdapter::Execute(ctx)
  -> callbacks.execute(holder, ctx)
  -> _borrow_eager_op_execution_context(ctx_handle)
  -> ge.custom_op._bridge.call_execute(instance_id, py_ctx)
  -> user_op.execute(...)
  -> py_ctx._invalidate()
```

### 6.4 Modifications to Submodules

- `api/python/ge/ge/custom_op/`: Added Python custom op API, registry, bootstrap, bridge helper, and native context binding.
- `api/python/ge/ge/runtime/`: Provides runtime tensor/shape/format types for reuse by the custom op context.
- `runtime/custom_op/`: Added Python bridge loader, adapter, and bridge C ABI.
- `inc/graph_metadef/graph/custom_op/`: Added capability and cast helper.
- GE initialization entry: Ensures the Python runtime is attempted to be ready before `LoadCustomOps()`. On failure, a warning is logged and execution continues. The Python custom op loader performs a hard fail only when Python entries actually exist.
- `compiler/graph/manager/graph_manager.cc`: `PreRun()` idempotently loads Python custom ops before refreshing ops kernel information.

### 6.5 Error Handling

#### System Errors

- Python runtime not loaded or not initialized: `LoadPythonCustomOps()` returns failure when a Python custom op entry exists.
- Bridge/native artifact missing or ABI mismatch: Warning is logged for each candidate. Failure is returned after all candidates fail.
- Holder creation failure: The adapter is deemed invalid and the execution path fails.
- Context query, output allocation, or workspace allocation failure: The native binding raises `RuntimeError`.

#### Interface Errors

- `op_type` is not a string or is an empty string: `register_op_impl` raises `TypeError`.
- The decorated object is not a subclass of `BaseCustomOp`: `TypeError` is raised.
- Only `BaseCustomOp` is inherited but no supported interface is provided: `TypeError` is raised.
- Duplicate `op_type` or `descriptor_key`: `ValueError` is raised.
- The `execute` signature is not `self, ctx`: `TypeError` is raised at class definition time.
- Accessing a borrowed view after expiration: `RuntimeError` is raised.

## 7. Security Check

### 7.1 Coding Standards

The implementation follows the existing Python pass and GE runtime style:

- Python internal modules are named with underscores and are not user-facing APIs.
- The C++ loader does not introduce Python dependencies in the base graph structure.
- Resource release has clear owners: the adapter manages holders, the bridge manages Python module state, and the loader manages SO handles.

### 7.2 Coding Check Items

| Check Item | Description | Involved |
|------------|-------------|----------|
| Resource lifecycle management | The Python interpreter, bridge SO, native module, holders, and borrowed context all have process-level or callback-level lifecycles | Involved |
| Whether new threads are created | The current implementation does not create new threads | Not involved |
| Memory safety | The context/tensor borrowed view prevents access outside the callback through shared validity | Involved |
| Log frequency | Loader and registration phase logs are low-frequency. The high-frequency execution path logs only on errors | Involved |
| Environment variables | Reuses `ASCEND_CUSTOM_OPP_PATH` and does not add product-level switches | Involved |

## 8. Compatibility Check

- The original C++ custom op `dynamic_cast` semantics are preserved through `CustomOpCast<T>()` fallback for regular C++ ops.
- The OM format is not changed. Old OMs are still loaded according to the original custom op partition and registry logic on new versions.
- New OMs do not carry Python implementations. The Python custom op execution capability cannot be assumed to be reproducible on older versions.
- The Python custom op depends on matching versions of `ge_py`, bridge/native SO, and the Python ABI in the runtime environment.
- `ASCEND_CUSTOM_OPP_PATH` is an existing environment variable. Adding Python file and package recognition does not affect C++ OPP paths that have no Python entries.

## 9. DT Design

### 9.1 Test Boundaries

- Python API test entries: `ge.custom_op`, `ge.custom_op._bridge`, `ge.custom_op.bootstrap`.
- Native context test entries: `_borrow_eager_op_execution_context` and `EagerOpExecutionContext` methods.
- C++ test entries: `CustomOpCast<T>`, `PythonCustomOpAdapter`, `LoadPythonCustomOps()`, `ShutdownCustomOpsForProcess()`.
- End-to-end sample entry: `examples/custom_op/args_refresh_add_custom/python/run.sh`.

### 9.2 Test Design

| Test Category | Key Test Items | Test Method | Case Type |
|---------------|----------------|-------------|-----------|
| Function | Successful registration, duplicate registration, invalid `op_type`, invalid inheritance, invalid `execute` signature | Python pytest | UT |
| Function | `execute(self, ctx)` signature verification and context pass-through | Python pytest fake context | UT |
| Function | Bridge descriptor retrieval, holder creation and destruction, context invalidation on exceptions | Python pytest | UT |
| Function | Capability bitmask and `CustomOpCast<T>()` behavior | C++ gtest | UT |
| Function | Loader skips when no Python entry exists and loads the bridge when entries exist | C++ gtest / stub | UT |
| Compatibility | C++ custom op bare capability inheritance still casts correctly | C++ gtest | UT |
| Feature cross | Ops kernel info is refreshed after online PreRun loading | GE graph execution related tests | UT/ST |
| Sample | Python EagerExecuteOp custom op graph construction and execution | `args_refresh_add_custom/python` | ST |

### 9.3 Test Framework Design

- Python UT is placed in `tests/ge/ut/ge/graph/pyge_tests/` and uses fake contexts to avoid device dependencies.
- C++ UT uses the existing GE gtest framework and stubs external dependencies of the runtime registry and loader.
- ST reuses the custom op sample directory and verifies kernel load, graph build, session run, and output metadata in a real CANN environment.

## 10. Design Document Check Results

- [x] Cross-feature impact analysis: Static shape, dynamic shape, dynamic shape static subgraph, offline atc, and online framework adaptation scenarios have been analyzed according to `cross_feature_check.md`.
- [x] Key feature design principles: `rt2_runtime.md`, `known_shape_runtime.md`, and `graph_metadef.md` have been loaded. The solution does not modify RT2 lowering data, DavinciModel interfaces, or graph infrastructure semantics.
- [x] Template section coverage: Introduction, general overview, functional requirements, non-functional requirements, performance, interface design, software design, security check, compatibility check, and DT design have been covered.
