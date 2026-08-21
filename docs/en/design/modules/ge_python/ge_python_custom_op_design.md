# GE Python Custom Operator Design Document

## 1. Introduction

### 1.1 Purpose

This document describes the requirements, design boundaries, runtime access methods, and external Python API of the GE Python custom operator. The intended audience includes GE Python API developers, custom operator sample maintainers, and business developers who need to write custom operator prototypes and capability implementations in Python.

### 1.2 Scope

The long-term goal of Python custom operators is to let users describe custom operator prototypes and implement custom operator capabilities in Python. V1 completes the execution and static-model address-refresh loops and serves as the baseline for V2, covering the following capabilities:

- Python users write `execute` logic through `ge.custom_op`. User classes do not need to inherit capability base classes, while the existing `EagerExecuteOp` inheritance form remains compatible.
- `execute` supports both the legacy form that directly receives an `EagerOpExecutionContext` and a schema-bound form whose inputs and attributes are assembled from canonical IR.
- Python plugins are discovered and imported through `ASCEND_CUSTOM_OPP_PATH`.
- GE loads Python custom ops idempotently before initialization, online compilation, and execution.
- The C++ runtime accesses the existing `CustomOpFactory` / `CustomOpRegistry` through `PythonCustomOpAdapter`.
- The Python native module `_ge_custom_op_native` provides `EagerOpExecutionContext` and `RuntimeAttrs` borrowed views.
- Python users implement the compile-time `AnnotatedArgsOp` callback through `declare_launch_args` and declare kernel launch arguments with `AnnotatedArgsContext`, `AnnotatedKernelArgs`, and `AnnotatedKernelLaunchInfo`.
- `ge.runtime` provides runtime data structures required by the context for return values or input parameters, such as `Tensor`, `StorageShape`, `StorageFormat`, `Shape`, and `TensorPlacement`.

V2 extends the V1 execution path with Python prototype and Meta inference capabilities. The current stage provides Python prototype creators, Adapter registration transactions, ownership management, and a Python `infer_meta` callback connected to both compile-time and RT2 dynamic-shape inference.

In V2, the Python function decorated with `register_op` performs Meta inference and is referred to as `infer_meta` throughout this document; the function itself does not have to be named `infer_meta`. Following the operator prototype, it receives input `TensorDesc` objects, including optional and dynamic inputs, together with attribute values, and returns one or more `TensorDesc` objects describing output shape and data type. It neither reads input Tensor data nor executes the operator kernel.

V2 specifically covers the following capabilities:

- Provide `ge.runtime.TensorDesc`, and use `ge.custom_op.register_op` to declare, validate, and collect custom operator prototypes from Python function signatures.
- Deep-copy Python prototypes into C++ and register them with `OperatorFactory`, including idempotent registration, conflict detection, ownership, unloading, and rollback on failure.
- Decouple prototype and implementation registration order, Adapter descriptors, callbacks, and holder lifecycles, and support infer-only operators that have a Python prototype and Meta inference but no Python `execute` implementation.
- Complete the `infer_meta` execution path, including assembling inputs and attributes from canonical IR, invoking Python, validating all returned values, and committing the results as one transaction.
- Invoke `infer_meta` once during compilation to obtain shape and dtype for all outputs, and atomically write back shape, dtype, and origin dtype. RT2 dynamic-shape inference reuses the same callback but updates only shape at runtime.
- Complete the public APIs, type declarations, samples, Chinese and English documentation, and NPU end-to-end validation, while preserving existing schema-bound `execute`, C++ prototype with Python implementation, and legacy `execute(ctx)` behavior.

The following items remain outside the V2 scope:

- Direct Python class implementations of `BaseCustomOp` capability interfaces such as `ShapeInferOp`, `CompilableOp`, `PortableOp`, and `ArgsUpdater`; V2 provides Meta inference through the `infer_meta` callback instead.
- Data-dependent inference that reads input Tensor data.
- InferShapeRange, format inference, symbolic inference, and shape-rule generation.
- Python argument binding for `compile`, `serialize`, and `deserialize`, as well as ES API generation.
- New schema-bound `execute` features or removal of the legacy `execute(ctx)` form.
- Python custom operator serialization and deserialization with OM and cross-process loading.
- External encapsulation of `KernelArgs` / `MallocReadOnlyDevArgs` on the Python side.
- Independent compatibility upgrades of the Bridge, native module, and Adapter. V2 still requires them to be built and replaced together, with changes taking effect after restart.

## 2. General Overview

### 2.1 Software Overview

GE supports extending operator prototypes, compilation capabilities, and runtime execution capabilities through custom operators. Traditional C++ custom operators define operator prototypes, implement `BaseCustomOp` and its capability interfaces, and register creators to `CustomOpFactory`. They are created and invoked by GE during the compilation and execution phases.

The Python custom operator aims to progressively provide Python implementations for operator prototypes and custom operator capabilities. The current V1 release enables execution and static-model address refresh. Users still provide operator definitions and kernels through the existing OPP / op proto mechanism. The Eager path wraps the runtime `EagerOpExecutionContext` as a Python borrowed view and calls the user's `execute` method for host-side scheduling. The AnnotatedArgs path wraps `AnnotatedArgsContext` during compilation and calls `declare_launch_args` to generate `args_format` for address refresh when a static model is loaded.

The execution entry supports two forms. `execute(ctx)` directly corresponds to C++ `EagerExecuteOp::Execute(gert::EagerOpExecutionContext *)`, while schema-bound `execute` lets the bridge assemble input and attribute arguments from canonical IR and the runtime context. Both forms share the same adapter, holder, and borrowed-view lifecycle.

### 2.2 Product Environment Introduction

The Python custom op is part of the GE Python system and shares the following design principles with the Python pass:

- Python code is located in the `ge_py` package.
- Version-sensitive native capabilities are carried by artifacts corresponding to the Python minor version.
- The C++ main body does not directly expose Python object lifecycles. Python-related logic is consolidated in the bridge/native components.

The actual module boundaries are as follows:

| Module | Location | Responsibility |
|--------|----------|----------------|
| Python API | `api/python/ge/ge/custom_op/` | Implementation method reflection, registration, compatibility base classes, plugin discovery, bridge helper |
| Runtime types | `api/python/ge/ge/runtime/` | Runtime data structures such as `Tensor`, `StorageShape`, and `StorageFormat` |
| Native context | `api/python/ge/ge/custom_op/native_bindings/` | `_ge_custom_op_native`, binding Eager, AnnotatedArgs, and InferShape contexts, argument builders, and `RuntimeAttrs` |
| Runtime loader | `runtime/custom_op/custom_op_loader.cc` | Unified loading of C++ custom ops and Python custom ops |
| Bridge loader | `runtime/custom_op/python_custom_op_bridge_loader.cc` | Artifact selection, loading `libge_python_custom_op_bridge.so`, and creator registration |
| Pybind bridge | `runtime/custom_op/python_custom_op_pybind_bridge.cc` | Importing the Python bridge module, creating holders, and calling back `execute` / `declare_launch_args` |
| Proto runtime | `runtime/custom_op/python_custom_op_proto.*` | Deep-copying C POD prototypes and registering `OperatorFactory` creators |
| Adapter | `runtime/custom_op/python_custom_op_adapter.*` | Serving as a C++ `BaseCustomOp` instance to access the existing runtime |
| Capability helper | `inc/graph_metadef/graph/custom_op/` | `CustomOpCapability` and `CustomOpCast<T>` |

### 2.3 Software Functions

V1 functions include:

- `@register_op_impl(op_type=...)` registers a Python custom operator implementation.
- `@register_op(op_type=..., mutates_args=...)` collects a custom operator prototype from a Python function signature.
- The bridge registers Python prototypes as `OperatorFactory` creators and collects canonical IR from the effective creators. It does not invoke `infer_meta` at this stage.
- `register_op_impl` reflects callable `execute` and `declare_launch_args` methods and declares the corresponding capabilities without requiring inheritance from `BaseCustomOp`, `EagerExecuteOp`, or `AnnotatedArgsOp`.
- The legacy `execute(self, ctx)` form directly receives an `EagerOpExecutionContext`; the schema-bound form receives inputs and attributes assembled from canonical IR.
- `EagerOpExecutionContext` supports input and output tensor queries, dynamic input instance counts, runtime attribute access, output and workspace allocation, and stream retrieval.
- `declare_launch_args` binds inputs and outputs as positional arguments and attributes as keyword-only arguments from canonical IR. The callback uses `get_declare_launch_args_ctx()` to create an argument builder, allocate workspace, and add kernel launches. The index passed to `append_input` / `append_output` is the flattened input/output instance index of the compute node.
- `ASCEND_CUSTOM_OPP_PATH` carries both the existing C++ custom op OPP paths and the Python custom op file or package paths.
- GE initialization and `GraphManager::PreRun()` idempotently load Python custom ops when needed, so that the corresponding op type is visible before `OpsKernelInfo` is refreshed.

### 2.4 Design Constraints

- The AscendIR graph structure, op proto format, and OM file format are not changed.
- Python runtime or pybind dependencies are not directly introduced in `graph_metadef/register`.
- Python entry failures affect loading only when Python custom op entries actually exist. The loading is skipped when no Python file or package is present.
- `EagerOpExecutionContext`, `AnnotatedArgsContext`, `RuntimeAttrs`, and borrowed views such as `Tensor` objects returned through them can be used only within the current callback. An `AnnotatedKernelArgs` object cannot be reused after `add_launch` consumes it.
- The return value of the Python `execute` method is not used as a status code. A normal return indicates success, and an exception indicates failure.
- The Python custom op currently declares `EagerExecuteOp` and `AnnotatedArgsOp` capabilities. Other C++ capability interfaces are retained as overrides in the adapter but are treated as unsupported.
- The schema-bound form depends on canonical IR from the existing operator prototype. While loading descriptors, the bridge collects canonical IR and calls `validate_op_impl_descriptor` before holder creation or any business callback to validate schema-bound signatures once. `execute` validates IR inputs and attributes, excludes IR outputs from its parameters, and does not constrain its return annotation or return value. `declare_launch_args` validates inputs, outputs, and attributes and requires a `None` return annotation. Runtime callbacks only assemble arguments and invoke business methods; they do not validate signatures, and validation state does not enter the holder lifecycle.
- Cross-SO prototype and Adapter descriptors are synchronously borrowed C POD views. Runtime callbacks must finish validation and deep copying before returning.
- A Python prototype may replace a built-in prototype. If `CustomOpFactory` already contains a C++ or Python custom operator with the same name, registration reports a custom-operator conflict.
- A schema-bound callback obtains the current context through `get_execute_ctx()`. The binding is valid only in the dynamic scope of that callback.
- The Python custom op native/bridge is related to the Python ABI at build time. Cross-Python minor version compatibility is not guaranteed.
- The bridge C ABI remains v1. The `execute` and `declare_launch_args` callbacks pass only the holder and corresponding context. The bridge queries canonical IR through the public run-package API instead of passing a private ABI projection.

### 2.5 Assumptions and Dependencies

- The GE / ATC / Executor entry calls `GePythonRuntimeManager::EnsureReady()` before `LoadCustomOps()`. If interpreter initialization fails, the system continues with a warning according to the existing entry strategy.
- Python entries in `ASCEND_CUSTOM_OPP_PATH` are `.py` files, non-underscore-prefixed `.py` files one level deep in a directory, or package directories with `__init__.py`.
- For operators that do not use Python `register_op`, the V1 compatibility path still obtains the operator prototype and shape/dtype inference through the existing C++ / OPP method. Python `register_op` operators use the V2 `infer_meta` path.
- The Python custom op sample depends on the ACL Python runtime and a Python environment that matches the run package.
- `declare_launch_args` depends on the Python registration environment only during compilation. A new OM stores the selected refresh mode in `_custom_task_args_mode` and the launch layout in `args_format`; static-model loading uses the explicit mode and does not load the Python implementation again. An OM without the attribute keeps the legacy registry lookup and `args_format` fallback.

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
  -> The bridge collects canonical IR and calls validate_op_impl_descriptor to validate each descriptor once
  -> CustomOpFactory::RegisterCustomOpCreator()

Runtime execution
  -> CustomOpRegistry creates PythonCustomOpAdapter
  -> The bridge holder collects and owns a canonical IR snapshot; the signature was validated during descriptor loading
  -> CustomOpCast<EagerExecuteOp>()
  -> PythonCustomOpAdapter::Execute(ctx)
  -> ge.custom_op._bridge.call_execute(instance_id, ir_meta, python_ctx)
  -> legacy execute(ctx) or schema-bound execute(*inputs, **attrs)

Offline compilation argument declaration
  -> CustomOpCast<AnnotatedArgsOp>()
  -> PythonCustomOpAdapter::DeclareLaunchArgs(ctx)
  -> The bridge holder uses its canonical IR snapshot to wrap AnnotatedArgsContext; runtime does not validate the signature again
  -> schema-bound declare_launch_args(*inputs, *outputs, **attrs)
  -> append_input/append_output use flattened instance indices; add_launch generates args_format

Static-model loading
  -> Read _custom_task_args_mode as the selected compile-time refresh mode
  -> For an old OM without the attribute, query the registry and then use args_format as a fallback
  -> Refresh input, output, workspace, and scalar addresses from args_format stored in the OM
  -> Do not load the Python implementation
```

### 3.2 Functional Requirements

#### 3.2.1 Python Eager Execution Interface

**Introduction**

Users implement a callable `execute` method in either a plain Python class or a subclass of a compatibility base class. The bridge selects the invocation form from the bound method signature: a single positional parameter named `ctx` uses the legacy form, while other signatures use the schema-bound form. Callable `execute` methods provided through `staticmethod`, `classmethod`, or inheritance follow the same rule.

```python
# Legacy form
def execute(self, ctx: EagerOpExecutionContext) -> None:
    ...

# Schema-bound form; parameter order follows canonical IR
def execute(self, x, optional_y, dynamic_z, *, alpha, axes) -> None:
    ...
```

**Input**

- Legacy form: `ctx` is an `EagerOpExecutionContext` borrowed view.
- Schema-bound inputs: `REQUIRED_INPUT` passes one `Tensor`, `OPTIONAL_INPUT` passes a `Tensor` or `None`, and `DYNAMIC_INPUT` passes a list of `Tensor` objects in runtime instance order.
- Schema-bound attributes: the bridge reads runtime attributes in canonical IR order and passes them as keyword arguments named by the IR attributes.

**Processing**

- While loading descriptors, the bridge collects canonical IR through the official public run-package API `GetRegisteredIrDef(op_type)` and calls `validate_op_impl_descriptor` before holder creation or any business callback to validate the Python `execute` input and attribute signature. IR outputs are excluded from the `execute` signature, and its return annotation and return value are not validated.
- Holder creation collects and owns another canonical IR snapshot for runtime use. A schema-bound invocation only reads inputs and attributes from `EagerOpExecutionContext` in that IR order and invokes Python; it does not validate the signature again.
- During a schema-bound callback, `get_execute_ctx()` returns the current context. When a nested invocation ends, the outer binding is restored.
- The Python bridge calls `ctx._invalidate()` in the `finally` block to invalidate the context and its derived borrowed views.

**Output**

- A normal return indicates successful execution.
- Users should express failure by raising exceptions.
- A schema-bound descriptor without canonical IR primarily fails during descriptor loading/registration, before any holder or callback context is created.

#### 3.2.2 Python AnnotatedArgs Address-Refresh Interface

**Introduction**

Users implement `declare_launch_args` to declare the ordered arguments for each kernel launch during ATC/GE compilation. This method supports only the schema-bound form: the bridge assembles inputs, outputs, and attributes from canonical IR, and the implementation obtains the current `AnnotatedArgsContext` through `get_declare_launch_args_ctx()`.

**Input**

- Required, optional, and dynamic inputs and outputs are expanded into Python arguments from canonical IR. A dynamic slot is passed as a list of `Tensor` instances in instance order.
- Attributes are passed as keyword arguments named by the IR attributes.
- The `instance_index` passed to `append_input(instance_index, tensor)` or `append_output(instance_index, tensor)` is the flattened compute-node input or output instance index, not an IR-slot index.

**Processing**

- The callback uses `create_kernel_args()` to create an ordered argument builder for one launch and may append inputs, outputs, workspaces, and scalars.
- Each `malloc_workspace(size)` call returns an address object with a workspace index. `append_workspace` records a refresh entry using that index.
- `add_launch(launch_info, args)` consumes the argument builder and writes the AnnotatedArgs description. A consumed builder cannot be reused.
- During descriptor loading, the bridge calls `validate_op_impl_descriptor` to verify that the Python signature matches canonical IR and has a `None` return annotation; runtime callbacks do not validate the signature again. It still checks the actual return value for `None` after every callback and invalidates the context and all derived borrowed views in `finally`.

**Output**

- Compilation stores the selected refresh mode in `_custom_task_args_mode` and the launch argument format in `args_format`.
- Static-model loading selects the refresh path from the explicit mode and consumes `args_format` for AnnotatedArgs without calling Python. Only an old OM without the attribute uses the registry and `args_format` fallback.

#### 3.2.3 Plugin Discovery and Registration

**Introduction**

The Python custom op uses the `@register_op_impl(op_type=...)` decorator to register implementation classes. Plugin discovery reuses `ASCEND_CUSTOM_OPP_PATH` and maintains the same configuration entry as the existing C++ OPP paths.

**Input**

- `op_type`: A non-empty string, consistent with the op type of the custom operator in the graph.
- Plugin path: `.py` files, plain directories, or Python packages in `ASCEND_CUSTOM_OPP_PATH`.

**Processing**

- At decoration time, `register_op_impl` only verifies that the decorated object is a concrete, non-abstract class and reflects its capabilities. It does not require a `BaseCustomOp` subclass; holder creation later instantiates the class with a no-argument constructor.
- The registry reflects callable methods on the class. `execute` maps to `eager_execute`, `declare_launch_args` maps to `annotated_args`, and the registry generates `descriptor_key = module_name:class_name:op_type`.
- Registration fails when the class implements no supported method. The decoration stage does not validate schema-bound business signatures. The bridge subsequently loads the descriptor, collects canonical IR, and calls `validate_op_impl_descriptor` to validate signatures once before registering the C++ creator.
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
| `interfaces` | Capability list containing `"eager_execute"`, `"annotated_args"`, or both |

#### 3.2.4 Native Context Interface

**Introduction**

`_ge_custom_op_native` binds `EagerOpExecutionContext`, `AnnotatedArgsContext`, `InferShapeContext`, `AnnotatedKernelArgs`, `AnnotatedKernelLaunchInfo`, and `RuntimeAttrs`. Types such as `Tensor`, `TensorDesc`, `StorageShape`, and `StorageFormat` returned by context methods are provided by `ge.runtime`.

**Input**

The bridge layer injects the Python borrowed view at the execution entry.

**Processing**

`EagerOpExecutionContext` exposes the following public methods:

| Method | Description |
|--------|-------------|
| `get_input_tensor(index)` | Obtains an input `Tensor` by input index |
| `get_input_num()` | Obtains the number of runtime input tensors of the current compute node |
| `get_dynamic_input_num(ir_index)` | Obtains the runtime instance count of a dynamic input IR slot |
| `get_attrs()` | Obtains the `RuntimeAttrs` borrowed view of the current node |
| `get_required_input_tensor(ir_index)` | Obtains a `REQUIRED_INPUT` type input `Tensor` based on the operator IR prototype definition |
| `get_optional_input_tensor(ir_index)` | Obtains an `OPTIONAL_INPUT` type input `Tensor` based on the operator IR prototype definition |
| `get_dynamic_input_tensor(ir_index, relative_index)` | Obtains a `DYNAMIC_INPUT` type input `Tensor` based on the operator IR prototype definition |
| `malloc_output_tensor(index, shape, format, dtype)` | Allocates device memory for an output tensor and initializes the basic information of the output tensor |
| `make_output_ref_input(output_index, input_index)` | Specifies that the memory address of an output references an input |
| `malloc_workspace(size)` | Allocates workspace memory with device placement and returns the address as an integer |
| `get_output_tensor(index)` | Obtains the output `Tensor` specified by index |
| `get_stream()` | Obtains the address integer of the associated execution stream |

`InferShapeContext` is used by the `register_op` decorated function when it runs as `infer_meta`. It reads shapes and data types for required, optional, and dynamic inputs, reads typed runtime attributes, and queries dynamic output instance counts. This context is valid only during the current `infer_meta` callback; output shapes and data types are carried together by the `TensorDesc` objects returned from `infer_meta`.

`AnnotatedArgsContext` exposes workspace allocation, stream-id query, kernel-argument builder creation, and launch addition. Its input/output tensor and attribute queries are used by the internal schema-bound assembly logic. `AnnotatedKernelArgs` exposes `append_input`, `append_output`, `append_workspace`, and `append_scalar`.

`RuntimeAttrs` provides the following typed readers by attribute IR index:

| Attribute type | Methods |
|----------------|---------|
| Scalar | `get_int`, `get_float`, `get_bool`, `get_str`, `get_data_type`, and `get_tensor` |
| List | `get_list_int`, `get_list_float`, `get_list_bool`, `get_list_str`, `get_list_data_type`, and `get_list_list_int` |
| Count | `get_attr_num()` |

**Output**

- Tensor-related methods return `ge.runtime.Tensor`.
- `infer_meta` inputs and return values use `ge.runtime.TensorDesc`; `TensorDesc.shape` uses `StorageShape`, and `TensorDesc.data_type` uses `ge.graph.DataType`.
- Shape and format input parameters use `ge.runtime.StorageShape` and `ge.runtime.StorageFormat`.
- dtype uses `ge.graph.DataType`.
- Stream and workspace addresses are represented as Python `int`.
- `RuntimeAttrs` and borrowed objects returned from it expire with the current context.

#### 3.2.5 C++ Adapter and Capability Detection

**Introduction**

Existing C++ custom ops express capabilities through interface inheritance. The Python custom op uses a single `PythonCustomOpAdapter`, so a capability bitmask is needed to maintain the capability detection semantics.

**Input**

- `interfaces` in the Python descriptor.
- `CustomOpCapabilityMask` parsed by the bridge.

**Processing**

- `PythonCustomOpAdapter` inherits `EagerExecuteOp`, `AnnotatedArgsOp`, `CompilableOp`, `ShapeInferOp`, `PortableOp`, `ArgsUpdater`, and `CustomOpCapabilityProvider`.
- `PythonCustomOpCallbacks::IsValid()` accepts `kEagerExecute`, `kAnnotatedArgs`, or both and verifies that each capability has its corresponding callback.
- The internal GE capability detection uses `CustomOpCast<T>()`. For regular C++ custom ops, it degrades to `dynamic_cast<T *>`. For the Python adapter, it checks the bitmask first.

**Output**

- When `kEagerExecute` is supported, `Execute(ctx)` forwards to Python.
- When `kAnnotatedArgs` is supported, `DeclareLaunchArgs(ctx)` forwards to Python.
- Unsupported `Compile`, `InferShape`, `InferDataType`, `Serialize`, `Deserialize`, and `UpdateHostArgs` return `GRAPH_FAILED` and log a message.

#### 3.2.6 Loading, Unloading, and Lifecycle

**Introduction**

Python custom op loading is managed by `runtime/custom_op` to avoid direct Python runtime dependencies in `graph_metadef/register`.

**Processing**

- `custom_op::LoadCustomOps()` first calls `OpLibRegistry::PreProcessForCustomOp()` to load C++ custom ops.
- `NeedLoadPythonCustomOps()` returns true only when Python files or packages are found under `ASCEND_CUSTOM_OPP_PATH`.
- `LoadPythonCustomOps()` resolves the loaded Python runtime key and selects the bridge/native artifact under `custom_op/python_custom_op_artifacts/<python_tag>-<platform>`.
- `libge_python_custom_op_bridge.so` exposes C ABI v1 through `GeGetPythonCustomOpBridgeApi()`.
- The bridge imports `_ge_custom_op_native` and `ge.custom_op._bridge` and obtains one prototype/implementation snapshot. It registers all prototypes first, then validates and registers Adapters.
- `CustomOpLoader::LoadCustomOps()` records whether Python custom ops are loaded, so repeated lifecycle load requests return success without invoking the bridge registration entry again. The dynamic `LoadPythonCustomOpsIfNeeded()` path intentionally does not use this flag, allowing newly added Python custom op paths to be discovered during runtime. The lower-level `LoadPythonCustomOps()` function performs one bridge registration attempt; callers are responsible for invoking `UnloadPythonCustomOps()` after a failed attempt so that partial registrations are cleaned up.
- `UnloadPythonCustomOps()` removes registered Adapter creators, clears the Python custom-op runtime registry in one operation, and then removes registered proto creators. It does not perform per-entry runtime unregistration or maintain pending-cleanup state in the bridge loader.
- `UnloadCustomOps()` uses an `active_users_` reference count to manage the lifecycle: each `LoadCustomOps()` increments the count by 1, each `UnloadCustomOps()` decrements it by 1, and Python custom ops are only unloaded (holders/registry cleaned up and bridge closed) when the count reaches zero. `ShutdownCustomOpsForProcess()` is retained as a compatibility wrapper that internally calls `UnloadCustomOps()`.

**Output**

- Python prototypes are registered as `OperatorFactory` creators, and Python implementations are registered as `CustomOpFactory` creators.
- When the adapter is destructed, the Python holder is destroyed and the runtime registry entry is released.
- Individual runtime registry unregistration remains guarded by active adapters; process unload removes Adapter creators first and clears the registry in bulk.

### 3.3 Non-Functional Requirements

#### 3.3.1 Maintainability

- The Python API, native context, bridge loader, and adapter are clearly layered to avoid scattering Python logic into the graph infrastructure.
- The Python pass and Python custom op only reuse the internal artifact/plugin loader design and do not forcibly extract unstable common interfaces.
- The external API is bounded by the `__all__` of `ge.custom_op`. Internal `_bridge`, `_native`, and `_artifact_utils` are not user-facing APIs.

#### 3.3.2 Testability

- Python UT covers plain-class and compatibility-base registration, reflection of both capabilities, schema-bound signature validation during descriptor loading, legacy/schema-bound invocation, `declare_launch_args` return validation, flattened instance indices, context scope, holder lifecycle, and environment-variable plugin loading.
- C++ UT should cover the capability helper, canonical IR lookup, adapter execute/declare forwarding, loader skip and load paths, bridge ABI v1 verification, and shutdown order.
- The sample `examples/custom_op/annotated_args_refresh_add_custom/python` verifies offline compilation, loading without Python, and address-refresh execution.

#### 3.3.3 Portability

- The native/bridge artifact is selected by Python tag, platform tag, and bridge ABI.
- Cross-Python minor version reuse is not currently committed. The build and runtime Python versions must match.

#### 3.3.4 Reliability

- When no Python custom op entry exists, loading is skipped and existing C++ custom ops are not affected.
- When a Python custom op entry exists but the interpreter is not loaded or not initialized, the loader returns failure to avoid continuing execution in a partially initialized state.
- Borrowed views are uniformly invalidated after `execute` or `declare_launch_args` completes, and schema-bound context bindings are cleared on both normal and exceptional paths, reducing the risk of dangling references across callbacks.

#### 3.3.5 Platform Requirements

The Python custom op does not differentiate between chips and does not introduce chip-specific branches. Device kernel capabilities are determined by the user-provided kernel and ACL/RT interfaces.

#### 3.3.6 Feature Cross Analysis

| Scenario | Applicability | Analysis |
|----------|---------------|----------|
| Static Shape | Applicable | The Eager path executes through the existing `EagerExecuteOp` call site. The AnnotatedArgs path records the selected refresh mode and `args_format` during compilation, then refreshes addresses when the static model is loaded, without changing DavinciModel interfaces. |
| Dynamic Shape | Applicable | `EagerOpExecutionContext` provides dynamic input instance counts, tensor queries, and runtime metadata. The bridge assembles dynamic input lists at execution time and adds no RT2 lowering data. |
| Dynamic Shape Static Subgraph | Applicable | No new `DavinciModelCreate` / `DavinciModelCreateV2` inputs are added, and the v2-to-v1 boundary data is not changed. If a custom op exists in a static subgraph, it is still called through the existing custom op registry and `EagerExecuteOp` path. |
| Offline Scenario (atc compilation) | Applicable | atc initializes and loads the Python custom op, invokes `declare_launch_args`, and stores the explicit refresh mode and `args_format` in the OM. The Python implementation itself is not stored. Model execution uses the serialized mode and layout and does not depend on the Python registration environment; an Eager `execute` implementation still cannot be deployed independently offline. |
| Online Scenario (framework adaptation) | Applicable | Both online initialization and `GraphManager::PreRun()` can load Python custom ops. The front end still needs to generate matching op types, operator prototypes, and necessary tensor descriptions; schema-bound invocation reuses canonical IR from that prototype. |

## 4. Performance

### 4.1 Model Compilation Duration

When no Python custom op entry exists, `NeedLoadPythonCustomOps()` only scans one level of Python files and packages under `ASCEND_CUSTOM_OPP_PATH` and then skips bridge loading. When a Python custom op exists, additional time is spent on Python plugin import, artifact selection, and bridge registration. This is the fixed initialization cost of using this capability.

### 4.2 OM Size and Memory Usage

The current implementation does not serialize Python implementations into the OM, does not add OM partitions, and does not change model file sizes. Additional in-process memory comes primarily from the Python interpreter, imported modules, bridge/native SO, and Python holders.

### 4.3 Execution Performance

The Python `execute` path enters the Python GIL and calls back user Python code. Its performance is not equivalent to that of C++ custom ops. The schema-bound form also traverses IR inputs and attributes and creates Python `list` / `dict` arguments, with cost growing linearly with the number of prototype parameters. This interface primarily provides development convenience and host-side scheduling and is not suitable as an ultimate execution performance path. The execution hot path does not print high-frequency logs. Logs, dynamic allocation, ACL calls, and kernel args management in user Python code are controlled by the user.

## 5. Interface Design

### 5.1 New/Modified Interface Description

For the Python external API, refer to `docs/zh/api/graph_engine_api/python/ge/custom_op/`. The current public interfaces are as follows:

| Interface | Description |
|-----------|-------------|
| `BaseCustomOp` | Compatibility base class for existing Python custom ops; new implementations are not required to inherit it |
| `EagerExecuteOp` | Compatibility Eager execution base class; new implementations may use a plain class |
| `execute` | User-implemented execution entry supporting both legacy and schema-bound forms |
| `EagerOpExecutionContext` | Execution context borrowed view |
| `InferShapeContext` | Input metadata access context for `infer_meta` callbacks |
| `RuntimeAttrs` | Attribute borrowed view returned by `EagerOpExecutionContext.get_attrs()` |
| `get_execute_ctx` | Obtains the execution context of the active schema-bound callback |
| `register_op` | Declares and collects a Python custom operator prototype |
| `register_op_impl` | Registers an implementation class and reflects its capability methods |
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
| Interface compatibility | Whether behavior changes before and after modification | Involved; existing inheritance and `execute(ctx)` behavior is preserved, while plain-class registration and schema-bound invocation are added |
| Interface compatibility | Whether the new interface works properly on older versions | Involved; it is unavailable when the older run package lacks the corresponding native/bridge |
| Interface constraints | Whether usage scenarios and call sequence constraints are involved | Involved; context/attrs are valid only in the current callback, and schema-bound invocation depends on canonical IR |
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
    cls: Type[Any]
```

`to_bridge_dict()` returns the stable fields required by the bridge and does not expose `cls`.

#### PythonCustomOpIrMeta

The Python-version-sensitive bridge uses the official `GetRegisteredIrDef` signature from public run-package headers, resolves the loaded run-package symbol through `dlsym(RTLD_DEFAULT, "GetRegisteredIrDef")`, and copies the returned op type, input, attribute, and output information into its private `PythonCustomOpIrMeta`. `PythonCustomOpBridgeHolder` owns the metadata, so neither C++ containers nor a private POD view cross the runtime/bridge callback boundary.

Runtime symbol resolution is used because `libge_runner.so`/`libge_runner_v2.so` already depend on `custom_op_runtime`, while the bridge is dynamically loaded by `custom_op_runtime`; directly linking the bridge back to a runner would create a reverse SO dependency. The function signature and data types are still governed by official public run-package headers, with only symbol binding deferred until after the bridge is loaded.

#### PythonCustomOpProto and PythonCustomOpAdapterDescriptor

The C++ runtime stores an owning prototype and an Adapter descriptor separately; the Adapter descriptor stores only the implementation key:

```cpp
struct PythonCustomOpAdapterDescriptor {
  std::string op_type;
  std::string impl_descriptor_key;
  CustomOpCapabilityMask capabilities{0U};
};
```

#### PythonCustomOpAdapterCallbacks

The bridge registers create/destroy/execute/declare_launch_args callbacks to the runtime. Schema-bound signature validation is not a C++ callback; the bridge calls `validate_op_impl_descriptor` internally before registering an Adapter. `IsValid()` accepts `kEagerExecute`, `kAnnotatedArgs`, or both; it requires create/destroy and verifies execute/declare_launch_args according to the capability mask.

#### BorrowedEagerOpExecutionContext

The native binding holds a `gert::EagerOpExecutionContext *` and a shared validity flag. `_invalidate()` sets validity to false and causes all borrowed runtime objects derived from this context to raise errors on subsequent access.

#### BorrowedAnnotatedArgsContext

The native binding holds a `gert::AnnotatedArgsContext *` and an independent validity flag. After `declare_launch_args` returns or raises, `_invalidate()` expires the context, workspace addresses, kernel-argument builders, and their derived borrowed tensors.

### 6.2 Key Technologies and Algorithms

- **Plugin discovery**: Reuses `ge._internal.plugin_loader` and splits environment variables by `os.pathsep`. Files are imported by dynamic module name, and directories are imported as one-level `.py` files and packages.
- **Artifact selection**: Reuses `python_artifact_utils` and `python_bridge_loader_utils` and matches `python_custom_op_artifacts` by the loaded Python runtime key.
- **Capability reflection**: The registry applies `getattr` and `callable` checks to the implementation class, maps `execute` to `eager_execute`, and maps `declare_launch_args` to `annotated_args`. The inheritance hierarchy of the Python user class does not participate in capability detection.
- **Capability filtering**: `CustomOpCast<T>()` first identifies `CustomOpCapabilityProvider` and then checks the bitmask to determine whether the target interface is supported.
- **IR argument assembly**: The bridge queries canonical IR through the public run-package API during descriptor loading for signature validation, then queries it again at holder creation and owns the resulting runtime snapshot. Callbacks read required, optional, and dynamic inputs/outputs and typed runtime attributes in that IR order, then construct positional and keyword arguments.
- **Registration transaction**: The runtime synchronously deep-copies prototype C POD data and registers its creator, collects canonical IR, and finally registers the implementation runtime entry and Adapter creator. If any step fails, the upper-level loader invokes unload to roll the completed steps back in reverse order.
- **Callback signature validation**: While loading a descriptor, the bridge calls `validate_op_impl_descriptor` to validate schema-bound signatures once, before holder creation or any business callback. For `execute`, it validates the total parameter count, the positional form and supplied type annotations of inputs, and the keyword-only form, names, and supplied type annotations of attributes. Outputs, the return annotation, and the return value are not validated. `declare_launch_args` validates inputs, outputs, attributes, and a `None` return annotation. Runtime callbacks do not validate signatures, and validation state does not enter the holder lifecycle.
- **Holder lifecycle**: The C++ adapter owns `PythonCustomOpHolder`. The Python side uses `_OP_IMPL_HOLDERS` to save instances by `instance_id`. When the adapter is destructed, the Python holder is destroyed.
- **Context binding**: Schema-bound callbacks establish dynamic scopes with `ContextVar`, and `get_execute_ctx()` / `get_declare_launch_args_ctx()` read their corresponding bindings. Resetting the token restores the outer context after nested invocation.
- **Context invalidation**: Both execute and declare bridge calls use `finally` to ensure context invalidation and do not rely on the user returning normally.

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
        -> load_and_get_op_descriptors()
        -> register_custom_ops(registrar)
        -> Register Python prototype creators and collect canonical IR
        -> Call validate_op_impl_descriptor
        -> Register implementation runtime entries and Adapter creators
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
  -> PythonCustomOpImplHolder(desc)
  -> callbacks.create(desc)
  -> The bridge resolves the public GetRegisteredIrDef symbol and queries canonical IR
  -> The bridge holder stores PythonCustomOpIrMeta without validating signatures again
  -> ge.custom_op._bridge.create_op_impl_holder(instance_id, descriptor_key)

PythonCustomOpAdapter::Execute(ctx)
  -> callbacks.execute(holder, ctx)
  -> _borrow_eager_op_execution_context(ctx_handle)
  -> The bridge holder builds the Python ir_meta object
  -> ge.custom_op._bridge.call_execute(instance_id, ir_meta, py_ctx)
  -> legacy: user_op.execute(ctx)
  -> schema-bound: user_op.execute(*inputs, **attrs)
  -> py_ctx._invalidate()

PythonCustomOpAdapter::DeclareLaunchArgs(ctx)
  -> callbacks.declare_launch_args(holder, ctx)
  -> _borrow_annotated_args_context(ctx_handle)
  -> ge.custom_op._bridge.call_declare_launch_args(instance_id, ir_meta, py_ctx)
  -> user_op.declare_launch_args(*inputs, *outputs, **attrs)
  -> py_ctx._invalidate()
```

### 6.4 Modifications to Submodules

- `api/python/ge/ge/custom_op/`: Added the Python custom op API, registry, bootstrap, bridge helper, a separate schema-callback signature-validation module, and native context binding.
- `api/python/ge/ge/runtime/`: Provides runtime tensor/shape/format types for reuse by the custom op context.
- `runtime/custom_op/`: Added the Python bridge loader and adapter while retaining bridge C ABI v1. The adapter forwards Eager and AnnotatedArgs callbacks, and the bridge obtains and caches canonical IR through the public run-package API.
- `inc/graph_metadef/graph/custom_op/`: Added capability and cast helper.
- GE initialization entry: Ensures the Python runtime is attempted to be ready before `LoadCustomOps()`. On failure, a warning is logged and execution continues. The Python custom op loader performs a hard fail only when Python entries actually exist.
- `compiler/graph/manager/graph_manager.cc`: `PreRun()` idempotently loads Python custom ops before refreshing ops kernel information.

### 6.5 Error Handling

#### System Errors

- Python runtime not loaded or not initialized: `LoadPythonCustomOps()` returns failure when a Python custom op entry exists.
- Bridge/native artifact missing or ABI mismatch: Warning is logged for each candidate. Failure is returned after all candidates fail.
- Holder creation failure: The adapter is deemed invalid and the execution path fails.
- Context query, output allocation, or workspace allocation failure: The native binding raises `RuntimeError`.
- Failure to collect canonical IR required by a schema-bound descriptor causes descriptor loading/registration to fail before any holder or callback context is created.
- A schema-bound `execute` or `declare_launch_args` signature that does not match canonical IR causes `validate_op_impl_descriptor` to raise `TypeError` during descriptor loading/registration; no business callback runs.
- A non-`None` `declare_launch_args` return, reuse of a consumed `AnnotatedKernelArgs`, or an out-of-range flattened input/output instance index causes the Python bridge/native binding to raise and terminate compilation.

#### Interface Errors

- `op_type` is not a string or is an empty string: `register_op_impl` raises `TypeError`.
- The `register_op` `op_type`, signature annotations, attribute defaults, or `mutates_args` are invalid: `TypeError` or `ValueError` is raised.
- The decorated object is not a class or is an abstract class: `TypeError` is raised.
- The implementation class provides no supported callable method: `TypeError` is raised and the message lists the supported methods.
- Duplicate `op_type` or `descriptor_key`: `ValueError` is raised.
- A schema-bound `execute` has no canonical IR: descriptor loading/registration fails before a callback context is created.
- `get_execute_ctx()` is called outside a schema-bound callback: `RuntimeError` is raised.
- `get_declare_launch_args_ctx()` is called outside its callback: `RuntimeError` is raised.
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

- Existing Python user implementations based on `EagerExecuteOp` inheritance and `execute(ctx)` continue through the original path. Capability reflection only relaxes registration requirements and does not change the legacy argument or lifecycle semantics.
- The original C++ custom op `dynamic_cast` semantics are preserved through `CustomOpCast<T>()` fallback for regular C++ ops.
- The OM format is not changed. Old OMs are still loaded according to the original custom op partition and registry logic on new versions.
- New OMs do not carry Python implementations. The Python custom op execution capability cannot be assumed to be reproducible on older versions.
- New OMs carry `_custom_task_args_mode` and AnnotatedArgs `args_format` rather than the Python implementation. A new runtime uses the explicit mode as the source of truth; only an old OM without the attribute falls back to registry lookup and then a non-empty `args_format`.
- The Python custom op depends on matching versions of `ge_py`, bridge/native SO, and the Python ABI in the runtime environment.
- `ASCEND_CUSTOM_OPP_PATH` is an existing environment variable. Adding Python file and package recognition does not affect C++ OPP paths that have no Python entries.

## 9. DT Design

### 9.1 Test Boundaries

- Python API test entries: `ge.custom_op`, `ge.custom_op.proto`, `ge.custom_op._bridge`, `ge.custom_op.bootstrap`.
- Native context test entries: Eager/AnnotatedArgs borrowed contexts, `AnnotatedKernelArgs`, and launch-info methods.
- C++ test entries: `CustomOpCast<T>`, `PythonCustomOpAdapter`, `AnnotatedKernelArgs`, `CustomTaskInfo`, `LoadPythonCustomOps()`, `LoadCustomOps()`/`UnloadCustomOps()`, and `ShutdownCustomOpsForProcess()`.
- End-to-end sample entry: `examples/custom_op/annotated_args_refresh_add_custom/python/run.sh`.

### 9.2 Test Design

| Test Category | Key Test Items | Test Method | Case Type |
|---------------|----------------|-------------|-----------|
| Function | Eager/AnnotatedArgs capability reflection for plain classes, compatibility bases, inherited methods, `staticmethod`, and `classmethod`, plus invalid registration | Python pytest | UT |
| Function | Prototype signature parsing, defaults, `mutates_args`, idempotent registration, and conflicts | Python pytest | UT |
| Function | Legacy `execute(ctx)` context pass-through; schema-bound required/optional/dynamic input and typed attribute assembly | Python pytest fake context | UT |
| Function | Schema-bound `execute` input/attribute signature validation during descriptor loading, output/return compatibility, and no repeated runtime validation | Python pytest fake context | UT |
| Function | Schema-bound `declare_launch_args` signature validation during descriptor loading, plus runtime input/output and attribute assembly, return validation, flattened instance indices, and builder consumption | Python pytest fake/native context | UT |
| Function | `get_execute_ctx()` / `get_declare_launch_args_ctx()` access in callbacks, exception cleanup, and nested invocation restoration | Python pytest | UT |
| Function | Bridge descriptor retrieval, holder creation/destruction, non-callable method rejection, and context invalidation | Python pytest | UT |
| Function | Canonical IR lookup/cache, bridge ABI v1, and adapter execute/declare forwarding | C++ gtest | UT |
| Function | Capability bitmask and `CustomOpCast<T>()` behavior | C++ gtest | UT |
| Function | Loader skips when no Python entry exists and loads the bridge when entries exist | C++ gtest / stub | UT |
| Compatibility | C++ custom op bare capability inheritance still casts correctly | C++ gtest | UT |
| Feature cross | Ops kernel info is refreshed after online PreRun loading | GE graph execution related tests | UT/ST |
| Sample | Python custom op legacy/schema-bound graph execution plus AnnotatedArgs address refresh after offline compilation without a Python runtime environment | `annotated_args_refresh_add_custom/python` | ST/hardware |

### 9.3 Test Framework Design

- Python UT is placed in `tests/ge/ut/ge/graph/pyge_tests/` and uses fake contexts to avoid device dependencies.
- C++ UT uses the existing GE gtest framework and stubs external dependencies of the runtime registry and loader.
- ST reuses the custom op sample directory and verifies kernel load, graph build, session run, and output metadata in a real CANN environment.

## 10. Design Document Check Results

- [x] Cross-feature impact analysis: Static shape, dynamic shape, dynamic shape static subgraph, offline atc, and online framework adaptation scenarios have been analyzed according to `cross_feature_check.md`.
- [x] Key feature design principles: `rt2_runtime.md`, `known_shape_runtime.md`, and `graph_metadef.md` have been loaded. The solution does not modify RT2 lowering data, DavinciModel interfaces, or graph infrastructure semantics.
- [x] Template section coverage: Introduction, general overview, functional requirements, non-functional requirements, performance, interface design, software design, security check, compatibility check, and DT design have been covered.
