# GE Pass Python Implementation V1 Design Document

## 1. Background

GE currently has two types of basic capabilities directly related to this requirement:

- The existing custom pass loading chain already exists. GE discovers and `dlopen`-s pass libraries through `opp/vendors/*/custom_fusion_passes/*.so`.
- The Python side already has `ge.es` graph construction capability and `ge.graph` basic graph interfaces.

The goal of this design is to introduce formal Python pass development capability without overturning the existing GE pass execution framework, so that users can both quickly develop and validate locally and distribute passes as standard Python packages to teams.

## 2. Goals and Scope

### 2.1 Goals

- Support users in developing GE passes using Python.
- Reuse the existing GE pass execution chain without adding a second pass scheduling framework.
- Reuse the existing `ge.es` Python graph construction capability without redesigning Python ES.
- First support environment variable-driven development mode integration, then supplement release mode auto-discovery later.
- Reserve extension points for subsequent Python ATC entry to reuse the same pass registration and discovery protocol.

### 2.2 V1 Scope

- Support three types of passes:
- `FusionBasePass`
- `PatternFusionPass`
- `DecomposePass`

- The bring-up and independent bridge separation phase still uses `FusionBasePass` regression as the minimum verification chain; after formal wrapper implementation, the first complete acceptance target shifts to `PatternFusionPass`, with `DecomposePass` added afterwards, but still within V1 scope of this design document.

- The current phase discovery mechanism is first converged to:
- Environment variable `ASCEND_GE_PY_PASS_PATH`

- Subsequent phase will add:
- `entry_points(group="ge.passes.plugins")`

- Current code has already converged to the environment variable main path, with `entry_points` as a subsequent phase capability supplement.

- Complete the minimum graph interface capabilities required for Python pass authoring.
- The current repository directly produces the `ge_py` main wheel and multi-version native sub-wheels.
- The existing 9 samples in `examples/fusion_pass` are planned to have Python equivalent versions provided.
- `REGISTER_CUSTOM_PASS` needs to be supported, but is not a first-batch main chain target; it is placed in the subsequent extension phase based on the same bridge / registry / session mechanism reuse.
- Current priority rectification items:
  - Discovery mechanism first converged to environment variable `ASCEND_GE_PY_PASS_PATH`
  - `entry_points` to be added later
  - `python_pass_bootstrap_test.py` migrated to `tests/ge/ut/ge/graph/pyge_tests/` and connected to the current frontend script
  - New file year uniformly uses `2026`; existing old files do not have their year batch-changed
  - First priority targets "`FusionBasePass` formal sample passing end-to-end through environment variable"; all capabilities involved in this goal must be formally completed; capabilities not involved can be deferred
  - After Phase 2 closure, the formal Python form of `PassContext` / `MatchResult` / `Pattern` / `PatternMatcherConfig` uniformly depends directly on `_ge_pass_native.so`, no longer retaining the bring-up phase compatibility shim

### 2.3 Non-Goals

- V1 does not force users to package passes into whl.
- V1 first batch does not take the legacy `REGISTER_CUSTOM_PASS` system as the main Python implementation target; it prioritizes coverage of the three pass types in the `PassRegistry` system. However, the architecture and documentation need to reserve subsequent integration capability.
- V1 does not create a second Python-only pass executor.

## 3. User Experience Design

### 3.1 Development Mode

After users install the `ge_py` provided by CANN, they only need to write ordinary `.py` files or ordinary Python packages, and tell GE/ATC through environment variables:

- `ASCEND_GE_PY_PASS_PATH=/abs/path1:/abs/path2`

The current phase focuses on stabilizing this path first. It does not require users to write their own wheel packaging logic, nor does it require users to understand `entry_points`.

### 3.2 Release Mode (Subsequent Phase)

When users need team sharing, version freezing, and auto-discovery, they can package passes into independent Python packages and declare:

- `entry_points = {"ge.passes.plugins": [...]}`

GE automatically discovers installed pass plugin packages at runtime.

### 3.3 Notes

The current phase environment variable is not a fallback, but the main path:

- `ASCEND_GE_PY_PASS_PATH=/abs/path1:/abs/path2`

`entry_points` auto-discovery will be added later.

## 4. Overall Architecture

### 4.1 Architecture Principles

- GE collects pass plugin loading through a unified upper-level loader during the initialization phase.
- Legacy custom passes continue to use the existing `.so + dlopen` mechanism.
- From a long-term productization and extensibility perspective, the Python pass bridge should be designed as an "independent internal bridge `.so`" from the beginning, rather than being directly compiled into `ge_compiler.so`.
- This still does not adopt the "bridge `.so` discovered and loaded as a pass plugin by `custom_fusion_passes`" approach; the recommendation is a private bridge `.so` explicitly loaded by the GE internal loader.
- The design goal is to keep `ge_compiler.so` as Python ABI neutral as possible, only retaining the stable pass runtime, registry, and adapter protocols; all logic directly depending on `Python.h` / `pybind11` / `libpython` should be converged into a replaceable independent bridge `.so`.
- This way, whether going through pre-compilation or fallback codegen, the replacement target is the independent bridge `.so`, not the `ge_compiler.so` in the run package.
- The Python side uniformly manages plugin discovery, module import, and the registry through `ge.passes.bootstrap`.
- The C++ side only cares about "getting executable pass descriptors and registering them to PassRegistry", not about where user Python files specifically are.

### 4.2 Core Components

- `PassPluginLoader`
- Unified pass plugin loading entry at the compiler layer
- Internally uniformly calls legacy `CustomPassHelper::Load()` and Python pass registration logic
- Maintains "one call entry closure", while not putting Python logic back into `graph_metadef/register`

- `ge.passes.bootstrap`
- Python-side unified discovery entry
- Currently prioritizes environment variable discovery, with `entry_points` to be added later

- `ge.passes.registry`
- Python-side registry
- Responsible for storing pass metadata, class objects, stages, types, and additional parameters

- `ge.passes._bridge`
- Protocol layer between Python and the C++ bridge
- Responsible for normalizing Python registry objects into C++ consumable data structures

- `ge.passes.runtime`
- Python-side native artifact runtime management entry
- Responsible for prebuilt artifact selection, fallback codegen triggering, and `_ge_pass_native.so` loading

- `_ge_pass_native`
- Python helper module exported by `PYBIND11_MODULE`
- Only carries `Graph` / `PassContext` / `MatchResult` and other native-backed wrappers and helpers
- Does not carry `FusionBasePass` / `PatternFusionPass` / `DecomposePass` user-inheritable pass base classes

- C++ pass adapter
- Provides corresponding C++ adapter classes for the three types of Python passes
- Calls back Python object methods in the adapter classes

- Independent bridge `.so` loaded as a pass plugin via `dlopen`
- The current main approach has explicitly abandoned this
- Any description in the document based on "adding a new bridge `.so` discovered by `custom_fusion_passes`" should be understood as "unified loader + private internal bridge `.so`", not a new pass plugin discovery chain

- Private internal bridge `.so`
- This is the recommended formal direction in this design, not an optional optimization
- It is not a pass plugin and does not participate in `custom_fusion_passes` discovery; it carries complete Python version-sensitive bridge logic, including interpreter initialization, GIL, object conversion, exception translation, and Python callbacks
- During formal delivery, it forms the same bridge artifact set with `_ge_pass_native.so`; both need to be included in pre-compilation and fallback management

### 4.3 Native Binding Strategy

V1 adopts a "two-layer binding" strategy, rather than migrating all Python interfaces to the same binding method at once:

- `ge.graph` / `ge.es`
- Continue to reuse the existing C wrapper + `ctypes` approach in the repository
- This minimizes changes and allows prioritizing reuse of existing Python graph interfaces and eager-style graph construction capabilities

- Python pass bridge / adapter layer
- Uses `pybind11` as the core implementation strategy
- The reason is that this part needs to more naturally handle `MatchResult` wrapping, Python object lifecycle, exception translation, and GIL management

- Version release strategy
- The main wheel is responsible for Python code, discovery logic, and runtime selection logic
- `cp39-cp314` provides pre-compiled `pybind11` native sub-wheels
- When no pre-compiled version is matched, runtime fallback codegen serves as the final fallback

That is, V1 is not "full pybind migration", but "graph interfaces continue with the existing ctypes/C wrapper, pass bridge adopts pybind11".

### 4.4 pybind11 Usage Method Selection

`pybind11` has two typical usage methods in this solution:

- embed mode
- The C++ process initializes the Python interpreter, `import`-s Python modules, and calls back Python objects

- extension mode
- Exports C++ capabilities as Python-directly `import`-able native modules through `PYBIND11_MODULE`

The existing `compiler/graph/fusion/pass/python_fusion_base_pass_pybind_bridge.cc` uses embed mode, not extension mode. The reasons are:

- The current main direction is "GE compiler calls Python pass", not "Python actively imports a C++ pass runtime then reversely drives the compiler"
- `FusionBasePass` in the first stage only needs C++ to safely create Python objects and call their `run()`, without first exposing C++ base classes to Python for inheritance
- embed mode can first reuse the existing `ge.passes.bootstrap / registry / _bridge` pure Python organization, reducing first-batch implementation cost

Therefore, the current bridge file does not have a `PYBIND11_MODULE` macro. This is not a missing feature, but an intentional mode difference.

However, from a long-term design perspective, the embed bridge should not continue to be directly compiled into `ge_compiler.so`. The current workspace has already split it into the independent `libge_python_pass_bridge.so`, which is also the formal boundary that should be continuously maintained. A more reasonable form is:

- `ge_compiler.so`
- Only holds the stable loader, descriptor / adapter protocols, and minimal C/C++ interaction surface

- Independent internal bridge `.so`
- Adopts whichever of embed or extension is more suitable for the specific implementation
- Undertakes all Python version-sensitive native logic

- `_ge_pass_native.so`
- Serves as a Python-directly `import`-able helper extension
- Provides native-backed wrappers and helpers for the Python layer, but does not define user-inheritable pass base classes

What needs to be further emphasized is that the current solution has two clear boundaries that cannot be mixed:

- User pass definition layer
- Continues to maintain pure Python form; users inherit `ge.passes.base.FusionBasePass` / `PatternFusionPass` / `DecomposePass`

- native helper / wrapper layer
- `_ge_pass_native.so` provides `Graph` / `PassContext` / `MatchResult` and other wrappers
- `libge_python_pass_bridge.so` provides embed path runtime bridging

That is, this solution does not require or recommend exposing C++ `FusionBasePass` / `PatternFusionPass` base classes to Python for inheritance through `PYBIND11_MODULE`.

What needs special explanation is that Python version sensitivity is not only present with `PYBIND11_MODULE`. As long as native code directly depends on:

- `Python.h`
- `pybind11`
- `libpython`

Whether embed or extension, it naturally has binary coupling with the Python minor version. `PYBIND11_MODULE` is just the export entry for extension mode, not the root cause of version sensitivity.

#### 4.4.1 Loading Location Differences Between `_ge_pass_native.so` and `libge_python_pass_bridge.so`

Although these two artifacts both belong to the Python pass bridge artifact set, they are on two different loading paths:

- `_ge_pass_native.so`
  - As a Python extension module, loaded by the Python interpreter through `import ge.passes._ge_pass_native`
  - When it enters the process, the host interpreter already exists; therefore, on Linux, `libpython.so` does not need to be explicitly written into the ELF `NEEDED`
  - This type of extension typically resolves `Py_*` / `PyObject_*` symbols directly to the current interpreter process at import time

- `libge_python_pass_bridge.so`
  - As an embed bridge from the GE internal loader perspective, explicitly `dlopen`-ed by the `ge_compiler` side
  - It is responsible for interpreter initialization, interpreter reuse, GIL management, Python module import, and exception translation
  - Because it cannot assume a Python interpreter already exists in the process, it needs to explicitly link `libpython.so`

Therefore, "`_ge_pass_native.so` does not explicitly depend on `libpython.so` in ELF" only indicates that its loading context is different; it does not mean it is naturally easier to reuse across versions than the embed bridge.

#### 4.4.2 ABI Compatibility Boundary: Whether `NEEDED libpython` Is Explicit Is Not the Same as Whether Python Version Constraints Apply

Two levels need to be distinguished:

- Whether `NEEDED libpythonX.Y.so` explicitly appears at the ELF level
- Whether the artifact is still bound to a certain CPython minor version C API / ABI

For the current solution:

- `_ge_pass_native.so`
  - Even if `libpython` is not explicitly carried in ELF
  - It is still an extension built against the `Python.h` / `pybind11` headers corresponding to the current `HI_PYTHON`
  - As long as the `Py_LIMITED_API` / `abi3` approach is not adopted, it is still bound to the specific CPython minor version ABI by default

- `libge_python_pass_bridge.so`
  - Because it uses the embed path, this binding is more directly manifested as explicit `NEEDED libpythonX.Y.so`
  - Its runtime constraints are also exposed earlier and more explicitly

Therefore:

- `_ge_pass_native.so` not explicitly `NEEDED libpython` does not mean it can safely be reused across multiple Python minor versions
- `libge_python_pass_bridge.so` has more explicit and harder restrictions, but both are Python version-sensitive artifacts in the ABI sense
- If cross-version reuse capability needs to be expanded in the future, the direction should be evaluating `abi3` / limited API, rather than inferring compatibility solely based on "the extension does not explicitly link `libpython`"

In conclusion, these two artifacts should still be treated as the same Python-version-sensitive artifact set, entering pre-compilation and fallback management together.

#### 4.4.3 Why Pass Base Classes Continue to Remain Pure Python Rather Than Being Migrated to Native Wrappers

The current solution deliberately separates the "user-inheritable pass contract layer" from the "lifecycle-sensitive native helper layer":

- User-inheritable pass contract layer
  - `FusionBasePass`
  - `PatternFusionPass`
  - `DecomposePass`
  - Remains pure Python

- Lifecycle-sensitive native helper layer
  - `PassContext`
  - `MatchResult`
  - `Pattern`
  - `PatternMatcherConfig`
  - `release_graph()` and other helpers
  - Converges to `_ge_pass_native.so`

The reasons for this layering are:

- Pass base classes are essentially user DSL / contracts
- If `FusionBasePass` / `PatternFusionPass` were also migrated to `_ge_pass_native.so`
  - It would pull user-side APIs into the Python ABI sensitive surface
  - It would increase import, release, and environment assembly constraints
  - It would reduce the evolvability of Python layer protocols, error messages, type constraints, and registration logic

- More importantly, doing so would not eliminate `libge_python_pass_bridge.so`
  - Because the "GE calling Python pass from C++" embed chain would still exist
  - That is, making pass base classes native would only expand the version-sensitive surface without reducing the need for the bridge

Therefore, the formal boundary of this solution remains:

- `libge_python_pass_bridge.so`
  - Responsible for the C++ perspective embed bridge

- `_ge_pass_native.so`
  - Responsible for the Python perspective native-backed wrappers and helpers

- `ge.passes.base` / `registry` / `bootstrap` / `_bridge`
  - Continue to carry the pure Python parts of the user DSL, registration protocol, and bridging protocol

This is also why the current solution recommends "base classes pure Python + helpers native" rather than "making pass base classes into wrapper modules too".

Additionally, `pybind_options` in the current build is only a CMake `INTERFACE` target used to suppress some compilation warnings introduced by pybind headers; it does not carry runtime logic and is not an independent pybind dependency artifact.

## 5. Runtime Chain

### 5.1 Initialization Phase

1. GE starts and triggers the unified entry `LoadPassPlugins()`
2. The loader internally first calls legacy `CustomPassHelper::Load()`
3. The loader determines whether the Python pass discovery chain needs to be started solely through the environment variable `ASCEND_GE_PY_PASS_PATH`; `options` no longer carries user pass routing parameters such as `ge.py_path / py_path`
4. The bridge loader inside `ge_compiler.so` calls `RegisterPythonFusionBasePassesFromPlugin()`
5. The bridge loader resolves the target Python runtime key
6. The bridge loader selects and loads `libge_python_pass_bridge.so` in the order of "prebuilt artifact, runtime fallback codegen"
7. The bridge loader obtains the stable entry exported by the bridge through `GeGetPythonFusionBasePassBridgeApi()` and passes the registrar callback to the bridge
8. `libge_python_pass_bridge.so` initializes or reuses the Python runtime and imports `ge.passes._bridge`
9. The bridge synchronizes `ASCEND_GE_PY_PASS_PATH` from the current process environment to Python `os.environ`, then calls `load_and_get_pass_descriptors()`
10. bootstrap discovers and imports user pass modules
11. User pass modules register passes to `ge.passes.registry` through decorators
12. The bridge reads the registry and dynamically registers descriptors back to `PassRegistry` through the registrar callback

The corresponding call relationship can be simplified as the following timeline:

```text
PassPluginLoader / ge_compiler.so
  -> python_fusion_base_pass_bridge_loader.cc
  -> dlopen(libge_python_pass_bridge.so)
  -> GeGetPythonFusionBasePassBridgeApi()
  -> register_fusion_base_passes(registrar)
  -> ge.passes._bridge.load_and_get_pass_descriptors()
  -> ge.passes.bootstrap.load_pass_plugins()
  -> registrar.register_pass(pass_desc, callbacks)
  -> RegisterPythonFusionBasePass(...)
  -> PassRegistry / runtime registry
```

Where:

- `registrar` is constructed by the loader, representing "how to register descriptors back to the compiler"
- `bridge` is responsible for discovering Python passes, and after getting descriptors, calls back `registrar`
- `RegisterPythonFusionBasePass(...)` is the actual landing point that attaches descriptors, callbacks, and creators back to the compiler-side registration center

In the early stage of Phase 1 split, `libge_python_pass_bridge.so` can first depend only on the pure Python `bootstrap / _bridge` protocol to complete minimum integration debugging; the formal approach after Phase 2 closure requires `_ge_pass_native.so` to be in place simultaneously, and the bridge and Python API run based on the same set of native helpers by default.

### 5.2 Execution Phase

1. `FusionPassExecutor` gets passes from `PassRegistry` according to the existing flow
2. If a pass is a Python pass, the corresponding C++ adapter instance is actually created
3. The adapter calls back the Python pass object during `Run` or related phases
4. The Python pass reads/writes the graph and builds the replacement graph through `ge.graph` and `ge.es` interfaces
5. The return value maps to GE `Status`

## 6. Python Public Interface Design

### 6.1 Package Structure

Add a new `ge.passes` package, providing the following public interfaces:

- `FusionBasePass`
- `PatternFusionPass`
- `DecomposePass`
- `register_fusion_pass`
- `register_decompose_pass`
- `PassStage`
- `PassContext`
- `Pattern`
- `NodeIo`
- `MatchResult`
- `PatternMatcherConfig`
- `PatternMatcherConfigBuilder`
- `create_pattern`
- `create_replacement`
- `FuseCheckResult`
- `can_fuse`
- `report_fuse`
- `load_pass_plugins`
- `get_registered_passes`

### 6.2 Method Style

Default to Python style naming.

### 6.3 Registration Interface

The suggested form is as follows:

```python
from ge.passes import FusionBasePass, PassStage, register_fusion_pass

@register_fusion_pass(name="ConvFormatPass", stage=PassStage.BEFORE_INFER_SHAPE)
class ConvFormatPass(FusionBasePass):
    def run(self, graph, context):
        return 0
```

```python
from ge.passes import DecomposePass, PassStage, register_decompose_pass

@register_decompose_pass(
    name="DecomposeGroupedConv",
    stage=PassStage.AFTER_INFER_SHAPE,
    op_types=["Conv2D"],
)
class DecomposeGroupedConv(DecomposePass):
    def meet_requirements(self, node):
        return True

    def replacement(self, node):
        ...
```

### 6.4 Python-side Return Value Convention

- `run` returns `StatusLike`
- `meet_requirements` returns `bool`
- `patterns` returns `list[Pattern | Graph]`
- `replacement` returns `Graph`
- If the user wishes to skip the current match, the user must return `False` in `meet_requirements`; returning `None` through `replacement` to express "abandon replacement" is not supported

Where `StatusLike` in the Python layer uniformly converts to GE `Status`.

## 7. Discovery Mechanism Design

### 7.1 Unified Entry

The Python side uniformly provides:

- `ge.passes.bootstrap.load_pass_plugins()`
- `ge.passes.bootstrap.get_registered_passes()`

Before each round of loading, the C++ side refreshes `ASCEND_GE_PY_PASS_PATH` in Python `os.environ` according to the current process environment, to prevent the environment cache in the resident Python interpreter from affecting the next round of pass discovery.

### 7.2 Discovery Priority

The current phase priority converges to:

1. Environment variable `ASCEND_GE_PY_PASS_PATH`

Subsequent phase will add:

2. `entry_points(group="ge.passes.plugins")`

### 7.3 Environment Variable Mode

- `ASCEND_GE_PY_PASS_PATH` supports multiple directories, separated by `:`
- Directories allow single-file modules or ordinary Python packages
- `bootstrap` is responsible for temporarily adding these directories to `sys.path`

### 7.4 entry_points Mode (Subsequent Phase)

- The group is fixed as `ge.passes.plugins`
- The value can point to a module path, or return a callable that provides a module
- After the module is imported, registration is completed through decorators

## 8. Three Types of Pass Bridge Design

### 8.1 FusionBasePass

The most direct type. The C++ adapter calls the Python object:

- `run(graph, context)`
- The return value is constrained to `None` / `bool` / `int` three types of status values
- In the formal pass contract, `context` is always `PassContext`
- Only `_bridge.py`'s direct bridge/pytest auxiliary entry allows passing `None`

This type is prioritized for connection, serving as the minimum closure of the entire chain.

### 8.2 PatternFusionPass

This type continues to reuse the existing C++ PatternMatcher mechanism. The Python side is only responsible for:

- Providing the pattern graph
- Judging whether conditions are satisfied based on `MatchResult`
- Constructing the replacement graph

This means the C++ side needs a Python adapter inheriting `PatternFusionPass`, calling back to Python at the following points:

- `Patterns()`
- `MeetRequirements()`
- `Replacement()`

There is a clear scheme constraint here:

- The Python user class is not required to directly inherit a C++ `PatternFusionPass` exposed through `PYBIND11_MODULE`
- Users continue inheriting the pure Python `ge.passes.base.PatternFusionPass`
- The responsibility of reusing the C++ base class public `Run()` flow is placed on `PythonPatternFusionPassAdapter`, not on the Python user class
- Python subclasses are prohibited from overriding `run()`; if mistakenly overridden, the base class directly throws `TypeError` at the class definition phase, avoiding the ambiguity of "implemented but never called"

The recommended form is:

- `PythonPatternFusionPassAdapter : public PatternFusionPass`
- The adapter overrides `Patterns()` / `MeetRequirements()` / `Replacement()`
- The override functions internally call back to the Python pass instance
- `Run()` directly reuses the existing C++ `PatternFusionPass::Run()`

The reasons for choosing this approach are:

- It maximizes reuse of the existing C++ `PatternMatcher`, rewrite, statistics, and error handling logic
- It does not force the Python user environment to first successfully import a native C++ base class module, maintaining the usability of the `ge.passes` pure Python API
- It keeps `FusionBasePass`, `PatternFusionPass`, and `DecomposePass` in a unified style on the user side
- It converges Python version sensitivity issues into the adapter / wrapper / native bridge layer as much as possible, rather than spreading them to the user pass base class definition layer

### 8.3 DecomposePass

This type continues to reuse the existing `DecomposePass` semantics. The Python side is only responsible for:

- `MeetRequirements(const GNode &)`
- `Replacement(const GNode &)`

`op_types` information needs to be retained during construction.

Same as `PatternFusionPass`, the Python user class does not directly take over the `Run()` main flow, but only implements hooks:

- `meet_requirements(node) -> bool`
- `replacement(node) -> Graph`

The additional contract constraints here need to be explicitly hardcoded in the Python base class:

- Python subclasses are prohibited from overriding `run()`; if mistakenly overridden, the base class directly throws `TypeError` at the class definition phase
- `replacement()` must return a replacement Graph
- If the user wishes to skip the current node, the user must return `False` in `meet_requirements()`; returning `None` through `replacement()` is not supported

Same as `PatternFusionPass`, the Python user class is not required to directly inherit the C++ `DecomposePass` exposed through pybind. The recommended form is:

- `PythonDecomposePassAdapter : public DecomposePass`
- The adapter reuses `DecomposePass::Run()` on the C++ side
- The adapter overrides `MeetRequirements()` / `Replacement()` and delegates to the Python pass instance

This retains the existing C++ `DecomposePass` main flow semantics while avoiding exposing construction parameters, `op_types`, and Python version-sensitive logic directly to the Python base class inheritance system.

### 8.4 Creator and Context Acquisition Design

The current `CreateFusionPassFn` is a naked function pointer:

- `using CreateFusionPassFn = FusionBasePass *(*)();`

V1 does not recommend directly changing it to `std::function<FusionBasePass *()>`. There are two reasons:

- Python passes come from dynamic registration of the bridge `.so`; if the creator holds a capturable lambda, the destruction chain easily couples with `dlclose` order
- Once `PassRegistry` or other global objects are destructed after the bridge `.so` has been unloaded, `std::function` internal object destruction may access already unloaded code, posing a `coredump` risk
The current approach no longer uses "independent bridge `.so` `dlclose` risk" as the main reason. The more accurate reasons are:

- The existing creator ABI is still a parameterless naked function pointer; directly changing to capturable objects would expand the impact scope
- `std::function` would couple runtime routing information, object destruction, and call paths into the creator itself, which is detrimental to maintaining the layering of "creator only does minimal routing, runtime resources placed in bridge/runtime registry"
- Retaining naked function pointer + TLS routing context remains the approach with the smallest impact scope and the most reliable solution

Therefore, a more prudent approach is recommended here:

- Retain `CreateFusionPassFn` as a naked function pointer
- Add a "creation-phase TLS context"
- Python pass runtime objects and metadata are placed in the process-level registry held by the bridge

The "identification information" mentioned here is not the Python runtime context itself, but rather **the stable key and metadata used for registry lookup**, for example:

- `pass_name`
- `pass_kind`, namely `fusion` / `pattern` / `decompose`
- `stage`
- Python module name
- Python class full name
- `op_types` in `decompose` scenarios

This information belongs to registration-phase static information, not execution-phase context. The actual Python interpreter state, module objects, and pass instances are not carried in the creator, but are uniformly managed in the bridge's global registry.

The recommended implementation approach is as follows:

1. Set the TLS creation context at the position that consumes `create_fn()`
- The final recommendation is to pass in `descriptor_key`
- When some existing call points can more easily obtain `pass_name` in the short term, a transitional mapping can be used first, but it is not recommended to fix `pass_name` as the final creator routing key

2. Provide a small number of generic creator functions for the three types of Python passes
- `CreatePythonFusionPass()`
- `CreatePythonPatternPass()`
- `CreatePythonDecomposePass()`

3. The generic creator reads the current `descriptor_key` from TLS
- Then finds the corresponding descriptor from the bridge registry accordingly
- Then constructs the corresponding adapter

4. The adapter obtains the Python pass instance or its holder from the bridge registry during execution

The advantages of this design are:

- No need to put Python object context into `create_fn`
- Does not introduce `std::function` destruction order risk
- Better compatibility with the current GE creator call approach

### 8.5 TLS Creation Context Refinement

The current implementation needs to retain a lightweight creation-phase TLS context. Its necessity is not "to pass more information", but because the current `FusionPassRegistrationData::CreatePassFn` is still a parameterless creator, and multiple descriptors on the Python pass side share the same adapter factory; without this TLS, the upper-layer `PassRegistry::CreatePass()` knows "which pass is being created", but the shared factory does not know which Python descriptor to bind to.

The current code form can be expressed as:

```cpp
struct PythonPassCreateContext {
  std::string descriptor_key;
  std::string pass_name;
  PythonPassKind kind;
};
```

And provides the following auxiliary facilities:

- `SetCurrentPythonPassCreateContext(descriptor_key)`
- `GetCurrentPythonPassCreateContext()`
- `ClearCurrentPythonPassCreateContext()`
- RAII scope guard, for example `PythonPassCreateScope`

The usage is as follows:

1. Before calling `create_fn()`, the caller sets the current `descriptor_key`
2. The generic creator reads `descriptor_key` from TLS
3. The generic creator then finds the corresponding descriptor in the bridge registry
4. Constructs the corresponding adapter
5. Automatically cleans up TLS after `create_fn()` returns

Where:

- `descriptor_key` is the actual routing primary key
- `pass_name` / `kind` currently mainly serves consistency verification, avoiding the shared factory from mistakenly binding to the wrong descriptor

If the call chain is further unified subsequently, it is still recommended to converge `PythonPassCreateContext` to "retaining only `descriptor_key` as the minimum necessary field", avoiding state duplication and keeping consistency with the runtime descriptor / runtime entry primary key model.

V1 recommends connecting this scope at the following call points:

- `compiler/graph/fusion/pass/fusion_pass_executor.cc`
- `FusionPassExecutor::InitPassesIfNeed`

Where:

- Both the first batch and subsequent main chains only cover `FusionPassExecutor`
- `graph_fusion.cc` is not within the subsequent support scope of this scheme

This avoids spreading the Python-ification scope to the legacy compatibility chain, while ensuring the creator / TLS / descriptor scheme evolves around the main chain in a closed loop.

### 8.6 Bridge Process-level Registry Refinement

The bridge internally recommends maintaining two levels of registration information, rather than continuing to let a single `holder_key` simultaneously serve as both "static identity" and "runtime instance" semantics. Logically it can be split into two parts:

- `PythonPassDescriptor`
- Registration-phase static information

- `PythonPassInstanceHolder`
- Execution-phase instance information
- Python pass instance
- Runtime state
- Exception state
- session / instance association information

`PythonPassDescriptor` is recommended to contain at least:

- `pass_name`
- `pass_kind`
- `stage`
- `module_name`
- `class_qualname`
- `op_types`
- `descriptor_key`

Where:

- `descriptor_key`
- Represents the static key of "which pass class this is"
- Recommended format is `module_name + class_qualname + pass_name`
- Used for registration deduplication, descriptor lookup, and log positioning

- `instance_id`
- Represents the dynamic key of "which runtime instance this is"
- Generated by adapter / session during creation
- Used for holder lookup, instance lifecycle management, and execution-phase isolation

The minimum implementation phase once used `holder_key` for both descriptor lookup and holder lookup. The current `FusionBasePass` has completed the split, and subsequent passes should also maintain this model:

- `descriptor_key`
- Static identity

- `instance_id`
- Dynamic instance identity

V1 recommends the registry be held and destructed only by the bridge itself, not exposed to other global singleton-held objects, avoiding cross-so lifecycle coupling.

This design deliberately does not make Python pass instances into process-level singletons. The reasons are:

- Users can more naturally treat `self` as a "temporary state container for this pass execution"
- It can avoid residual state pollution across graphs and across executions
- It can reduce lock and reentrancy requirements caused by instance sharing under multi-threaded concurrency

### 8.7 Python Pass Adapter Refinement

V1 recommends providing adapters for the three types of passes respectively:

- `PythonFusionBasePassAdapter`
- `PythonPatternFusionPassAdapter`
- `PythonDecomposePassAdapter`

Common characteristics of the three:

- Only receives `descriptor_key` or `pass_name` during construction
- Does not directly hold raw pointers to Python temporary objects during construction
- Completes descriptor binding during construction and creates an independent `instance_id` for the current adapter
- The adapter exclusively owns its Python pass instance during its lifecycle, and does not reuse long-term shared holders
- During execution, finds the holder corresponding to the current adapter in the bridge instance repository through `instance_id`
- Releases the holder through `instance_id` during destruction
- Uniformly performs GIL acquisition, exception translation, and state mapping during execution

This way, even if the adapter itself is held by GE long-term, it only depends on the holder stably managed by the bridge, and does not depend on creator closure objects.

The division of responsibilities among the three types of adapters needs further clarification:

- `PythonFusionBasePassAdapter`
- Directly overrides `Run()`, internally calls Python `run(graph, context)`

- `PythonPatternFusionPassAdapter`
- Inherits C++ `PatternFusionPass`
- Does not override the base class public `Run()` main flow
- Only overrides `Patterns()` / `MeetRequirements()` / `Replacement()` three hooks, and delegates to Python within the hooks

- `PythonDecomposePassAdapter`
- Inherits C++ `DecomposePass`
- Does not override the base class public `Run()` main flow
- Only overrides `MeetRequirements()` / `Replacement()`, and delegates to Python within the hooks

This is also the core reason why the current design does not rush to expose C++ pass base classes to Python for inheritance through `PYBIND11_MODULE`: the ones that truly need to reuse the C++ non-pure-virtual main flow are the adapters, not the Python pass classes written by users.

### 8.8 Execution-phase Session Design

To avoid imposing unnecessary restrictions on Python pass authoring, V1 recommends introducing a "one session per execution" model:

- One `Run` call of one adapter corresponds to one `PythonPassExecutionSession`
- A new Python pass instance is created within the session, and a unique `instance_id` is assigned
- Within the same session, multiple Python callbacks share the same instance
- When the session ends, the instance and its temporary wrapper objects are uniformly released

This means:

- `FusionBasePass`
- One `Run` corresponds to one Python instance

- `PatternFusionPass`
- `Patterns`, `MeetRequirements`, and `Replacement` within one `Run` share the same Python instance

- `DecomposePass`
- Processing of multiple matching nodes within one `Run` shares the same Python instance

After this design, Python users can naturally use:

- `self.xxx` as temporary cache during one execution
- Ordinary Python objects as auxiliary state
- Ordinary exceptions as failure signals

Without needing to understand bridge internal details such as "whether this instance is reused across graphs".

### 8.9 Memory Management Refinement

The V1 memory management goals are:

- Does not require Python users to manually release any bridge objects
- Does not require Python users to explicitly use interfaces such as `with`, `close()`, `release()`
- Does not allow double-free or dangling pointers triggered by users storing objects in local variables or member variables

It is recommended to handle objects at three levels separately.

#### 8.9.1 Registration-phase Objects

Registration-phase objects include:

- descriptor
- Python module objects
- Python class objects
- descriptor registry

These objects are held by the bridge registry uniformly, and the bridge layer is responsible for reference counting and cleanup. Transparent to Python users.

#### 8.9.2 Execution-phase Objects

Execution-phase objects include:

- instance holder
- Python pass instances
- `Graph` / `Node` / `MatchResult` / `NodeIo` Python wrapper objects created in callbacks
- Possible temporary `TensorDesc` / `Shape` / `Tensor` wrapper objects
- `instance_id`

These objects are all bound to the execution session, not to the global descriptor.

When the session ends:

- Python pass instance is released
- Execution-phase wrapper cache is released
- Execution-phase validity token is invalidated

#### 8.9.3 Borrowed Graph Objects

`Graph`, `Node`, `Tensor` and other objects are often borrowed views of the current GE execution graph. To ensure the Python experience does not degrade, it is recommended:

- Python wrapper objects internally hold an execution-phase owner token
- While the owner token is valid, all accesses work normally
- Once a user saves an object across sessions and accesses it again, a crash is not allowed; instead, a clear Python exception is thrown, for example:
- `RuntimeError: graph handle has expired`

The effect of this approach is:

- Does not require adding a hard restriction in documentation telling users "do not cache these objects"
- Even if users write code this way, they should get an understandable Python error, not a coredump

#### 8.9.4 TensorDesc / Shape Value Semantics

`TensorDesc` and `Shape` objects are recommended to be exposed to Python with value semantics:

- Python obtains independent objects
- Can be safely stored in local variables or `self`
- Does not depend on the original borrowed graph handle continuing to survive

This better matches Python user expectations and can also reduce dangling reference issues.

#### 8.9.5 Bridge Unloading and Destruction Order

GE provides two levels of unloading semantics, corresponding to the "end of one round of business" and "process exit" lifecycles respectively:

##### Unload -- Business-level Unloading

After one round of graph compilation completes, GE calls `UnloadPassPlugins()` to clean up the pass registration state for this round, but does not close the Python interpreter or unload the bridge so. This way the next round of business can reuse the already initialized Python runtime, avoiding the overhead of repeated initialization/finalization.

Current implementation chain:

```
UnloadPassPlugins()
  → PassPluginLoader::Unload()                              [pass_plugin_loader.cc]
    ├─ UnloadPythonFusionBasePasses()                        // Only executed when python_pass_loaded_ is true
    │   → BridgeLoader::Unload()                             [bridge_loader.cc]
    │     ├─ api_->reset_bridge_state()                      // Notify bridge to clean up Python-side state and release bridge module references
    │     ├─ ClearPythonFusionBasePassRuntimeRegistry()      // Clean up C++-side runtime registry
    │     └─ PassRegistry::ClearPythonPasses()               // Clean up C++-side pass registry
    │   python_pass_loaded_ = false
    └─ CustomPassHelper::Unload()                            // Clean up C++ custom passes
```

Unload does not touch the Python interpreter lifecycle or bridge so handle, ensuring the next `Load()` can directly reuse them.

##### Process-level Shutdown

When the process exits, GE performs complete resource release through `UnloadPassPlugins()` when the reference count drops to zero. All entry points now uniformly call `UnloadPassPlugins()`:

- `GEFinalizeV2()` -- when the online mode process ends
- `aclgrphBuildFinalize()` -- when offline compilation ends
- `GeGenerator::Finalize()` -- when the generator mode ends
- `atc main_impl` -- when ATC ends

`UnloadPassPlugins()` performs process-level cleanup when the reference count reaches zero:

```
UnloadPassPlugins()
  → PassPluginLoader::Unload()                               [pass_plugin_loader.cc]
    ├─ active_users_--
    ├─ if (active_users_ == 0):
    │   ├─ if (python_pass_loaded_):
    │   │   UnloadPythonFusionBasePasses()                   // Clean up registration state first
    │   │     → BridgeLoader::Unload()
    │   │       ├─ api_->reset_bridge_state()
    │   │       ├─ ClearPythonFusionBasePassRuntimeRegistry()
    │   │       └─ PassRegistry::ClearPythonPasses()
    │   │   python_pass_loaded_ = false
    │   │
    │   ├─ if (cpp_pass_loaded_):
    │   │   cpp_pass_loaded_ = false
    │   │   CustomPassHelper::Unload()                        // Clean up C++ custom passes
    │   │
    │   └─ if (!shutdown_done_):
    │       shutdown_done_ = true
    │       ShutdownPythonFusionBasePassesForProcess()        // Process-level Python bridge cleanup
    │         → BridgeLoader::ShutdownForProcess()            [bridge_loader.cc]
    │           ├─ if (api_ != nullptr):
    │           │   api_->shutdown_bridge()
    │           │     → PybindBridge::Shutdown()              [pybind_bridge.cc]
    │           │       ├─ ResetBridgeStateUnlocked()
    │           │       └─ if (owns_interpreter_):
    │           │           py::finalize_interpreter()
    │           │     owns_interpreter_ = false
    │           ├─ api_ = nullptr
    │           ├─ if (handle_ != nullptr):
    │           │   dlclose(handle_)
    │           │   handle_ = nullptr
    │           └─ loaded_path_.clear()
```

##### Idempotency Guarantee

Since `UnloadPassPlugins()` may be called repeatedly from multiple entry points, the entire chain guarantees idempotency through the following guards:

1. **PassPluginLoader layer** -- `active_users_` reference count: decremented on each call, unloading only occurs when it reaches zero; `shutdown_done_` flag ensures process-level cleanup only executes once
2. **BridgeLoader layer** -- `api_` / `handle_` null pointer guard: set to `nullptr` after first execution, subsequent calls skip shutdown and dlclose
3. **PybindBridge layer** -- `Py_IsInitialized()` guard: does not enter Python cleanup logic after the interpreter has been finalized; `owns_interpreter_` guard ensures only the self-initialized interpreter is finalized

##### Unloading Order Core Constraints

The current implementation follows these order principles:

1. **Clean up C++ registry first, then dlclose bridge so** -- `UnloadPythonFusionBasePasses()` first cleans up `PassRegistry` and `PythonFusionBasePassRuntimeRegistry`, then executes `ShutdownForProcess()` to perform `dlclose`. This ensures no C++ object still holds bridge-side callback function pointers during dlclose.
2. **Clean up Python objects first, then finalize interpreter** -- `PybindBridge::Shutdown()` first calls `ResetBridgeStateUnlocked()` to clean up the Python-side registry, holders, and dynamically loaded pass modules, and within reset releases the `bridge_module_` reference and calls `gc.collect()` to break circular references, and only then calls `py::finalize_interpreter()`.
3. **Finalize interpreter first, then dlclose so** -- `shutdown_bridge()` executes before `dlclose(handle_)` in `BridgeLoader::ShutdownForProcess()`, ensuring the Python interpreter is no longer running during dlclose.
4. **If the interpreter has been finalized externally** -- `Py_IsInitialized()` returns 0, the bridge skips all Python cleanup logic and only cleans up C++-side state, and does not perform `DECREF` on already released Python objects.

The priority here is to guarantee "no crash" rather than aggressively reclaiming all trailing memory. The CPython internal arena allocator may still have residual memory not reclaimed after `Py_Finalize()`; this is known CPython behavior and does not affect normal process exit.

### 8.10 Lock and GIL Strategy Refinement

The lock and GIL design goals are:

- Does not expose lock concepts to Python users
- Does not require Python pass authors to understand or manage GIL themselves
- Controls lock granularity to the minimum within the bridge, avoiding serializing the entire GE pass execution path

It is recommended to divide into three types of locks.

#### 8.10.1 Bridge Management Lock

Used to protect:

- Registry initialization
- Plugin discovery
- Holder lazy loading
- Unload / finalize state transitions

This type of lock only surrounds the bridge's own state management, not user pass logic execution.

#### 8.10.2 Execution Session Lock

Each execution session can have its own lightweight state protection, but it is not recommended to let multiple sessions share coarse-grained mutexes.

The goal is to allow:

- Different passes executing without blocking each other
- Non-Python pure C++ matching/graph modification logic continues to run along the original path

#### 8.10.3 Python GIL

Unified rules are as follows:

- Acquire GIL before entering Python
- Release GIL immediately after leaving Python
- Pure C++ graph matching, graph traversal, and data organization logic does not hold GIL

Specific strategies for the three types of passes:

- `FusionBasePass`
- Holds GIL when calling back `run`
- Releases GIL immediately after Python returns

- `PatternFusionPass`
- C++ pattern matching process does not hold GIL
- Briefly holds GIL when calling `Patterns`, `MeetRequirements`, `Replacement`

- `DecomposePass`
- C++ search for matching nodes does not hold GIL
- Briefly holds GIL when calling `meet_requirements`, `replacement`

This clearly separates Python execution and C++ execution boundaries, reducing unnecessary global serialization.

#### 8.10.4 Callback Reentry Strategy

V1 recommends supporting "multiple sessions concurrent, single session serial" by default:

- No concurrent Python callbacks within one execution session
- If different sessions are concurrently triggered by the GE upper layer, they naturally serialize into Python through GIL

For Python users this means:

- No need to write additional locks for passes because of the bridge
- If users use module-level global mutable state themselves, they still need to ensure logic correctness themselves

The bridge does not additionally restrict users from writing this way, but also does not provide automatic transaction semantics for user-owned global shared state.

### 8.11 Pythonic Experience Constraints

The V1 design principle is "converging lifecycle and concurrency complexity inside the bridge", minimizing non-Pythonic rules imposed on Python users. Specific requirements are as follows:

- Does not require users to manually manage memory
- Does not require users to manually manage locks or GIL
- Does not require users to write passes through specific context managers
- Does not require users to artificially break apart ordinary Python code to avoid reuse issues

Within what is achievable, Python users should be able to write as ordinary classes:

- Use constructors to initialize fixed configuration
- Use `self` to save temporary state within one execution
- Use ordinary Python exceptions to indicate errors
- Use ordinary return values to indicate results

The boundaries that need to be truthfully explained are only two types:

- Registration protocol boundary
- Users still need to declare passes through decorators or equivalent registration interfaces; this belongs to the framework integration protocol and is not a non-Pythonic restriction

- Expired object boundary
- If users save borrowed graph view objects long-term across executions and access them again later, they will get a Python exception rather than being silently supported indefinitely

These two types of boundaries are necessary for framework integration, but should not force users into "having to write code in non-Pythonic patterns".

### 8.12 `REGISTER_CUSTOM_PASS` Subsequent Support Design

`REGISTER_CUSTOM_PASS` needs to be supported, but it is recommended to place it in the extension phase after the three types of `PassRegistry` passes are stabilized. The reasons are:

- Its execution path differs from the `FusionPassExecutor` system, currently mostly going through `CustomPassHelper` / legacy custom pass chain
- The first batch prioritizes connecting the three types of passes, which can more quickly stabilize the common foundation of descriptor, session, bridge, holder, GIL, and exception isolation
- After the foundation is stabilized, connecting `REGISTER_CUSTOM_PASS` can significantly improve code reuse and avoid building a second Python bridge

The recommended reuse approach is as follows:

- Continue to reuse the same Python discovery mechanism
  - `ge.passes.bootstrap`
  - The current phase uses the environment variable as the main path, with `entry_points` to be added later

- Continue to reuse the same Python registry and descriptor mechanism
  - Add `legacy_custom` in `PythonPassKind`
  - Descriptor adds metadata required by legacy custom passes

- Continue to reuse the same pybind bridge
  - Does not start a second Python runtime initialization
  - Does not start a second holder / session management

- Continue to reuse the "static `descriptor_key` + dynamic `instance_id`" model
  - Avoid legacy custom passes reintroducing shared Python instance state limitations

- Add `PythonLegacyCustomPassAdapter` on the C++ side
  - Adapt the interfaces and execution entry required by `REGISTER_CUSTOM_PASS`
  - Only differs from the three types of passes at the outermost interface adaptation level

In other words, the subsequent support for `REGISTER_CUSTOM_PASS` should be:

- Reuse the same `bootstrap`
- Reuse the same `registry`
- Reuse the same `pybind bridge`
- Reuse the same `session / holder / instance_id`
- Only add a legacy custom path adapter layer and a small number of descriptor fields

This ensures the subsequent extension does not need to build a second parallel system.

### 8.13 PatternFusionPass Bridge Protocol

This section defines the cross-language call protocol between the C++ pybind bridge and `_bridge.py` for `PatternFusionPass`.

#### 8.13.1 Protocol Functions

`_bridge.py` needs to implement the following three functions for `libge_python_pass_bridge.so` to call back in embed mode:

1. **`get_pass_patterns(instance_id: str) -> list`**
   - Called back by the C++ side `PythonPatternFusionPassAdapter::Patterns()` through the bridge
   - `_bridge.py` calls the Python pass instance's `patterns()` method
   - Returns a Pattern object list; each Pattern is constructed by `_ge_pass_native.so`

2. **`call_meet_requirements(instance_id: str, match_result_handle: int) -> bool`**
   - Called back by the C++ side `PythonPatternFusionPassAdapter::MeetRequirements()` through the bridge
   - `match_result_handle` is the `uintptr_t` representation of the C++ `MatchResult*`
   - `_bridge.py` restores it to a borrowed MatchResult wrapper through `_ge_pass_native.so`
   - Calls the Python pass instance's `meet_requirements()` method
   - Returns whether the condition is satisfied

3. **`call_replacement(instance_id: str, match_result_handle: int) -> int`**
   - Called back by the C++ side `PythonPatternFusionPassAdapter::Replacement()` through the bridge
   - `match_result_handle` is the same as above
   - Calls the Python pass instance's `replacement()` method
   - Requires the Python pass to return a replacement Graph
   - If the current match should not continue, it must return `False` in the `meet_requirements()` phase
   - `_bridge.py` is responsible for validating the return value type and transferring Graph ownership to C++

#### 8.13.2 Ownership and Lifecycle Convention

##### Pattern Ownership Transfer (get_pass_patterns)

- The Python side constructs Pattern objects through `_ge_pass_native.so`
- Pattern has `unique_ptr` semantics; before the function returns, the Python side must call `release()` to transfer ownership
- The C++ side takes over the raw pointer through `unique_ptr<Pattern>` and is responsible for subsequent destruction
- **Constraint**: The Python side must not continue to hold Pattern references after the function returns

##### MatchResult Borrowed Semantics (call_meet_requirements / call_replacement)

- `MatchResult` ownership always remains on the C++ side (held by `PatternFusionPass::Run()`)
- The MatchResult obtained by the Python side is a borrowed view and does not own ownership
- **Constraint**: The Python side must not continue to hold or cache MatchResult references after the callback returns
- The native binding should provide a borrowed wrapper and throw `RuntimeError` on expired access rather than crashing
- `uintptr_t` passing is a transitional approach, to be replaced with a type-safe method when the MatchResult native binding is ready

##### Replacement Graph Ownership Transfer (call_replacement)

- The Python side constructs the replacement Graph
- Graph ownership is transferred to the C++ side through `release()` (`GraphUniqPtr` / `unique_ptr<Graph>`)
- **Constraint**: The Python side must not continue to hold the Graph reference after the function returns

#### 8.13.3 DecomposePass Bridge Protocol

`DecomposePass` reuses the C++ `DecomposePass::Run()` matching and replacement main flow, so `_bridge.py` only needs to add two per-node callback protocol functions:

1. **`call_decompose_meet_requirements(instance_id: str, node_handle: int) -> bool`**
   - Called back by C++ `PythonDecomposePassAdapter::MeetRequirements()` through the bridge
   - `node_handle` is the `uintptr_t` representation of the current `GNode*`
   - `_bridge.py` restores it to a short-lived Python `Node` view through the `_ge_pass_native.so` `borrow_node()` helper
   - Calls the Python pass instance's `meet_requirements()` method

2. **`call_decompose_replacement(instance_id: str, node_handle: int) -> int`**
   - Called back by C++ `PythonDecomposePassAdapter::Replacement()` through the bridge
   - `_bridge.py` calls the Python pass instance's `replacement()` method
   - Requires the Python pass to return a replacement Graph
   - If the current node should not continue, it must return `False` in the `meet_requirements()` phase
   - `_bridge.py` is responsible for validating the return value type and transferring ownership to C++ through `release_graph()`

3. **`_bridge.py` internal instance dispatch uses explicit base classes, not `Any`**
   - The instance type saved in the holder converges to `FusionBasePass`
   - `PatternFusionPass` / `DecomposePass` protocol entries perform `isinstance` convergence before calling
   - The bridge parameters and return values of `replacement()` are handled according to the formal `Graph` / `Node` interfaces, no longer passed as unconstrained `Any`

##### Node Borrowed Semantics (call_decompose_meet_requirements / call_decompose_replacement)

- `DecomposePass::Run()` enumerates matching nodes on the C++ side
- The `Node` seen by the Python side is a short-lived view constructed by `_ge_pass_native.so` based on the current `GNode`
- The Python side should not cache this `Node` across callbacks
- Because this view essentially still points to the real graph node, reading name, type, attributes, and tensor descriptions remains consistent with the current graph

It is recommended to split legacy custom pass integration into two layers:

- Discovery and registration layer
  - Continue to reuse `ge.passes.bootstrap`, `ge.passes.registry`, `ge.passes._bridge`
  - The Python side only needs to add `legacy_custom` descriptor fields and decorator/registration entry

- Execution adaptation layer
  - Add `PythonLegacyCustomPassAdapter` on the existing `CustomPassHelper` / legacy custom pass chain
  - This adapter continues to reuse the same pybind bridge and the same `descriptor_key -> instance_id` lifecycle protocol

The direct benefits of this approach are:

- Python pass discovery, module loading, exception translation, GIL, session, and holder reclamation only maintain one implementation
- Legacy custom passes only add outermost interface adaptation without duplicating Python runtime management logic
- If both `PassRegistry` passes and `REGISTER_CUSTOM_PASS` need to be supported simultaneously, both sides still share the same Python user development experience

## 9. Python Graph Interface Completion Design

### 9.1 Required Capabilities

To support rewriting the existing `examples/fusion_pass` in Python, at minimum the following need to be completed:

- `Graph` borrowed/non-owning handle mode
- `Node.get_input_desc`
- `Node.get_output_desc`
- `Node.update_input_desc`
- `Node.update_output_desc`
- `Node.get_input_const_tensor`
- `Shape`
- `TensorDesc`
- `GeUtils.InferShape`
- `GeUtils.CheckNodeSupportOnAicore`

### 9.2 Borrowed Handle

Many `Graph`, `Node`, `Tensor` objects passed back from C++ to Python at runtime should not be destructed by Python. The current `Graph._create_from(handle, owns_handle, owner)` already supports borrowed / non-owning form. During execution, the bridge maps runtime `GraphPtr` to the formal `ge.graph.Graph` view through:

- `_create_from(handle, owns_handle=False, owner=...)`

and uses `owner` to hold the creation-phase token, preventing the Python wrapper from prematurely destructing and accidentally releasing the underlying runtime graph.

## 10. Packaging and Release Design

### 10.1 Artifacts

The current repository is responsible for building:

- `ge_py` main wheel
- Multi-version native sub-wheels
- Bridge artifact set and its loading logic

The bridge artifact set here refers to:

- `libge_python_pass_bridge.so`
  - The private bridge so `dlopen`-ed by the GE internal loader
  - Responsible for the embed path, interpreter management, GIL, Python callbacks, and exception translation

- `_ge_pass_native.so`
  - The helper extension imported by Python
  - Responsible for `Graph` / `PassContext` / `MatchResult` and other native-backed wrappers and helpers
  - Does not carry user-inheritable pass base classes

### 10.2 Version Strategy

Formal support matrix:

- `cp39`
- `cp310`
- `cp311`
- `cp312`
- `cp313`
- `cp314`

These sub-wheels carry the `pybind11` extensions related to the Python pass bridge / adapter, not replacements for the entire `ge.graph` Python wrapper.

The release pipeline needs to cover build capability for the above Python versions, but this is not equivalent to "a single machine must have all Python versions installed simultaneously". It is recommended to organize by a multi-version build matrix:

- Each build job/container is only responsible for one Python minor version
- Each job produces the native sub-wheel for the corresponding tag
- The main wheel still maintains a single build

In other words, the requirement is "the overall release pipeline covers `cp39-cp314`", not "every development machine or build machine must have all versions simultaneously".

Current implementation status:

- `build_python_pass_native_matrix.py` has built-in support for the version set `cp39/cp310/cp311/cp312/cp313/cp314`; CMake no longer maintains a separate version list, and `--tag` is only used for manually specifying a build version
- The `ge_python` formal target produces the main wheel and triggers `ge_python_pass_native_wheel_matrix` to best-effort build native sub-wheels for available Python minor versions
- `ge_python_native_wheel` is a single-version development target, only outputting the artifact set and `ge_py_pass_bridge` native sub-wheel corresponding to the current `HI_PYTHON`
- The repository provides `build_python_pass_native_matrix.py` and the `ge_python_pass_native_wheel_matrix` target in the `python_pass_native_build` build tool directory, for automatically sniffing available `python3.9` through `python3.14` in CI/local environments; the matrix does not reconfigure the entire repository CMake, but reuses compile/link metadata generated by the parent build, only replacing Python include/lib before recompiling `libge_python_pass_bridge.so` and `_ge_pass_native.so`
- The interpreter discovery order is: explicitly passed `--python`, `python3.9` through `python3.14`/`python3`/`python` in `PATH`, `bin/python` in Conda environments; Conda environments are preferably obtained through `conda env list --json`, falling back to scanning the `envs` directory derived from `CONDA_PREFIX`/`CONDA_EXE` and common paths `~/miniconda3/envs`, `~/anaconda3/envs`, `~/.conda/envs`
- The matrix build depends on the parent build first completing the current `HI_PYTHON` `ge_python_native_wheel`, and reuses the parent build's compiler, compilation options, include/link metadata, built GE dependency libraries, and `CMAKE_CXX_COMPILER_LAUNCHER`; this flow does not re-enter `cmake/package.cmake` or recalculate `BUILD_COMPONENT`
- The main wheel only assembles pure Python code and no longer embeds the default native artifact set for the current build Python
- Native sub-wheels are generated through standard `bdist_wheel`, carrying a bridge/native artifact set under `ge/passes/python_pass_artifacts/<python_tag>-<platform>`
- The ge-compiler run package continues to install the original `ge_py-0.0.1-py3-none-any.whl`, and additionally installs `ge_py_pass_bridge-*.whl` from the matrix output directory into `ge-compiler/lib64`
- The `ge_python_pass_fallback_codegen` target generates `ge/passes/fallback_codegen/` resources through `gen_fallback_resources.py`, including `build_config.json` and a Python resource module that can temporarily expand `src/{bridge,native}` and `include/{bridge,native}`
- Runtime selection priority is "prebuilt artifact set > runtime fallback codegen"

The run package installation script needs to note: the run package can carry multiple `ge_py_pass_bridge` native sub-wheels, but a single installation should only install one sub-wheel compatible with "the Python interpreter executing the installation script". It is recommended to let pip automatically select based on wheel tags:

```bash
PYTHON_BIN=${PYTHON_BIN:-python3}
LIB_DIR=<run-package>/ge-compiler/lib64

"${PYTHON_BIN}" -m pip install \
  --no-index \
  --find-links "${LIB_DIR}" \
  "${LIB_DIR}/ge_py-0.0.1-py3-none-any.whl" \
  ge-py-pass-bridge
```

In this approach, the `ge_py` main wheel is explicitly installed through a file path, and `ge-py-pass-bridge` selects the native sub-wheel matching the current Python tag and platform tag from the same directory through `--find-links`. For example, Python 3.11 will only select the `cp311-cp311` wheel. The installation script should not pass all `cp39-cp314` native wheel file paths to pip at once, otherwise wheels incompatible with the current interpreter will be judged as uninstallable.

#### 10.2.1 Artifact Set Manifest and Selection Mechanism

The Python pass bridge artifact set uses a fixed directory layout:

```text
ge/passes/python_pass_artifacts/<python_tag>-<platform>/manifest.json
ge/passes/python_pass_artifacts/<python_tag>-<platform>/libge_python_pass_bridge.so
ge/passes/python_pass_artifacts/<python_tag>-<platform>/_ge_pass_native.so
```

The current manifest fields are as follows:

```json
{
  "python_tag": "cp311",
  "platform": "linux-x86_64",
  "bridge_abi": 1,
  "artifacts": {
    "bridge": "libge_python_pass_bridge.so",
    "native": "_ge_pass_native.so"
  }
}
```

Field descriptions:

| Field | Description | Matching Method |
|-------|-------------|-----------------|
| `python_tag` | The CPython minor version tag bound to the artifact, for example `cp39`, `cp310`, `cp311`, `cp312` | Must match the target Python runtime key in the current process |
| `platform` | The platform tag bound to the artifact, currently composed of the system name and machine architecture, for example `linux-x86_64` | Must match the current running platform |
| `bridge_abi` | The C ABI protocol version between the loader in `ge_compiler.so` and `libge_python_pass_bridge.so` | Must match `kPythonFusionPassBridgeAbiVersion` |
| `artifacts.bridge` | The bridge `.so` path relative to the manifest directory | Must resolve to a real file, as the `dlopen` target |
| `artifacts.native` | The `_ge_pass_native.so` path relative to the manifest directory | Must resolve to a real file, and be passed to the bridge through `set_artifact_config` to ensure Python import uses the same-source native |

Selection flow:

1. The loader first resolves the current process target Python runtime key. If the current process has already loaded CPython C API symbols, it obtains `cpXY` through `Py_GetVersion()`; if no visible CPython symbols exist in the process, it probes `python3` and `python` in `PATH` according to the internal native entry scenario.
2. The loader derives the `ge` package directory from its own `.so` path and scans manifests under `ge/passes/python_pass_artifacts`.
3. The manifest first undergoes JSON structure and artifact file existence validation, then filters by `python_tag`, `platform`, and `bridge_abi`.
4. Matched artifact sets enter the candidate list with priority.
5. If no match is found or candidate loading fails, runtime fallback codegen is triggered, calling `ge.passes.runtime.run_fallback_codegen()`; if fallback artifacts are still unavailable, the loader directly returns failure.
6. After bridge `dlopen` succeeds, the loader reads the actual Python runtime key of the current process and performs consistency validation against the target key before loading, to avoid accidentally starting a different CPython minor version in the same process.

`kPythonFusionPassBridgeAbiVersion` only describes the protocol version of the loader and bridge C API, for example it only needs to be rolled when function table fields, function semantics, or call timing undergo incompatible changes. Before the project formally releases the Python pass bridge, this value remains `1` during full feature development and does not change frequently due to internal field additions or phased development commits.

### 10.3 Installation Strategy

V1 does not perform installation-time codegen, nor does it plan a separate `prepare` pre-compilation command. The reasons include:

- Standard wheels lack a stable, universal post-install compilation mechanism
- The Python environment at installation time is not necessarily the final runtime environment
- Installation machines do not necessarily have compilers and development headers
- Once installation-time automatic compilation fails, it may also bring down the availability of the pure Python parts

Therefore, a two-layer strategy is recommended:

1. Mainstream versions use pre-compiled native sub-wheels
2. Non-covered versions use runtime fallback codegen as fallback

That is, the installation phase is only responsible for installing the main wheel and available native sub-wheels in place; if the current Python version does not match a prebuilt artifact, the runtime triggers fallback on demand.

For the packaged default path in the run package, the following fixed constraints also need to be satisfied:

- `libge_python_pass_bridge.so` is placed with the native artifact wheel at `ge/passes/python_pass_artifacts/<python_tag>-<platform>/`
- The GE internal loader resolves bridge/native paths in the artifact set through manifest parsing
- `_ge_pass_native.so` and `libge_python_pass_bridge.so` follow the same artifact set assembly constraints

### 10.4 Fallback

When the current Python version does not match a pre-compiled sub-wheel:

- The main wheel directly enters the runtime fallback flow
- Generates and compiles the corresponding version bridge artifact set in the local cache directory
- If successful, enables Python pass
- If failed, disables Python pass, but does not affect the original C++ pass flow

The fallback boundary needs to be specially constrained:

- Fallback codegen does not generate user pass code
- Fallback codegen does not rewrite `ge_compiler.so`
- The goal of fallback codegen is to generate a replaceable bridge artifact set, not a local patch
- Fallback artifacts need to cover both `libge_python_pass_bridge.so` and `_ge_pass_native.so`
- Fallback artifacts carry complete Python version-sensitive bridge logic and reuse the stable protocol agreed upon with `ge_compiler.so`

### 10.5 Local Validation Constraints

The current development environment Python is `3.13`, while the formal matrix is `cp39-cp314`. Therefore, local validation can only cover Python tags available in the current environment; the full matrix requires CI or release pipelines to build separately in environments with the corresponding Python versions.

### 10.6 pybind Module Boundary

V1 recommends keeping pybind-side content within the "pass bridge required capability" scope, not expanding to the entire `ge.graph`. It is recommended to split into the following boundaries:

- Pure Python code in the main wheel
- `ge.passes.base`
- `ge.passes.registry`
- `ge.passes.bootstrap`
- `ge.passes.runtime`
- `ge.passes._bridge`

- Stable core inside `ge_compiler.so`
- `PassPluginLoader`
- `PassRegistry` registration and runtime routing
- `descriptor_key + instance_id` lifecycle protocol
- Adapter callbacks, exception translation, error code mapping
- Stable interface for interacting with the independent internal bridge `.so`

- Prebuilt / fallback generated independent internal bridge `.so`
- Python version-sensitive `Python.h` / `pybind11` binding code
- Python interpreter initialization, GIL management, module import, exception translation
- Unified runtime path calling `ge.passes._bridge`

- Prebuilt / fallback generated `_ge_pass_native.so`
- Construction and conversion of formal Python wrappers such as `Graph` / `PassContext` / `MatchResult` / `NodeIo`
- Helper and factory interfaces directly imported by the Python side
- Provides formal native-backed object sources for `ge.passes.base` / `ge.passes.pattern`

In other words:

- The responsibility of `libge_python_pass_bridge.so` is "supporting Python pass integration into GE"
- The responsibility of `_ge_pass_native.so` is "exposing native-backed objects and helpers for Python passes"
- Neither is "rewriting the existing Python graph API"

Differences between Python versions should not in principle be reflected in maintaining multiple `.cc` files with different semantics, but mainly reflected in:

- Using the same template or the same source code
- Different Python include / libpython / extension suffix / rpath and other build parameters at compilation time
- Individual compatibility macros, conditional compilation, or generated metadata differences

That is, it is not recommended to maintain multiple hand-written bridge source code branches by `cp39/cp310/cp311/cp312/cp313/cp314` long-term.

### 10.7 pybind Sub-Wheel Organization Recommendations

It is recommended to adopt a "main wheel + internal bridge native wheel" organization:

- Main wheel package name
- Keep `ge_py`

- Native bridge wheel
- Carried by a separate package name, for example logically named `ge_py_pass_bridge`
- Wheel tag corresponds to `cp39-cp314`
- Uses standard `bdist_wheel` generation, avoiding manual assembly of wheel metadata / RECORD / tags

`ge.passes.runtime` in the main wheel is responsible for:

1. Identifying the current Python version
2. Parsing and loading matching bridge artifact metadata
3. If no prebuilt module is matched, entering fallback codegen
4. Providing the final bridge artifact path and entry information to the GE internal loader
5. The GE internal loader `dlopen`-s `libge_python_pass_bridge.so` and establishes callback entries
6. `libge_python_pass_bridge.so` then loads `_ge_pass_native.so` from the same directory

This way the main wheel logic is fixed, and the native wheel is only responsible for bridge implementation under different Python versions.

A compatibility principle needs to be further clarified here:

- The bridge artifact ultimately selected at runtime should be overridden with "prebuilt artifact > fallback artifact" priority
- It should not be required to decide which generated artifact to use at `ge_compiler.so` link time
- A more reasonable approach is for the Python side or configuration side to first resolve the target bridge artifact, then have the GE internal loader `dlopen` it
- Therefore, generated artifacts need to be dynamically loaded through stable module names or stable C ABI, rather than making the `.so` built into the run package the only hard-linked target

From an extensibility perspective, the bridge should not continue to be compiled into `ge_compiler.so`. If such implementation currently exists, it can only be treated as a temporary bring-up deviation, not the formal architecture.

### 10.8 Fallback Codegen Boundary

Fallback codegen is recommended to directly generate the entire bridge artifact set, not just a single local wrapper helper. The input is fixed as:

- Current Python version
- Unified source code or templates for `libge_python_pass_bridge.so` / `_ge_pass_native.so`
- Stable header files exported by the current repository
- Build parameters such as include / libpython / extension suffix corresponding to the current Python environment
- Stable link dependencies exposed by the current run package

The output is fixed as:

- `ge/passes/python_pass_artifacts/<python_tag>-<platform>/libge_python_pass_bridge.so`
- `ge/passes/python_pass_artifacts/<python_tag>-<platform>/_ge_pass_native.so`
- Corresponding manifest / metadata files

The recommended link and loading relationship is as follows:

1. `ge_compiler.so` does not depend on a specific fallback artifact at link time
2. The runtime resolves the final bridge artifact path to use
3. The GE internal loader `dlopen`-s that bridge `.so`
4. The bridge `.so` interacts with `ge_compiler.so` through stable ABI

This ensures "generated `.so` takes effect with priority" rather than always being overridden by the version preset in the run package.

### 10.9 Current Engineering and Subsequent Codegen Compatibility Strategy

The current engineering has already split `python_fusion_base_pass_pybind_bridge.cc` out of `ge_compiler.so`, with the first batch changed to have `python_fusion_base_pass_bridge_loader.cc` responsible for runtime loading of `libge_python_pass_bridge.so`. For compatibility with subsequent codegen evolution, subsequent implementations still need to continuously observe the following boundaries:

- Do not solidify Python version-sensitive logic into the `ge_compiler.so` shipped with the run package
- `ge_compiler.so` only retains the stable loader, registry, adapter protocols, and minimal interaction surface
- The independent bridge `.so` carries complete Python-sensitive logic and becomes the unified replacement target for pre-compilation / fallback
- Core semantics such as `descriptor_key + instance_id`, adapter callback protocol, error code translation, and holder lifecycle remain stable on the compiler side

In other words, if the goal is productized support for subsequent extensions, then the independent bridge `.so` should not be an optimization item to "move later", but should be the formal architecture boundary from the beginning.

### 10.10 Python Interpreter Source and Fallback Selection Constraints

The Python pass bridge cannot assume "there is only one way to start Python in the process". When subsequently doing multi-version pre-compilation and fallback codegen, interpreter source, artifact version selection, and finalize order must be treated as the same lifecycle problem.

The following typical scenarios currently need to be covered:

| Scenario | Current Process | Python Interpreter State | Bridge Behavior | Fallback / Codegen Constraint |
|----------|----------------|-------------------------|-----------------|-------------------------------|
| External Python launcher, for example the actual `python -m ge.pyatc` | The process where the bridge resides is still `atc.bin` | The outer Python only starts `atc.bin`, which is not equivalent to `atc.bin` already having an interpreter; actual logs show TBE initializes Python inside `atc.bin` first | The bridge sees `Py_IsInitialized()` as true in `atc.bin` and reuses the TBE interpreter, `owns_interpreter=false` | Artifact selection must be based on the interpreter version in the current `atc.bin` process; the launcher Python or TBE worker Python cannot be directly used as the bridge version basis |
| Python process directly calls GE/ATC entry through `ctypes` / C API | Current Python main process | The interpreter is already initialized before entering GE | Both TBE and the bridge should see `Py_IsInitialized()==1` and reuse the current interpreter; neither owns the interpreter | Artifact selection must use the current Python process's `sys.version_info` as the sole authoritative version; `python3` in `PATH` does not participate in version selection for this scenario |
| Internal native entry, for example `atc.bin`, and Python pass uses Python first | `atc.bin` | The interpreter is not initialized before the bridge enters | The bridge can call `py::initialize_interpreter()` and record `owns_interpreter=true` | Artifact selection must be completed before initialization based on a clear target Python environment, for example a configured Python executable or unified runtime selector; different sources of `python3` cannot be implicitly randomly used |
| TBE initializes Python first in `atc.bin` | `atc.bin` | TBE's `TbeInitialize -> PythonAdapterManager::Initialize -> py_decouple.cc` has already initialized the interpreter | The bridge must reuse the existing interpreter, `owns_interpreter=false` | Prebuilt or fallback artifacts must match the Python minor version already started by TBE in the process; if the selected artifact is inconsistent with the in-process version, it must report an error and disable Python pass; a second interpreter cannot be started |
| TBE parallel compilation worker, for example `TBE(pid,python3)` in logs | Independent `python3` subprocess | Only affects the worker's own address space | Has no shared interpreter relationship with the bridge in `atc.bin` | The worker process Python version cannot be used as the bridge artifact selection basis; the bridge only looks at the current process |

#### 10.10.1 External Python Launcher Actual Flow

For the current `python -m ge.pyatc` type external startup scenario, if the implementation is `subprocess` / `execve` starting the native `atc.bin`, it should be understood as "Python launcher starts native `atc.bin`", not "GE runs directly in the launcher Python process". Therefore, the current process effective for the Python pass bridge is still `atc.bin`.

It needs to be clarified that after `subprocess.Popen(["atc.bin", ...])` goes through `execve`, the `atc.bin` subprocess does not inherit the parent Python process's interpreter state, GIL, `sys.modules`, Python objects, or loaded libpython handles; it only inherits environment variables, current working directory, some file descriptors, and other process attributes. Therefore, the outer launcher Python version can only indirectly affect `atc.bin` through environment variables such as `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `ASCEND_GE_PY_PASS_PATH`, and cannot directly serve as the bridge artifact version basis.

The actual log-corresponding flow is TBE starting Python first:

```text
User shell
  -> python -m ge.pyatc
      -> Outer Python launcher prepares parameters / environment
      -> Starts atc.bin

atc.bin process
  -> TbeInitialize()
      -> PythonAdapterManager::Initialize()
          -> HandleManager::Initialize()
              -> dlsym(RTLD_DEFAULT, "Py_Initialize")
                  -> If symbol can be found:
                       -> LoadFuncs(RTLD_DEFAULT)
                       -> If Py_IsInitialized() == 1:
                            -> Reuse existing interpreter
                       -> If Py_IsInitialized() == 0:
                            -> TE_Py_Initialize()
                  -> If symbol cannot be found:
                       -> LaunchDynamicLib()
                       -> Derive libpython from python3 / python in PATH
                       -> dlopen(libpythonX.Y.so.1.0, RTLD_GLOBAL)
                       -> TE_Py_Initialize()
          -> import te_fusion.* / parallel_compilation
          -> init_multi_process_env()
          -> Start independent python3 worker subprocess

atc.bin process
  -> LoadPassPlugins()
      -> dlopen(libge_python_pass_bridge.so)
      -> bridge EnsureBridgeReady()
          -> Py_IsInitialized() == 1
          -> Reuse TBE-initialized interpreter
          -> owns_interpreter = false
          -> import ge.passes._ge_pass_native / ge.passes._bridge
          -> Register Python pass descriptors
```

In this scenario, TBE is the interpreter owner, and the Python pass bridge is the interpreter user. Neither the outer launcher Python nor the TBE worker Python is the interpreter where the bridge resides.

#### 10.10.2 Python Process `ctypes` / C API Direct Call Scenario

If there is subsequently a mode that does not start a subprocess but directly calls GE/ATC entry functions within the Python main process through `ctypes.CDLL(...)`, C API wrapper, or similar approaches, GE, TBE, and the Python pass bridge all run in the same Python process address space:

```text
python main process
  -> import ge.pyatc / ctypes.CDLL(...)
  -> Call main_impl / GE build entry

Same python process
  -> TbeInitialize()
      -> dlsym(RTLD_DEFAULT, "Py_Initialize")
          -> Should be able to find current CPython symbols
      -> Py_IsInitialized() == 1
      -> TBE reuses current Python interpreter
      -> pyEnvStatusBeforeTbe = true
      -> TBE does not own the interpreter, should not Py_Finalize

Same python process
  -> LoadPassPlugins()
      -> dlopen(libge_python_pass_bridge.so)
      -> bridge EnsureBridgeReady()
          -> Py_IsInitialized() == 1
          -> Reuse current Python interpreter
          -> owns_interpreter = false
          -> import ge.passes._ge_pass_native / ge.passes._bridge
          -> Register Python pass descriptors
```

In this scenario, the interpreter owner is the outer Python main process. Both TBE and the Python pass bridge can only serve as interpreter users and must not call `Py_Finalize()` / `py::finalize_interpreter()`.

Version selection in this mode is the most direct; the core principle is reusing the existing interpreter in the current process:

- The current Python process's `sys.version_info` is the sole authoritative version.
- `libge_python_pass_bridge.so` and `_ge_pass_native.so` must match that Python minor version.
- TBE's `py_decouple.cc` must reuse the current interpreter when it can already find CPython symbols through `RTLD_DEFAULT` and `Py_IsInitialized()==1`; the `PATH` / `python3-config` branch should only be entered when the current process has no visible CPython symbols.
- If the current process's initialized Python version is found to be inconsistent with the bridge artifact manifest, it must report an error and disable Python pass; having two CPython minor versions coexisting in the same process is not an acceptable fallback approach.

#### 10.10.3 Internal `atc.bin` Scenario: TBE Starts First

In the `atc.bin` native process, if TBE uses Python first, TBE's `py_decouple.cc` is responsible for finding and initializing libpython:

```text
atc.bin
  -> TbeInitialize()
      -> dlsym(RTLD_DEFAULT, "Py_Initialize")
          -> Symbol found:
               -> Indicates the current process already has visible CPython C API
               -> Then call Py_IsInitialized()
                    -> true: Reuse existing interpreter
                    -> false: TBE calls TE_Py_Initialize()
          -> Symbol not found:
               -> popen("python3 -V")
               -> If fails, popen("python -V")
               -> Parse Python major.minor
               -> Assemble so name:
                    Python 3.7  -> libpython3.7m.so.1.0
                    Python 3.8+ -> libpython3.X.so.1.0
               -> popen("python3-config --prefix")
               -> Assemble $prefix/lib/libpythonX.Y[ m ].so.1.0
               -> mmDlopen(absolute path, RTLD_NOW | RTLD_GLOBAL)
               -> If fails:
                    -> mmDlopen("libpythonX.Y[ m ].so.1.0", RTLD_NOW | RTLD_GLOBAL)
                    -> Depends on LD_LIBRARY_PATH / RUNPATH / ld.so.cache / system lib paths
               -> TE_Py_Initialize()

atc.bin
  -> LoadPassPlugins()
      -> dlopen(libge_python_pass_bridge.so)
      -> Py_IsInitialized() == 1
      -> Bridge reuses TBE interpreter
      -> owns_interpreter = false
```

In this scenario, fallback / codegen must match the Python minor version in the `atc.bin` process that TBE has already initialized. If the bridge artifact is inconsistent with that version, Python pass must be disabled or a clear error must be reported; attempting to load another CPython is not allowed.

#### 10.10.4 Internal `atc.bin` Scenario: Python Pass Starts First

If in some flow the Python pass bridge uses Python before TBE, the interpreter owner becomes the bridge:

```text
atc.bin
  -> LoadPassPlugins()
      -> Resolve bridge artifact set
      -> dlopen(libge_python_pass_bridge.so)
          -> libpython resolved by bridge's ELF NEEDED / RUNPATH / LD_LIBRARY_PATH / ld.so.cache
      -> bridge EnsureBridgeReady()
          -> Py_IsInitialized() == 0
          -> py::initialize_interpreter()
          -> owns_interpreter = true
          -> import ge.passes._ge_pass_native / ge.passes._bridge
          -> Register Python pass descriptors

atc.bin
  -> TbeInitialize()
      -> dlsym(RTLD_DEFAULT, "Py_Initialize")
          -> Can find symbol
      -> Py_IsInitialized() == 1
      -> TBE reuses bridge-initialized interpreter
      -> pyEnvStatusBeforeTbe = true
      -> TBE does not own the interpreter, should not Py_Finalize
```

In this scenario, special care is needed for shutdown: although the bridge has `owns_interpreter=true`, if TBE subsequently reuses that interpreter, the bridge must not call `py::finalize_interpreter()` before TBE cleans up its own Python module / parallel compilation. Therefore, the subsequent design must split "bridge cleaning up Python pass state" and "interpreter final finalize" into two actions, or introduce a unified process-level Python runtime manager.

There are several judgment principles that must be hardcoded:

- `Py_IsInitialized()` only describes the CPython interpreter state in the **current process**, not the external `python3` subprocess state.
- `Py_IsInitialized()` returning true only indicates the current process already has an interpreter; it cannot alone determine which wheel to select; the version still needs to be explicitly read through in-process `sys.version_info` or `Py_GetVersion()`.
- Being able to find symbols through `dlsym(RTLD_DEFAULT, "Py_IsInitialized")` or `Py_Initialize` only indicates the current process has visible CPython C API symbols, which is not equivalent to the interpreter being initialized.
- Only one CPython minor version is allowed in the same process; any module that finds the current process already has an interpreter must base subsequent Python pass artifact selection on that interpreter version, and cannot re-select a different minor version artifact based on another `python3` in `PATH`.
- `libge_python_pass_bridge.so` and `_ge_pass_native.so` must match the same CPython minor version and be enabled together as the same manifest-described artifact set.
- If the bridge initializes the interpreter itself, the bridge can call `py::finalize_interpreter()` during process-level shutdown; if the interpreter comes from the user Python main process or TBE, the bridge can only clean up its own held modules, holders, and registry state, and must not finalize the interpreter.

Therefore, the recommended strategy for the fallback runtime selector is:

1. If the interpreter in the current process is already initialized, use the in-process Python version as the sole authoritative version.
2. If the interpreter in the current process is not yet initialized, use the explicitly configured target Python executable or unified selector result as the target version, and use that version to generate / select the artifact set.
3. After bridge initialization, read the in-process Python version again and perform consistency validation against the artifact manifest.
4. When version inconsistency is found, prioritize reporting a clear error and disabling Python pass; mixing two CPython minor versions in the same process is not an acceptable recovery path.

A shutdown order problem needs to be specially prevented here:

```text
TBE initializes Python
  -> Python pass bridge reuses interpreter and caches py::object / holders
  -> TBE calls Py_Finalize first
  -> Python pass bridge subsequently continues to access py::objects
```

The above order is illegal. The more accurate principle is not "the bridge always finalizes first" or "TBE always finalizes first", but:

- All Python objects, modules, and holders held by all modules must be cleaned up before the interpreter is finalized.
- The module that actually calls `Py_Finalize()` / `py::finalize_interpreter()` must be the interpreter owner, and must wait until all other Python users in the process have completed cleanup before executing.
- The bridge's "cleaning up Python pass state" and "finalizing the interpreter" must be separable by design; the two must not be permanently bound as one action.

Therefore, the shutdown constraints differ under different startup orders:

```text
TBE initializes Python first:
UnloadPassPlugins() (when reference count reaches zero)
  -> reset / clear Python pass holders, modules, registry
GELib::Finalize()
  -> TBE / op store finalize
  -> TBE cleans up its own Python modules / parallel compilation
  -> TBE calls Py_Finalize when it owns the interpreter

Python pass bridge initializes Python first:
UnloadPassPlugins() (when reference count reaches zero)
  -> reset / clear Python pass holders, modules, registry
  -> Must not immediately py::finalize_interpreter unless confirmed TBE and other Python users have not initialized
GELib::Finalize()
  -> TBE / op store finalize
  -> TBE only cleans up its own Python modules, does not Py_Finalize
Finally the bridge owner or process-level Python runtime manager finalizes the interpreter
```

If `GEFinalize`, `GeGenerator::Finalize`, `aclgrphBuildFinalize`, TBE plugin manager, or pass plugin loader order is subsequently adjusted, this constraint must be re-checked. As long as the bridge does not own the interpreter, it cannot assume it can control the interpreter lifecycle; as long as the bridge owns the interpreter but TBE has already reused it, the bridge must not finalize the interpreter before TBE cleans up its own Python modules. The more robust long-term direction is to introduce a process-level Python runtime manager or reference-counted owner protocol, abstracting "interpreter initialization / reuse / final finalize" out of individual bridge or TBE modules.

## 11. ATC Extension Design

To make unified reservations for subsequent Python ATC entry, the current recommendation is to split into two layers:

- Current main path
- `ASCEND_GE_PY_PASS_PATH`

- Subsequent productization entry
- CLI / internal options can add explicit parameters later, but all should ultimately converge to the `ge.passes.bootstrap.load_pass_plugins()` unified discovery protocol

- Selection logic related to the native companion module
- Users should not be required to directly specify a pybind bridge source code or generation script
- It is more appropriate for `ge.passes.runtime` to uniformly decide "prebuilt / fallback" artifact selection

The subsequent Python ATC entry should not design a second discovery mechanism, but directly reuse:

- `ge.passes.bootstrap.load_pass_plugins()`
- `ge.passes.bootstrap.get_registered_passes()`

## 12. File-level Development Plan

### 12.1 Python Package and Discovery Layer

Modify and add the following files:

- `api/python/ge/setup.py`
- `api/python/ge/ge/__init__.py`
- `api/python/ge/ge/passes/__init__.py`
- `api/python/ge/ge/passes/base.py`
- `api/python/ge/ge/passes/registry.py`
- `api/python/ge/ge/passes/pattern.py`
- `api/python/ge/ge/passes/replacement.py`
- `api/python/ge/ge/passes/bootstrap.py`
- `api/python/ge/ge/passes/_bridge.py`

Responsibilities are as follows:

- Define the three types of pass Python base classes
- Provide decorators and registry
- Implement the environment variable main path discovery logic, and reserve extension for subsequent `entry_points`
- Unify the bridge external interface
- Carry wheel selection and fallback management

The formal direction of `base.py` needs to be clearly split into two layers:

- User inheritance layer
- Continue to expose pure Python `FusionBasePass` / `PatternFusionPass` / `DecomposePass`
- Users are not required to inherit native C++ base classes here

- Wrapper source layer
- `PassContext` / `MatchResult` / `PatternMatcherConfig` and other objects directly obtain native-backed implementations from `_ge_pass_native`
- `_ge_pass_native` import failure is an environment or artifact assembly issue, no longer treated as a formal fallback path for the Python API

### 12.2 Python Graph Wrapper Completion

Modify and add the following files:

- `api/python/ge/ge/graph/__init__.py`
- `api/python/ge/ge/graph/graph.py`
- `api/python/ge/ge/graph/node.py`
- `api/python/ge/ge/graph/tensor.py`
- `api/python/ge/ge/graph/tensor_desc.py`
- `api/python/ge/ge/_capi/pygraph_wrapper.py`

Responsibilities are as follows:

- Complete borrowed handle
- Expose tensor desc/shape
- Add node input/output desc capabilities
- Add constant tensor read interface

### 12.3 C Wrapper and Native Bridge

Modify and add the following files:

- `api/python/ge/ge_api_c_wrapper/c_graph.cc`
- `api/python/ge/ge_api_c_wrapper/c_gnode.cc`
- `api/python/ge/ge_api_c_wrapper/c_tensor.cc`
- `api/python/ge/ge_api_c_wrapper/c_match_result.cc`
- `api/python/ge/ge_api_c_wrapper/ge_api_c_wrapper_utils.h`
- `compiler/graph/fusion/pass/pass_plugin_loader.cc`
- `compiler/graph/fusion/pass/pass_plugin_loader.h`
- `compiler/graph/fusion/pass/python_fusion_base_pass_bridge_c_api.h`
- `compiler/graph/fusion/pass/python_fusion_base_pass_bridge_loader.cc`
- `compiler/graph/fusion/pass/python_fusion_base_pass_pybind_bridge.cc`
- `compiler/graph/fusion/pass/python_fusion_base_pass_pybind_bridge.h`
- Add `_ge_pass_native.so` source code, export headers, and build script
- `api/python/ge/ge_api_c_wrapper/CMakeLists.txt`
- `api/python/ge/CMakeLists.txt`
- `compiler/CMakeLists.txt`

Responsibilities are as follows:

- Provide C interfaces for Python graph wrappers
- Provide `pybind11`-based Python pass bridge / helper so
- Connect wheel packaging and installation

It is recommended to further split responsibilities:

- `c_graph.cc` / `c_gnode.cc` / `c_tensor.cc` / `c_match_result.cc`
- Continue to serve the `ge.graph` / `ge.es` ctypes approach

- Independent bridge `.so`
- Responsible for Python runtime initialization, descriptor synchronization, holder management, adapter native logic, and Python/C++ object conversion
- Responsible for receiving prebuilt / fallback artifacts

- `_ge_pass_native.so`
- Responsible for `Graph` / `PassContext` / `MatchResult` and other native-backed wrappers and helpers
- Responsible for interfacing with `base.py` object sources
- Does not carry user pass base classes, and does not require users to directly import C++ pass base classes for inheritance

- `pass_plugin_loader.cc/.h`
- Responsible for locating and `dlopen`-ing bridge `.so`
- Responsible for stable ABI interfacing with bridge `.so`

- `python_fusion_base_pass_bridge_c_api.h`
- Defines the stable C ABI between bridge loader and `libge_python_pass_bridge.so`
- Current entry is `GeGetPythonFusionBasePassBridgeApi()`

- `python_fusion_base_pass_bridge_loader.cc`
- Located on the `ge_compiler.so` side
- Responsible for `dladdr` positioning, `dlopen/dlsym`, caching bridge API, and passing registrar callback to bridge
- Currently explicitly uses `RTLD_GLOBAL` to load bridge, so that embedded CPython can resolve `libpython` symbols when subsequently importing standard library / native extensions

- `python_fusion_base_pass_pybind_bridge.cc/.h`
- Located on the `libge_python_pass_bridge.so` side
- Responsible for Python runtime initialization, descriptor synchronization, holder lifecycle, and `create/run/destroy` callback implementation
- Exposes stable ABI to `pass_plugin_loader`, reuses `bootstrap / _bridge` protocol on the Python side

The formal architecture boundary should be "bridge artifact set replaceable, `ge_compiler.so` stable". Where:

- `libge_python_pass_bridge.so` is the main entry from the GE internal loader perspective
- `_ge_pass_native.so` is the helper extension from the Python perspective
- Both must be managed as the same version, same build key companion artifacts
- The current first batch has moved `python_fusion_base_pass_pybind_bridge.cc` out of the `ge_compiler` target and added a new `ge_python_pass_bridge` target to produce `libge_python_pass_bridge.so`
- `ge_compiler.so` currently only retains stable semantics such as loader, adapter, registry/runtime entry; the bridge so is the one that directly depends on `Python3::Python` and `pybind_options`

### 12.4 Pass Registration Core Refactoring

Modify the following files:

- `compiler/graph/fusion/pass/pass_registry.cc`
- `compiler/graph/fusion/pass/fusion_pass_executor.cc`
- Add creation-phase context management files, for example:
- `compiler/graph/fusion/pass/pass_create_context.h`
- `compiler/graph/fusion/pass/pass_create_context.cc`

Responsibilities are as follows:

- Inject TLS creation context at the `create_fn()` call point
- Let the generic creator find the corresponding Python descriptor by `pass_name`
- Keep existing C++ pass behavior unchanged

Recommended responsibility further refinement:

- `pass_create_context.h/.cc`
- Define TLS context and RAII scope

- `fusion_pass_executor.cc`
- Add scope around `create_fn()` in `InitPassesIfNeed`

- Bridge registration function
- Register Python pass descriptor as "fixed creator function + pass_name metadata"

Notes:

- `graph_fusion.cc` belongs to the legacy compatibility chain and is not included in the subsequent support scope of this scheme
- `REGISTER_CUSTOM_PASS` subsequent support goes through an independent extension phase, but still reuses the same descriptor / bridge / session mechanism

### 12.5 A/B Division and Integration Boundary

The current recommendation is to advance in parallel along the following boundaries:

- A is responsible for `libge_python_pass_bridge.so`, `pass_plugin_loader`, `ge_compiler.so` stable ABI, adapter routing, fallback loading, and existing `ge.graph.Graph` borrowed view integration
- B is responsible for `_ge_pass_native.so`, `base.py`, `PassContext` / `MatchResult` native-backed wrappers, and Python sample / Python API completion

B needs to clearly deliver:

- `_ge_pass_native.so` build script and module exports
- `PassContext` borrowed view wrapper
- `MatchResult` minimum usable wrapper
- Necessary helper / factory interfaces for `libge_python_pass_bridge.so` to construct Python objects
- `base.py` / `pattern.py` in `PassContext` / `MatchResult` / `Pattern` / `PatternMatcherConfig` native-backed direct export
- Python sample and Python API completion minimum capability list

A needs to clearly deliver:

- `graph.py` in `Graph._create_from(handle, owns_handle, owner)` borrowed / non-owning semantics
- `python_fusion_base_pass_pybind_bridge.cc` in `BuildPythonGraph()` formal integration with existing `ge.graph.Graph`
- `libge_python_pass_bridge.so` and `_ge_pass_native.so` bridge integration point

Regarding the `Graph` boundary, the following principles need to be specially fixed:

- `Graph` prioritizes reusing the currently existing `ge.graph.Graph`
- `_ge_pass_native.so` no longer introduces a second user-visible `Graph` type
- A is responsible for connecting runtime `GraphPtr` to the existing `ge.graph.Graph` as a borrowed view
- B does not directly own the `Graph` type itself, but provides supporting capabilities around `PassContext` / `MatchResult` / helpers

After B completes, `base.py` should converge to:

- `FusionBasePass` / `PatternFusionPass` / `DecomposePass` still maintain pure Python base classes
- `PassContext` / `MatchResult` / `PatternMatcherConfig` directly re-export types provided by `_ge_pass_native`
- `Pattern` directly exports types provided by `_ge_pass_native` through `ge.passes.pattern`
- No longer retain compatibility shims for `_ge_pass_native` missing scenarios

When A locally splits bridge `.so` and loader, Phase 1 can first not depend on `_ge_pass_native` for minimum validation, but this is only a temporary bring-up strategy and should not be retained in formal code after Phase 2 closure:

- Continue to use the existing `FusionBasePass` pure Python contract class
- `_bridge.py` and bridge `.so` first validate descriptor, holder, create/run/destroy, `dlopen`, and fallback artifact selection
- `PatternFusionPass` formal end-to-end validation waits for B's `_ge_pass_native` to land before merging

Additional constraints:

- A does not own `base.py`, only consumes stable Python interfaces exposed by B
- B does not directly own `libge_python_pass_bridge.so`, only provides stable Python / native helper capabilities needed by the bridge

### 12.6 ATC Parameter Integration

Modify the following files:

- `api/atc/main_impl.cc`

Responsibilities are as follows:

- Currently does not add new user pass routing parameters
- If CLI / options are added later, they should ultimately converge to `ASCEND_GE_PY_PASS_PATH` or `ge.passes.bootstrap` unified discovery protocol

### 12.7 Documentation, Samples, Testing

Add and modify the following files:

- `examples/fusion_pass/README.md`
- `examples/fusion_pass/python_pass_development_guide.md`
- `examples/fusion_pass/*/python/*`
- `tests/ge/ut/ge/graph/pyge_tests/*_test.py`
- `tests/ge/python_pass/*`

Responsibilities are as follows:

- Provide Python samples
- Provide user usage documentation
- Complete unit tests and integration validation

## 13. Collaboration and Advancement Approach

Notes:

- This design document is responsible for maintaining long-term stable architectural boundaries, A/B collaboration constraints, interface freeze points, and acceptance principles
- `PLAN.md` is the sole source for phase progress, checklists, and completion status
- Phase objectives, A/B sub-objectives, and completed/pending status will only be updated in `PLAN.md`; this design document will no longer maintain similar progress information

### 13.1 Overall Collaboration Principles

It is recommended to continuously advance along two parallel workflows, with the basic principle of "freezing interfaces first, separating write sets as much as possible, and unifying phase status back to `PLAN.md`".

Parallel collaboration follows these constraints:

- Freeze public interfaces first, then develop concurrently:
  - Descriptor/callback protocol between `ge.passes._bridge` and native bridge
  - Minimum interfaces visible to Python in `ge.graph` / `ge.es` / `_ge_pass_native.so`
- Progress tracking is maintained uniformly in `PLAN.md`
- The design document only retains long-term valid boundaries, not process records of "who completed what"
- Phase acceptance can be executed gradually by phase, but whether a "phase is complete" is determined by the checklist in `PLAN.md`

### 13.2 A/B Workflow Boundaries

The long-term boundaries of the two workflows are as follows:

- A focuses on compiler / native bridge / loader / adapter / fallback / existing `ge.graph.Graph` integration
- B focuses on `_ge_pass_native.so`, `base.py`, `PassContext` / `MatchResult`, Python samples, and Python API completion

The fixed boundaries directly related to current implementation are as follows:

- A is responsible for `libge_python_pass_bridge.so`, `pass_plugin_loader`, `ge_compiler.so` stable ABI, and existing `ge.graph.Graph` borrowed view integration
- B is responsible for `_ge_pass_native.so`, `base.py`, native-backed form of `PassContext` / `MatchResult`, and subsequent completion of samples / Python API
- `Graph` prioritizes reusing the existing `ge.graph.Graph`
- `_ge_pass_native.so` no longer introduces a second user-visible `Graph` type
- A does not own `base.py`
- B does not own `libge_python_pass_bridge.so`

### 13.3 Phase Advancement and Delivery Approach

Subsequent phases should collaborate in the following order:

1. First freeze phase boundaries and completion definitions, and write them into `PLAN.md`
2. Then freeze minimum interfaces between A/B
3. Develop in parallel according to write sets
4. Prioritize completing test samples and targeted validation during integration
5. After passing, update checklist status back to `PLAN.md`

For phase definitions themselves, it is recommended to maintain the following long-term order:

- `FusionBasePass` formally connected
- `PatternFusionPass` connected
- `DecomposePass` connected
- Fallback and prebuilt version connected
- Python equivalent implementation of samples
- Supporting documentation, validation, and delivery materials completed

Detailed sub-items, status, and blockers for these phases are uniformly referenced in `PLAN.md` and will not be expanded again in the design document.

### 13.4 Phase Acceptance and Documentation Synchronization

At the conclusion of each phase, it is recommended to complete the following actions synchronously:

- Update the completion status of the corresponding checklist in `PLAN.md`
- If interface boundaries change, update this design document
- If only status changes, do not modify phase descriptions in this design document
- No longer add independent phase progress/acceptance markdown files; process progress is uniformly written back to `PLAN.md`
- Preserve minimum validation commands, result summaries, and known blockers for phase acceptance

This ensures:

- `PLAN.md` reflects real-time progress
- The design document remains stable and is not polluted by process status
- A/B always has a single source of progress during integration

## 14. Validation and Acceptance Requirements

### 14.1 Test Layering

V1 testing is recommended to be organized in four layers:

- Python unit tests:
  - `registry`, `decorator`, `bootstrap`
  - Environment variable primary path, subsequent `entry_points` auto-discovery
  - Descriptor normalization
  - Stale handle checking for borrowed handles
- C++ / native unit tests:
  - Bridge initialization and repeated initialization
  - TLS creation context
  - Dynamic registration
  - Session lifecycle
  - Exception isolation
- Integration tests:
  - `FusionBasePass`
  - `PatternFusionPass`
  - `DecomposePass`
  - `MatchResult`
  - `GeUtils.InferShape`
  - `GeUtils.CheckNodeSupportOnAicore`
- Packaging and installation tests:
  - Main wheel installation
  - Native sub-wheel selection
  - Fallback compilation when no matching version
  - Development path / direct module passing

### 14.2 Phase Acceptance Principles

The completion definition, A/B sub-objectives, status, and blockers for each phase are uniformly based on `PLAN.md`, and this design document will no longer maintain hard-coded acceptance nodes for "phase one/phase two/phase three".

Phase acceptance is recommended to uniformly adopt the following structure:

- Completion definition:
  - Directly reference the "phase completion definition" for the corresponding phase in `PLAN.md`
- Required validations:
  - At least cover one positive main chain
  - At least cover exception paths or failure isolation for newly added interfaces in that phase
  - At least cover lifecycle, repeated loading, or resource release parts directly related to that phase
- Conclusion requirements:
  - `PLAN.md` corresponding checklist status update completed
  - Affected interface boundaries, loading relationships, or responsibility divisions in the design document updated
  - Validation commands, result summaries, and known blockers preserved in phase acceptance records

Additionally, one constraint needs to be followed:

- A capability belongs to whichever phase it is in, and is accepted in that phase, not merged into other phases in advance
- For example, productization capabilities like `entry_points`, prebuilt versions, multi-version native artifacts, and fallback should be based on the corresponding phase in `PLAN.md`, rather than being written into the completion criteria of earlier phases

### 14.3 Milestone Organization Recommendations

Considering the project advances on a "two-week" rhythm, formal milestones are recommended to be organized dynamically according to `PLAN.md` current priorities, rather than hard-coding "certain two phases must be accepted together" in this design document.

It is recommended that each milestone follows these principles:

- One milestone should try to conclude only `1~2` themes that can form a closed loop
- Prioritize organizing around the current main objective, for example:
  - `FusionBasePass` closed loop
  - `PatternFusionPass` closed loop
  - `DecomposePass` closed loop
  - Fallback / prebuilt version closed loop
  - Sample and delivery materials closed loop
- Each milestone should output:
  - Completed checklist delta for this round
  - Targeted validation commands and result summaries
  - Remaining blockers
  - Handoff prerequisites for the next milestone

Extension items are recommended to be organized as separate milestones, not strongly bound to V1 main chain acceptance. For example, `REGISTER_CUSTOM_PASS` is more suitable for separate acceptance after the main chain stabilizes.

### 14.4 Recommended Acceptance Deliverables

Each formal acceptance is recommended to preserve at least the following deliverables:

- Test result summary table:
  - Test case name
  - Covered capability points
  - Execution result
  - Failure reason and conclusion
- Sample execution records:
  - Input model or script
  - Triggered pass
  - Key logs or result summaries
- Known issues list:
  - Whether it blocks the next phase
  - Workaround approach
  - Planned fix phase

### 14.5 Overall Acceptance Dimensions

V1 final acceptance is recommended to be based on the following dimensions:

- Discovery and loading:
  - Discovery mechanisms currently within scope are consistent with `PLAN.md` and consistent with user documentation
  - If `entry_points`, prebuilt versions, or fallback are included in this round, corresponding chains have independent acceptance records
- Three types of pass main chains:
  - `FusionBasePass`, `PatternFusionPass`, `DecomposePass` are discoverable, registrable, creatable, and executable within their respective scopes
- Python / native wrapper:
  - `Graph`, `PassContext`, `MatchResult`, and helpers required by the phase have corresponding capabilities
- Stability:
  - Python pass import failures can be isolated
  - Python pass execution exceptions can be isolated
  - Lifecycle, repeated loading, and resource release semantics have validation records
- Delivery and materials:
  - `PLAN.md`, design documents, samples, validation records, and limitation descriptions remain consistent

### 14.6 Phase Progress Reference

Current phase completion status, A/B sub-objectives, incomplete items, and blockers are not repeatedly maintained in this design document; `PLAN.md` is the sole reference.

This design document only retains the following long-term acceptance requirements:

- Each phase conclusion must synchronously update `PLAN.md`
- If interfaces, responsibility boundaries, or loading relationships change, this design document must be synchronously updated
- If only task status changes, only update `PLAN.md`
- Phase acceptance requires preserving validation commands, result summaries, and known blockers

## 15. Risks and Points of Attention

- If `CreateFusionPassFn` is directly changed to `std::function`, it may introduce `dlclose` and global destructor ordering coupling risks; prioritize adopting the "function pointer + TLS creation context + process-level registry" approach.
- `Graph` borrowed wrapper has been implemented, but subsequently added runtime wrappers must still comply with the constraint of "not taking runtime handle ownership by default" to avoid reintroducing double-free risks.
- The local development environment is Python 3.13, while the formal wheel planning is `cp39-cp314`; local testing can only cover Python tags available in the current environment, and the full matrix requires release pipelines to validate separately by Python minor version.
- V1 needs to clearly distinguish two types of native strategies in documentation and implementation: `ge.graph` continues to use ctypes/C wrapper, Python pass bridge uses `pybind11`.
- If more Python version-sensitive logic continues to be compiled directly into `ge_compiler.so`, the evolution space for subsequent bridge artifact sets and fallback will be compressed; therefore, subsequent implementations must converge version differences into replaceable `libge_python_pass_bridge.so` / `_ge_pass_native.so`.
- Within `atc.bin`, Python may be initialized first by TBE, or Python may be initialized first by the Python pass bridge; fallback selection must be based on the current in-process interpreter or unified selector, and bridge state cleanup should be designed separately from interpreter finalize, ensuring the interpreter is only finalized by the owner after all Python users have cleaned up.
- `PatternFusionPass` Python implementation is not a simple function callback; it must ensure that the pattern/match/replacement three-stage semantics are consistent with the existing C++ framework.
- Before B's `_ge_pass_native` is implemented, local validation can temporarily not depend on it, but this can only be used for independent bridge `.so` splitting and `FusionBasePass` regression; it cannot replace the final acceptance of `PatternFusionPass`.
- Providing Python equivalents for all 9 samples will expand wrapper coverage; priority should be given to ensuring main chain availability before gradually completing edge interfaces.

## 16. Reuse Boundaries with Subsequent Custom Operator Python Implementation

This scheme is not just for Python pass as a single point service, but is laying a generic foundation for "GE/CANN main framework safely calling Python extension capabilities".

Subsequently, if "custom operator Python implementation" is advanced, a large part of the infrastructure in this scheme can be directly reused, but operator definition, delivery, and sinking-related capabilities still need new dedicated designs.

### 16.1 Directly Reusable Capabilities

The following capabilities can be directly reused in subsequent custom operator Python implementation:

- Python plugin discovery protocol: Currently based on the environment variable primary path as baseline, with `entry_points` auto-discovery to be added later; this protocol can be abstracted into a generic Python extension discovery framework.
- Main wheel + native sub-wheel + fallback codegen: This "pure Python main package + multi-Python version native companion package + local fallback compilation when version not matched" release mode is essentially independent of pass/custom op.
- Bridge lifecycle management: Python interpreter initialization, repeated initialization idempotency, exception isolation, unload order control, logging, and diagnostics can all be reused.
- GIL and lock strategies: The strategy of "management lock + session lightweight state + GIL precise surround for Python callbacks" defined in the current document applies to the vast majority of subsequent Python extension integration points.
- Execution session / holder model: Runtime objects and Python wrapper owner tokens, stale handle checks, borrowed objects converting to Python exceptions rather than crashes after becoming invalid; this mechanism also applies to custom operator callback period objects.
- Python registry and descriptor model: The registry, descriptor, bootstrap, and bridge exit in the current pass design can evolve into a generic "Python extension registry".
- Pythonic constraints: User experience goals of not requiring manual object release, not requiring manual GIL management, and not requiring explicit close/release from users should continue to be maintained.

Looking only at the foundation layer of "Python capabilities formally integrated into GE/CANN", the reuse rate at this layer is estimated to reach `60%~70%`.

### 16.2 Partially Reusable Capabilities

The following capabilities can reuse a portion but need to be tailored according to custom operator semantics:

- `ge.graph` / `ge.es` wrapper: If subsequent custom operator Python implementation needs to express schema, infer logic, graph-level replacement, or eager graph entry, these graph wrappers still have value; but they cannot replace operator schema's own modeling.
- Plugin path and installation path management: The current repository already has custom opp path and plugin manager capabilities, such as `custom_op_lib_path_` related logic in `graph_metadef/base/common/plugin/plugin_manager.cc`, which indicates there is an existing foundation for "plugin discovery" and "custom deliverable installation paths"; but Python custom op still needs to define clearer mapping rules between Python packages and OPP packages.
- Error propagation and degradation strategy: The "single plugin failure does not bring down the main chain" approach in current passes still applies, but custom ops are often closer to the compilation main flow than passes; which errors allow degradation and which errors must hard fail needs to be re-graded.
- ATC parameter entry: The parameter integration approach reserved for `--py_pass_path`, `--py_pass_module` in this scheme can be extended to Python custom op later; but parameter naming, priority, and relationship with existing custom opp directory configuration still need to be determined.

This layer is more like "mechanism reuse", not "implementation direct copy". The reuse ratio is roughly `20%~30%`.

### 16.3 Capabilities Needing New Addition for Custom Operators

The following parts basically cannot be directly obtained from the Python pass scheme and need separate design:

- Custom operator schema/OpDef registration model: Including inputs, outputs, attributes, constraints, default values, version, and namespace.
- Shape/type inference Python registration and execution model.
- Kernel delivery chain: For example, AscendC/Triton/TBE/host-side implementation, compilation artifacts, binary layout, version validation.
- OPP package layout and installation protocol: Directory organization relationships between Python packages, op proto, kernel binaries, tiling files, and configuration files.
- Compile/runtime recognition logic: How ATC and online compilation phases discover, validate, and sink Python custom ops.
- Custom operator and framework adaptor connection: For example, the PyTorch/TensorFlow graph entry methods shown in current `examples/custom_op`; this part is clearly beyond the scope of pass responsibilities.

If the subsequent goal is only "Python writing schema + infer + registration", the reuse from the current pass scheme will be higher; if the goal also includes "Python side fully driving kernel development, packaging, and publishing", the new work will increase significantly.

### 16.4 Reuse Conclusion

It is recommended to view "Python pass implementation" and "Python custom op implementation" as two phases:

- The first phase completes Python pass first, stabilizing the generic foundation of Python plugin integration, lifecycle, memory management, lock/GIL, packaging, and fallback.
- The second phase adds custom operator-specific models on this foundation, including schema, infer, kernel delivery, and OPP layout.

Rough estimate from the project dimension:

- Infrastructure layer reuse: `60%~70%`
- Overall project reuse: `40%~50%`

This ratio is sufficient to demonstrate that the current Python pass design is not a one-time solution, but is proactively building a public foundation that subsequent Python custom ops can continuously reuse.
