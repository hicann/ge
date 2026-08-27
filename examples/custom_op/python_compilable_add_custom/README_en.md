# Python CompilableAddCustom online and offline compilation sample

This sample exercises the GE Python `CompilableOp` interface through both
compilation paths with one shared Python `compile`/`declare_launch_args`
implementation:

- Online: GE `CustomGraphOptimizer` calls `compile` while a Python `Session`
  graph is being built, then executes the graph immediately.
- Offline: a C++ graph builder emits AIR, ATC invokes the same Python callback
  to produce an OM, and a C++ ACL runner loads that OM after the Python plugin
  has been removed from `ASCEND_CUSTOM_OPP_PATH`.

## Pipeline

```text
Python plugin load
    -> register_op_impl discovers compile / declare_launch_args
    -> CustomGraphOptimizer calls compile(x, y, z)
    -> get_compile_platform_info() reads NpuArch/SoC
    -> BiSheng + llvm-objcopy produce owned kernel bytes
    -> declare_launch_args publishes the launch descriptor
    ├── run_online.sh  -> GE Session online execution
    └── run_offline.sh -> AIR -> ATC -> OM -> ACL without the Python plugin
```

## Prerequisites

- A CANN installation matching the GE build (`source /path/to/cann/set_env.sh`).
- `cmake`, `atc`, `bisheng`, `llvm-objcopy`, and Python 3; offline execution also
  needs the ACL development libraries.
- An NPU for online execution and OM execution. AIR-to-OM compilation itself
  can run on the host.
- The sample kernel is float32-only and requires an element count divisible by
  1024.

## Run

### Online

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/python_compilable_add_custom
bash run_online.sh
```

The script builds the proto/ES wrapper, sets `ASCEND_CUSTOM_OPP_PATH` to both
the OPP root and the Python plugin directory, and runs the Python `Session`. A
successful run prints:

```text
PY_COMPILE_MODULE_LOADED=1
PY_COMPILE_CALLBACK_ENTER=1 mode=online ...
PY_COMPILE_ONLINE_NPU=PASS
```

### Offline

```bash
source /path/to/cann/set_env.sh
cd examples/custom_op/python_compilable_add_custom
bash run_offline.sh
```

The script defaults to `Ascend910B1`; set `PYTHON_COMPILABLE_ADD_SOC_VERSION`
before running it when compiling for another SoC.

The script:

1. Builds the C++ `REG_OP` deliverable, AIR exporter, and ACL OM runner.
2. Sets `ASCEND_CUSTOM_OPP_PATH=<OPP-root>:<Python-plugin-dir>` and generates AIR.
3. Runs ATC; its log must contain `PY_COMPILE_CALLBACK_ENTER=1 mode=offline`.
4. Removes the Python plugin path, keeps only the OPP root, loads the OM, and
   checks `x + y = 3`.

The final C++ runner prints:

```text
PY_COMPILE_OFFLINE_OM=PASS
```

This demonstrates that Python `compile` participates in model compilation but
is not needed to execute the resulting OM.

## Layout

```text
python_compilable_add_custom
├── CMakeLists.txt                              # proto, ES wrapper, offline tools
├── run_online.sh                               # online compile and execution
├── run_offline.sh                              # AIR -> ATC -> OM -> ACL
├── kernel/add_custom.asc                       # Ascend C source compiled by callback
├── python/es_custom/__init__.py              # package entry template for generated ES wrapper
├── proto/add_custom.h                          # PythonCompilableAddCustom prototype
├── proto/add_custom.cc                         # shape/data-type inference
├── src/ge/python_compilable_add_custom.py     # compile + declare_launch_args
├── src/run.py                                  # online graph and execution
├── src/offline_graph_build.cc                  # offline AIR exporter
└── src/offline_model_exec.cc                   # OM execution without Python
```

## Implementation notes

- `compile` uses `get_compile_platform_info()` to obtain
  `get_platform_resource("version", "NpuArch")` and `get_soc_version()`;
  compiler failures propagate as graph-compilation failures.
- The generated `.aicore.o` is cached by shape/dtype key in the Python holder.
  The cache is cleared when the SoC or NPU architecture changes, so a binary
  from another target cannot be reused. The on-disk key also includes the
  SoC, source contents, and Ascend C include path, so changing the source or
  target platform triggers a rebuild. A cache miss in `declare_launch_args` is
  an explicit error; launch
  declaration never silently recompiles.
- `ASCEND_CUSTOM_OPP_PATH` must contain both the OPP root (for the C++ `REG_OP`
  prototype) and the Python plugin directory (for the Python custom-op loader).
- Both paths reuse GE's existing `AnnotatedArgs` launch mechanism. No execution-
  time Python callback is added and no Python state is written into the OM.

## Verification

Verify the complete paths with `bash run_online.sh` and `bash run_offline.sh`.
