# Python Declarative Address Refresh Add Custom Operator Online Sample

## Sample Overview

- Graph building entry: `GE` (Python ES graph-building APIs)
- Operator language: `Python` (prototype and callbacks) + `Ascend C` (kernel)
- Compilation mode: `bisheng` compiles the Ascend C kernel; the Python `compile` callback generates and caches the binary by shape
- Core path: `Python prototype registration -> ES graph building -> Session online execution -> compile/declare_launch_args/execute callbacks`
- Comparison target: declarative address refresh `AnnotatedAddCustom` versus no-refresh `NoRefreshAddCustom`

This sample defines two functionally identical Add custom operators. Both take `[8192]` float32 inputs and
dispatch the same `add_custom` Ascend C kernel:

- `AnnotatedAddCustom` provides `compile` and `declare_launch_args` through `register_op_impl`. `compile`
  invokes `bisheng` to compile `add_custom.asc` and caches the binary by input shape/dtype;
  `declare_launch_args` declares two input address slots and one output address slot, and GE refreshes
  addresses according to that layout during repeated execution.
- `NoRefreshAddCustom` provides `execute` through `register_op_impl`. On every graph execution it allocates
  the output tensor, loads the binary through the ACL Python API, reassembles the kernel arguments, and
  serves as the performance baseline.

The C++ `REG_OP` declarations in `proto/add_custom.h` are used only by `gen_esb` to generate the
graph-building APIs. The runtime prototypes and `infer_meta` functions are supplied by Python decorators,
and all operator execution is implemented by Python callbacks. `online/cpp/ge/custom_op.cpp` is not compiled.

`run.py` builds the two graphs above. Each graph alternates between two sets of device buffers, performs
warm-up, 100 benchmark iterations, and precision checks, then prints the latency of both paths and the
speedup.

## Applicable Scenarios

- Understanding how Python `register_op`/`register_op_impl` declare kernel tasks and args address layouts.
- Comparing declarative address refresh with no-refresh paths in repeated online execution.
- Verifying the complete chain of Python-side prototype, compile callback, and execute callback.

## Prerequisites

### CANN

- The CANN environment is installed and configured, for example by running `source ${ASCEND_HOME_PATH}/set_env.sh`.
- The environment provides the required `ACL`, `GE`, and `Graph` headers and libraries.
- Refer to the [Installation Guide](../../../../../docs/en/quick_install.md) to install the toolkit and ops packages.

### Frameworks and Plugins

- Python 3 and `pip`.
- The ES graph-building APIs are provided by the `es_custom` wheel built by this sample. No extra
  framework plugin is required.

### Environment Variables

- `ASCEND_HOME_PATH`
- `run.sh` automatically sets `PYTHONPATH`, `LD_LIBRARY_PATH`, `ASCEND_CUSTOM_OPP_PATH`,
  `GE_PYTHON_CUSTOM_OP_SOURCE`, and `GE_PYTHON_CUSTOM_OP_BINARY`.

### Additional Dependencies

- `cmake`
- `bisheng` and `llvm-objcopy` (shipped with the CANN toolchain, used to compile and extract the kernel binary)

## Quick Run

Run the following commands in the `examples/custom_op/annotated_args_refresh_add_custom/online/python` directory:

### Recommended

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` automatically compiles the kernel, builds and installs the OPP/ES wheel, and runs the online
comparison:

1. `bisheng` compiles `add_custom.asc` and `llvm-objcopy` extracts the `.aicore_binary` section into the
   kernel binary.
2. Builds the custom OPP registration library and the Python ES wheel, then installs it.
3. Runs the online precision and performance comparison of the two graphs.

On success the terminal prints something like:

```text
[OnlinePython] AnnotatedAddCustom precision check PASS
[OnlinePython] NoRefreshAddCustom precision check PASS
[Perf] input shape: [8192], dtype: float32
[Perf] iters: 100
[Perf] AnnotatedAddCustom: xxx us (avg xxx us/iter)
[Perf] NoRefreshAddCustom: xxx us (avg xxx us/iter)
[Perf] Annotated speedup: xxx x
[OnlinePython] NPU_EXECUTION=PASS
```

### Step by Step

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
mkdir -p build
# 1. Compile the Ascend C kernel and extract the binary
bisheng -c ../cpp/add_custom_kernel/add_custom.asc -o build/add_custom.host.o --npu-arch=dav-2201
llvm-objcopy -O binary --only-section=.aicore_binary build/add_custom.host.o build/add_custom.aicore.o
# 2. Build the custom OPP and ES wheel
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target build_es_custom -j"$(nproc)"
# 3. Install the ES wheel
python3 -m pip install --target build/whl_package build/es_output/whl/es_custom-1.0.0-py3-none-any.whl
# 4. Set environment variables and run the comparison
export PYTHONPATH="$(pwd)/build/whl_package:$(pwd)/src:$PYTHONPATH"
export LD_LIBRARY_PATH="$(pwd)/build/es_output/lib64:$LD_LIBRARY_PATH"
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/build/opp:$(pwd)/src/ge"
export GE_PYTHON_CUSTOM_OP_SOURCE="$(pwd)/../cpp/add_custom_kernel/add_custom.asc"
export GE_PYTHON_CUSTOM_OP_BINARY="$(pwd)/build/add_custom.aicore.o"
DEVICE_ID=0 python3 src/run.py
```

The NPU architecture defaults to `dav-2201` and can be overridden with `ADD_CUSTOM_NPU_ARCH`; `DEVICE_ID`
defaults to `0`.

## Directory Structure and Key Files

```text
annotated_args_refresh_add_custom
└── online/python
    ├── CMakeLists.txt                     # Build the custom OPP library and ES wheel
    ├── README.md
    ├── README_en.md
    ├── run.sh                             # Build and run the online comparison
    ├── proto
    │   └── add_custom.h                   # C++ graph prototype consumed by gen_esb
    └── src
        ├── run.py                         # Validate and benchmark the two online graphs
        └── ge
            └── annotated_add_custom.py    # Python prototypes, infer_meta, and implementations
```

Key files:

- `src/ge/annotated_add_custom.py`
  Implements `AnnotatedAddCustom` and `NoRefreshAddCustom`. The former generates the binary in the compile
  callback and declares the address slots during task generation; the latter dispatches the same kernel
  through the eager execution path.
- `proto/add_custom.h`
  Registers the `AnnotatedAddCustom` and `NoRefreshAddCustom` graph operator types for `gen_esb` to
  generate the Python ES graph-building APIs.
- `src/run.py`
  Builds the two graphs, alternates two sets of input/output device addresses, performs 5 warm-up
  iterations, 100 benchmark iterations, and validates precision.
- `CMakeLists.txt`
  Builds the custom OPP registration library (`libcust_opapi.so`) and the Python ES wheel.

## Core Path

### Online Execution

1. `run.py` builds `python_annotated_graph` and `python_no_refresh_graph`, both with `[8192]` float32 inputs.
2. The `compile` callback of `AnnotatedAddCustom` reads `add_custom.asc`, invokes `bisheng`, extracts the
   `.aicore_binary` section, and caches the binary by input shape/dtype.
3. The `declare_launch_args` callback of `AnnotatedAddCustom` sets the kernel name, binary, and block
   dimension, then declares `InputAddr{0}`, `InputAddr{1}`, and `OutputAddr{0}`.
4. The `execute` callback of `NoRefreshAddCustom` allocates the output tensor, loads the binary through
   the ACL Python API, assembles the args, and dispatches the kernel.
5. The two graphs alternate between two sets of device addresses and report total time, average time,
   and speedup.

### Declarative Address Refresh

```text
Compile phase:
  compile(x, y, z)
    ├─ Read add_custom.asc
    ├─ bisheng + llvm-objcopy extract .aicore_binary   -> kernel binary
    └─ Cache the binary by input shape/dtype

Task generation phase:
  declare_launch_args(x, y, z)
    ├─ KernelArgs: InputAddr{0}, InputAddr{1}, OutputAddr{0}
    ├─ AnnotatedKernelLaunchInfo { kernel_name, kernel_bin, block_dim, stream_id }
    └─ ctx.add_launch(launch_info, args)

Execution phase:
  GE refreshes the current input/output addresses according to the saved args layout
```

### No-Refresh Baseline

```text
Every graph execution:
  NoRefreshAddCustom.execute(x, y)
    ├─ ctx.malloc_output_tensor()
    ├─ acl.rt.binary_load_from_file() / binary_get_function()
    ├─ kernel_args_init / append(x, y, z) / finalize
    └─ acl.rt.launch_kernel_with_config()
```

This implementation declares no address slots in the args. Both graphs share identical computation logic
and kernel, so the performance difference reflects the effect of declarative address refresh.

## Build Artifacts

- `build/add_custom.aicore.o`
  The device-side binary of the Ascend C Add kernel, compiled with `bisheng` and extracted from the `.aicore_binary` section.
- `build/opp/`
  The custom operator OPP registration library (`op_proto/custom/libcust_opapi.so` plus platform copies), exposed to GE through `ASCEND_CUSTOM_OPP_PATH`.
- `build/es_output/`
  ES graph-building deliverables: `lib64/libes_custom.so` and `whl/es_custom-1.0.0-py3-none-any.whl` (installed into `build/whl_package/`).

## Result Verification

On success you can observe:

- The terminal output contains `AnnotatedAddCustom precision check PASS` and `NoRefreshAddCustom precision check PASS`.
- The terminal output contains `AnnotatedAddCustom`, `NoRefreshAddCustom`, and `Annotated speedup`.
- The terminal output contains `NPU_EXECUTION=PASS` and `Online Python custom-op pipeline PASS`.

If it fails, check first:

- Whether `ASCEND_HOME_PATH` is set and the CANN environment is loaded correctly.
- Whether `bisheng`, `llvm-objcopy`, and `cmake` are available.
- Whether `build/es_output/whl/es_custom-1.0.0-py3-none-any.whl` and `build/add_custom.aicore.o` were generated.
- Whether an NPU is available in the current environment.

## Notes / Limitations

- The kernel binary is compiled by `bisheng` inside the `compile` callback; the graph compilation phase
  includes this overhead, which is excluded from the 100 benchmark iterations.
- `ge.graphRunMode` is set to `1` (`PRIORITY_GRAPH`) to ensure the online execution path.
- The benchmark alternates two sets of device buffers to trigger input/output address changes.
- The speedup is affected by the NPU model and system load, and is for reference only.
- The NPU architecture defaults to `dav-2201` (Atlas A2); override it with `ADD_CUSTOM_NPU_ARCH` for
  other models.

## Appendix

### Operator Specification

| Item | Content |
| --- | --- |
| Operator type | `AnnotatedAddCustom` / `NoRefreshAddCustom` |
| Inputs | `x`, `y` |
| Output | `z` |
| Input/output shape | `[8192]` |
| Input/output data type | `float32` |
| Format | `ND` |
| Kernel name | `add_custom` (Ascend C, compiled with `bisheng`) |
| Block size | `1024` |

### Key Interfaces

| Interface | Operator | Purpose |
| --- | --- | --- |
| `ge.custom_op.register_op` + `infer_meta` | Both operators | Prototype registration and output shape/dtype inference |
| `register_op_impl` + `compile` | `AnnotatedAddCustom` | Compile the kernel with `bisheng` and cache the binary by shape |
| `register_op_impl` + `declare_launch_args` | `AnnotatedAddCustom` | Declare the kernel launch and input/output address slots |
| `register_op_impl` + `execute` | `NoRefreshAddCustom` | Allocate the output tensor and dispatch the kernel through the ACL API |
| `acl.rt.launch_kernel_with_config` | `NoRefreshAddCustom` | Dispatch the kernel directly through the ACL Python API |
