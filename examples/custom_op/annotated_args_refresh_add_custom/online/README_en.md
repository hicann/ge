# Declarative Argument Address Refresh Add Custom Operator Online Sample

## Sample Overview

- Graph construction entry: `GE`
- Operator programming language: `Ascend C` (RTC runtime compilation)
- Compilation method: `.cpp` files are compiled into the host-side custom operator, while the kernel source is compiled into a device binary by RTC at runtime
- Core pipeline: `Ascend C kernel source -> RTC runtime compilation -> GE deliverable -> in-process graph construction -> online execution through Session::ExecuteGraphWithStreamAsync`
- Comparison target: declarative address refresh `AnnotatedAddCustom` versus no-refresh `NoRefreshAddCustom`

This sample defines two functionally identical Add custom operators. Both accept `[4096, 4096]` float32 inputs (16M elements, 64MB) and launch the same `add_custom` Ascend C kernel:

- `AnnotatedAddCustom` derives from `CompilableOp`, `AnnotatedArgsOp`, and `ShapeInferOp`. `Compile` generates the kernel binary through RTC. `DeclareLaunchArgs` uses `AnnotatedKernelArgs(InputAddr, InputAddr, OutputAddr)` to declare two input address slots and one output address slot, allowing GE to refresh addresses according to that layout during repeated execution.
- `NoRefreshAddCustom` derives from `EagerExecuteOp` and `ShapeInferOp`. It allocates and copies device args during model loading but does not declare address slots in the args, serving as the performance baseline.

`session_run` constructs only these two graphs. Each graph alternates between two sets of device memory, performs warmup, collects 100 iterations of performance data, verifies accuracy, and prints both elapsed times plus the `no-refresh / annotated` speedup.

## Applicable Scenarios

- Learn how `AnnotatedArgsOp::DeclareLaunchArgs` declares a kernel task and its argument address layout.
- Compare declarative address refresh with no refresh during repeated online execution.

## Prerequisites

### CANN

- The CANN environment is installed and configured, for example by running `source ${ASCEND_HOME_PATH}/set_env.sh`.
- The environment provides the required `ACL`, `GE`, and `Graph` headers and libraries.
- Refer to the [Installation Guide](../../../../docs/en/quick_install.md) to install the toolkit and ops packages.

### Frameworks and Plugins

- This sample does not depend on PyTorch, TensorFlow, or TorchAir.
- `add_custom_kernel/add_custom.asc` is compiled through RTC at runtime and requires no precompilation.

### Environment Variables

- `ASCEND_HOME_PATH`
- `run.sh` appends this sample's `output/` to `ASCEND_CUSTOM_OPP_PATH`

### Additional Dependencies

- `cmake`
- `g++`

## Quick Run

Run the following commands in `examples/custom_op/annotated_args_refresh_add_custom/online`:

### Recommended Method

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` configures, builds, and installs the sample, then appends this directory's `output/` to `ASCEND_CUSTOM_OPP_PATH`:

1. Build the custom-op deliverable and `annotated_args_refresh_session_run`
2. Run the online accuracy and performance comparison for the two graphs

Successful output is similar to:

```text
[Perf] input shape: [4096, 4096], float32, 64MB
[Perf] iters: 100
[Perf] AnnotatedAddCustom: xxx us (avg xxx us/iter)
[Perf] NoRefreshAddCustom: xxx us (avg xxx us/iter)
[Perf] Annotated speedup: xxx x
```

### Step-by-Step Method

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cmake --install build
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"

cd build
./annotated_args_refresh_session_run
cd ..
```

`cmake --install build` installs the proto header and kernel source from this directory into this directory's `output/op_graph/`. `export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"` adds the custom operator package root to the environment variable, after which GE loads the deliverable from `output/op_graph/lib/<os>/<arch>/libcust_opapi.so`.

## Directory Structure and Key Files

```text
annotated_args_refresh_add_custom
└── online
    ├── CMakeLists.txt
    ├── README.md
    ├── README_en.md
    ├── run.sh
    ├── add_custom_kernel
    │   ├── add_custom.asc            // Shared Ascend C Add kernel
    │   └── add_custom_kernel.h       // Kernel name and block size
    ├── ge
    │   ├── add_custom_ir.h              // Proto registration for both operators
    │   ├── custom_op.cpp             // Compile, DeclareLaunchArgs, Execute, and shape inference
    │   └── utils
    │       ├── log.h                 // Logging macros
    │       ├── rtc_kernel_loader.h   // RTC kernel loader interface
    │       └── rtc_kernel_loader.cpp // RTC compilation and loading implementation
    └── session_run
        └── main.cc                   // Online accuracy and performance comparison
```

Key files:

- `ge/custom_op.cpp`
  Implements `AnnotatedAddCustom` and `NoRefreshAddCustom`. The former generates a binary at compile time and declares address slots during task generation; the latter launches the same kernel through the eager execution path.
- `ge/add_custom_ir.h`
  Registers the `AnnotatedAddCustom` and `NoRefreshAddCustom` graph operator types.
- `add_custom_kernel/add_custom.asc`
  Shared element-wise float32 Add kernel with a block size of `1024`.
- `ge/utils/rtc_kernel_loader.cpp`
  RTC loader used by `NoRefreshAddCustom` to read source, compile it, load the binary, and obtain a function handle.
- `session_run/main.cc`
  Constructs two graphs, alternates two sets of input/output device addresses, warms up each graph 5 times, measures 100 iterations, and verifies accuracy.

## Core Pipeline

### Online Execution

1. `session_run/main.cc` constructs `annotated_graph` and `no_refresh_graph`, both using `[4096, 4096]` float32 inputs.
2. `AnnotatedAddCustom::Compile` reads `add_custom.asc`, obtains the NPU architecture from the compile context, runs RTC compilation, and caches the binary by input storage shape.
3. `AnnotatedAddCustom::DeclareLaunchArgs` sets the kernel name, binary, and block dimension, then declares `InputAddr{0}`, `InputAddr{1}`, and `OutputAddr{0}`.
4. `NoRefreshAddCustom::Execute` compiles and loads the same kernel through `RtcKernelLoader` during model loading, allocates device args, and calls `aclrtLaunchKernelV2`.
5. Each graph alternates between two sets of device addresses and reports total time, average time, and speedup.

### Declarative Address Refresh

```text
Compile time:
  Compile()
    ├─ Read add_custom.asc
    ├─ aclrtcCompileProg()                         -> Generate kernel binary
    └─ Cache the binary by input storage shape

Task generation:
  DeclareLaunchArgs(ctx)
    ├─ AnnotatedKernelArgs(InputAddr{0}, InputAddr{1}, OutputAddr{0})
    ├─ AnnotatedKernelLaunchInfo { kernel_name, kernel_bin, block_dim }
    └─ ctx.AddLaunch(launch_info, std::move(args))

Execution:
  GE refreshes current input/output addresses according to the saved args layout
```

### No-Refresh Baseline

```text
During model loading:
  NoRefreshAddCustom::Execute()
    ├─ RtcKernelLoader::Load()                     -> RTC compiles and loads the shared kernel
    ├─ MallocOutputTensor()
    ├─ aclrtMalloc() + aclrtMemcpy()               -> Prepare device args
    └─ aclrtLaunchKernelV2()
```

This implementation does not declare address slots in its args. Because both graphs use identical computation and the same kernel, the measured difference reflects the declarative address-refresh effect.

## Build Products

- `output/op_graph/lib/linux/x86_64/libcust_opapi.so`
  GE custom-op deliverable on Linux x86_64; aarch64 uses `output/op_graph/lib/linux/aarch64/libcust_opapi.so`.
- `output/op_graph/lib/<os>/<arch>/add_custom.asc`
  Kernel source used for RTC compilation.
- `output/op_graph/include/add_custom_ir.h`
  Graph-side operator proto header.
- `build/annotated_args_refresh_session_run`
  Online accuracy and performance comparison executable.

## Result Validation

When successful:

- Each graph prints `Precision check passed`.
- The terminal output contains `AnnotatedAddCustom`, `NoRefreshAddCustom`, and `Annotated speedup`.
- The library, kernel source, and proto header under `output/op_graph/` all come from this directory.

If execution fails, check:

- Whether `ASCEND_HOME_PATH` is set and the CANN environment is sourced.
- Whether `ASCEND_CUSTOM_OPP_PATH` includes this sample's `output/`.
- Whether an NPU is available.
- Whether `output/op_graph/lib/<os>/<arch>/libcust_opapi.so`, `add_custom.asc`, and `output/op_graph/include/add_custom_ir.h` were generated.

## Precautions and Limitations

- RTC compilation adds overhead during graph compilation or model loading; the 100 measured iterations exclude those stages.
- `ge.graphRunMode` is set to `1` (`PRIORITY_GRAPH`) to use the online execution path.
- The benchmark alternates between two sets of device memory to change input/output addresses.
- Speedup depends on the NPU model and system load and is provided for reference only.

## Appendix

### Operator Specifications

| Item | Content |
| --- | --- |
| Operator type | `AnnotatedAddCustom` / `NoRefreshAddCustom` |
| Inputs | `x`, `y` |
| Output | `z` |
| Input/output shape | `[4096, 4096]` |
| Input/output data type | `float32` |
| Format | `ND` |
| Kernel name | `add_custom` (Ascend C, RTC runtime compilation) |
| Block size | `1024` |

### Key Interfaces

| Interface | Operator | Purpose |
| --- | --- | --- |
| `CompilableOp::Compile` | `AnnotatedAddCustom` | RTC-compile and cache a kernel binary by shape |
| `AnnotatedArgsOp::DeclareLaunchArgs` | `AnnotatedAddCustom` | Declare the kernel launch and input/output address slots |
| `EagerExecuteOp::Execute` | `NoRefreshAddCustom` | Prepare output and device args, then launch the kernel during model loading |
| `ShapeInferOp::InferShape` | Both operators | Set output shape equal to input shape |
| `ShapeInferOp::InferDataType` | Both operators | Set output data type equal to input data type |
