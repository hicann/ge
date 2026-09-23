# HostCpu Host Scheduling Add Custom Op Sample

## Overview

This sample implements `AddCustom` as a **complete custom op that registers both Device and HostCPU backends**:

- `REG_OP(AddCustom)` registers a dedicated op type;
- The Device backend (`EagerExecuteOp`) compiles the Ascend C kernel (`add_custom.asc`) with RTC at first execution and launches it on the NPU;
- The HostCPU backend (`HostCpuExecuteOp`) performs float/float16 vector addition on the host.

`HostcpuEngineUpdatePass` selects the backend at runtime: small dynamic-shape graphs are scheduled to HostCPU (scenario 1), while static large-shape graphs run on the Device (scenario 2).

## Prerequisites

- Refer to the [Installation Guide](../../../../../docs/en/quick_install.md) to install the `toolkit` and `ops` packages.
- Set the environment variables (assuming that the packages are installed in `/usr/local/Ascend/`):
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## Quick Run

Run in `examples/custom_op/host_cpu_add_custom/host_scheduling/cpp`:

```bash
bash run.sh
```

By default, both scenarios are run. You can also specify a single scenario:

```bash
bash run.sh --scenario=host     # Run scenario 1 only
bash run.sh --scenario=device   # Run scenario 2 only
bash run.sh --scenario=all      # Run both scenarios (default)
```

The script configures, builds, and installs. Expected output includes:

```text
=== Scenario1: HostCpu Custom (Sub + AddCustom + dynamic Sub) ===
[ShapeInferOp] InferDataType for AddCustom
[ShapeInferOp] InferShape for AddCustom
[HostCpuExecuteOp] Execute for AddCustom
output shape: [4]
output values (first 4): 6 8 10 12

=== Scenario2: Device (Data input + large shape + static graph) ===
[ShapeInferOp] InferDataType for AddCustom
[ShapeInferOp] InferShape for AddCustom
[EagerExecuteOp] Execute for AddCustom
output shape: [1024]
output values (first 10): 6 8 10 12 14 16 18 20 22 24
```

## Key Files

```text
host_scheduling/cpp
├── CMakeLists.txt
├── run.sh
├── add_custom_kernel
│   ├── add_custom.asc            // Ascend C vector-add kernel source, compiled by RTC at runtime
│   └── add_custom_kernel.h       // Kernel name and block size constants
├── ge
│   ├── add_custom_ir.h           // REG_OP(AddCustom) IR definition
│   ├── add_custom_ir.cc          // op proto library for add_es_library ES wrapper generation
│   ├── custom_op.cpp             // Device / HostCPU backend implementations and registration
│   └── utils
│       ├── rtc_kernel_loader.h   // RTC kernel loader (compile + binary load)
│       ├── rtc_kernel_loader.cpp
│       └── log.h
└── session_run
    └── main.cc                   // Two scenarios with ES graph construction (es::AddCustom) and Session::RunGraph
```

## Implementation Steps

`AddCustom` in `ge/custom_op.cpp` is the core implementation:

- `REG_OP(AddCustom)` registers a dedicated op type with inputs/outputs `x`/`y`/`z`, supporting `DT_FLOAT` and `DT_FLOAT16`.
- At build time, `add_es_library` (based on `add_custom_op_proto`) generates the `es_custom` wrapper library, and `session_run` keeps the ES-style graph construction with `es::AddCustom`.
- `EagerExecuteOp::Execute` (Device backend): on first execution, `RtcKernelLoader` compiles and loads `add_custom.asc` with the arch dynamically detected from the current device; the args are allocated on the device through `MallocReadOnlyDevArgs` and the kernel is launched via `aclrtLaunchKernelV2`. Subsequent executions reuse the loaded kernel handle.
- `HostCpuExecuteOp::Execute` (HostCPU backend): performs float/float16 vector addition on the host.
- `ShapeInferOp` copies input shape and dtype to the output.
- Both backends are registered via `REG_OP_BACKEND(AddCustom, "AddCustom", OpBackend::kDevice)` and `REG_OP_BACKEND(AddCustom, "AddCustom", OpBackend::kHostCPU)`.
- In Scenario 1, `HostcpuEngineUpdatePass` detects that the op has both HostCPU and Device backends, the graph is dynamic, and the shapes are small (4 <= 8), so it schedules the op to HostCPU.
- In Scenario 2, static graph with a large shape means `HostcpuEngineUpdatePass` does not trigger, and the op runs through the Device backend `EagerExecuteOp`.

## Notes

`run.sh` appends `output/` to `ASCEND_CUSTOM_OPP_PATH`; `add_custom.asc` is installed next to `libcust_opapi.so` so that the RTC loader can read it at runtime.
