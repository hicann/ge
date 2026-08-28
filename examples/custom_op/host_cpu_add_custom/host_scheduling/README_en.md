# HostCpu Host Scheduling Add Custom Op Sample

## Overview

This sample registers `HostCpuExecuteOp` and `ShapeInferOp` for the built-in `Add` operator, validating that `HostcpuEngineUpdatePass` schedules the built-in op to HostCpu at runtime. It covers HostCpu-hit and AICore-execution scenarios.

## Prerequisites

- Refer to the [Installation Guide](../../../../docs/en/quick_install.md) to install the `toolkit` and `ops` packages.
- Set the environment variables (assuming that the packages are installed in `/usr/local/Ascend/`):
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## Quick Run

Run in `examples/custom_op/host_cpu_add_custom/host_scheduling`:

```bash
bash run.sh
```

By default, both scenarios are run. You can also specify a single scenario:

```bash
bash run.sh --scenario=host     # Run scenario 1 only
bash run.sh --scenario=aicore   # Run scenario 2 only
bash run.sh --scenario=all      # Run both scenarios (default)
```

The script configures, builds, and installs. Expected output includes:

```text
=== Scenario1: HostCpu Custom (Sub + Add + dynamic Sub) ===
[ShapeInferOp] InferDataType for Add
[ShapeInferOp] InferShape for Add
[HostCpuExecuteOp] Execute for Add
output shape: [4]
output values (first 4): 6 8 10 12

=== Scenario2: AiCore (Data input + large shape + static graph) ===
[ShapeInferOp] InferDataType for Add
[ShapeInferOp] InferShape for Add
output shape: [1024]
output values (first 10): 6 8 10 12 14 16 18 20 22 24
```

## Key Files

```text
host_scheduling
├── CMakeLists.txt
├── run.sh
├── ge
│   └── custom_op.cpp         // Registers HostCpuExecuteOp / ShapeInferOp for built-in Add
└── session_run
    └── main.cc               // Two scenarios with ES graph construction and Session::RunGraph
```

## Implementation Steps

`AddHostCpu` in `ge/custom_op.cpp` is the core implementation:

- `HostCpuExecuteOp::Execute` performs float vector addition on the host side.
- `ShapeInferOp` copies input shape and dtype to the output.
- Binds to the built-in Add via `REG_OP_BACKEND(AddHostCpu, "Add", OpBackend::kHostCPU)`, registering only kHostCPU backend.
- In Scenario 1, `HostcpuEngineUpdatePass` detects that Add's input/output shapes are small (4 <= 8) and marks it for HostCpu execution.
- In Scenario 2, static graph with large shape means `HostcpuEngineUpdatePass` does not trigger, and Add runs on AICore normally.

## Notes

- This sample only covers the runtime host scheduling path; the constant-folding sample lives in `../constant_folding`, and the offline OM sample in `../offline`.
- `run.sh` appends `output/` to `ASCEND_CUSTOM_OPP_PATH`.
