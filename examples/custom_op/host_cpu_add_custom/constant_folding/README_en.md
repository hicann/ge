# HostCpu Constant Folding Add Custom Op Online Sample

## Overview

This sample defines a minimal `AddCustom` custom operator to demonstrate how a HostCpu custom op participates in constant folding. The graph uses `EsGraphBuilder::CreateConst` for construction, where two `Const` nodes feed into `AddCustom`, and constant folding computes the result at compile time.

## Prerequisites

- Follow the [Installation Guide](../../../../docs/en/quick_install.md) to install the `toolkit` and `ops` packages.
- Set the environment variables (assuming that the packages are installed in `/usr/local/Ascend/`):
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## Quick Run

Run in `examples/custom_op/host_cpu_add_custom/constant_folding`:

```bash
bash run.sh
```

The script configures, builds, installs, and generates `libes_custom.so`. Expected output includes:

```text
HostCpuExecuteOp::Execute for AddCustom
output shape: [1]
output values: 3
```

If constant folding does not hit, this HostCPU-only operator does not enter the device Eager execution path.

### Dump Graph Verification

Enable graph dumping to visually verify constant folding:

```bash
export DUMP_GE_GRAPH=2
cd build
./host_cpu_add_custom_constant_folding_session_run
cd ..
```

Open `ge_proto_*_AfterInfershape.pbtxt` — the graph should no longer contain the `AddCustom` node (folded into `Const`).

### Log Verification

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
cd build
./host_cpu_add_custom_constant_folding_session_run
cd ..
```

Search for `Constant folding computation for node` in the logs — `return code: 0` indicates success.

## Key Files

```text
constant_folding
├── CMakeLists.txt
├── run.sh
├── ge
│   ├── add_custom_ir.h       // AddCustom operation prototype
│   ├── add_custom_ir.cc      // Compiles the AddCustom operation prototype and generates the ES custom API
│   └── custom_op.cpp         // HostCpuExecuteOp / ShapeInferOp implementation
└── session_run
    └── main.cc               // ES graph construction and Session::RunGraph
```

## Implementation Steps

`AddCustom` in `ge/custom_op.cpp` is the core implementation:

- `HostCpuExecuteOp::Execute` performs float addition on the host side, called by `ConstantFoldingPass` at compile time.
- `AddCustom` registers only the `kHostCPU` backend and provides no device/Eager implementation.
- `ShapeInferOp` copies input shape and dtype to the output.
- `Session::GEInitialize` uses the GE default optimization settings: the default optimization level is `O3`, and constant folding is enabled by default. `ConstantFoldingPass` can therefore detect constant inputs and invoke the HostCpu implementation.

## Notes

- This sample only covers the constant-folding path; the runtime host scheduling sample lives in `../host_scheduling`, and the offline OM sample in `../offline`.
- `AddCustom` is intentionally minimal and float32-only so the HostCpu constant-folding path stays easy to verify.
- `ASCEND_CUSTOM_OPP_PATH` is appended automatically by `run.sh` with this sample's `output/`.
