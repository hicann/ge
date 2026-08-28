# HostCpu AddCustom Custom Op Offline OM Sample

## Overview

This offline sample demonstrates the offline compilation and deployment flow for the `AddCustom` custom operator: building a graph with ES API to generate `.air`, converting to `.om` via ATC, then loading and executing through ACL API.

## Prerequisites

- Refer to the [Installation Guide](../../../../docs/en/quick_install.md) to install the `toolkit` and `ops` packages.
- Set the environment variables (assuming that the packages are installed in `/usr/local/Ascend/`):
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

## Quick Run

Run in `examples/custom_op/host_cpu_add_custom/offline`:

```bash
bash run.sh
```

The script will:

1. Build `output/op_graph/lib/<os>/<arch>/libcust_opapi.so`.
2. Run `single_add_graph_build` to generate `output/single_add.air`.
3. Run `atc` to generate `output/single_add_<os>_<arch>.om`.
4. Run `single_add_model_exec` to load and execute the OM.

Expected output includes:

```text
[HostCpuExecuteOp] Execute for AddCustom
[INFO] Model executed successfully!
output values: 6 8 10 12
[INFO] Output verification passed!
```

## Key Files

```text
offline
├── CMakeLists.txt
├── run.sh
├── ge
│   ├── add_custom_ir.h       // AddCustom operation prototype
│   ├── add_custom_ir.cc      // Compiles the AddCustom operation prototype
│   └── custom_op.cpp         // HostCpuExecuteOp / ShapeInferOp / PortableOp implementation
├── graph_build
│   └── main.cc               // Builds graph and exports AIR
└── model_exec
    └── main.cc               // Loads and executes OM through ACL
```

## Implementation Steps

`AddCustom` in `ge/custom_op.cpp` is the core implementation:

- `HostCpuExecuteOp::Execute` performs float vector addition on the host side.
- `ShapeInferOp` copies input shape and dtype to the output.
- `PortableOp::Serialize/Deserialize` provide the instance persistence implementation required by offline OM.
- Registers only kHostCPU backend via `REG_OP_BACKEND(AddCustom, "AddCustom", ge::OpBackend::kHostCPU)`.

## Notes

- `run.sh` uses `--soc_version=Ascend910B1` by default. Adjust it for your hardware if needed.
- The graph input shape is fixed to `[4]` float32, with input data `[1,2,3,4]` + `[5,6,7,8]` and expected output `[6,8,10,12]`.
- `ASCEND_CUSTOM_OPP_PATH` is appended automatically by `run.sh` with this sample's `output/`.
