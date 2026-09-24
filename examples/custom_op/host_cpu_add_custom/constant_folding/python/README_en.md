# HostCpu Constant Folding Add Custom Op Python Online Sample

## Overview

This sample defines a minimal `AddCustom` custom operator to demonstrate how a Python HostCpu custom op participates in constant folding. The operation prototype, `infer_meta` and HostCpu `execute` are all provided by Python decorators. The graph uses `GraphBuilder.create_const_float` for construction, where two `Const` nodes feed into `AddCustom`, and constant folding calls the Python `execute` at compile time.

## Prerequisites

- Follow the [Installation Guide](../../../../../docs/en/quick_install.md) to install the `toolkit` and `ops` packages.
- `cmake`, `python3` and a working `pip` are required.
- Set the environment variables (assuming that the packages are installed in `/usr/local/Ascend/`):
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```
- The GE Python package loads `pybind11`-based precompiled binary components at run time (such as `ge.custom_op` and `ge.runtime`). The CANN package prioritizes artifacts matching the current Python version; if no matching artifact exists, it automatically enters the fallback compilation flow, which requires `pybind11` in the current Python environment (`python3 -m pip install pybind11`).

## Quick Run

Run in `examples/custom_op/host_cpu_add_custom/constant_folding/python`:

```bash
bash run.sh
```

The script runs in three steps: Step 1/3 configures and builds the `add_custom_op_proto` target (which produces `build/opp/op_graph/lib/<os>/<arch>/libcust_opapi.so`) together with the ES wheel, Step 2/3 installs the wheel, and Step 3/3 runs the sample. Expected output includes:

```text
[Python] InferMeta for AddCustom
[Python] HostCpu execute for AddCustom
output shape: [1]
output values: 3.0
```

Select the NPU through `DEVICE_ID` (`0` by default):

```bash
DEVICE_ID=1 bash run.sh
```

### Dump Graph Verification

Enable graph dumping to visually verify constant folding:

```bash
export DUMP_GE_GRAPH=2
```

Open `ge_onnx_*_AfterInfershape.pbtxt` — the graph should no longer contain the `AddCustom` node (folded into `Const`).

### Log Verification

```bash
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=0
```

Search for `Constant folding computation for node` in the logs — `return code: 0` indicates success.

## Key Files

```text
constant_folding/python
├── CMakeLists.txt                  // Builds the custom OPP registry library and the ES wheel
├── run.sh
├── proto
│   ├── add_custom.h                // AddCustom graph-building prototype, used by gen_esb only
│   └── add_custom.cc
└── src
    ├── run.py                      // Python graph construction and Session.run_graph
    └── ge
        └── add_custom.py           // register_op infer-meta and register_op_impl HostCpu execute
```

## Implementation Steps

`src/ge/add_custom.py` is the core implementation:

- `add_custom_infer_meta`, decorated with `@register_op(op_type="AddCustom")`, infers the output shape and dtype, which is the counterpart of `ShapeInferOp` in the C++ sample.
- `@register_op_impl(op_type="AddCustom")` registers the implementation class, whose `execute` is declared as the HostCpu backend through `@register_kernel(backend=OpBackend.HOST)`; no device implementation is provided.
- `execute` obtains the `HostCpuOpExecutionContext` through `get_execute_ctx()`, allocates the output with `malloc_output_tensor`, and reads or writes host memory through `ctypes` at `Tensor.addr` to perform float32 addition.
- `GeApi.ge_initialize` uses the GE default optimization settings: the default optimization level is `O3`, and constant folding is enabled by default. `ConstantFoldingPass` can therefore detect constant inputs and invoke the Python HostCpu implementation.

The C++ `REG_OP` in `proto/add_custom.h` is only used by `gen_esb` to generate the `ge.es.custom.AddCustom` graph-building API; the runtime prototype and `infer_meta` come from the Python decorators.

## Notes

- `AddCustom` is intentionally minimal and float32-only so the Python HostCpu constant-folding path stays easy to verify.
- `run.sh` sets `ASCEND_CUSTOM_OPP_PATH` to `build/opp:src/ge`, where `src/ge` lets GE discover the Python custom op plugin.
- The Python HostCpu implementation is called by constant folding at compile time, so `[Python] HostCpu execute for AddCustom` appears in the graph-building logs and the operator is not executed at run time.
