# HostCpu Host Scheduling AddCustom Custom Op Python Sample

## Overview

This sample registers the custom operator `AddCustom` with dual Python backend implementations: the HostCpu backend performs the elementwise add directly on host memory, while the device backend launches the PyPTO kernel through the zero-copy tensor bridge. At run time `HostcpuEngineUpdatePass` schedules small dynamic-shape graphs to the Python HostCpu callback, and static large-shape graphs run on the device, covering both scheduling paths.

## Prerequisites

### CANN

- Follow the [Installation Guide](../../../../../docs/en/quick_install.md#1-environment-preparation) to install the `toolkit` and `ops` packages.
- `cmake`, `python3` and a working `pip` are required.
- Set the environment variables (assuming that the packages are installed in `/usr/local/Ascend/`):

  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- The GE Python package loads `pybind11`-based precompiled binary components at run time (such as `ge.custom_op` and `ge.runtime`). The CANN package prioritizes artifacts matching the current Python version; if no matching artifact exists, it automatically enters the fallback compilation flow, which requires `pybind11` in the current Python environment (`python3 -m pip install pybind11`).

### Frameworks and Plugins

- The device (PyPTO) path requires `PyTorch` and `torch_npu`. `run.sh` checks them before running:

  ```bash
  python3 -c "import torch, torch_npu, pypto, torchair; from torchair.llm_datadist import create_npu_tensors"
  ```

References:

- [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)
- [Ascend Extension for PyTorch Community Guide](https://hiascend.com/document/redirect/Pytorch-index)

## Quick Run

Run in `examples/custom_op/host_cpu_add_custom/host_scheduling/python`:

```bash
bash run.sh
```

Both scenarios run by default. Use `--scenario` to run a single one:

```bash
bash run.sh --scenario=host     # Scenario 1 only
bash run.sh --scenario=device   # Scenario 2 only
bash run.sh --scenario=all      # Both scenarios (default)
```

Select the NPU through `DEVICE_ID` (`0` by default). Expected output includes:

```text
=== Scenario1: HostCpu Custom (Sub + AddCustom + dynamic Sub) ===
[Python] HostCpu execute for AddCustom
output shape: [4]
output values (first 4): 6.0 8.0 10.0 12.0
[HostSchedulingPython] scenario1 output verification passed

=== Scenario2: Device (Data input + large shape + static graph) ===
[Python] Device execute for AddCustom
output shape: [1024]
output values (first 10): 6.0 8.0 10.0 12.0 14.0 16.0 18.0 20.0 22.0 24.0
[HostSchedulingPython] scenario2 output verification passed
```

## Key Files

```text
host_scheduling/python
├── CMakeLists.txt         // Builds the custom OPP registry library and the ES Python API (gen_esb)
├── run.sh                 // Checks PyPTO dependencies, builds the ES API and runs both scenarios
├── proto
│   ├── add_custom.h       // C++ graph prototype of AddCustom (input of gen_esb)
│   └── add_custom.cc      // Proto compilation unit (OP_PROTO_LIB)
└── src
    ├── pypto_add_kernel.py // @pypto.jit elementwise add kernel (shared by the device path)
    ├── run.py              // Graph construction and Session.run_graph for both scenarios
    └── ge
        └── add_custom.py   // AddCustom prototype, infer_meta, HostCpu/device execute callbacks and the zero-copy bridge
```

## Implementation Steps

`src/ge/add_custom.py` is the core implementation:

- `@register_op(op_type="AddCustom")` registers the prototype and `infer_meta`; the output metadata is derived from the shape/dtype checks of the two inputs.
- `@register_op_impl(op_type="AddCustom")` declares both backend implementations in one class:
  - `execute` + `@register_kernel(backend=OpBackend.HOST)`: the HostCpu backend. It obtains the `HostCpuOpExecutionContext` through `get_execute_ctx()`, allocates the output with `malloc_output_tensor`, and reads or writes host memory through `ctypes` at `Tensor.addr` for the float32 add (same as the C++ counterpart).
  - `execute` + `@register_kernel(backend=OpBackend.DEVICE)`: the device backend. It reuses the PyPTO sample's `execute` callback plus the zero-copy bridge, launching the PyPTO kernel under device-wide synchronization.
- `src/run.py` initializes GE with `ge.exec.static_model_ops_lower_limit=-1` so the execute callback runs on every graph execution; the PyPTO kernel is JIT compiled by `@pypto.jit` on the first device execute call, and the compilation cache is reused in-process afterwards.
- In scenario 1, `HostcpuEngineUpdatePass` detects the small input/output shape of `AddCustom` (4 <= 8) and finds the Python HostCpu implementation through `CustomOpFactory`; in scenario 2, the static graph with a large shape does not trigger `HostcpuEngineUpdatePass`, so the device implementation takes effect and the PyPTO kernel computes the add on the NPU.

## Notes

- `run.sh` sets `ASCEND_CUSTOM_OPP_PATH` to `build/opp:src/ge`: the former exposes the C++ proto to GE, and the latter lets the Python custom-op loader discover the plugin.
- `run.sh` sets `PYTHONPATH` to the ES Python package directory and `src` (the plugin imports `pypto_add_kernel` from it), and points `LD_LIBRARY_PATH` at `build/es_output/lib64`.
- Scenario 1 depends on dynamic shape and small-shape propagation; keep the `_host_tensor` and `_graph_unknown_flag` attributes, otherwise `AddCustom` is not scheduled to HostCpu.
- The PyPTO kernel is annotated with a fixed 1024-element shape, so scenario 2 inputs must contain exactly 1024 elements. The kernel is JIT compiled by `@pypto.jit` on the first device execute call, and the compilation cache is reused in-process afterwards.
- Scenario 2 relies on `ge.exec.static_model_ops_lower_limit=-1` to force the RT2 dynamic execution path: PyPTO kernels can only be launched on the current torch stream and cannot join the GE task stream, so the callback must execute immediately under device-wide synchronization, which requires the callback to fire after the input data is ready. Without this option, the default static sink path invokes the device callback once during task generation (before inputs are written), the kernel computes on unprepared data, and the output is all zeros.
- Python import order requires `import torch` / `import torch_npu` before `import pypto`; otherwise `libc10.so` static TLS errors may occur (`src/ge/add_custom.py` already follows this order).
