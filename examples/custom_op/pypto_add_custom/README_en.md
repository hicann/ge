# PyPTO Operator Enters GE Graph Through Custom Operator

## Sample Overview

- Graph building entry: `GE` (Python ES graph-building APIs, RT2 dynamic execution path)
- Operator language: `PyPTO` (Python Tensor operator programming)
- Compilation mode: `JIT` (warm-up compiled on the main thread, reused from the in-process cache)
- Model sinking capability: `N/A`
- Core path: `PyPTO kernel -> Python custom-op execute callback -> torchair zero-copy bridge -> GE graph execution`
- Difference from other samples: this sample wires a PyPTO JIT kernel into the Python `execute` callback of a GE custom operator. No `.so` or kernel binary is compiled, and model sinking is not involved.

This sample registers the `PyptoAddCustom` custom operator: `@register_op` registers the prototype
and `infer_meta`, and `@register_op_impl` registers the schema-bound `execute(x, y)` implementation.
At run time the execute callback allocates the output tensor through the GE
`EagerOpExecutionContext`, wraps the input/output device memory as torch tensors through a
zero-copy torchair bridge (`as_torch_tensors`), and invokes the Add kernel defined with `@pypto.jit`. The PyPTO kernel is
warm-up compiled by `src/run.py` on the main thread before GE starts; the execute callback only
consumes the in-process compilation cache.

## Applicable Scenarios

- Understanding how a PyPTO kernel enters GE graph execution through a Python custom operator.
- Referencing the zero-copy bridge between GE tensors and torch tensors.
- Validating the numerics of a PyPTO JIT kernel on the GE online execution path.

## Prerequisites

### CANN

- Follow [installation guide](../../../docs/en/quick_install.md#1-environment-preparation) to complete toolkit and ops package installation.
- CANN version must be >= 9.2.0. The `pypto` Python package is released with the CANN toolkit (located in `python/site-packages/pypto`).

### Frameworks and Plugins

- `PyTorch` and `torch_npu` have been installed.

References:

- [Ascend Extension for PyTorch](https://gitcode.com/Ascend/pytorch)
- [Ascend Extension for PyTorch on Ascend community](https://hiascend.com/document/redirect/Pytorch-index)

### Environment Variables

- `ASCEND_HOME_PATH`
- `LD_LIBRARY_PATH` and other CANN runtime related variables
- `run.sh` automatically sets `PYTHONPATH`, `LD_LIBRARY_PATH`, and `ASCEND_CUSTOM_OPP_PATH`

### Additional Dependencies

- `cmake`, Python 3/`pip`

## Quick Run

Run the following commands in the `examples/custom_op/pypto_add_custom` directory:

### Recommended

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` automatically builds, installs, and runs the online graph:

1. Builds the custom OPP registration library and the ES Python graph-building APIs
   (all artifacts are placed under `build/`).
2. Runs `src/run.py`: warm-up compiles the PyPTO kernel on the main thread, builds the
   `PyptoAddCustom` graph, and executes it online.

On success the terminal prints something like:

```text
[OnlinePython] PyPTO kernel warm-up PASS
[OnlinePython] PyptoAddCustom precision check PASS
[Perf] input shape: [8192], dtype: float32
[Perf] iters: 100
[Perf] PyptoAddCustom: xxx us (avg xxx us/iter)
[OnlinePython] NPU_EXECUTION=PASS
```

## Directory Structure and Key Files

```text
pypto_add_custom
├── CMakeLists.txt                     # Build the custom OPP library and ES wheel
├── README.md
├── README_en.md
├── run.sh                             # Build and run the online GE graph
├── proto
│   └── add_custom.h                   # C++ graph prototype consumed by gen_esb
└── src
    ├── tensor_bridge.py               # Zero-copy bridge from GE tensors to torch tensors (torchair)
    ├── pypto_add_kernel.py            # @pypto.jit Add kernel (shared)
    ├── run.py                         # Warm-up + GE graph building and online execution
    └── ge
        └── add_custom.py                # Python prototype, infer_meta, and execute callback
```

Key files:

- `src/ge/add_custom.py`
  Implements `PyptoAddCustom`: `@register_op` registers the prototype and `infer_meta`; the
  `@register_op_impl` `execute` allocates the output tensor, bridges the input/output device memory
  through the torchair bridge, and runs the PyPTO kernel under device synchronization.
- `src/tensor_bridge.py`
  The zero-copy bridge from GE device memory to torch tensors. It reuses the torchair
  ``as_torch_tensors`` entry (the same machinery behind torchair's Python custom-op
  callbacks), while the device memory stays owned by GE.
- `src/pypto_add_kernel.py`
  The Add kernel defined with `@pypto.jit`, including the `pypto.set_vec_tile_shapes(1, 1024)`
  tile configuration; shared by the execute callback and the direct test.
- `src/run.py`
  Warm-up compiles the kernel on the main thread before GE starts, then builds the graph, validates
  precision, and measures 100 iterations.
- `proto/add_custom.h`
  Registers the `PyptoAddCustom` graph operator type for `gen_esb` to generate the Python ES
  graph-building APIs.

## Core Path

1. `run.py` imports torch/torch_npu/pypto on the main thread and warm-up compiles the PyPTO kernel
   to populate the in-process compilation cache.
2. `run.py` initializes GE with `ge.exec.static_model_ops_lower_limit=-1` so the graph takes the
   RT2 dynamic execution path, where the Python execute callback runs for every graph execution.
3. `infer_meta` validates the inputs and returns the output metadata.
4. `execute` allocates the output through `ctx.malloc_output_tensor` and wraps the x/y/z device
   device memory as torch tensors through the zero-copy torchair bridge.
5. `execute` invokes the PyPTO kernel (current torch stream) under device synchronization; the
   result is consumed by the subsequent tasks of the GE graph.

## Build Artifacts

- `build/opp/`
  The custom operator OPP registration library (`op_proto/custom/libcust_opapi.so` plus `op_graph/lib/<os>/<arch>/` platform copies), exposed to GE through `ASCEND_CUSTOM_OPP_PATH`.
- `build/es_custom_build/python_package/es_custom/`
  The build-generated ES Python graph-building APIs (including the auto-generated `__init__.py` and the wrappers generated by `gen_esb`), loaded through `PYTHONPATH`.
- `build/es_output/`
  ES deliverables: `lib64/libes_custom.so` (loaded through `LD_LIBRARY_PATH`) and `whl/es_custom-1.0.0-py3-none-any.whl`.

## Result Verification

On success you can observe:

- The terminal output contains `PyPTO kernel warm-up PASS`.
- The terminal output contains `PyptoAddCustom precision check PASS`.
- The terminal output contains `NPU_EXECUTION=PASS` and `Online Python PyPTO custom-op pipeline PASS`.

If it fails, check first:

- Whether `ASCEND_HOME_PATH` is set and the CANN environment is loaded correctly.
- Whether torch/torch_npu/pypto can be imported (CANN >= 9.2.0).
- Whether `build/opp/`, `build/es_custom_build/python_package/es_custom/`, and `build/es_output/lib64/` were generated.
- Whether an NPU is available in the current environment.

## PyPTO Support Status

- The current `pypto` Python package only supports **JIT direct execution**: the first call of a
  `@pypto.jit` kernel automatically parses, compiles, and executes it, and the compiled result is
  cached in the process.
- **No standalone compilation interface yet**: a PyPTO kernel cannot be pre-compiled into an
  independently publishable kernel binary file (for example the `.aicore.o` produced by compiling
  Ascend C with `bisheng`).
- **No binary export interface yet**: the JIT compilation artifact is an internal PyPTO format and
  cannot be converted into the `kernel_bin` required by the GE `AnnotatedArgs` declarative refresh
  mechanism. Therefore it cannot be wired into the `compile`/`declare_launch_args` path, nor sunk
  into an offline om model through ATC.
- The **Python `execute` callback + torchair bridge** shown in this sample is the currently viable
  path to integrate PyPTO kernels with GE.

## Notes / Limitations

- Verified products: `Atlas A2 training series`. The sample contains no hardcoded NPU architecture; PyPTO JIT detects the actual device architecture at compile time. Other platforms supported by PyPTO and torch_npu in CANN 9.2 (such as Atlas A3 or the Ascend 950 series) should also work, but are untested.
- The sample validates results with a `float32` one-dimensional input of `8192` elements. The number
  of input elements must be divisible by the tile size (1024).
- **The kernel must be warm-up compiled on the main thread before GE starts**: triggering JIT
  compilation inside a GE callback hangs because the compiler subprocess conflicts with the GE
  runtime.
- `run.py` forces the RT2 dynamic execution path through `ge.exec.static_model_ops_lower_limit=-1`
  so the execute callback runs for every graph execution. On the default static path the callback
  runs only once at task generation time, which is incompatible with the immediate launch model of
  PyPTO.
- **The torchair bridge requires a working ``import torchair.llm_datadist``** (``src/tensor_bridge.py``
  uses the public ``create_npu_tensors`` entry directly, without compatibility fallbacks). Some
  torch/torchair version combinations fail during the torchair package init (for example a missing
  ``hint_int`` in ``torch.fx``); upgrade torch_npu and torchair to matching versions before running
  this sample in that case.
- Python import order requires `import torch` / `import torch_npu` before `import pypto`; otherwise a
  `libc10.so` static TLS error may occur.
- The execute callback guarantees data visibility with device synchronization, so the per-iteration
  latency (about 4 ms) is higher than the Ascend C + AnnotatedArgs path (see
  `annotated_args_refresh_add_custom`, about 0.4 ms). This sample focuses on wiring the path, not
  on performance.

## Appendix

### Operator Specification

| Item | Content |
| --- | --- |
| Operator type | `PyptoAddCustom` |
| Inputs | `x`, `y` |
| Output | `z` |
| Input shape | `8192` |
| Output shape | `8192` |
| Data type | `float32` |
| Format | `ND` |
| Kernel name | `pypto_add_kernel` (PyPTO JIT) |
| Tile size | `1024` |

### Key Interfaces

| Interface | Purpose |
| --- | --- |
| `ge.custom_op.register_op` + `infer_meta` | Prototype registration and output shape/dtype inference |
| `ge.custom_op.register_op_impl` + `execute` | Allocate the output tensor and run the PyPTO kernel |
| `ge.custom_op.get_execute_ctx` | Access the execute callback context (output allocation) |
| `pypto.jit` + `pypto.set_vec_tile_shapes` | PyPTO kernel definition and tile configuration |
| `torchair as_torch_tensors` | Zero-copy bridge from GE device memory to torch tensors |
