# TileLang Add Custom Operator Online Compilation Sample

## Sample Overview

- **Graph construction entry**: GE native (Session API)
- **Operator programming language**: TileLang
- **Compilation method**: GE compile phase invokes TileLang Python compiler via `CompilableOp::Compile` callback (subprocess), compiling kernel source to `.so` online
- **Core pipeline**: `TileLang kernel source → GE Compile callback → subprocess compilation → dlopen load → Execute`
- **Scenario**: Scenario B — online compilation + online execution (`CompilableOp` + `EagerExecuteOp` + `ShapeInferOp`)

This sample demonstrates how to compile TileLang kernel source online during GE's compile phase (`CompileGraph`) via the `CompilableOp` interface, rather than pre-compiling the `.so`. Contrast with the [tilelang_add_custom](../tilelang_add_custom/README.md) (eager mode) sample.

## Differences from Eager Sample

| Dimension | Eager (`tilelang_add_custom`) | Online Compilation (this sample) |
|-----------|-------------------------------|----------------------------------|
| Interface combo | `EagerExecuteOp` + `ShapeInferOp` | `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp` |
| Compilation timing | Pre-compiled by `run.sh` | Online during `CompileGraph` |
| Load timing | Lazy `dlopen` at first `Execute` | `dlopen` in `Compile`, cached for `Execute` |
| Compilation trigger | Manual `python3 add_custom_kernel.py` | GE `CustomGraphOptimizer` calls `Compile` |
| Shape caching | None (fixed N=4096) | Keyed by element count, supports multiple element counts |
| Thread safety | `std::once_flag` | `std::mutex` (`Compile` may be called in parallel) |

## Directory Structure

```text
tilelang_add_custom_online/
├── README.md
├── README_en.md
├── CMakeLists.txt
├── run.sh
├── add_custom_kernel/
│   └── add_custom_kernel.py
├── ge/
│   ├── add_custom.h
│   └── custom_op.cpp
└── session_run/
    └── main.cc
```

## Core Pipeline

```text
GE compile phase (CompileGraph):
  CustomGraphOptimizer calls Compile(ctx)
    ├─ Read input shape → build binary key
    ├─ If key not cached:
    │   ├─ Locate add_custom_kernel.py (in OPP package, same dir as libcust_opapi.so)
    │   ├─ popen("python3 add_custom_kernel.py <N> <output.so>")
    │   ├─ TileLang compiler compiles kernel source → .so (host-wrapper)
    │   └─ dlopen .so + dlsym("call") → cache function pointer
    └─ Return GRAPH_SUCCESS

GE execution phase (ExecuteGraphWithStreamAsync):
  Execute(ctx) called
    ├─ Read input shape → build binary key
    ├─ Get cached call function pointer
    ├─ Allocate output Tensor
    └─ call(x_ptr, y_ptr, z_ptr, stream) → NPU execution
```

## Prerequisites

### CANN

- CANN environment properly installed and configured (`source ${ASCEND_HOME_PATH}/set_env.sh`)

### TileLang-Ascend

```bash
pip install tilelang
# TileLang-Ascend backend: install from https://github.com/tile-ai/tilelang-ascend
```

If installed from source, set:

```bash
export TILELANG_ASCEND_HOME=/path/to/tilelang-ascend
```

## Quick Start

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` executes 3 steps:

1. Build `libcust_opapi.so` and `tilelang_online_session_run`, install `add_custom_kernel.py` to OPP package
2. Verify kernel source is in OPP package
3. Run test program (`CompileGraph` triggers TileLang online compilation, then executes and verifies)

> **Note**: Unlike the eager sample, this sample does NOT pre-compile the TileLang kernel in `run.sh`. Compilation happens when `session_run` calls `CompileGraph`, triggered by GE's `CompilableOp::Compile` callback.

## Operator Specification

| Item | Value |
|------|-------|
| Op type | `AddCustomOnline` |
| Inputs | `x` (float32), `y` (float32) |
| Output | `z` (float32) |
| Input shape | `[4096]` (fixed) |
| Format | ND |
| Kernel name | `main_kernel` (wrapped by `call`) |
| BLOCK_SIZE | 1024 |

## Notes

- **Same-machine NPU compilation required**: TileLang-Ascend uses `torch.npu.get_device_name()` for runtime platform detection and does not support specifying target architecture offline.
- Kernel source `.py` is installed in the OPP package at `op_graph/lib/<os>/<arch>/`, alongside `libcust_opapi.so`. `Compile` locates it via `dladdr`.
- Compiled `.so` uses `mkstemps` for unique temp file, unlinked immediately after reading.
- `ge.graphRunMode=1` ensures online execution (PRIORITY_GRAPH mode).
- `CompileGraph` must be called before `ExecuteGraphWithStreamAsync`, otherwise `Execute` cannot find the compiled kernel.
- Online compilation requires Python + TileLang in the runtime environment, suitable for development; for production deployment, consider the eager sample's pre-compilation approach.
