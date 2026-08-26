# TileLang Add Custom Operator Sample

## Sample Overview

- **Graph construction entry**: GE native (Session API)
- **Operator programming language**: TileLang
- **Compilation method**: TileLang pre-compiles to host-wrapper `.so`, loaded via `dlopen` at runtime
- **Core pipeline**: `TileLang kernel → pre-compiled .so → GE deliverable → in-process graph → Session::ExecuteGraphWithStreamAsync online execution`
- **Scenario**: Scenario A — online execution with pre-compiled kernel

This sample demonstrates how to integrate a TileLang kernel into GE's graph compilation and execution flow via the language-independent custom operator mechanism, using an element-wise Add operator as an example.

## Directory Structure

```text
tilelang_add_custom/
├── README.md
├── README_en.md
├── CMakeLists.txt                         # Build libcust_opapi.so + session_run + install add_kernel.so
├── run.sh                                 # One-click build and run
├── add_custom_kernel/
│   └── add_custom_kernel.py               # TileLang kernel + compile to add_kernel.so
├── ge/
│   ├── add_custom.h                       # REG_OP proto definition
│   └── custom_op.cpp                      # EagerExecuteOp + ShapeInferOp implementation
└── session_run/
    └── main.cc                            # GE native graph + Session execution + precision check
```

## Core Pipeline

```text
TileLang kernel source (add_custom_kernel.py)
    ↓ TileLang-Ascend compiler (TVM + Ascend C codegen + Bisheng)
add_kernel.so (host-wrapper, exports call function)
    ↓ dlopen + dlsym("call")
GE custom operator (AddCustom, EagerExecuteOp)
    ↓ call(x_ptr, y_ptr, z_ptr, stream) — wraps main_kernel<<<>>> launch
NPU execution
```

## Prerequisites

### CANN

- CANN environment properly installed and configured (`source ${ASCEND_HOME_PATH}/set_env.sh`)

### TileLang-Ascend

```bash
pip install tilelang
# TileLang-Ascend backend: install from https://github.com/tile-ai/tilelang-ascend
```

If TileLang-Ascend is installed from source (not via pip), set:

```bash
export TILELANG_ASCEND_HOME=/path/to/tilelang-ascend
```

## Quick Start

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

## Operator Specification

| Item | Value |
|------|-------|
| Op type | `AddCustom` |
| Inputs | `x` (float32), `y` (float32) |
| Output | `z` (float32) |
| Input shape | `[4096]` (fixed) |
| Format | ND |
| Kernel name | `main_kernel` (wrapped by `call`) |
| BLOCK_SIZE | 1024 |

## Notes

- The kernel is compiled with fixed N=4096. Execute validates input shape size and returns failure on mismatch.
- `ge.graphRunMode=1` ensures online execution (PRIORITY_GRAPH mode).
- Only float32 is supported. To support more data types, adjust `REG_OP` DATATYPE and TileLang kernel dtype parameter.
- TileLang-Ascend platform detection is based on `torch.npu.get_device_name()`. Ascend910 maps to A2 platform.
- The kernel `.so` is installed in the OPP package at `op_graph/lib/<os>/<arch>/`, alongside `libcust_opapi.so`.
