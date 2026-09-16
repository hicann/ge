# TileLang Add Custom Operator Sample

## Sample Overview

- **Graph construction entry**: GE native (Session API)
- **Operator programming language**: TileLang
- **Compilation method**: TileLang pre-compiles a host-wrapper `.so`, loaded via `dlopen` by GE at runtime
- **Core pipeline**: `TileLang kernel → pre-compiled .so → GE deliverable → in-process graph construction → Session::ExecuteGraphWithStreamAsync online execution`
- **Scenario**: Scenario A — dynamic graph online execution (pre-compiled kernel + host scheduling)

This sample uses an element-wise Add operator to demonstrate how to integrate a TileLang-written kernel into GE's graph compilation and execution flow via the language-independent custom operator mechanism.

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
    ↓ call(x_ptr, y_ptr, z_ptr, stream) — wraps main_kernel<<<>>> launch internally
NPU execution
```

The `.so` compiled by TileLang-Ascend exports a function with the signature:

```c
extern "C" void call(uint8_t* A_handle, uint8_t* B_handle, uint8_t* C_handle, aclrtStream stream)
```

This function internally wraps the launch logic of `main_kernel<<<>>>` (including hardware scheduling address acquisition, tiling, etc.), so GE does not need to assemble args manually.

## Prerequisites

### CANN

- CANN environment properly installed and configured (`source ${ASCEND_HOME_PATH}/set_env.sh`)
- The environment provides ACL, GE, and Graph related headers and libraries

### TileLang-Ascend

Install the TileLang main package and the TileLang-Ascend backend:

```bash
pip install tilelang                    # main package
# TileLang-Ascend backend: install from https://github.com/tile-ai/tilelang-ascend
```

If TileLang-Ascend is installed from source (not via `pip install`), set the environment variable:

```bash
export TILELANG_ASCEND_HOME=/path/to/tilelang-ascend
```

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ASCEND_HOME_PATH` | Yes | CANN toolkit path |
| `TILELANG_ASCEND_HOME` | No | TileLang-Ascend source installation path (not needed if installed via pip) |
| `ASCEND_CUSTOM_OPP_PATH` | Auto | Set automatically by `run.sh` |

## Quick Start

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

`run.sh` executes 4 steps in sequence:

1. Compile the TileLang kernel, producing `add_kernel.so`
2. Build `libcust_opapi.so` and `tilelang_session_run`, install `add_kernel.so` into the OPP package
3. Verify the kernel `.so` is in the OPP package
4. Run the test program

Expected terminal output on success:

```text
[INFO] Step 1/4: compile TileLang kernel
Kernel .so saved to: add_kernel.so
[INFO] Step 2/4: build custom op library and session_run
...
[INFO] kernel .so installed in OPP package.
[INFO] Step 4/4: run session test
Precision check passed, max_error=0
[INFO] Sample pipeline finished.
```

## Key Files

### `ge/custom_op.cpp`

GE deliverable implementing `EagerExecuteOp` + `ShapeInferOp`:

- **Execute**:
  1. On the first call, loads `add_kernel.so` via `dlopen` (path located via the `ASCEND_CUSTOM_OPP_PATH` environment variable) and obtains the `call` function pointer via `dlsym`
  2. Validates that both inputs have a shape size of 4096
  3. Allocates the output Tensor and calls `call(x_ptr, y_ptr, z_ptr, stream)`
- **InferShape / InferDataType**: output shape and dtype are the same as the input
- Uses `std::once_flag` for thread-safe lazy loading
- The kernel `.so` path is located via the `ASCEND_CUSTOM_OPP_PATH` environment variable, independent of the working directory

### `ge/add_custom.h`

`REG_OP(AddCustom)` declares the operator's input/output specification, used by GE native graph construction to create nodes.

### `add_custom_kernel/add_custom_kernel.py`

TileLang kernel source that defines the element-wise Add and compiles it into `add_kernel.so`.

### `session_run/main.cc`

GE native graph construction test program:

1. `GEInitialize` + create a `Session`
2. Build the `Data → AddCustom` computation graph
3. `AddGraph` → `CompileGraph` → `LoadGraph`
4. Allocate device memory, H2D copy of input data
5. Execute with `ExecuteGraphWithStreamAsync`
6. D2H copy of the output, element-wise precision check (including NaN check)

## Operator Specification

| Item | Value |
|------|-------|
| Op type | `AddCustom` |
| Inputs | `x` (float32), `y` (float32) |
| Output | `z` (float32) |
| Input shape | `[4096]` (fixed) |
| Output shape | `[4096]` |
| Format | ND |
| Kernel name | `main_kernel` (wrapped by `call`) |
| BLOCK_SIZE | 1024 |

## Step-by-Step Run

```bash
# 1. Compile the TileLang kernel
cd add_custom_kernel && python3 add_custom_kernel.py && cd ..

# 2. Build (including installing add_kernel.so into the OPP package)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build

# 3. Set environment variables
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"

# 4. Run
./build/tilelang_session_run
```

## Notes

- The kernel is compiled with fixed N=4096. Execute validates the input shape size and returns failure on mismatch.
- `ge.graphRunMode=1` ensures the online execution path (PRIORITY_GRAPH mode).
- Only float32 is supported. To support more data types, adjust the `REG_OP` `DATATYPE` constraint and the TileLang kernel dtype parameter.
- TileLang-Ascend platform detection is based on `torch.npu.get_device_name()`. Ascend910 maps to the A2 platform.
- The kernel `.so` is installed in the OPP package at `op_graph/lib/<os>/<arch>/`, alongside `libcust_opapi.so`. The path is located via the `ASCEND_CUSTOM_OPP_PATH` environment variable.
