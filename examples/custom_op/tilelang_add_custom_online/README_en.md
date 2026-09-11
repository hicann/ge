# TileLang Add Custom Operator Online Compilation Sample

## Sample Overview

- **Graph construction entry**: GE native (Session API)
- **Operator programming language**: TileLang
- **Compilation method**: GE compile phase invokes the TileLang Python compiler via `CompilableOp::Compile` callback (subprocess), compiling kernel source to `.so` online
- **Core pipeline**: `TileLang kernel source → GE Compile callback → subprocess compilation → dlopen load → Execute`
- **Scenario**: Scenario B — online compilation + online execution (`CompilableOp` + `EagerExecuteOp` + `ShapeInferOp`)

This sample uses an element-wise Add operator to demonstrate how to compile TileLang kernel source online during GE's compile phase (`CompileGraph`) via the `CompilableOp` interface, rather than pre-compiling the `.so` and loading it. Contrast with the [tilelang_add_custom](../tilelang_add_custom/README_en.md) (eager mode) sample.

## Differences from Eager Sample

| Dimension | Eager (`tilelang_add_custom`) | Online Compilation (this sample) |
|-----------|-------------------------------|----------------------------------|
| Interface combo | `EagerExecuteOp` + `ShapeInferOp` | `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp` |
| Compilation timing | `run.sh` pre-compiles `.so` | Online during `CompileGraph` |
| Load timing | First `Execute` call triggers lazy `dlopen` | `dlopen` in `Compile` callback, `Execute` uses the cache directly |
| Compilation trigger | Manual `python3 add_custom_kernel.py` | GE `CustomGraphOptimizer` calls `Compile` |
| Shape caching | None (fixed N=4096) | Keyed by element count, supports multiple input sizes |
| Thread safety | `std::once_flag` | `std::mutex` (`Compile` may be called in parallel) |

## Directory Structure

```text
tilelang_add_custom_online/
├── README.md
├── README_en.md
├── CMakeLists.txt                         # Build libcust_opapi.so + session_run + install .py source
├── run.sh                                 # One-click build and run (no kernel pre-compilation)
├── add_custom_kernel/
│   └── add_custom_kernel.py               # TileLang kernel source (accepts N and output path arguments)
├── ge/
│   ├── add_custom.h                       # REG_OP proto definition
│   └── custom_op.cpp                      # CompilableOp + EagerExecuteOp + ShapeInferOp implementation
└── session_run/
    └── main.cc                            # GE native graph + CompileGraph (triggers online compilation) + execution + precision check
```

## Core Pipeline

```text
GE compile phase (CompileGraph):
  CustomGraphOptimizer calls back Compile(ctx)
    ├─ Read input element count → build binary key
    ├─ If key not cached:
    │   ├─ Locate add_custom_kernel.py (in OPP package, same dir as libcust_opapi.so)
    │   ├─ popen("python3 add_custom_kernel.py <N> <output.so>") (same-machine NPU compilation)
    │   ├─ TileLang compiler compiles kernel source → produces .so (host-wrapper)
    │   └─ dlopen .so + dlsym("call") → cache function pointer (temp file unlinked immediately after reading)
    └─ Return GRAPH_SUCCESS

GE execution phase (ExecuteGraphWithStreamAsync):
  Execute(ctx) called back
    ├─ Read input element count → build binary key
    ├─ Get cached call function pointer
    ├─ Allocate output Tensor
    └─ call(x_ptr, y_ptr, z_ptr, stream) → NPU execution
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
pip install tilelang
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

`run.sh` executes 3 steps in sequence:

1. Build `libcust_opapi.so` and `tilelang_online_session_run`, install `add_custom_kernel.py` into the OPP package
2. Verify the kernel source is in the OPP package
3. Run the test program (`CompileGraph` triggers TileLang online compilation, then executes and verifies precision)

> **Note**: Unlike the eager sample, this sample does NOT pre-compile the TileLang kernel in `run.sh`. Compilation happens when `session_run` calls `CompileGraph`, triggered by GE's `CompilableOp::Compile` callback.

Expected terminal output on success:

```text
[INFO] Step 1/3: build custom op library and session_run
...
[INFO] Step 2/3: run session test (CompileGraph triggers TileLang online compilation)
CompileGraph (triggers TileLang online compilation)...
Compiling TileLang kernel: python3 ".../add_custom_kernel.py" 4096 ".../tilelang_add_custom_online_4096.so" 2>&1
Kernel .so saved to: ...
TileLang kernel compiled and loaded, key=4096, so=...
Precision check passed, max_error=0
[INFO] Step 3/3: sample pipeline finished.
```

## Key Files

### `ge/custom_op.cpp`

GE deliverable implementing `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp`:

- **Compile**:
  1. Read the input shape size from `ctx->GetInputTensor(0)` and build a binary key
  2. Lock and check the cache; if the key already exists, return directly (multiple shapes supported)
  3. Locate `add_custom_kernel.py` via `dladdr` on the directory of `libcust_opapi.so`
  4. Call `python3 add_custom_kernel.py <N> <output.so>` via `popen` to compile the TileLang kernel
  5. `dlopen` the compiled `.so`, obtain the function pointer via `dlsym("call")`, and cache it in `kernel_entries_`
- **Execute**:
  1. Read the input shape size and build the key
  2. Get the function pointer cached during `Compile` from `kernel_entries_`
  3. Allocate the output Tensor and call `call(x_ptr, y_ptr, z_ptr, stream)`
- **InferShape / InferDataType**: output shape and dtype are the same as the input
- Uses `std::mutex` for thread safety (`CustomGraphOptimizer` may call `Compile` in parallel)

### `ge/add_custom.h`

`REG_OP(AddCustomOnline)` declares the operator's input/output specification, used by GE native graph construction to create nodes.

### `add_custom_kernel/add_custom_kernel.py`

TileLang kernel source, accepting command-line arguments:

- 1st argument: `N` (total element count, default 4096, must be a multiple of BLOCK_SIZE=1024)
- 2nd argument: `output_path` (path of the produced `.so`)

### `session_run/main.cc`

GE native graph construction test program:

1. `GEInitialize` + create a `Session`
2. Build the `Data → AddCustomOnline` computation graph
3. `AddGraph` → `CompileGraph` (triggers `CompilableOp::Compile` → TileLang online compilation)
4. `LoadGraph`
5. Allocate device memory, H2D copy of input data
6. Execute with `ExecuteGraphWithStreamAsync`
7. D2H copy of the output, element-wise precision check (including NaN check)

## Operator Specification

| Item | Value |
|------|-------|
| Op type | `AddCustomOnline` |
| Inputs | `x` (float32), `y` (float32) |
| Output | `z` (float32) |
| Input shape | `[4096]` (fixed) |
| Output shape | `[4096]` |
| Format | ND |
| Kernel name | `main_kernel` (wrapped by `call`) |
| BLOCK_SIZE | 1024 |

## Step-by-Step Run

```bash
# 1. Build (including installing add_custom_kernel.py into the OPP package)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
cmake --install build

# 2. Set environment variables
export ASCEND_CUSTOM_OPP_PATH="$(pwd)/output:$ASCEND_CUSTOM_OPP_PATH"

# 3. Run (CompileGraph triggers online compilation)
./build/tilelang_online_session_run
```

## Notes

- **Same-machine NPU compilation required**: TileLang-Ascend currently uses `torch.npu.get_device_name()` for runtime platform detection and does not support specifying the target architecture offline. This sample only applies to scenarios where the compilation machine and the target machine have the same NPU.
- The kernel source `.py` is installed in the OPP package at `op_graph/lib/<os>/<arch>/`, alongside `libcust_opapi.so`. `Compile` locates it via `dladdr`.
- The compiled `.so` uses `mkstemps` for a unique temp file and calls `unlink` immediately after reading, leaving no residue.
- `ge.graphRunMode=1` ensures the online execution path (PRIORITY_GRAPH mode).
- `CompileGraph` must be called before `ExecuteGraphWithStreamAsync`, otherwise `Execute` cannot find the compiled kernel.
- Only float32 is supported. To support more data types, adjust the `REG_OP` `DATATYPE` constraint and the TileLang kernel dtype parameter.
- TileLang-Ascend platform detection is based on `torch.npu.get_device_name()`. Ascend910 maps to the A2 platform.
- Online compilation requires Python + TileLang in the runtime environment, suitable for the development phase; for production deployment, consider the eager sample's pre-compilation approach.
