# TileLang Add Custom Operator Offline OM Model Sinking Sample

## Sample Overview

- **Graph construction entry**: GE native (`Graph::SaveToFile` generates AIR, then ATC compiles OM)
- **Operator programming language**: TileLang
- **Compilation method**: ATC compile phase invokes the TileLang Python compiler via `CompilableOp::Compile` callback (subprocess), compiling kernel source to `.so` online, then `PortableOp::Serialize` serializes the `.so` bytes into the OM model
- **Core pipeline**: `Graph → AIR → ATC compile (Compile + Serialize) → OM → ACL load (Deserialize + Execute)`
- **Scenario**: Scenario C — offline OM model sinking (`CompilableOp` + `PortableOp` + `EagerExecuteOp` + `ShapeInferOp`)

This sample uses an element-wise Add operator to demonstrate how to serialize TileLang compilation products into an OM model file via the `PortableOp` interface, enabling an offline deployment path. Contrast with [tilelang_add_custom_online](../tilelang_add_custom_online/README_en.md) (online compilation, Scenario B).

## Differences from Online Compilation Sample

| Dimension | Online (`tilelang_add_custom_online`) | Offline OM sinking (this sample) |
|-----------|---------------------------------------|----------------------------------|
| Interface combo | `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp` | + `PortableOp` |
| Model format | No OM, direct `Session::ExecuteGraphWithStreamAsync` | OM model file |
| Compilation product lifecycle | In-process cache, lost on process exit | Serialized to OM file, persists across processes |
| Execution | GE Session online execution | ACL `aclmdlLoadFromFile` + `aclmdlExecute` |
| Deployment | Requires Python + TileLang in runtime | OM file is self-contained, no Python + TileLang needed at deployment |

## Directory Structure

```text
tilelang_add_custom_offline/
├── README.md
├── README_en.md
├── CMakeLists.txt                         # Build libcust_opapi.so + graph_build + model_exec
├── run.sh                                 # One-click build and run
├── add_custom_kernel/
│   └── add_custom_kernel.py               # TileLang kernel source (accepts N and output path arguments)
├── ge/
│   ├── add_custom.h                       # REG_OP proto definition
│   └── custom_op.cpp                      # CompilableOp + PortableOp + EagerExecuteOp + ShapeInferOp implementation
├── graph_build/
│   └── main.cc                            # Graph construction + Graph::SaveToFile generates AIR (for ATC compilation)
└── model_exec/
    └── main.cc                            # ACL loads OM + execution + precision check (triggers Deserialize + Execute)
```

## Core Pipeline

```text
=== graph_build phase (Graph::SaveToFile → ATC) ===

graph_build generates the AIR file → ATC loads AIR and compiles OM

GE calls back Compile(ctx)
  ├─ Read input element count → build binary key
  ├─ popen("python3 add_custom_kernel.py <N> <output.so>") (same-machine NPU compilation)
  ├─ TileLang compiler compiles kernel source → produces .so (host-wrapper)
  ├─ Read .so file bytes → so_data
  ├─ mkstemps temp file unlinked immediately after reading
  └─ dlopen .so + dlsym("call") → cache function pointer

GE calls back Serialize(buffer)
  ├─ Little-endian format: [magic][version][count]
  │         [key_len][key][so_size][so_data] ...
  └─ Write all .so bytes in kernel_entries_ into the buffer → embedded into OM

aclgrphSaveModel → save the OM file

=== model_exec phase (aclmdlLoadFromFile) ===

ACL loads OM → GE calls back Deserialize(buffer)
  ├─ Validate magic/version/count, check boundaries and duplicate keys
  ├─ Restore kernel entries one by one, load .so from memory via memfd_create (no disk file)
  ├─ dlopen memfd + dlsym("call") → cache function pointer
  ├─ Check there is no trailing dirty data
  └─ Atomically replace kernel_entries_ after all entries succeed (transactional)

aclmdlExecute → GE calls back Execute(ctx)
  ├─ Get the call function pointer from kernel_entries_
  ├─ Allocate output Tensor
  └─ call(x_ptr, y_ptr, z_ptr, stream) → NPU execution
```

## Prerequisites

### CANN

- CANN environment properly installed and configured (`source ${ASCEND_HOME_PATH}/set_env.sh`)

### TileLang-Ascend

Install the TileLang main package and the TileLang-Ascend backend:

```bash
pip install tilelang
# TileLang-Ascend backend: install from https://github.com/tile-ai/tilelang-ascend
```

> **Note**: TileLang-Ascend is only needed in the graph_build phase (compiling OM); the model_exec phase (loading and executing OM) does not need it.

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

1. Build `libcust_opapi.so`, `graph_build`, and `model_exec`, install the `.py` source into the OPP package
2. Run `graph_build` (`Graph::SaveToFile` generates the AIR file)
3. Run `atc` (compile AIR → OM, triggers `Compile` + `Serialize`)
4. Run `model_exec` (`aclmdlLoadFromFile` triggers `Deserialize`, `aclmdlExecute` triggers `Execute`)

Expected terminal output on success:

```text
[INFO] Step 1/4: build custom op library, graph_build and model_exec
...
[INFO] Step 2/4: generate AIR file (graph definition)
Saving AIR file (for ATC offline compilation)...
AIR file saved to: .../tilelang_add_offline.air
[INFO] Step 3/4: compile AIR to OM via ATC (triggers Compile + Serialize)
ATC compiling ...
Compiling TileLang kernel: python3 ".../add_custom_kernel.py" 4096 "..." 2>&1
TileLang kernel compiled and loaded, key=4096, so_size=...
Serialized 1 kernel(s), total buffer size=...
[INFO] OM model generated: ... bytes
[INFO] Step 4/4: execute OM model (triggers Deserialize + Execute)
Loading OM model (triggers Deserialize): .../tilelang_add_offline.om
Deserialized 1 kernel(s)
Executing model (triggers Execute)...
Precision check passed, max_error=0
[INFO] Sample pipeline finished.
```

## Operator Specification

| Item | Value |
|------|-------|
| Op type | `AddCustomOffline` |
| Inputs | `x` (float32), `y` (float32) |
| Output | `z` (float32) |
| Input shape | `[4096]` (fixed) |
| Output shape | `[4096]` |
| Format | ND |
| Kernel name | `main_kernel` (wrapped by `call`) |
| BLOCK_SIZE | 1024 |

## Serialization Format

`PortableOp::Serialize` uses a custom binary format to embed the TileLang `.so` compilation product into OM:

```text
Offset  Length  Field     Description
0       4       magic     Fixed 0x4F504B4E (custom format identifier, little-endian)
4       4       version   Fixed 1 (little-endian)
8       4       count     Number of kernel entries (little-endian)
12      ---     entries   Repeated count times:
        4       key_len   Key byte length (little-endian)
        N       key       Element count string (e.g. "4096")
        4       so_size   .so file byte length (little-endian)
        M       so_data   Full .so binary content
```

`Deserialize` reads this format, loads each `.so` from memory via `memfd_create` (no disk file), and checks integrity constraints such as duplicate keys and trailing dirty data.

## Key Files

### `ge/custom_op.cpp`

GE deliverable implementing `CompilableOp` + `PortableOp` + `EagerExecuteOp` + `ShapeInferOp`:

- **Compile**: invokes Python via subprocess to compile TileLang → reads `.so` bytes → `dlopen` → cache
- **Serialize**: serializes the `.so` bytes in `kernel_entries_` into a binary buffer
- **Deserialize**: restores `.so` bytes from the buffer → loads from memory via `memfd_create` (no disk file) → `dlopen` → cache
- **Execute**: calls `call(x, y, z, stream)` with the cached function pointer
- Uses `std::mutex` for thread safety

### `graph_build/main.cc`

Generates the AIR file with `Graph::SaveToFile` for offline ATC compilation:

1. `GEInitialize` + build the computation graph
2. `graph->SaveToFile(air_path)` — generate the AIR file

When ATC compiles AIR → OM, `Compile` + `Serialize` are triggered automatically.

### `model_exec/main.cc`

Loads and executes the OM model with ACL APIs:

1. `aclInit` + `aclrtSetDevice`
2. `aclmdlLoadFromFile(om_path)` — triggers Deserialize
3. `aclmdlGetDesc` to get the model description
4. Allocate device memory, H2D copy of inputs
5. `aclmdlExecute` — triggers Execute
6. D2H copy of the output, precision check

## Notes

- **Same-machine NPU compilation required**: TileLang-Ascend currently uses `torch.npu.get_device_name()` for runtime platform detection and does not support specifying the target architecture offline. Therefore the Python compiler invoked in the `Compile` callback is not passed `--soc_version`, and the compilation product is bound to the compilation machine's NPU. This sample only applies to scenarios where the compilation machine and the target machine have the same NPU; it cannot be used for cross-platform ATC offline compilation. For cross-platform compilation, wait for TileLang to support offline target specification, then update this sample.
- The `graph_build` phase requires Python + TileLang-Ascend in the runtime environment; the `model_exec` phase does not (OM is self-contained with the compiled `.so`).
- `ge.graphRunMode=1` ensures the online execution path (PRIORITY_GRAPH mode).
- The serialization format is custom; GE only passes the buffer through without parsing, and the format is fully controlled by the operator. All `uint32_t` fields use little-endian format.
- Only float32 is supported. To support more data types, adjust the `REG_OP` `DATATYPE` constraint and the TileLang kernel dtype parameter.
- **OM compilation depends on the ATC tool**: `graph_build` generates the AIR file, and ATC compiles AIR → OM (triggering `Compile` + `Serialize`). The `SOC_VERSION` environment variable can override the default `Ascend910_9362`.
- The compiled `.so` uses `mkstemps` for a unique temp file during `Compile` and calls `unlink` immediately after reading; `Deserialize` loads it from memory via `memfd_create`, without touching disk.
- If ATC is unavailable in the current environment (version mismatch, etc.), refer to the `compilable_add_custom` sample to compile the OM in an environment where ATC is available.
