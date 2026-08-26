# TileLang Add Custom Operator Offline OM Model Sinking Sample

## Sample Overview

- **Graph construction entry**: GE native (`Graph::SaveToFile` generates AIR, then ATC compiles OM)
- **Operator programming language**: TileLang
- **Compilation method**: ATC compile phase invokes TileLang Python compiler via `CompilableOp::Compile` callback (subprocess), then `PortableOp::Serialize` embeds `.so` bytes into OM model
- **Core pipeline**: `Graph → AIR → ATC compile (Compile + Serialize) → OM → ACL load (Deserialize + Execute)`
- **Scenario**: Scenario C — offline OM model sinking (`CompilableOp` + `PortableOp` + `EagerExecuteOp` + `ShapeInferOp`)

This sample demonstrates how to serialize TileLang compilation products into an OM model file via the `PortableOp` interface, enabling offline deployment. Contrast with [tilelang_add_custom_online](../tilelang_add_custom_online/README_en.md) (online compilation, Scenario B).

## Differences from Online Compilation Sample

| Dimension | Online (`tilelang_add_custom_online`) | Offline OM (this sample) |
|-----------|---------------------------------------|--------------------------|
| Interface combo | `CompilableOp` + `EagerExecuteOp` + `ShapeInferOp` | + `PortableOp` |
| Model format | No OM, direct `Session::ExecuteGraphWithStreamAsync` | OM model file |
| Compilation product lifecycle | In-process cache, lost on process exit | Serialized to OM file, persists across processes |
| Execution | GE Session online execution | ACL `aclmdlLoadFromFile` + `aclmdlExecute` |
| Deployment | Requires Python + TileLang in runtime | OM file is self-contained, no Python + TileLang needed at deployment |

## Quick Start

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

## Operator Specification

| Item | Value |
|------|-------|
| Op type | `AddCustomOffline` |
| Inputs | `x` (float32), `y` (float32) |
| Output | `z` (float32) |
| Input shape | `[4096]` (fixed) |
| Format | ND |
| Kernel name | `main_kernel` (wrapped by `call`) |
| BLOCK_SIZE | 1024 |

## Notes

- **Same-machine NPU compilation required**: TileLang-Ascend uses `torch.npu.get_device_name()` for runtime platform detection and does not support specifying target architecture offline. This sample only works when the compilation machine has the same NPU as the target. Cross-platform ATC offline compilation is not supported until TileLang adds offline target specification.
- `graph_build` phase requires Python + TileLang-Ascend; `model_exec` phase does not (OM is self-contained).
- Serialization format is custom (little-endian); GE only transparently passes the buffer.
- `Deserialize` uses `memfd_create` to load `.so` from memory (no disk files), with boundary checks, duplicate key detection, trailing data validation, and transactional rollback on failure.
