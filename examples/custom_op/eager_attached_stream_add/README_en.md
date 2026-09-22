# Eager Custom Operator Attached Stream Request End-to-End Sample

This sample verifies the complete flow of `EagerOpExecutionContext::RequestAttachedStream`: the Eager custom operator requests a framework-managed physical attached stream by key in the Execute callback and launches the kernel on the attached stream.

## Sample Overview

- Graph construction entry: `GE`
- Operator programming language: `Ascend C` (RTC runtime compilation)
- Core flow: `RTC compiles the kernel -> GE deliverables -> in-process graph construction -> Session online execution -> kernel launch on the attached stream -> precision check`
- Difference from other samples: this sample focuses on the `EagerOpExecutionContext::RequestAttachedStream` interface. The operator requests a framework-managed attached stream in the Execute callback and launches the kernel on the attached stream (instead of the main stream).

## Verification Points

The sample operator (`EagerAttachedStreamAddOp`) performs the following verifications in Execute:

| # | Verification Item | Pass Condition |
|---|-------------------|----------------|
| ① | Request an attached stream by key | `RequestAttachedStream("eager_aux")` returns non-nullptr |
| ② | Kernel launch on the attached stream | `aclrtLaunchKernelV2(..., attached stream)` succeeds |
| ③ | Replay precision | z = x + y is correct in all rounds of model replay |

> Interface semantics such as same-key reuse and different-key isolation are covered by unit tests, see
> `tests/ge/ut/ge/graph/load/attached_stream_collection_unittest.cc`.

## Core Verification Flow

```text
Session::AddGraph
  └─ Model loading (DoTaskSink)
      └─ CustomTaskInfo::Distribute
          └─ EagerExecuteOp::Execute (called once)
              ├─ ctx->RequestAttachedStream("eager_aux")     → returns the framework-managed physical attached stream
              └─ aclrtLaunchKernelV2(..., attached stream)    → kernel launched on the attached stream
Session::ExecuteGraphWithStreamAsync
  └─ Model replay (tasks on the attached stream are executed repeatedly)
      └─ z = x + y precision check
```

## Quick Start

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

On success, the terminal output is:

```text
[INFO] [AttachedStream] kernel launched on attached stream 0x..., blocks=8
[INFO] [online] precision check passed (8192 elements)
[INFO] ========== Eager AttachedStream Online E2E: ALL PASS ==========
```

## Key Files

| File | Description |
|------|-------------|
| `ge/custom_op.cpp` | `EagerAttachedStreamAddOp`: requests the attached stream and launches the kernel in Execute |
| `ge/add_custom.h` | Operator proto registration (REG_OP) |
| `session_run/main.cc` | Graph construction + Session execution + multi-round precision checks |
| `add_custom_kernel/add_custom.asc` | Ascend C Add kernel source |
| `ge/utils/rtc_kernel_loader.cpp` | RTC compiles and loads the kernel |

## Operator Specification

| Item | Content |
|------|---------|
| Operator type | `EagerAttachedStreamAddOp` |
| Inputs | `x`, `y` |
| Output | `z` |
| Shape | `[8192]` float32 |
| Kernel | `add_custom` (RTC-compiled, block 1024) |
| Attached stream key | `"eager_aux"` |
