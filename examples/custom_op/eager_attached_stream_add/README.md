# Eager 自定义算子辅流申请端到端样例

本样例验证 `EagerOpExecutionContext::RequestAttachedStream` 的完整链路：Eager 自定义算子在 Execute 回调中按 key 申请框架托管的物理辅流，并将 kernel 下发到辅流上执行。

## 样例概述

- 构图入口：`GE`
- 算子编程语言：`Ascend C`（RTC 运行时编译）
- 核心链路：`RTC 编译 kernel -> GE 交付件 -> 进程内构图 -> Session 在线执行 -> 辅流 kernel 下发 -> 精度校验`
- 与其他样例的区别：本样例聚焦 `EagerOpExecutionContext::RequestAttachedStream` 接口，算子在 Execute 回调中申请框架托管辅流并将 kernel 下发到辅流（而非主流）上执行。

## 验证点

样例算子（`EagerAttachedStreamAddOp`）在 Execute 中执行以下验证：

| # | 验证项 | 通过条件 |
|---|--------|---------|
| ① | 按 key 申请辅流 | `RequestAttachedStream("eager_aux")` 返回非 nullptr |
| ② | 辅流 kernel 下发 | `aclrtLaunchKernelV2(..., 辅流)` 成功 |
| ③ | 重放精度 | 模型重放多轮后 z = x + y 全部正确 |

> 同 key 复用、不同 key 隔离等接口语义由单元测试覆盖，见
> `tests/ge/ut/ge/graph/load/attached_stream_collection_unittest.cc`。

## 核心验证链路

```text
Session::AddGraph
  └─ 模型加载（DoTaskSink）
      └─ CustomTaskInfo::Distribute
          └─ EagerExecuteOp::Execute（调用一次）
              ├─ ctx->RequestAttachedStream("eager_aux")     → 返回框架托管的物理辅流
              └─ aclrtLaunchKernelV2(..., 辅流)               → kernel 下发到辅流
Session::ExecuteGraphWithStreamAsync
  └─ 模型重放（辅流上的 task 重复执行）
      └─ z = x + y 精度校验
```

## 快速运行

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
bash run.sh
```

运行成功时终端输出：

```text
[INFO] [AttachedStream] kernel launched on attached stream 0x..., blocks=8
[INFO] [online] precision check passed (8192 elements)
[INFO] ========== Eager AttachedStream Online E2E: ALL PASS ==========
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `ge/custom_op.cpp` | `EagerAttachedStreamAddOp`：Execute 中申请辅流并下发 kernel |
| `ge/add_custom.h` | 算子 proto 注册（REG_OP） |
| `session_run/main.cc` | 构图 + Session 执行 + 多轮精度校验 |
| `add_custom_kernel/add_custom.asc` | Ascend C Add kernel 源码 |
| `ge/utils/rtc_kernel_loader.cpp` | RTC 编译和加载 kernel |

## 算子规格

| 项目 | 内容 |
|------|------|
| 算子类型 | `EagerAttachedStreamAddOp` |
| 输入 | `x`, `y` |
| 输出 | `z` |
| shape | `[8192]` float32 |
| kernel | `add_custom`（RTC 编译，block 1024） |
| 辅流 key | `"eager_aux"` |
