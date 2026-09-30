# RequestAttachedStream

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/eager\_op\_execution\_context.h\>
- 库文件：liblowering.so

## 功能说明

按 key 申请框架托管的物理辅流，供 Eager 类自定义算子在[`Execute`](../../ge/EagerExecuteOp/Execute.md)回调中申请辅流，并在辅流上下发 kernel。

- 首次申请时，框架创建物理流并托管其生命周期；同一模型执行实例内使用相同 key 的后续申请直接返回已创建的流句柄，支持跨节点、跨算子类型复用。
- 不同 key 返回不同的流句柄；不同模型执行实例之间的辅流相互隔离，即使 key 相同也不共享。

## 函数原型

```c++
rtStream RequestAttachedStream(const ge::AscendString &key)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| key | 输入 | 辅流复用 key。不能为空字符串，为空时返回nullptr。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | rtStream | 成功返回框架托管的辅流句柄；key为空、当前执行链路不支持辅流申请、或物理流创建/绑定失败时返回nullptr。 |

## 约束说明

- 主辅流之间的执行顺序由算子自行管理，如：通过 event（`aclrtRecordEvent`/`aclrtStreamWaitEvent`）自行同步，框架不保证执行顺序。
- 算子必须在[`Execute`](../../ge/EagerExecuteOp/Execute.md)返回前把辅流上的工作 join 回`GetStream()`返回的执行流（辅流 record event、执行流 wait event）。RT2 链路的内存回收只感知框架逻辑流，若不 join 就返回，本轮分配的 args 与输出内存可能在辅流 kernel 执行完成前被回收复用。
- 流的生命周期由框架管理，算子不得销毁返回的辅流（禁止调用`aclrtDestroyStream`）。
- 此接口不支持并发调用。
- `CANN-FMK-`是框架内部使用的辅流key前缀。用户自定义算子应避免以该前缀命名key，以免与框架内部申请的辅流相互覆盖。

## 调用示例

```c++
ge::graphStatus Execute(gert::EagerOpExecutionContext *ctx) override {
  // 按 key 申请框架托管的辅流
  gert::rtStream aux_stream = ctx->RequestAttachedStream("eager_aux");
  if (aux_stream == nullptr) {
    return ge::GRAPH_FAILED;
  }
  // 同 key 再次申请：同一模型实例内返回同一句柄
  gert::rtStream reuse_stream = ctx->RequestAttachedStream("eager_aux");
  if (reuse_stream != aux_stream) {
    return ge::GRAPH_FAILED;
  }

  const auto main_stream = static_cast<aclrtStream>(ctx->GetStream());
  // 主流录制 event，保证辅流上的 kernel 在输入就绪后执行
  aclError ret = aclrtRecordEvent(ev_main_to_aux, main_stream);
  if (ret != ACL_ERROR_NONE) {
    return ge::GRAPH_FAILED;
  }
  ret = aclrtStreamWaitEvent(static_cast<aclrtStream>(aux_stream), ev_main_to_aux);
  if (ret != ACL_ERROR_NONE) {
    return ge::GRAPH_FAILED;
  }

  // 将 kernel 下发到辅流（而非主流）上执行
  ret = aclrtLaunchKernelV2(func_handle, num_blocks, dev_args, sizeof(args), nullptr,
                             static_cast<aclrtStream>(aux_stream));
  if (ret != ACL_ERROR_NONE) {
    return ge::GRAPH_FAILED;
  }

  // 辅流录制 event，主流等待，保证输出在 kernel 完成后使用
  ret = aclrtRecordEvent(ev_aux_to_main, static_cast<aclrtStream>(aux_stream));
  if (ret != ACL_ERROR_NONE) {
    return ge::GRAPH_FAILED;
  }
  ret = aclrtStreamWaitEvent(main_stream, ev_aux_to_main);
  if (ret != ACL_ERROR_NONE) {
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}
```
